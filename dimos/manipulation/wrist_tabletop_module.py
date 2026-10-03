# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrist camera over a tabletop: colour-segmented object clouds and masking-tape slots.

Two jobs the container skills need that the open-vocabulary scene registry does
not do well:

* ``scan_object_cloud``: the planning-frame point cloud of the largest blob in an
  HSV range, straight from the latest aligned colour + depth frame. The scene
  registry accumulates the cloud of anything re-detected within its distance
  threshold of a stored object (and keeps every first sighting), so after a
  container is set down near an old pose its cloud is a union of poses. A fresh
  per-frame segmentation has no memory.
* ``add_tape_view`` / ``fit_slots``: masking-tape pixels from one or more wrist
  views, back-projected through the depth image onto the table, fitted as a grid
  of lines along the planning frame's X and Y axes. Three slots between four
  lines along Y and two lines along X is the layout this was learned on; the
  fitter returns whatever grid it finds and names the slots along +X.

The camera's planning-frame pose comes from the TF buffer (``tf`` port), so the
hand-eye transform must be published (ManipulationModule ``static_transforms``).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

Hsv = tuple[int, int, int]


class WristTabletopConfig(ModuleConfig):
    planning_frame: str = "world"
    # HSV (OpenCV ranges: H 0..179) of the object scan_object_cloud looks for by default.
    object_hsv_low: Hsv = (15, 120, 100)
    object_hsv_high: Hsv = (40, 255, 255)
    # Planning-frame Z band the object's points may occupy (drops depth speckle).
    object_z_range: tuple[float, float] = (-0.04, 0.16)
    min_object_points: int = 400
    # Masking tape on a wooden table reads bluish-grey next to the orange wood.
    tape_hsv_low: Hsv = (80, 20, 120)
    tape_hsv_high: Hsv = (125, 110, 255)
    # Planning-frame Z band of the table surface.
    table_z_range: tuple[float, float] = (-0.05, 0.03)
    # Depth range (metres along the optical axis) accepted from the sensor.
    depth_range: tuple[float, float] = (0.15, 1.5)
    # Pixel stride when back-projecting masks.
    stride: int = Field(default=2, ge=1)
    # Slot grid: names given to the cells between consecutive X lines, from -X to +X,
    # and the half width of the tape (the usable slot is between tape inner edges).
    slot_names: list[str] = Field(default_factory=lambda: ["left", "middle", "right"])
    tape_half_width: float = 0.012
    # Minimum fraction of the strongest histogram peak a line must reach, and the
    # minimum separation between lines.
    line_min_fraction: float = 0.25
    line_min_separation: float = 0.04
    tf_time_tolerance: float = 0.2


def color_mask(bgr: NDArray[np.uint8], low: Hsv, high: Hsv) -> NDArray[np.uint8]:
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return np.asarray(opened, dtype=np.uint8)


def largest_component(mask: NDArray[np.uint8]) -> NDArray[np.bool_] | None:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == index, dtype=bool)


def back_project(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    world_from_optical: NDArray[np.float64],
    depth_range: tuple[float, float],
    stride: int,
) -> NDArray[np.float32]:
    """Planning-frame XYZ of the masked pixels that have a valid depth."""
    ys, xs = np.nonzero(mask[::stride, ::stride])
    ys = ys * stride
    xs = xs * stride
    z = depth_m[ys, xs]
    ok = (z > depth_range[0]) & (z < depth_range[1])
    xs, ys, z = xs[ok], ys[ok], z[ok]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    optical = np.c_[(xs - cx) / fx * z, (ys - cy) / fy * z, z, np.ones(len(z))]
    world = (world_from_optical @ optical.T).T[:, :3]
    return np.asarray(world, dtype=np.float32)


def histogram_peaks(
    values: NDArray[np.floating],
    low: float,
    high: float,
    bin_width: float = 0.005,
    min_fraction: float = 0.25,
    separation: float = 0.04,
) -> list[tuple[float, int]]:
    """Local maxima of a 1-D histogram, each refined to the median of the values
    within 1.5 cm of the bin, strongest first then sorted by position."""
    if len(values) == 0:
        return []
    counts, edges = np.histogram(values, bins=np.arange(low, high + bin_width, bin_width))
    centres = (edges[:-1] + edges[1:]) / 2
    peaks: list[tuple[float, int]] = []
    for index in np.argsort(counts)[::-1]:
        if counts[index] < max(min_fraction * counts.max(), 30):
            break
        if all(abs(centres[index] - position) >= separation for position, _ in peaks):
            selected = np.abs(values - centres[index]) < 0.015
            peaks.append((float(np.median(values[selected])), int(selected.sum())))
    return sorted(peaks)


def fit_slot_grid(
    points: NDArray[np.floating],
    slot_names: list[str],
    z_range: tuple[float, float] = (-0.05, 0.03),
    min_fraction: float = 0.25,
    separation: float = 0.04,
) -> dict[str, Any]:
    """Lines along Y (at X positions) and along X (at Y positions) -> named slots.

    Slots are the cells between consecutive X lines, named from -X to +X, and
    span the outermost two Y lines. ``y`` is (far, near) with near = larger Y.
    """
    table = points[(points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])]
    if len(table) == 0:
        return {"x_lines": [], "y_lines": [], "slots": {}}
    x_lines = histogram_peaks(
        table[:, 0], -1.0, 1.0, min_fraction=min_fraction, separation=separation
    )
    on_x_line = np.zeros(len(table), dtype=bool)
    for x, _ in x_lines:
        on_x_line |= np.abs(table[:, 0] - x) < 0.02
    y_lines = histogram_peaks(
        table[~on_x_line, 1], -1.5, 1.5, min_fraction=min_fraction, separation=separation
    )
    xs = [x for x, _ in x_lines]
    ys = [y for y, _ in y_lines]
    result: dict[str, Any] = {
        "x_lines": xs,
        "y_lines": ys,
        "x_counts": x_lines,
        "y_counts": y_lines,
        "table_z": float(np.median(table[:, 2])),
        "slots": {},
    }
    if len(xs) >= 2 and len(ys) >= 2:
        for index, name in enumerate(slot_names[: len(xs) - 1]):
            x0, x1 = xs[index], xs[index + 1]
            y_far, y_near = ys[0], ys[-1]
            result["slots"][name] = {
                "x": [x0, x1],
                "y": [y_far, y_near],
                "center": [(x0 + x1) / 2.0, (y_far + y_near) / 2.0],
                "width": x1 - x0,
                "length": y_near - y_far,
            }
    return result


class WristTabletopModule(Module):
    """Colour-blob object clouds and tape-slot mapping from the wrist camera."""

    config: WristTabletopConfig

    color_image: In[Image]
    depth_image: In[Image]
    camera_info: In[CameraInfo]
    tf: In[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._color: Image | None = None
        self._depth: Image | None = None
        self._info: CameraInfo | None = None
        self._tape_points: list[NDArray[np.float32]] = []
        self._slots: dict[str, Any] = {}

    @rpc
    def start(self) -> None:
        super().start()
        self.color_image.subscribe(self._on_color)
        self.depth_image.subscribe(self._on_depth)
        self.camera_info.subscribe(self._on_info)

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_color(self, image: Image) -> None:
        with self._lock:
            self._color = image

    def _on_depth(self, image: Image) -> None:
        with self._lock:
            self._depth = image

    def _on_info(self, info: CameraInfo) -> None:
        with self._lock:
            self._info = info

    # frame access

    def _frame(self) -> tuple[Image, Image, CameraInfo] | None:
        with self._lock:
            if self._color is None or self._depth is None or self._info is None:
                return None
            return self._color, self._depth, self._info

    def _world_from_optical(self, image: Image) -> NDArray[np.float64] | None:
        transform = self.tfbuffer.get(
            self.config.planning_frame,
            image.frame_id,
            image.ts,
            self.config.tf_time_tolerance,
            forward_tolerance=self.config.tf_time_tolerance,
        )
        if transform is None:
            logger.warning(
                f"WristTabletop: no transform {self.config.planning_frame} <- {image.frame_id}"
            )
            return None
        q = transform.rotation
        t = transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        matrix = np.eye(4)
        matrix[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        matrix[:3, 3] = [t.x, t.y, t.z]
        return matrix

    def _masked_world_points(
        self, low: Hsv, high: Hsv, largest_only: bool
    ) -> NDArray[np.float32] | None:
        frame = self._frame()
        if frame is None:
            logger.warning("WristTabletop: no camera frames yet")
            return None
        color, depth, info = frame
        world_from_optical = self._world_from_optical(color)
        if world_from_optical is None:
            return None
        bgr = color.to_opencv()
        depth_m = depth.to_opencv().astype(np.float64)
        if depth_m.max() > 20.0:  # 16-bit millimetres
            depth_m = depth_m / 1000.0
        if depth_m.shape[:2] != bgr.shape[:2]:
            logger.warning("WristTabletop: depth is not aligned to colour")
            return None
        mask = color_mask(bgr, low, high)
        if largest_only:
            component = largest_component(mask)
            if component is None:
                return None
            mask_bool = component
        else:
            mask_bool = mask > 0
        intrinsics = np.array(info.K, dtype=float).reshape(3, 3)
        return back_project(
            mask_bool,
            depth_m,
            intrinsics,
            world_from_optical,
            self.config.depth_range,
            self.config.stride,
        )

    # rpcs

    @rpc
    def scan_object_cloud(
        self,
        hsv_low: tuple[int, int, int] | None = None,
        hsv_high: tuple[int, int, int] | None = None,
    ) -> PointCloud2 | None:
        """Planning-frame cloud of the largest blob in the HSV range (default: the configured object)."""
        low = tuple(hsv_low) if hsv_low is not None else self.config.object_hsv_low
        high = tuple(hsv_high) if hsv_high is not None else self.config.object_hsv_high
        points = self._masked_world_points(low, high, largest_only=True)  # type: ignore[arg-type]
        if points is None:
            return None
        z0, z1 = self.config.object_z_range
        points = points[(points[:, 2] > z0) & (points[:, 2] < z1)]
        if len(points) < self.config.min_object_points:
            return None
        frame = self._frame()
        stamp = float(frame[0].ts) if frame is not None and frame[0].ts else None
        return PointCloud2.from_numpy(points, frame_id=self.config.planning_frame, timestamp=stamp)

    @rpc
    def add_tape_view(self) -> int:
        """Back-project the tape pixels of the current frame onto the table; returns points kept."""
        points = self._masked_world_points(
            self.config.tape_hsv_low, self.config.tape_hsv_high, largest_only=False
        )
        if points is None:
            return 0
        z0, z1 = self.config.table_z_range
        points = points[(points[:, 2] > z0) & (points[:, 2] < z1)]
        self._tape_points.append(points)
        return len(points)

    @rpc
    def clear_tape_views(self) -> None:
        self._tape_points = []

    @rpc
    def fit_slots(self) -> dict[str, Any]:
        """Fit the slot grid to all tape views added so far and remember it."""
        if not self._tape_points:
            return {"x_lines": [], "y_lines": [], "slots": {}}
        points = np.concatenate(self._tape_points)
        result = fit_slot_grid(
            points,
            self.config.slot_names,
            self.config.table_z_range,
            self.config.line_min_fraction,
            self.config.line_min_separation,
        )
        result["n_points"] = len(points)
        result["tape_half_width"] = self.config.tape_half_width
        self._slots = result
        logger.info(
            f"WristTabletop: slots {list(result['slots'])} from {len(points)} tape points, "
            f"x lines {np.round(result['x_lines'], 3).tolist()}, y lines {np.round(result['y_lines'], 3).tolist()}"
        )
        return result

    @rpc
    def get_slots(self) -> dict[str, Any]:
        return self._slots


wrist_tabletop = WristTabletopModule.blueprint
