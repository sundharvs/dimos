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

"""Rim grasps for open containers (bins, boxes, cups) from a gravity-aligned cloud.

A centroid grasp closes on air inside an open container, and the container is
usually wider than the jaws anyway. What a parallel gripper can hold is the wall:
the jaws straddle the rim from above, closing across the wall's thickness.

The rim is the top band of the object's point cloud. Its outline is fitted as a
rotated rectangle in XY; each candidate sits on one side of that rectangle, with
the jaw closing axis normal to the side and the TCP lowered ``insertion_depth``
below the rim so the finger pads, not the tips, carry the wall. Candidates are
ranked by how close the grasp point is to the footprint centroid in XY, because
the lift torque, and so the chance of the container slipping or tipping, grows
with that lever arm. Every tunable is an RPC-settable parameter so an outer loop
can search over them without restarting the stack.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.manipulation_msgs.GraspCandidate import GraspCandidate
from dimos.msgs.manipulation_msgs.GraspCandidateArray import GraspCandidateArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Header import Header
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class RimGraspConfig(ModuleConfig):
    # Thickness of the band below the highest points that counts as the rim.
    rim_band: float = Field(default=0.03, gt=0.0)
    # How far below the rim top the TCP (fingertip frame) goes.
    insertion_depth: float = Field(default=0.03, ge=0.0)
    # Offset of the grasp point along the chosen side, as a fraction of the side
    # length from its centre (-0.5..0.5). 0 is the middle of the wall.
    along_fraction: float = Field(default=0.0, ge=-0.5, le=0.5)
    # Shift of the TCP across the wall, metres, positive = toward the outside.
    # Compensates a biased rim estimate (e.g. a lip the camera sees from above).
    across_offset: float = 0.0
    # Which sides to propose: "long", "short" or "all". Centre-of-mass distance
    # ranks within the set unless wall_select says otherwise.
    sides: str = "all"
    # How to rank the proposed sides: "lever" (closest to the footprint centroid
    # first), "highest" / "lowest" (by that wall's own top height; a scoop-front
    # bin's opening is its lowest wall), or "normal" (outward normal closest to
    # prefer_normal_xy in the cloud frame).
    wall_select: str = "lever"
    prefer_normal_xy: tuple[float, float] = (0.0, -1.0)
    # TCP never goes below this world z (the measured table clearance).
    min_z: float | None = None
    # Extra yaw added to the closing axis, radians.
    yaw_offset: float = 0.0
    # Points above the top quantile are treated as outliers (sensor speckle).
    top_quantile: float = Field(default=0.98, gt=0.5, le=1.0)
    # Minimum points in the rim band to trust the fit.
    min_rim_points: int = Field(default=30, ge=4)


class RimGraspModule(Module, GraspGenSpec):
    """Top-down rim grasps on the walls of an open container.

    The input frame's XY plane must be horizontal and its +Z axis must point up.
    """

    config: RimGraspConfig

    @rpc
    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump(
            include={
                "rim_band",
                "insertion_depth",
                "along_fraction",
                "across_offset",
                "sides",
                "wall_select",
                "prefer_normal_xy",
                "min_z",
                "yaw_offset",
                "top_quantile",
                "min_rim_points",
            }
        )

    @rpc
    def set_params(self, **params: Any) -> dict[str, Any]:
        """Update grasp parameters in place; returns the full parameter set."""
        allowed = self.get_params()
        for key, value in params.items():
            if key not in allowed:
                raise ValueError(f"unknown rim grasp parameter {key!r}")
            setattr(self.config, key, value)
        logger.info(f"RimGrasp params: {self.get_params()}")
        return self.get_params()

    @rpc
    def propose_grasps(self, object_pointcloud: PointCloud2) -> GraspCandidateArray:
        if object_pointcloud.ts is None or not math.isfinite(float(object_pointcloud.ts)):
            raise ValueError("object pointcloud must have a finite timestamp")
        if not object_pointcloud.frame_id:
            raise ValueError("object pointcloud frame_id must not be empty")
        points = object_pointcloud.points_f32()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < self.config.min_rim_points:
            raise ValueError("object pointcloud has too few XYZ points for a rim fit")
        if not np.all(np.isfinite(points)):
            raise ValueError("object pointcloud XYZ values must be finite floats in metres")

        candidates = self.rim_candidates(points)
        return GraspCandidateArray(
            Header(float(object_pointcloud.ts), object_pointcloud.frame_id),
            [GraspCandidate(pose, score=score) for pose, score in candidates],
        )

    @rpc
    def describe_rim(self, object_pointcloud: PointCloud2) -> dict[str, Any]:
        """The fitted rim rectangle, for logging and for an outer loop to inspect."""
        points = object_pointcloud.points_f32()
        fit = self._fit_rim(points)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}

    # ------------------------------------------------------------------ geometry
    def _fit_rim(self, points: NDArray[np.float32]) -> dict[str, Any]:
        z = points[:, 2]
        top = float(np.quantile(z, self.config.top_quantile))
        band = points[z >= top - self.config.rim_band]
        if len(band) < self.config.min_rim_points:
            raise ValueError(
                f"only {len(band)} points in the top {self.config.rim_band * 1000:.0f} mm band"
            )
        xy = band[:, :2].astype(np.float64)
        centroid_all = np.mean(points[:, :2], axis=0)
        # Minimum-area rotated rectangle around the rim outline.
        rect = _min_area_rect(xy)
        return {
            "rim_top_z": top,
            "n_rim_points": len(band),
            "rect_center": rect["center"],
            "rect_axes": rect["axes"],  # 2x2, rows are unit axis directions
            "rect_extents": rect["extents"],  # full lengths along the axes
            "footprint_centroid": centroid_all,
            "object_min_z": float(z.min()),
        }

    def rim_candidates(self, points: NDArray[np.float32]) -> list[tuple[Pose, float]]:
        fit = self._fit_rim(points)
        center = np.asarray(fit["rect_center"])
        axes = np.asarray(fit["rect_axes"])
        extents = np.asarray(fit["rect_extents"])
        centroid = np.asarray(fit["footprint_centroid"])
        top = float(fit["rim_top_z"])

        long_axis = int(np.argmax(extents))
        sides: list[tuple[np.ndarray, np.ndarray, float, float, str]] = []
        for axis_index in (0, 1):
            normal = axes[axis_index]
            along = axes[1 - axis_index]
            half = extents[axis_index] / 2.0
            length = extents[1 - axis_index]
            kind = "short" if axis_index == long_axis else "long"
            # A side whose normal is axis i is at +-half along that normal.
            for sign in (1.0, -1.0):
                sides.append((sign * normal, along, half, length, kind))
        wanted = self.config.sides
        xy_all = points[:, :2].astype(np.float64)
        out: list[tuple[Pose, float]] = []
        for normal, along, half, length, kind in sides:
            if wanted != "all" and kind != wanted:
                continue
            point = (
                center
                + normal * (half + self.config.across_offset)
                + along * (self.config.along_fraction * length)
            )
            # Walls differ in height (a scoop-front bin): measure this wall's own
            # top from the rim-band points along the whole side. A side the
            # camera barely saw falls back to the overall rim top rather than
            # to the floor inside the container.
            side_dist = np.abs((xy_all - center) @ normal - half)
            along_pos = np.abs((xy_all - center) @ along)
            near = (side_dist < 0.015) & (along_pos < length / 2.0)
            near &= points[:, 2] >= top - 2.0 * self.config.rim_band
            if np.count_nonzero(near) >= self.config.min_rim_points:
                wall_top = float(np.quantile(points[near, 2], self.config.top_quantile))
                wall_top = max(wall_top, top - 2.0 * self.config.rim_band)
            else:
                wall_top = top
            z = wall_top - self.config.insertion_depth
            if self.config.min_z is not None:
                z = max(z, self.config.min_z)
            # Jaw closing axis is the gripper Y; the tool points down (roll pi).
            # Gripper Y after roll pi about X is still body Y, so yaw puts the
            # closing axis onto the wall normal.
            yaw = (
                math.atan2(float(normal[1]), float(normal[0]))
                - math.pi / 2.0
                + self.config.yaw_offset
            )
            yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
            pose = Pose(
                Vector3(float(point[0]), float(point[1]), float(z)),
                Quaternion.from_euler(Vector3(-math.pi, 0.0, yaw)),
            )
            lever = float(np.linalg.norm(point - centroid))
            if self.config.wall_select == "highest":
                score = wall_top
            elif self.config.wall_select == "lowest":
                score = -wall_top
            elif self.config.wall_select == "normal":
                pref = np.asarray(self.config.prefer_normal_xy, dtype=float)
                pref = pref / max(np.linalg.norm(pref), 1e-9)
                score = float(np.dot(normal, pref))
            else:
                score = 1.0 / (1.0 + lever * 10.0)
            out.append((pose, score))
        out.sort(key=lambda item: -item[1])
        return out

    @rpc
    def describe_walls(self, object_pointcloud: PointCloud2) -> list[dict[str, Any]]:
        """Each candidate wall: outward normal, kind, its own top height and lever arm."""
        points = object_pointcloud.points_f32()
        walls = []
        for pose, score in self.rim_candidates(points):
            walls.append(
                {
                    "xyz": [pose.position.x, pose.position.y, pose.position.z],
                    "yaw": float(pose.orientation.to_euler().z),
                    "score": score,
                }
            )
        return walls


def _min_area_rect(xy: NDArray[np.float64]) -> dict[str, Any]:
    """Minimum-area bounding rectangle of 2D points via the convex hull edges."""
    import cv2

    pts = xy.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    angle = math.radians(angle_deg)
    axis0 = np.array([math.cos(angle), math.sin(angle)])
    axis1 = np.array([-math.sin(angle), math.cos(angle)])
    return {
        "center": np.array([cx, cy], dtype=float),
        "axes": np.stack([axis0, axis1]),
        "extents": np.array([w, h], dtype=float),
    }


rim_grasp = RimGraspModule.blueprint
