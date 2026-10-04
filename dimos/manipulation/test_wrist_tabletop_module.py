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

import numpy as np
import pytest

from dimos.manipulation.wrist_tabletop_module import (
    WristTabletopConfig,
    WristTabletopModule,
    back_project,
    color_mask,
    fit_slot_grid,
    histogram_peaks,
    largest_component,
)


def _tape_grid(x_lines: list[float], y_lines: list[float], z: float = -0.015) -> np.ndarray:
    rng = np.random.default_rng(1)
    points = []
    for x in x_lines:
        for y in np.linspace(min(y_lines), max(y_lines), 400):
            points.append((x + rng.normal(0, 0.004), y, z))
    for y in y_lines:
        for x in np.linspace(min(x_lines), max(x_lines), 400):
            points.append((x, y + rng.normal(0, 0.004), z))
    return np.asarray(points, dtype=np.float32)


def test_histogram_peaks_find_the_lines() -> None:
    values = np.concatenate(
        [np.random.default_rng(2).normal(c, 0.004, 500) for c in (-0.3, -0.1, 0.1)]
    )
    peaks = histogram_peaks(values, -0.5, 0.5)
    assert [round(p, 2) for p, _ in peaks] == [-0.3, -0.1, 0.1]


def test_fit_slot_grid_names_three_slots_from_minus_x() -> None:
    x_lines = [-0.31, -0.13, 0.04, 0.20]
    y_lines = [-0.71, -0.38]
    grid = fit_slot_grid(_tape_grid(x_lines, y_lines), ["left", "middle", "right"])
    assert [round(x, 2) for x in grid["x_lines"]] == x_lines
    assert [round(y, 2) for y in grid["y_lines"]] == y_lines
    assert list(grid["slots"]) == ["left", "middle", "right"]
    middle = grid["slots"]["middle"]
    assert middle["center"] == pytest.approx([-0.045, -0.545], abs=0.01)
    assert middle["y"][0] < middle["y"][1]  # (far, near)
    assert grid["table_z"] == pytest.approx(-0.015, abs=0.002)


def test_fit_slot_grid_without_a_grid_is_empty() -> None:
    grid = fit_slot_grid(np.zeros((0, 3), dtype=np.float32), ["a"])
    assert grid["slots"] == {}
    # a single line is not a grid either
    grid = fit_slot_grid(_tape_grid([0.0], [-0.5, -0.4]), ["a"])
    assert grid["slots"] == {}


def test_color_mask_and_largest_component() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :] = (40, 90, 150)  # orange-ish wood (BGR)
    image[5:15, 5:25] = (30, 200, 230)  # a yellow patch
    image[30:34, 50:54] = (30, 200, 230)  # a smaller yellow speck
    mask = color_mask(image, (15, 120, 100), (40, 255, 255))
    component = largest_component(mask)
    assert component is not None
    assert component[10, 15] and not component[32, 52]
    assert largest_component(np.zeros((40, 60), dtype=np.uint8)) is None


def test_back_project_puts_the_image_centre_on_the_optical_axis() -> None:
    depth = np.full((40, 60), 0.5)
    mask = np.zeros((40, 60), dtype=bool)
    mask[20, 30] = True
    intrinsics = np.array([[100.0, 0.0, 30.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])
    # camera looking straight down from z = 0.5: optical z -> world -z
    world_from_optical = np.array(
        [[1.0, 0.0, 0.0, 0.1], [0.0, -1.0, 0.0, 0.2], [0.0, 0.0, -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
    )
    points = back_project(mask, depth, intrinsics, world_from_optical, (0.1, 1.0), 1)
    assert points.shape == (1, 3)
    assert points[0] == pytest.approx([0.1, 0.2, 0.0], abs=1e-6)


class _Frame:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def to_opencv(self) -> np.ndarray:
        return self._array


def _tabletop(color: np.ndarray | None) -> WristTabletopModule:
    module = WristTabletopModule.__new__(WristTabletopModule)
    object.__setattr__(module, "config", WristTabletopConfig())
    frame = None if color is None else (_Frame(color), None, None)
    object.__setattr__(module, "_frame", lambda: frame)
    return module


def test_object_view_fraction_is_the_share_of_the_frame_in_the_object_colour() -> None:
    yellow_bgr = (0, 220, 255)
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :45] = yellow_bgr
    fraction = WristTabletopModule.object_view_fraction(_tabletop(image))
    assert fraction == pytest.approx(0.75, abs=0.02)


def test_object_view_fraction_without_the_object_or_a_frame() -> None:
    blank = np.zeros((40, 60, 3), dtype=np.uint8)
    assert WristTabletopModule.object_view_fraction(_tabletop(blank)) == 0.0
    assert WristTabletopModule.object_view_fraction(_tabletop(None)) is None
