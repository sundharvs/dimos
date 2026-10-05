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

import math

import numpy as np
from numpy.typing import NDArray
import pytest

from dimos.manipulation.container_grasp_module import height_under, rim_offset, wall_pinch


def _bin(
    cx: float, cy: float, yaw: float, length: float = 0.29, width: float = 0.115
) -> NDArray[np.float64]:
    """A bin's cloud seen from above: floor, and walls 9.5 cm tall."""
    rng = np.random.default_rng(0)
    floor = np.column_stack(
        [
            rng.uniform(-length / 2, length / 2, 1500),
            rng.uniform(-width / 2, width / 2, 1500),
            np.zeros(1500),
        ]
    )
    walls = []
    for side in (-1.0, 1.0):
        walls.append(
            np.column_stack(
                [
                    rng.uniform(-length / 2, length / 2, 400),
                    np.full(400, side * width / 2),
                    rng.uniform(0.0, 0.095, 400),
                ]
            )
        )
        walls.append(
            np.column_stack(
                [
                    np.full(150, side * length / 2),
                    rng.uniform(-width / 2, width / 2, 150),
                    rng.uniform(0.0, 0.095, 150),
                ]
            )
        )
    points = np.vstack([floor, *walls])
    c, s = math.cos(yaw), math.sin(yaw)
    xy = points[:, :2] @ np.array([[c, s], [-s, c]])
    return np.column_stack([xy[:, 0] + cx, xy[:, 1] + cy, points[:, 2]])


def test_wall_pinch_takes_the_long_wall_nearest_the_base() -> None:
    pinch = wall_pinch(_bin(0.30, -0.015, math.pi / 2))

    assert pinch is not None
    assert pinch.x == pytest.approx(0.30 - 0.115 / 2, abs=0.006)
    assert pinch.y == pytest.approx(-0.015, abs=0.01)
    assert pinch.yaw == pytest.approx(math.pi / 2, abs=0.05)
    assert pinch.rim_z == pytest.approx(0.095, abs=0.005)
    assert pinch.length == pytest.approx(0.29, abs=0.02)
    assert pinch.width == pytest.approx(0.115, abs=0.01)


def test_wall_pinch_follows_a_turned_bin() -> None:
    yaw = math.radians(60)
    pinch = wall_pinch(_bin(0.30, 0.05, yaw))

    assert pinch is not None
    assert pinch.yaw == pytest.approx(yaw, abs=0.05)
    # Half a width from the centre, on the base's side of it.
    assert math.hypot(pinch.x - 0.30, pinch.y - 0.05) == pytest.approx(0.115 / 2, abs=0.008)
    assert math.hypot(pinch.x, pinch.y) < math.hypot(0.30, 0.05)


def test_wall_pinch_needs_points() -> None:
    assert wall_pinch(np.zeros((10, 3))) is None


def test_rim_offset_finds_the_near_rim_from_inside_the_bin() -> None:
    # Hovering 3.7 cm inside the near wall of a bin along y, base at the origin.
    cloud = _bin(0.25 + 0.115 / 2, 0.0, math.pi / 2)

    offset = rim_offset(cloud, 0.287, 0.0, math.pi / 2, 0.073, toward_base=-1.0)

    # The across axis of a wall along +y is +x, so the near rim is at -3.7 cm.
    assert offset == pytest.approx(-0.037, abs=0.003)


def test_rim_offset_is_none_over_bare_table() -> None:
    table = np.column_stack([np.linspace(0.1, 0.5, 500), np.zeros(500), np.zeros(500)])

    assert rim_offset(table, 0.3, 0.0, math.pi / 2, 0.073, toward_base=-1.0) is None


def test_height_under_tells_a_held_floor_from_the_table() -> None:
    table = np.column_stack([np.linspace(0.1, 0.5, 100), np.zeros(100), np.zeros(100)])
    floor = np.column_stack([np.linspace(0.18, 0.3, 300), np.zeros(300), np.full(300, 0.11)])

    assert height_under(table, 0.25, 0.0, 0.1) == (0.0, 49)
    held = height_under(np.vstack([table, floor]), 0.25, 0.0, 0.1)
    assert held is not None and held[0] == pytest.approx(0.11)
    assert height_under(table, 2.0, 0.0, 0.1) is None
