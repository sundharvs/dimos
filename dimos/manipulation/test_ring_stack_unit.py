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

from collections.abc import Iterator
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from dimos.manipulation.ring_stack_module import (
    Held,
    RingStackModule,
    carry_pose,
    fit_ring,
    fit_rod_top,
    is_direct,
    leaning_orientation,
    rim_hold,
)

OVERVIEW = [0.0, 0.5, -0.9, 0.0, 1.2, 0.0]


def _torus(
    centre: tuple[float, float], ring_radius: float, tube_radius: float, table_z: float
) -> np.ndarray:
    """The upper half of a ring lying on a table, plus the table seen through its hole."""
    around, across = np.meshgrid(np.linspace(0, 2 * np.pi, 90), np.linspace(0, np.pi, 30))
    distance = ring_radius + tube_radius * np.cos(across)
    ring = np.column_stack(
        [
            centre[0] + (distance * np.cos(around)).ravel(),
            centre[1] + (distance * np.sin(around)).ravel(),
            table_z + tube_radius + (tube_radius * np.sin(across)).ravel(),
        ]
    )
    hole = np.column_stack(
        [
            centre[0] + np.linspace(-0.01, 0.01, 40),
            centre[1] + np.linspace(-0.01, 0.01, 40),
            np.full(40, table_z),
        ]
    )
    return np.vstack([ring, hole]).astype(np.float32)


def _held(pick_lean: float, side: int = 1, half_turn: bool = False) -> Held:
    """A ring of radius 32 mm gripped at the origin's bearing with a given lean."""
    to_tool = leaning_orientation(0.0, pick_lean, half_turn).to_rotation_matrix().T
    offset = to_tool @ np.array([0.0, side * 0.032, -0.011])
    normal = to_tool @ np.array([0.0, 0.0, 1.0])
    return Held(tuple(offset), tuple(normal), 0.043, 0.024, half_turn, 0.7, 0.33)


def _plan(path: list[list[float]], succeeded: bool = True) -> Any:
    points = [SimpleNamespace(positions=positions) for positions in path]
    plan = SimpleNamespace(trajectory=SimpleNamespace(points=points))
    return SimpleNamespace(succeeded=succeeded, message="", plan=plan)


def test_fit_ring_finds_centre_and_tube_circle_despite_the_hole() -> None:
    ring = fit_ring(_torus((0.3, -0.1), 0.032, 0.011, table_z=-0.02))

    assert ring is not None
    assert (ring.x, ring.y) == pytest.approx((0.3, -0.1), abs=0.001)
    assert ring.grip_radius == pytest.approx(0.032, abs=0.002)
    assert ring.outer_radius == pytest.approx(0.043, abs=0.002)
    assert ring.top_z == pytest.approx(0.002, abs=0.001)


def test_fit_ring_rejects_a_solid_disc() -> None:
    radius, angle = np.meshgrid(np.linspace(0, 0.04, 30), np.linspace(0, 2 * np.pi, 60))
    disc = np.column_stack(
        [(radius * np.cos(angle)).ravel(), (radius * np.sin(angle)).ravel(), np.zeros(radius.size)]
    )

    assert fit_ring(disc.astype(np.float32)) is None


def test_fit_rod_top_ignores_the_rods_side() -> None:
    radius, angle = np.meshgrid(np.linspace(0, 0.018, 12), np.linspace(0, 2 * np.pi, 40))
    face = np.column_stack(
        [
            0.4 + (radius * np.cos(angle)).ravel(),
            0.05 + (radius * np.sin(angle)).ravel(),
            np.full(radius.size, 0.17),
        ]
    )
    # Only the side facing the camera is seen, which would pull a mean toward it.
    side = np.column_stack(
        [np.full(60, 0.382), np.linspace(0.04, 0.06, 60), np.linspace(0.0, 0.16, 60)]
    )

    rod = fit_rod_top(np.vstack([face, side]).astype(np.float32))

    assert rod is not None
    assert (rod.x, rod.y, rod.top_z) == pytest.approx((0.4, 0.05, 0.17), abs=0.002)


@pytest.mark.parametrize("side", [1, -1])
def test_rim_hold_puts_the_jaw_axis_through_the_rings_centre(side: int) -> None:
    centre = (0.3, 0.1)

    hold = rim_hold(centre, 0.032, 0.0, side)

    assert hold is not None
    to_centre = np.array([centre[0] - hold.x, centre[1] - hold.y])
    outward = np.array([math.cos(hold.bearing), math.sin(hold.bearing)])
    assert np.linalg.norm(to_centre) == pytest.approx(0.032)
    assert to_centre @ outward == pytest.approx(0.0, abs=1e-9)
    assert hold.bearing == pytest.approx(math.atan2(hold.y, hold.x))
    # Seen from the base, the centre is to the left of the tool point for +1.
    assert side * (outward[0] * to_centre[1] - outward[1] * to_centre[0]) > 0


def test_rim_hold_refuses_a_ring_around_the_base() -> None:
    assert rim_hold((0.01, 0.0), 0.032, 0.0, 1) is None


@pytest.mark.parametrize("half_turn", [False, True])
def test_carry_pose_keeps_a_ring_level_when_picked_with_the_carry_lean(half_turn: bool) -> None:
    lean = math.radians(25.0)
    held = _held(lean, half_turn=half_turn)

    pose = carry_pose(held, 0.37, 0.03, top_z=0.17, gap=0.004, lean=lean)

    assert pose.tilt == pytest.approx(0.0, abs=1e-6)
    rotation = leaning_orientation(pose.bearing, lean, half_turn).to_rotation_matrix()
    centre = np.array([pose.x, pose.y, pose.z]) + rotation @ np.asarray(held.offset)
    assert centre[:2] == pytest.approx([0.37, 0.03], abs=1e-6)
    # Level, the ring's bottom is half its height under its centre.
    assert centre[2] - held.height / 2 == pytest.approx(0.174)
    assert pose.bearing == pytest.approx(math.atan2(pose.y, pose.x), abs=1e-6)


def test_carry_pose_raises_a_tilted_ring_until_its_low_side_clears() -> None:
    lean = math.radians(22.0)
    held = _held(0.0)

    pose = carry_pose(held, 0.37, 0.03, top_z=0.17, gap=0.004, lean=lean)

    assert pose.tilt == pytest.approx(lean)
    rotation = leaning_orientation(pose.bearing, lean).to_rotation_matrix()
    centre = np.array([pose.x, pose.y, pose.z]) + rotation @ np.asarray(held.offset)
    assert centre[:2] == pytest.approx([0.37, 0.03], abs=1e-6)
    low_side = centre[2] - held.outer_radius * math.sin(lean) - held.height / 2 * math.cos(lean)
    assert low_side == pytest.approx(0.174)


def test_is_direct_accepts_a_straight_path_and_rejects_a_detour() -> None:
    start, goal = [0.5, 0.9, -0.5, 0.0, 0.8, 0.0], [0.85, 0.9, -0.5, 0.0, 0.8, 0.0]
    halfway = [0.68, 0.9, -0.5, 0.0, 0.8, 0.0]
    # What the planner returned for a 20 degree turn of the base on 2026-10-03.
    detour = [-0.9, 1.6, -0.3, 0.4, 0.2, 0.0]

    assert is_direct(_plan([start, halfway, goal]), slack=0.2)
    assert not is_direct(_plan([start, detour, goal]), slack=0.2)
    assert not is_direct(SimpleNamespace(plan=None), slack=0.2)


@pytest.fixture
def module() -> Iterator[RingStackModule]:
    instance = RingStackModule(
        overview_joints=OVERVIEW,
        ring_view_joints=OVERVIEW,
        ring_view_distance=0.3,
        rod_view_joints=OVERVIEW,
        rod_view_distance=0.4,
        settle_timeout=0.0,
        view_settle=0.0,
    )
    instance._scene = MagicMock()
    instance._manipulation = MagicMock()
    instance._world = MagicMock()
    yield instance
    instance.stop()


def _group() -> Any:
    return SimpleNamespace(id="arm", joint_names=tuple(f"joint{i}" for i in range(1, 7)))


def test_a_detour_is_not_executed(module: RingStackModule) -> None:
    manipulation: Any = module._manipulation
    manipulation.plan_to_joints.return_value = _plan(
        [[0.5, 0.9, -0.5, 0.0, 0.8, 0.0], [-0.9, 1.6, -0.3, 0.4, 0.2, 0.0], [0.85] + [0.0] * 5]
    )

    failure = module._move_joints(_group(), [0.85, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert failure is not None
    assert failure.error_code == "PLANNING_FAILED"
    manipulation.clear_planned_path.assert_called_once()
    manipulation.execute.assert_not_called()


def test_a_direct_plan_is_executed(module: RingStackModule) -> None:
    manipulation: Any = module._manipulation
    manipulation.plan_to_joints.return_value = _plan([[0.5] + [0.0] * 5, [0.85] + [0.0] * 5])
    manipulation.execute.return_value = SimpleNamespace(succeeded=True, message="")

    assert module._move_joints(_group(), [0.85, 0.0, 0.0, 0.0, 0.0, 0.0]) is None
    manipulation.execute.assert_called_once()


@pytest.mark.parametrize(("jaws", "slipping"), [(0.33, False), (0.31, False), (0.27, True)])
def test_jaws_closing_on_a_held_ring_mean_it_is_slipping(
    module: RingStackModule, jaws: float, slipping: bool
) -> None:
    manipulation: Any = module._manipulation
    manipulation.get_state.return_value = SimpleNamespace(
        groups={"arm": SimpleNamespace(gripper_position=jaws)}
    )

    failure = module._slipping(_group(), _held(0.0))

    assert (failure is not None) == slipping
