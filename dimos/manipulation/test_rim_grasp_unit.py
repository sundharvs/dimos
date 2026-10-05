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
from numpy.typing import NDArray
import pytest

from dimos.manipulation.grasp_verification import GripperSettle
from dimos.manipulation.rim_grasp_module import (
    RimGraspModule,
    RimGraspModuleConfig,
    check_held,
    find_rim,
    rim_grasp_pose,
)

Points = NDArray[np.float64]

CONFIG = RimGraspModuleConfig(settle_time=0.0)


def _table() -> Points:
    xs, ys = np.meshgrid(np.arange(0.15, 0.45, 0.01), np.arange(-0.2, 0.2, 0.01))
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])


def _wall(x: float, y0: float, y1: float, bottom: float, top: float) -> Points:
    ys, zs = np.meshgrid(np.arange(y0, y1, 0.004), np.arange(bottom, top + 1e-9, 0.004))
    return np.column_stack([np.full(ys.size, x), ys.ravel(), zs.ravel()])


def _bin_on_table() -> Points:
    return np.vstack([_table(), _wall(0.30, -0.08, 0.10, 0.0, 0.10)])


def _bin_lifted(lift: float) -> Points:
    return np.vstack([_table(), _wall(0.30, -0.08, 0.10, lift, 0.10 + lift)])


def test_find_rim_returns_the_middle_of_the_top_edge() -> None:
    rim = find_rim(_bin_on_table(), CONFIG)

    assert rim is not None
    assert rim.x == pytest.approx(0.30, abs=0.005)
    assert rim.y == pytest.approx(0.01, abs=0.01)
    assert rim.rim_z == pytest.approx(0.10, abs=0.01)
    assert abs(rim.wall_yaw) == pytest.approx(math.pi / 2, abs=0.05)
    assert rim.length == pytest.approx(0.16, abs=0.03)


def test_find_rim_follows_a_rotated_wall() -> None:
    cloud = _bin_on_table()
    turn = np.array([[math.cos(0.4), -math.sin(0.4)], [math.sin(0.4), math.cos(0.4)]])
    cloud[:, :2] = cloud[:, :2] @ turn.T

    rim = find_rim(cloud, CONFIG)

    assert rim is not None
    assert rim.wall_yaw == pytest.approx(0.4 - math.pi / 2, abs=0.05)


def test_find_rim_ignores_a_bare_table_and_far_points() -> None:
    far = _wall(2.0, -0.1, 0.1, 0.0, 0.1)

    assert find_rim(_table(), CONFIG) is None
    assert find_rim(np.vstack([_table(), far]), CONFIG) is None
    assert find_rim(np.empty((0, 3)), CONFIG) is None


def test_check_held_tells_a_lifted_container_from_one_left_behind() -> None:
    assert check_held(_bin_lifted(CONFIG.lift_height), 0.30, 0.0, 0.10, CONFIG).held
    assert not check_held(_bin_on_table(), 0.30, 0.0, 0.10, CONFIG).held
    assert not check_held(_table(), 0.30, 0.0, 0.10, CONFIG).held


def test_find_rim_prefers_a_long_wall_over_a_taller_short_one() -> None:
    ys, zs = np.meshgrid(np.arange(0.25, 0.33, 0.004), np.arange(0.0, 0.108, 0.004))
    end_wall = np.column_stack([ys.ravel(), np.full(ys.size, 0.12), zs.ravel()])

    rim = find_rim(np.vstack([_bin_on_table(), end_wall]), CONFIG)

    assert rim is not None
    assert abs(rim.wall_yaw) == pytest.approx(math.pi / 2, abs=0.05)
    assert rim.x == pytest.approx(0.30, abs=0.005)


def test_rim_grasp_pose_puts_the_fingertips_below_the_rim_and_leans_outward() -> None:
    pose = rim_grasp_pose(0.30, 0.0, 0.10, math.pi / 2, CONFIG)

    tool_axis = pose.orientation.rotate_vector(pose.position.__class__(0.0, 0.0, 1.0))
    tips = np.array([pose.position.x, pose.position.y, pose.position.z]) + (
        CONFIG.fingertips_past_tcp * np.array([tool_axis.x, tool_axis.y, tool_axis.z])
    )
    assert tips == pytest.approx([0.30, 0.0, 0.10 - CONFIG.grasp_depth], abs=1e-6)
    assert tool_axis.x == pytest.approx(math.sin(CONFIG.lean), abs=1e-6)
    assert tool_axis.z == pytest.approx(-math.cos(CONFIG.lean), abs=1e-6)


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> Iterator[RimGraspModule]:
    def settle(read: Any, target: float, config: Any, **_: Any) -> GripperSettle:
        return GripperSettle(True, target, True, 0.1)

    monkeypatch.setattr("dimos.manipulation.rim_grasp_module.await_gripper_settle", settle)
    instance = RimGraspModule(settle_time=0.0)
    instance._scene = MagicMock()
    instance._manipulation = MagicMock()
    instance._manipulation.list_planning_groups.return_value = [
        SimpleNamespace(id="arm/tool", has_gripper=True, tip_frame="tool")
    ]
    instance._manipulation.plan_to_poses.return_value = SimpleNamespace(succeeded=True, message="")
    instance._manipulation.execute.return_value = SimpleNamespace(succeeded=True, message="")
    instance._manipulation.move_linear.return_value = SimpleNamespace(
        plan=SimpleNamespace(succeeded=True, message=""),
        execution=SimpleNamespace(succeeded=True, message=""),
    )
    instance._manipulation.set_gripper_position.return_value = SimpleNamespace(
        succeeded=True, message=""
    )
    yield instance
    instance.stop()


def _clouds(module: RimGraspModule, *clouds: Points) -> None:
    module._scene.get_full_scene_pointcloud.side_effect = [  # type: ignore[attr-defined]
        SimpleNamespace(frame_id="world", points_f32=lambda cloud=cloud: cloud) for cloud in clouds
    ]


def test_pick_up_by_rim_succeeds_when_depth_shows_the_container_lifted(
    module: RimGraspModule,
) -> None:
    _clouds(module, _bin_on_table(), _bin_lifted(module.config.lift_height))

    result = module.pick_up_by_rim()

    assert result.success, result.message
    moves = [call.args[2] for call in module._manipulation.move_linear.call_args_list]  # type: ignore[attr-defined]
    assert moves == [-module.config.pregrasp_offset, module.config.lift_height]
    assert module.release_rim().success


def test_pick_up_by_rim_reports_a_container_left_on_the_table(module: RimGraspModule) -> None:
    _clouds(module, _bin_on_table(), _bin_on_table())

    result = module.pick_up_by_rim()

    assert result.error_code == "GRASP_VERIFICATION_FAILED"
    assert module.release_rim().error_code == "INVALID_STATE"


def test_pick_up_by_rim_does_not_move_without_a_rim(module: RimGraspModule) -> None:
    _clouds(module, _table(), _table(), _table())

    result = module.pick_up_by_rim()

    assert result.error_code == "OBJECT_NOT_DETECTED"
    module._manipulation.plan_to_poses.assert_not_called()  # type: ignore[attr-defined]
    module._manipulation.set_gripper_position.assert_not_called()  # type: ignore[attr-defined]


def test_grasp_rim_tries_the_other_half_turn_when_the_first_is_unreachable(
    module: RimGraspModule,
) -> None:
    module._manipulation.plan_to_poses.side_effect = [  # type: ignore[attr-defined]
        SimpleNamespace(succeeded=False, message="JOINT_LIMITS"),
        SimpleNamespace(succeeded=True, message=""),
    ]
    _clouds(module, _bin_lifted(module.config.lift_height))

    assert module.grasp_rim(0.30, 0.0, 0.10, math.pi / 2).success
    assert module._manipulation.plan_to_poses.call_count == 2  # type: ignore[attr-defined]
