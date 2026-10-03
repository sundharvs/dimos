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
from pathlib import Path

import pytest

from dimos.manipulation.container_pick_module import (
    ContainerPickConfig,
    PathGuard,
    inside,
    nearest_equivalent_yaw,
    wrap_angle,
)
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.assets.model import RobotModel

# A planar two-link arm standing on the table: base yaw, then a shoulder that
# raises a 0.4 m upper arm and a 0.4 m forearm with a tool link at the end.
_URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="link3"/>
  <link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="tool_joint" type="fixed">
    <parent link="link3"/><child link="tool"/>
    <origin xyz="0 0 0.4"/>
  </joint>
</robot>
"""


@pytest.fixture
def guard(tmp_path: Path) -> PathGuard:
    urdf = tmp_path / "two_link.urdf"
    urdf.write_text(_URDF)
    config = ContainerPickConfig(
        model=RobotModel.from_file(urdf),
        hand_links=["tool"],
        elbow_links=["link3"],
        workspace_box=((-0.9, 0.9), (-0.9, 0.2), (-0.1, 1.0)),
        elbow_box=((-0.9, 0.9), (-0.9, 0.5), (-0.1, 1.0)),
        reach_max=0.85,
        base_joint="joint1",
        base_joint_max_excursion=1.0,
        path_max_length=3.0,
    )
    return PathGuard(config)


def _path(*configs: tuple[float, float, float]) -> list[JointState]:
    return [JointState(name=["joint1", "joint2", "joint3"], position=list(q)) for q in configs]


def test_helpers() -> None:
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi) or wrap_angle(
        3 * math.pi
    ) == pytest.approx(-math.pi)
    assert nearest_equivalent_yaw(math.pi - 0.1, -0.2) == pytest.approx(-0.1 - 0.0, abs=1e-9) or (
        abs(wrap_angle(nearest_equivalent_yaw(math.pi - 0.1, -0.2) + 0.2)) < 0.2
    )
    assert inside(__import__("numpy").array([0.0, 0.0, 0.0]), ((-1, 1), (-1, 1), (-1, 1)))
    assert not inside(__import__("numpy").array([2.0, 0.0, 0.0]), ((-1, 1), (-1, 1), (-1, 1)))


def test_a_short_reach_over_the_table_is_accepted(guard: PathGuard) -> None:
    # arm leaning forward over -y: joint1 = -pi/2 points +x of link1 toward -y
    start = (-math.pi / 2, 0.3, 0.6)
    goal = (-math.pi / 2, 0.5, 0.8)
    assert guard.check_path(_path(start, goal)) is None


def test_a_swing_of_the_base_joint_is_rejected(guard: PathGuard) -> None:
    start = (-math.pi / 2, 0.4, 0.6)
    goal = (math.pi / 2, 0.4, 0.6)  # swings the whole arm round behind the base
    why = guard.check_path(_path(start, goal))
    assert why is not None
    assert "excursion" in why or "box" in why


def test_leaving_the_hand_box_is_named(guard: PathGuard) -> None:
    # hand reaching to +y (behind the base) beyond the 0.2 m allowance
    start = (-math.pi / 2, 0.4, 0.6)
    behind = (math.pi / 2 - 0.9, 1.3, 0.2)
    why = guard.check_path(_path(start, behind))
    assert why is not None


def test_long_joint_paths_are_rejected(guard: PathGuard) -> None:
    # dithers the forearm in place: every pose is inside the boxes, but the
    # summed joint travel is far beyond what a short approach needs
    waypoints = [(-math.pi / 2, 0.3, 0.6 + (0.05 if i % 2 else -0.05)) for i in range(40)]
    why = guard.check_path(_path(*waypoints))
    assert why is not None and "path length" in why


def test_empty_path_is_rejected(guard: PathGuard) -> None:
    assert guard.check_path([]) == "empty path"
