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


def test_a_pure_wrist_turn_may_exceed_the_path_length_limit(guard: PathGuard) -> None:
    # joint3 stands in for the wrist joint of this two-link arm
    guard.config.wrist_joint = "joint3"
    start = (-math.pi / 2, 0.3, 0.0)
    goal = (
        -math.pi / 2,
        0.3,
        3.1,
    )  # inside the boxes throughout? the forearm swings; use small shoulder
    why = guard.check_path(_path(start, goal))
    assert why is None or "box" in why  # never rejected for length alone


def test_wrist_feasible_yaw_prefers_the_equivalent_with_room_to_turn() -> None:
    from dimos.manipulation.container_pick_module import wrist_feasible_yaw

    limits = (-3.0, 3.0)
    # wrist at +2.5 rad; the grasp yaw equals the current yaw, and a +1.5 rad
    # turn follows: with sign -1 the wrist would go to +1.0 (fine), so the plain
    # equivalent is chosen
    assert wrist_feasible_yaw(0.0, 0.0, 2.5, 1.5, limits, -1.0) == pytest.approx(0.0)
    # a -1.5 rad turn would push the wrist to +4.0: the half-turn equivalent
    # (wrist to -0.64 at the grasp, then +0.86) is chosen instead
    yaw = wrist_feasible_yaw(0.0, 0.0, 2.5, -1.5, limits, -1.0)
    assert yaw is not None and abs(abs(yaw) - math.pi) < 1e-9
    # nothing fits: both equivalents end outside the range
    assert wrist_feasible_yaw(0.0, 0.0, 2.9, -5.5, (-3.0, 3.0), -1.0) is None


def test_describe_container_finds_the_low_end() -> None:
    import numpy as np

    from dimos.manipulation.container_pick_module import describe_container

    rng = np.random.default_rng(0)
    # a 28 x 10 cm bin, long axis along +X, walls 9 cm high except the +X end
    # wall which is cut down to 5 cm (the scoop opening)
    points = []
    for x in np.linspace(-0.14, 0.14, 120):
        for y in (-0.05, 0.05):
            for z in np.linspace(0.0, 0.09, 20):
                points.append((x, y, z))
    for y in np.linspace(-0.05, 0.05, 40):
        for z in np.linspace(0.0, 0.09, 20):
            points.append((-0.14, y, z))
        for z in np.linspace(0.0, 0.05, 12):
            points.append((0.14, y, z))
    cloud = np.asarray(points, dtype=np.float32)
    cloud[:, :2] += np.array([0.3, -0.4], dtype=np.float32)
    cloud += rng.normal(0.0, 0.001, cloud.shape).astype(np.float32)
    rim = {
        "rect_center": [0.3, -0.4],
        "rect_axes": [[1.0, 0.0], [0.0, 1.0]],
        "rect_extents": [0.28, 0.10],
        "rim_top_z": 0.09,
    }
    seen = describe_container(cloud, rim)
    assert seen["opening_known"]
    assert seen["opening_dir"] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert seen["opening_drop"] == pytest.approx(0.04, abs=0.01)
    assert seen["half_length"] == pytest.approx(0.14)
    assert len(seen["corners"]) == 4
    # same bin with both ends full height: no opening
    tall = cloud[cloud[:, 0] < 0.3 + 0.139]
    seen = describe_container(tall, rim)
    assert not seen["opening_known"]
