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
import pytest

from dimos.manipulation.grasping.rim_grasp import RimGraspConfig, RimGraspModule
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def _module(**overrides: object) -> RimGraspModule:
    module = RimGraspModule.__new__(RimGraspModule)
    module.config = RimGraspConfig(**overrides)  # type: ignore[arg-type]
    return module


def _bin_cloud(
    center: tuple[float, float] = (0.0, -0.5),
    size: tuple[float, float] = (0.28, 0.10),
    height: float = 0.09,
    yaw: float = 0.0,
    low_wall: str | None = None,
    floor_z: float = -0.012,
    step: float = 0.004,
) -> np.ndarray:
    """Four thin walls on a floor, like a scoop-front shelf bin seen from above."""
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    points = []
    xs = np.arange(-half_x, half_x + step, step)
    ys = np.arange(-half_y, half_y + step, step)
    zs = np.arange(floor_z, floor_z + height + step, step)
    for z in zs:
        for x in xs:
            for y_sign, name in ((-1.0, "y-"), (1.0, "y+")):
                top = floor_z + (height * 0.6 if low_wall == name else height)
                if z <= top:
                    points.append((x, y_sign * half_y, z))
        for y in ys:
            for x_sign, name in ((-1.0, "x-"), (1.0, "x+")):
                top = floor_z + (height * 0.6 if low_wall == name else height)
                if z <= top:
                    points.append((x_sign * half_x, y, z))
    for x in xs:
        for y in ys:
            points.append((x, y, floor_z))
    pts = np.asarray(points, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rotated = pts[:, :2] @ np.array([[c, s], [-s, c]])
    pts[:, 0] = rotated[:, 0] + center[0]
    pts[:, 1] = rotated[:, 1] + center[1]
    return pts.astype(np.float32)


def _cloud_msg(points: np.ndarray) -> PointCloud2:
    return PointCloud2.from_numpy(points, frame_id="world", timestamp=1.0)


def test_candidates_sit_on_the_walls_and_close_across_them() -> None:
    module = _module(insertion_depth=0.03)
    points = _bin_cloud()
    candidates = module.rim_candidates(points)
    assert len(candidates) == 4
    for pose, _score in candidates:
        x, y = pose.position.x, pose.position.y
        on_long_wall = abs(abs(y + 0.5) - 0.05) < 0.01 and abs(x) < 0.02
        on_short_wall = abs(abs(x) - 0.14) < 0.01 and abs(y + 0.5) < 0.02
        assert on_long_wall or on_short_wall
        # 3 cm below the 9 cm wall standing on a floor at -0.012
        assert pose.position.z == pytest.approx(-0.012 + 0.09 - 0.03, abs=0.006)
        yaw = pose.orientation.to_euler().z
        # the jaw closing axis (body Y after roll pi) must be normal to the wall
        closing = np.array([math.sin(yaw), -math.cos(yaw)])
        normal = np.array([0.0, 1.0]) if on_long_wall else np.array([1.0, 0.0])
        assert abs(abs(float(closing @ normal)) - 1.0) < 0.05


def test_lever_ranking_prefers_long_walls_and_sides_filter_works() -> None:
    module = _module()
    points = _bin_cloud()
    best_pose, _ = module.rim_candidates(points)[0]
    assert abs(best_pose.position.x) < 0.02  # a long wall, 5 cm lever, not 14 cm
    module_short = _module(sides="short")
    for pose, _ in module_short.rim_candidates(points):
        assert abs(abs(pose.position.x) - 0.14) < 0.01


def test_lowest_wall_selection_finds_the_scoop_front_and_uses_its_own_height() -> None:
    module = _module(wall_select="lowest", insertion_depth=0.03)
    points = _bin_cloud(low_wall="y-")
    pose, _ = module.rim_candidates(points)[0]
    assert pose.position.y == pytest.approx(-0.55, abs=0.01)
    assert pose.position.z == pytest.approx(-0.012 + 0.09 * 0.6 - 0.03, abs=0.008)
    highest = _module(wall_select="highest")
    pose_high, _ = highest.rim_candidates(points)[0]
    assert pose_high.position.y > -0.55 + 0.02 or abs(pose_high.position.x) > 0.1


def test_rotated_bin_keeps_wall_normals_and_min_z_floor() -> None:
    yaw = math.radians(35.0)
    module = _module(min_z=0.08)
    points = _bin_cloud(yaw=yaw)
    for pose, _ in module.rim_candidates(points):
        assert pose.position.z >= 0.08 - 1e-6
        rel = math.degrees(abs(pose.orientation.to_euler().z - yaw)) % 90.0
        assert min(rel, 90.0 - rel) < 5.0


def test_describe_rim_reports_the_rectangle() -> None:
    module = _module()
    rim = module.describe_rim(_cloud_msg(_bin_cloud()))
    assert sorted(rim["rect_extents"]) == pytest.approx([0.10, 0.28], abs=0.012)
    assert rim["rim_top_z"] == pytest.approx(0.078, abs=0.006)
    assert rim["n_rim_points"] > 100


def test_propose_grasps_rejects_tiny_clouds() -> None:
    module = _module()
    with pytest.raises(ValueError):
        module.propose_grasps(_cloud_msg(np.zeros((5, 3), dtype=np.float32)))
