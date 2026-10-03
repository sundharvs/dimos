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

import cv2
import numpy as np
from numpy.typing import NDArray

from dimos.manipulation.calibration.auto_poses import (
    board_center,
    bootstrap_turns,
    look_at_views,
)
from dimos.manipulation.calibration.charuco import BoardSpec
from dimos.manipulation.calibration.hand_eye_solver import (
    GOOD_AXIS_SPREAD,
    angle_between,
    axis_spread,
    compare_methods,
    inv_T,
    make_T,
)
from dimos.manipulation.calibration.test_hand_eye import BASE_T_BOARD, GRIPPER_T_CAM

SPEC = BoardSpec()
CENTER = board_center(BASE_T_BOARD, SPEC.size_m)
# The camera 40 cm over the board, a little off its centre, looking straight down.
START_OPTICAL = make_T(np.diag([1.0, -1.0, -1.0]), CENTER + np.array([0.04, -0.03, 0.40]))
START_GRIPPER = START_OPTICAL @ inv_T(GRIPPER_T_CAM)


def _degrees(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _off_axis_deg(base_T_optical: NDArray[np.float64]) -> float:
    """How far from the optical axis the board centre appears."""
    return _degrees(base_T_optical[:3, 2], CENTER - base_T_optical[:3, 3])


def _observe(base_T_gripper: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    board = inv_T(base_T_gripper @ GRIPPER_T_CAM) @ BASE_T_BOARD
    jitter = make_T(
        cv2.Rodrigues(np.radians(rng.normal(0.0, 0.05, 3)))[0], rng.normal(0.0, 0.0005, 3)
    )
    return board @ jitter


def test_bootstrap_turns_keep_the_board_in_view() -> None:
    turns = bootstrap_turns(8.0, 15.0)
    assert len(turns) == 6
    gripper_poses = [START_GRIPPER @ turn for turn in turns]
    for pose in gripper_poses:
        assert np.allclose(pose[:3, 3], START_GRIPPER[:3, 3])
        # A RealSense's half field of view is about 21 degrees vertically.
        assert _off_axis_deg(pose @ GRIPPER_T_CAM) < 15.0
    assert axis_spread([START_GRIPPER, *gripper_poses]) > GOOD_AXIS_SPREAD


def test_views_aim_at_the_board_from_a_cone() -> None:
    views = look_at_views(BASE_T_BOARD, SPEC.size_m, START_OPTICAL, 16, 25.0, 30.0)
    assert len(views) == 16
    start_range = float(np.linalg.norm(START_OPTICAL[:3, 3] - CENTER))
    up = np.array([0.0, 0.0, 1.0])
    tilts = []
    for view in views:
        assert _off_axis_deg(view) < 1e-6
        tilt = _degrees(view[:3, 3] - CENTER, up)
        assert tilt <= 25.0 + 1e-6
        tilts.append(tilt)
        assert 0.84 <= np.linalg.norm(view[:3, 3] - CENTER) / start_range <= 1.16
        assert np.isclose(np.linalg.det(view[:3, :3]), 1.0)
    assert min(tilts) > 12.0
    # No two views are near-duplicates: each adds a turn worth having.
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            assert angle_between(a, b) > 8.0
    assert axis_spread([view @ inv_T(GRIPPER_T_CAM) for view in views]) > GOOD_AXIS_SPREAD


def test_bootstrap_then_views_recovers_the_mount() -> None:
    """The automatic run end to end, with the camera mount unknown to the planner."""
    rng = np.random.default_rng(3)
    arm = [START_GRIPPER, *(START_GRIPPER @ turn for turn in bootstrap_turns(8.0, 15.0))]
    seen = [_observe(pose, rng) for pose in arm]

    coarse = compare_methods(arm, seen)[0]
    assert np.linalg.norm(coarse.camera_pose[:3, 3] - GRIPPER_T_CAM[:3, 3]) < 0.02

    # Views are aimed with the coarse estimate but observed through the true mount.
    start_optical = START_GRIPPER @ coarse.camera_pose
    views = look_at_views(coarse.board_pose, SPEC.size_m, start_optical, 16, 25.0, 30.0)
    for view in views:
        pose = view @ inv_T(coarse.camera_pose)
        assert _off_axis_deg(pose @ GRIPPER_T_CAM) < 5.0
        arm.append(pose)
        seen.append(_observe(pose, rng))

    best = compare_methods(arm, seen)[0]
    assert np.linalg.norm(best.camera_pose[:3, 3] - GRIPPER_T_CAM[:3, 3]) < 0.002
    assert angle_between(best.camera_pose, GRIPPER_T_CAM) < 0.2
