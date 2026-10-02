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
import pytest

from dimos.manipulation.calibration.charuco import BoardSpec, CharucoDetector
from dimos.manipulation.calibration.hand_eye_module import solve_and_report
from dimos.manipulation.calibration.hand_eye_solver import (
    angle_between,
    axis_spread,
    compare_methods,
    inv_T,
    make_T,
    residuals,
    solve,
)

# A wrist camera a few centimetres off link7, looking along the tool axis.
GRIPPER_T_CAM = make_T(
    cv2.Rodrigues(np.array([0.05, -0.1, 1.5]))[0],
    np.array([0.067, -0.031, 0.007]),
)
BASE_T_BOARD = make_T(np.diag([1.0, -1.0, -1.0]), np.array([0.5, 0.05, 0.0]))


def _rotation(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
    return cv2.Rodrigues(np.radians([rx, ry, rz]))[0]


def _poses(
    tilts: list[tuple[float, float, float]],
    noise_m: float = 0.0,
    noise_deg: float = 0.0,
    seed: int = 0,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Gripper poses looking down at the board, and the board as the camera would see it."""
    rng = np.random.default_rng(seed)
    looking_down = np.diag([1.0, -1.0, -1.0])
    base_T_gripper, cam_T_board = [], []
    for i, tilt in enumerate(tilts):
        a = make_T(
            looking_down @ _rotation(*tilt),
            np.array([0.45 + 0.02 * (i % 3), 0.03 * (i % 4 - 1.5), 0.45 + 0.02 * (i % 2)]),
        )
        b = inv_T(a @ GRIPPER_T_CAM) @ BASE_T_BOARD
        if noise_m or noise_deg:
            jitter = make_T(_rotation(*rng.normal(0.0, noise_deg, 3)), rng.normal(0.0, noise_m, 3))
            b = b @ jitter
        base_T_gripper.append(a)
        cam_T_board.append(b)
    return base_T_gripper, cam_T_board


VARIED_TILTS = [
    (0, 0, 0),
    (20, 0, 0),
    (0, 20, 0),
    (0, 0, 30),
    (-15, 10, 0),
    (10, -15, 20),
    (0, 15, -25),
    (-20, -10, 10),
    (15, 15, -15),
    (-10, 20, 25),
    (25, -5, -10),
    (-5, -20, 15),
]


def test_recovers_mount_exactly_without_noise() -> None:
    a, b = _poses(VARIED_TILTS)
    for result in compare_methods(a, b):
        assert np.allclose(result.camera_in_gripper, GRIPPER_T_CAM, atol=1e-6), result.method
        assert result.residual.pos_rms < 1e-3
        assert np.allclose(result.board_in_base, BASE_T_BOARD, atol=1e-6)


def test_noisy_solve_is_close_and_spread_reflects_noise() -> None:
    a, b = _poses(VARIED_TILTS, noise_m=0.0005, noise_deg=0.1)
    best = compare_methods(a, b)[0]
    assert np.linalg.norm(best.camera_in_gripper[:3, 3] - GRIPPER_T_CAM[:3, 3]) < 0.003
    assert angle_between(best.camera_in_gripper, GRIPPER_T_CAM) < 0.5
    assert 0.1 < best.residual.pos_rms < 5.0


def test_eye_to_hand_substitution_is_caught_by_the_spread() -> None:
    """Feeding inverted arm poses -- the fixed-camera recipe -- must look wrong."""
    a, b = _poses(VARIED_TILTS)
    wrong = solve([inv_T(t) for t in a], b, method="PARK")
    assert residuals(wrong.camera_in_gripper, a, b).pos_rms > 20.0


def test_single_axis_rotation_is_refused() -> None:
    a, b = _poses([(0, 0, yaw) for yaw in range(0, 60, 6)])
    assert axis_spread(a) < 0.01
    with pytest.raises(ValueError, match="rotation diversity"):
        solve(a, b)
    with pytest.raises(ValueError, match="rotation diversity"):
        compare_methods(a, b)


def test_too_few_poses_is_refused() -> None:
    a, b = _poses(VARIED_TILTS[:2])
    with pytest.raises(ValueError, match="not enough"):
        solve(a, b)


def test_report_removes_the_cameras_own_optical_edge() -> None:
    a, b = _poses(VARIED_TILTS)
    camera_T_optical = make_T(_rotation(-90, 0, -90), np.array([0.0, 0.015, 0.0]))
    _, payload = solve_and_report(
        a,
        b,
        camera_T_optical,
        gripper_frame="link7",
        camera_frame="camera_link",
        optical_frame="camera_color_optical_frame",
    )
    expected = GRIPPER_T_CAM @ inv_T(camera_T_optical)
    assert np.allclose(np.array(payload["T_gripper_camera"]), expected, atol=1e-6)
    assert np.allclose(payload["translation"], expected[:3, 3], atol=1e-6)


def test_detector_recovers_a_rendered_board_pose() -> None:
    spec = BoardSpec.from_mm(5, 7, 34.0)
    width, height = 1280, 720
    camera_matrix = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)

    # Render the board flat, then warp it into the camera through the homography
    # of a known tilted pose.
    px_per_m = 4000.0
    board_w, board_h = spec.size_m
    flat = spec.cv_board().generateImage((round(board_w * px_per_m), round(board_h * px_per_m)))
    cam_T_board = make_T(_rotation(20, -15, 5), np.array([-0.08, -0.1, 0.45]))
    board_corners = np.array(
        [[0.0, 0.0, 0.0], [board_w, 0.0, 0.0], [board_w, board_h, 0.0], [0.0, board_h, 0.0]]
    )
    projected, _ = cv2.projectPoints(
        board_corners,
        cv2.Rodrigues(cam_T_board[:3, :3])[0],
        cam_T_board[:3, 3],
        camera_matrix,
        dist_coeffs,
    )
    flat_corners = np.array(
        [[0, 0], [flat.shape[1], 0], [flat.shape[1], flat.shape[0]], [0, flat.shape[0]]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        flat_corners, projected.reshape(4, 2).astype(np.float32)
    )
    image = cv2.warpPerspective(
        flat, homography, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    pose = CharucoDetector(spec).detect(image, camera_matrix, dist_coeffs)
    assert pose is not None
    assert pose.n_corners == spec.n_corners
    assert pose.reproj_px < 0.5
    assert np.linalg.norm(pose.board_in_camera[:3, 3] - cam_T_board[:3, 3]) < 0.002
    assert angle_between(pose.board_in_camera, cam_T_board) < 0.5


def test_detector_sees_nothing_in_a_blank_frame() -> None:
    detector = CharucoDetector(BoardSpec())
    blank = np.full((480, 640), 255, dtype=np.uint8)
    assert detector.detect(blank, np.eye(3), np.zeros(5)) is None
