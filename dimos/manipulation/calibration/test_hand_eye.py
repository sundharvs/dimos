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

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
import pytest

from dimos.manipulation.calibration.charuco import BoardSpec, CharucoDetector
from dimos.manipulation.calibration.hand_eye_module import solve_and_report
from dimos.manipulation.calibration.hand_eye_solver import (
    Residual,
    angle_between,
    axis_spread,
    compare_methods,
    consistent_methods,
    inv_T,
    make_T,
    method_disagreement_mm,
    residuals,
    solve,
)
from dimos.manipulation.calibration.intrinsics import load_zed_left_intrinsics

# A wrist camera a few centimetres off link7, looking along the tool axis.
GRIPPER_T_CAM = make_T(
    cv2.Rodrigues(np.array([0.05, -0.1, 1.5]))[0],
    np.array([0.067, -0.031, 0.007]),
)
BASE_T_BOARD = make_T(np.diag([1.0, -1.0, -1.0]), np.array([0.5, 0.05, 0.0]))
# A fixed side camera 1 m off the table, looking back at the workspace, and a
# board bolted to the gripper a few centimetres past the flange.
BASE_T_SIDE_CAM = make_T(
    cv2.Rodrigues(np.array([2.0, -0.6, 0.4]))[0],
    np.array([0.9, -0.6, 0.55]),
)
GRIPPER_T_BOARD = make_T(cv2.Rodrigues(np.array([0.1, 0.0, 0.3]))[0], np.array([0.0, 0.03, 0.12]))


def _rotation(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
    return cv2.Rodrigues(np.radians([rx, ry, rz]))[0]


def _poses(
    tilts: list[tuple[float, float, float]],
    noise_m: float = 0.0,
    noise_deg: float = 0.0,
    seed: int = 0,
    eye_to_hand: bool = False,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Gripper poses, and the board as the wrist (or the fixed side) camera would see it."""
    rng = np.random.default_rng(seed)
    looking_down = np.diag([1.0, -1.0, -1.0])
    base_T_gripper, cam_T_board = [], []
    for i, tilt in enumerate(tilts):
        a = make_T(
            looking_down @ _rotation(*tilt),
            np.array([0.45 + 0.02 * (i % 3), 0.03 * (i % 4 - 1.5), 0.45 + 0.02 * (i % 2)]),
        )
        if eye_to_hand:
            b = inv_T(BASE_T_SIDE_CAM) @ a @ GRIPPER_T_BOARD
        else:
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
        assert np.allclose(result.camera_pose, GRIPPER_T_CAM, atol=1e-6), result.method
        assert result.residual.pos_rms < 1e-3
        assert np.allclose(result.board_pose, BASE_T_BOARD, atol=1e-6)


def test_noisy_solve_is_close_and_spread_reflects_noise() -> None:
    a, b = _poses(VARIED_TILTS, noise_m=0.0005, noise_deg=0.1)
    best = compare_methods(a, b)[0]
    assert np.linalg.norm(best.camera_pose[:3, 3] - GRIPPER_T_CAM[:3, 3]) < 0.003
    assert angle_between(best.camera_pose, GRIPPER_T_CAM) < 0.5
    assert 0.1 < best.residual.pos_rms < 5.0


def test_eye_to_hand_substitution_is_caught_by_the_spread() -> None:
    """Solving wrist-camera data with the fixed-camera recipe must look wrong."""
    a, b = _poses(VARIED_TILTS)
    wrong = solve(a, b, method="PARK", mode="eye_to_hand")
    assert residuals(wrong.camera_pose, a, b).pos_rms > 20.0


def test_eye_to_hand_recovers_fixed_camera_and_board_on_gripper() -> None:
    a, b = _poses(VARIED_TILTS, eye_to_hand=True)
    for result in compare_methods(a, b, mode="eye_to_hand"):
        assert np.allclose(result.camera_pose, BASE_T_SIDE_CAM, atol=1e-6), result.method
        assert np.allclose(result.board_pose, GRIPPER_T_BOARD, atol=1e-6)
        assert result.residual.label == "board-on-gripper"
        assert result.residual.pos_rms < 1e-3


def test_eye_to_hand_noisy_and_wrong_mode_is_caught() -> None:
    a, b = _poses(VARIED_TILTS, noise_m=0.0005, noise_deg=0.1, eye_to_hand=True)
    best = compare_methods(a, b, mode="eye_to_hand")[0]
    assert np.linalg.norm(best.camera_pose[:3, 3] - BASE_T_SIDE_CAM[:3, 3]) < 0.005
    assert angle_between(best.camera_pose, BASE_T_SIDE_CAM) < 0.5
    wrong = solve(a, b, method="PARK", mode="eye_in_hand")
    assert wrong.residual.pos_rms > 20.0


def test_eye_to_hand_report_is_in_the_base_frame() -> None:
    a, b = _poses(VARIED_TILTS, eye_to_hand=True)
    camera_T_optical = make_T(_rotation(-90, 0, -90), np.zeros(3))
    _, payload = solve_and_report(
        a,
        b,
        camera_T_optical,
        mode="eye_to_hand",
        parent_frame="world",
        camera_frame="camera_link",
        optical_frame="camera_optical",
    )
    assert payload["frame_id"] == "world"
    expected = BASE_T_SIDE_CAM @ inv_T(camera_T_optical)
    assert np.allclose(np.array(payload["T_parent_camera"]), expected, atol=1e-6)


def test_zed_factory_intrinsics_pick_the_resolution(tmp_path: Path) -> None:
    conf = tmp_path / "SN1.conf"
    conf.write_text(
        "[LEFT_CAM_2K]\nfx=1063.98\nfy=1064.08\ncx=1104.58\ncy=633.793\n"
        "k1=-0.058\nk2=0.032\np1=0.0005\np2=-0.0009\nk3=-0.012\n"
        "[LEFT_CAM_HD]\nfx=531.99\nfy=532.04\ncx=638.79\ncy=364.9\n"
        "k1=-0.058\nk2=0.032\np1=0.0005\np2=-0.0009\nk3=-0.012\n"
    )
    camera_matrix, dist_coeffs = load_zed_left_intrinsics(conf, 2208)
    assert camera_matrix[0, 0] == pytest.approx(1063.98)
    assert camera_matrix[1, 2] == pytest.approx(633.793)
    assert dist_coeffs.tolist() == pytest.approx([-0.058, 0.032, 0.0005, -0.0009, -0.012])
    assert load_zed_left_intrinsics(conf, 1280)[0][0, 0] == pytest.approx(531.99)
    with pytest.raises(ValueError, match="no \\[LEFT_CAM_FHD\\]"):
        load_zed_left_intrinsics(conf, 1920)
    with pytest.raises(ValueError, match="not a ZED left-image width"):
        load_zed_left_intrinsics(conf, 848)


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
        mode="eye_in_hand",
        parent_frame="link7",
        camera_frame="camera_link",
        optical_frame="camera_color_optical_frame",
    )
    expected = GRIPPER_T_CAM @ inv_T(camera_T_optical)
    assert payload["frame_id"] == "link7"
    assert np.allclose(np.array(payload["T_parent_camera"]), expected, atol=1e-6)
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


def test_a_failed_solver_does_not_count_as_disagreement() -> None:
    """ANDREFF on real data: far off, but its own spread says it failed."""
    a, b = _poses(VARIED_TILTS, noise_m=0.0005, noise_deg=0.1)
    results = compare_methods(a, b)
    worst = results[-1]
    shifted = worst.camera_pose.copy()
    shifted[:3, 3] += [0.03, 0.0, 0.0]
    failed = replace(
        worst,
        camera_pose=shifted,
        residual=Residual(worst.residual.pos_mm * 4 + 5.0, worst.residual.rot_deg),
    )
    with_failure = [*results[:-1], failed]
    assert [r.method for r in consistent_methods(with_failure)] == [r.method for r in results[:-1]]
    assert method_disagreement_mm(with_failure) < 5.0

    report, payload = solve_and_report(
        a,
        b,
        np.eye(4),
        mode="eye_in_hand",
        parent_frame="link7",
        camera_frame="camera_link",
        optical_frame="camera_link",
    )
    assert "consistent solvers" in report
    assert payload["consistent_methods"]
