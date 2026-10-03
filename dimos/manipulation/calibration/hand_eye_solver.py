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

"""Hand-eye calibration: where a camera sits relative to the arm.

TWO ARRANGEMENTS, ONE EQUATION
    Eye-in-hand -- the camera rides on the gripper, the board is fixed in the
    workspace. The board's base-frame pose, through the arm and the camera,

        T_base_board = T_base_gripper . T_gripper_cam . T_cam_board

    must be the same at every pose, so X = T_gripper_cam is the unknown.

    Eye-to-hand -- the camera is fixed, the board rides on the gripper. Now the
    board's pose on the gripper is the constant:

        T_gripper_board = T_base_gripper^-1 . T_base_cam . T_cam_board

    which is the same equation with the arm pose inverted and X = T_base_cam.
    `solve()` makes that substitution; OpenCV's calibrateHandEye is written for
    eye-in-hand. Getting the mode backwards returns a plausible transform that
    is wrong everywhere -- the residual below is what catches it.

JUDGE IT BY THE SPREAD OF THE CONSTANT, NOT BY REPROJECTION
    A board can reproject beautifully and still disagree with the arm. Once X is
    known, the constant (board in base, or board on gripper) is recoverable from
    every pose, and every one must agree. The spread is a physical error in
    millimetres and degrees that shares no assumption with the solver.

ROTATION DIVERSITY IS A REQUIREMENT
    AX = XB determines X only if the wrist rotates about axes that are not all
    parallel. Pure translation constrains nothing. `axis_spread()` measures it
    and `solve()` refuses below a floor, because an under-determined fit still
    returns a matrix and still reports a small residual on the very motions
    that failed to constrain it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from typing import Literal

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

Matrix4 = NDArray[np.float64]
HandEyeMode = Literal["eye_in_hand", "eye_to_hand"]

METHODS = ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS")
MIN_POSES = 3
# Below this the axes nearly share a plane and X is under-determined.
MIN_AXIS_SPREAD = 0.15
# Above this the set is well spread; between the two, more axes still help.
GOOD_AXIS_SPREAD = 0.4
# A solver whose spread exceeds the best one's by both of these has failed on
# the data rather than disagreed about it.
OUTLIER_SPREAD_RATIO = 1.5
OUTLIER_SPREAD_MARGIN_MM = 1.0


def make_T(rotation: ArrayLike, translation: ArrayLike) -> Matrix4:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def inv_T(transform: Matrix4) -> Matrix4:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    return make_T(rotation.T, -rotation.T @ translation)


def angle_between(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Geodesic angle in degrees between two rotations (or the rotations of two poses)."""
    cos = (np.trace(np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def axis_spread(base_T_gripper: Sequence[Matrix4]) -> float:
    """How well the relative wrist rotation axes span 3D, in [0, 1]. Higher is better.

    sigma_min / sigma_max of the stacked UNIT rotation axes between consecutive
    poses: one shared axis gives 0, three orthogonal axes give 1, and axes
    confined to a plane read 0 however many poses are taken. Only direction
    counts -- weighting by angle would let one big turn drag the number down
    even when it adds a genuinely new axis.
    """
    axes = []
    for previous, current in itertools.pairwise(base_T_gripper):
        delta = np.asarray(current)[:3, :3] @ np.asarray(previous)[:3, :3].T
        rotation_vector = cv2.Rodrigues(delta)[0].reshape(3)
        norm = np.linalg.norm(rotation_vector)
        if norm > np.deg2rad(2.0):  # near-pure translations carry no axis
            axes.append(rotation_vector / norm)
    if len(axes) < 2:
        return 0.0
    singular = np.linalg.svd(np.asarray(axes), compute_uv=False)
    return float(singular[-1] / max(singular[0], 1e-9))


@dataclass(frozen=True)
class Residual:
    """How much the recovered constant pose disagrees with itself across samples."""

    pos_mm: NDArray[np.float64]  # per-pose distance from the median sample
    rot_deg: NDArray[np.float64]
    label: str = "board-in-base"

    @property
    def n(self) -> int:
        return len(self.pos_mm)

    @property
    def pos_rms(self) -> float:
        return float(np.sqrt(np.mean(self.pos_mm**2)))

    @property
    def rot_rms(self) -> float:
        return float(np.sqrt(np.mean(self.rot_deg**2)))

    def describe(self) -> str:
        return (
            f"{self.label} spread over {self.n} poses: "
            f"{self.pos_rms:.2f} mm RMS (max {self.pos_mm.max():.2f}), "
            f"{self.rot_rms:.2f} deg RMS (max {self.rot_deg.max():.2f})"
        )


def _solver_arm_poses(base_T_gripper: Sequence[Matrix4], mode: HandEyeMode) -> list[Matrix4]:
    """The arm poses as the eye-in-hand equation wants them for this mode."""
    poses = [np.asarray(t, dtype=float).reshape(4, 4) for t in base_T_gripper]
    return poses if mode == "eye_in_hand" else [inv_T(t) for t in poses]


def board_poses(
    camera_pose: Matrix4,
    base_T_gripper: Sequence[Matrix4],
    cam_T_board: Sequence[Matrix4],
    mode: HandEyeMode = "eye_in_hand",
) -> list[Matrix4]:
    """The constant pose each sample implies: board in base, or board on gripper."""
    arm = _solver_arm_poses(base_T_gripper, mode)
    return [a @ camera_pose @ b for a, b in zip(arm, cam_T_board, strict=True)]


def _median_index(boards: Sequence[Matrix4]) -> int:
    positions = np.array([board[:3, 3] for board in boards])
    return int(np.argmin(np.linalg.norm(positions - np.median(positions, axis=0), axis=1)))


def residuals(
    camera_pose: Matrix4,
    base_T_gripper: Sequence[Matrix4],
    cam_T_board: Sequence[Matrix4],
    mode: HandEyeMode = "eye_in_hand",
) -> Residual:
    """Spread of the constant pose across the samples. Anything it reports is error.

    Measured against the median sample, not the mean, so one badly seen board
    does not drag the reference it is judged against toward itself.
    """
    boards = board_poses(camera_pose, base_T_gripper, cam_T_board, mode)
    reference = boards[_median_index(boards)]
    return Residual(
        pos_mm=np.array([np.linalg.norm(b[:3, 3] - reference[:3, 3]) * 1000.0 for b in boards]),
        rot_deg=np.array([angle_between(b, reference) for b in boards]),
        label="board-in-base" if mode == "eye_in_hand" else "board-on-gripper",
    )


@dataclass(frozen=True)
class HandEyeResult:
    """The camera pose, where it puts the board, and how good it is.

    camera_pose is T_gripper_cam eye-in-hand and T_base_cam eye-to-hand;
    board_pose is T_base_board eye-in-hand and T_gripper_board eye-to-hand.
    """

    camera_pose: Matrix4
    board_pose: Matrix4
    mode: HandEyeMode
    method: str
    residual: Residual
    axis_spread: float

    @property
    def n_poses(self) -> int:
        return self.residual.n

    def describe(self) -> str:
        t = self.camera_pose[:3, 3]
        where = "camera on gripper" if self.mode == "eye_in_hand" else "camera in base"
        return (
            f"{where}: [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m "
            f"({self.method}, {self.n_poses} poses)\n  {self.residual.describe()}"
            f"\n  rotation diversity {self.axis_spread:.3f}"
        )


def solve(
    base_T_gripper: Sequence[Matrix4],
    cam_T_board: Sequence[Matrix4],
    method: str = "PARK",
    min_axis_spread: float = MIN_AXIS_SPREAD,
    mode: HandEyeMode = "eye_in_hand",
) -> HandEyeResult:
    """Solve for T_gripper_cam (eye-in-hand) or T_base_cam (eye-to-hand).

    PARK separates rotation from translation and is the steadiest on small
    sets; ANDREFF and DANIILIDIS solve both at once and do better with large,
    varied rotations. `compare_methods()` runs them all.
    """
    arm = [np.asarray(t, dtype=float).reshape(4, 4) for t in base_T_gripper]
    b = [np.asarray(t, dtype=float).reshape(4, 4) for t in cam_T_board]
    if len(arm) != len(b):
        raise ValueError(f"{len(arm)} arm poses against {len(b)} board poses")
    if len(arm) < MIN_POSES:
        raise ValueError(
            f"{len(arm)} poses is not enough; AX=XB needs at least {MIN_POSES}, "
            "and 15-20 well spread is what makes it stable"
        )

    # Measured on the arm's own poses in either mode: it is the wrist's motion
    # that has to be diverse.
    spread = axis_spread(arm)
    if spread < min_axis_spread:
        raise ValueError(
            f"rotation diversity {spread:.3f} is below {min_axis_spread}: the wrist "
            "turned about too nearly one axis, which leaves the camera under-determined. "
            "Re-collect turning the wrist about genuinely different axes -- "
            "translation alone constrains nothing."
        )

    a = _solver_arm_poses(arm, mode)
    rotation, translation = cv2.calibrateHandEye(
        [t[:3, :3] for t in a],
        [t[:3, 3] for t in a],
        [t[:3, :3] for t in b],
        [t[:3, 3] for t in b],
        method=getattr(cv2, f"CALIB_HAND_EYE_{method.upper()}"),
    )
    camera_pose = make_T(rotation, translation)
    if not np.all(np.isfinite(camera_pose)):
        raise ValueError(f"{method.upper()} returned a non-finite transform")

    boards = board_poses(camera_pose, arm, b, mode)
    return HandEyeResult(
        camera_pose=camera_pose,
        board_pose=boards[_median_index(boards)],
        mode=mode,
        method=method.upper(),
        residual=residuals(camera_pose, arm, b, mode),
        axis_spread=spread,
    )


def compare_methods(
    base_T_gripper: Sequence[Matrix4],
    cam_T_board: Sequence[Matrix4],
    min_axis_spread: float = MIN_AXIS_SPREAD,
    mode: HandEyeMode = "eye_in_hand",
) -> list[HandEyeResult]:
    """Every solver on the same data, smallest spread first.

    Five independent algorithms agreeing to a millimetre is strong evidence the
    data determines X. Disagreement is the data's fault, not the solver's.
    Raises the first solver's error when every one fails, since they share the
    same preconditions and that message says what to fix.
    """
    results = []
    first_error: ValueError | None = None
    for method in METHODS:
        try:
            results.append(solve(base_T_gripper, cam_T_board, method, min_axis_spread, mode))
        except (ValueError, cv2.error) as error:
            if first_error is None:
                first_error = ValueError(str(error))
    if not results:
        raise first_error or ValueError("every hand-eye solver failed")
    return sorted(results, key=lambda result: result.residual.pos_rms)


def consistent_methods(results: Sequence[HandEyeResult]) -> list[HandEyeResult]:
    """The solvers whose spread is near the best one's, best first.

    A solver whose own spread is far worse has failed on this data -- ANDREFF's
    linear solve is the usual one -- and its answer says nothing about whether
    the data pins X down.
    """
    if not results:
        return []
    limit = max(
        OUTLIER_SPREAD_RATIO * results[0].residual.pos_rms,
        results[0].residual.pos_rms + OUTLIER_SPREAD_MARGIN_MM,
    )
    return [r for r in results if r.residual.pos_rms <= limit]


def method_disagreement_mm(results: Sequence[HandEyeResult]) -> float:
    """Largest camera-position gap between the best solver and any consistent one."""
    consistent = consistent_methods(results)
    if not consistent:
        return 0.0
    best = consistent[0].camera_pose[:3, 3]
    return max(
        (float(np.linalg.norm(r.camera_pose[:3, 3] - best)) * 1000.0 for r in consistent),
        default=0.0,
    )
