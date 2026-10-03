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

"""Collect arm/board pose pairs from the live stack and solve for where a camera is.

TWO MODES
    eye_in_hand -- the camera rides on the gripper; fix the board flat on the
    table. Solves gripper -> camera.
    eye_to_hand -- the camera is fixed; mount the board rigidly on the gripper.
    Solves base -> camera.

BY HAND
    It reads the gripper pose from TF and the board from the camera. Move the
    arm with teleop, then capture.

AUTOMATICALLY (eye-in-hand only)
    Jog until the board is in view, then auto_calibrate() (A in the window)
    drives the arm through the poses in auto_poses.py, capturing at each, and
    solves. Small turns about the gripper's own axes come first, for a coarse
    solve; the rest are views from a cone about the board, aimed using it. Moves
    are planned against the robot model only -- nothing else around it is known
    -- so stay at the arm. abort() (X) cancels the current motion.

WHAT MAKES A GOOD SET
    Take 15-20 poses, turning the wrist about a genuinely different axis at
    each one -- roll, pitch, yaw -- while keeping the board in view, and vary
    the distance too. Translating the arm around constrains nothing. Rotation
    diversity is shown live: below 0.15 the solve is refused, and it should be
    above 0.4 before computing.

WHAT COMES OUT
    The solver finds the camera's optical frame, because that is the frame the
    board pose is measured in. The camera publishes camera_link -> optical
    itself, so the edge a blueprint configures (parent -> camera_link) is that
    result with the camera's own edge removed. Both, the per-method table and
    every captured sample land in `output_path`.

Keys in the calibration window: SPACE capture, U undo, C compute, A auto, X abort.
Or from `dimos shell`: app.HandEyeCalibrationModule.capture() / undo() / solve() /
auto_calibrate() / abort().
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, ClassVar, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from reactivex.disposable import Disposable

try:
    import pygame
    import pygame.image
except ImportError:
    pygame = None  # type: ignore[assignment]

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT, STATE_DIR
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.manipulation.calibration.auto_poses import (
    board_center,
    board_normal_towards,
    bootstrap_turns,
    look_at_views,
)
from dimos.manipulation.calibration.charuco import BoardPose, BoardSpec, CharucoDetector, draw
from dimos.manipulation.calibration.hand_eye_solver import (
    GOOD_AXIS_SPREAD,
    MIN_AXIS_SPREAD,
    MIN_POSES,
    HandEyeMode,
    angle_between,
    axis_spread,
    compare_methods,
    consistent_methods,
    inv_T,
    method_disagreement_mm,
)
from dimos.manipulation.calibration.intrinsics import load_zed_left_intrinsics
from dimos.manipulation.manipulation_spec import ExecutionStatus, ManipulationSpec
from dimos.manipulation.planning.spec.models import GeneratedPlan
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.perception.fiducial.marker_pose import camera_info_to_cv_matrices
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_OUTPUT = STATE_DIR / "calibration" / "hand_eye.json"
# Five solvers on the same data should agree. When they do not, the data is thin.
MAX_METHOD_DISAGREEMENT_MM = 5.0
# A capture turned less than this from every earlier one adds little.
MIN_TURN_DEG = 10.0
# Text panel under the camera view: enough for a solve summary and the snippet.
_PANEL_LINES = 14
_PANEL_PX = 22 * _PANEL_LINES + 16


class HandEyeCalibrationConfig(ModuleConfig):
    base_frame: str = "world"
    gripper_frame: str = "link7"
    # eye_in_hand reports gripper -> camera_frame, eye_to_hand base -> camera_frame.
    mode: HandEyeMode = "eye_in_hand"
    camera_frame: str = "camera_link"
    # A ZED factory .conf whose raw left-camera intrinsics replace camera_info,
    # for a ZED read as a plain webcam. None uses camera_info as published.
    intrinsics_file: str | None = None
    squares_x: int = 5
    squares_y: int = 7
    # The MEASURED square size of the print. Zero refuses to capture, because a
    # nominal size silently scales every distance in the result.
    square_mm: float = 0.0
    marker_mm: float | None = None
    dictionary: str = "DICT_5X5_1000"
    # A board seen this badly is usually motion blur or a grazing angle.
    max_reproj_px: float = 1.0
    min_axis_spread: float = MIN_AXIS_SPREAD
    # Robot TF arrives at 10 Hz, so a lookup at the image stamp needs a period.
    tf_tolerance_s: float = 0.1
    # The camera and the arm are stamped by different clocks; requiring the arm
    # to have been still across this window makes their offset irrelevant.
    still_window_s: float = 0.3
    max_still_motion_mm: float = 0.5
    max_still_motion_deg: float = 0.1
    max_image_age_s: float = 1.0
    output_path: str = str(DEFAULT_OUTPUT)
    show_window: bool = True

    # Automatic collection, eye-in-hand only. The bootstrap turns are about the
    # gripper's own origin, so they swing the view by about range * tan(angle).
    auto_bootstrap_tilt_deg: float = 8.0
    auto_bootstrap_roll_deg: float = 15.0
    auto_views: int = 16
    # Views tilt up to this far from the board normal, alternating with half of it.
    auto_max_tilt_deg: float = 25.0
    auto_max_roll_deg: float = 30.0
    # Multiples of the start range from the board centre, cycled through views.
    auto_distance_scales: list[float] = Field(default_factory=lambda: [1.0, 0.85, 1.15])
    auto_speed_scale: float = 0.2
    auto_settle_s: float = 0.6
    # The planner knows only the robot, so keep the tool point this far above
    # the board's plane.
    auto_min_clearance_m: float = 0.08
    # Refuse a plan that winds any joint further than this, e.g. a wrist flip.
    auto_max_joint_step_deg: float = 90.0
    auto_execution_timeout_s: float = 30.0


_MoveOutcome = Literal["moved", "skipped", "stop"]


class HandEyeCalibrationModule(Module):
    """Hand-eye calibration of a wrist or fixed camera against a ChArUco board."""

    # Owns a pygame window, and SDL allows one per process.
    dedicated_worker: ClassVar[bool] = True

    config: HandEyeCalibrationConfig

    color_image: In[Image]
    camera_info: In[CameraInfo]
    tf: In[TFMessage]

    # Moves the arm for automatic collection only.
    _manipulation: ManipulationSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._auto_abort = threading.Event()
        self._auto_thread: threading.Thread | None = None
        self._messages: list[str] = [
            "SPACE capture  U undo  C compute  A auto  X abort  -- jog in the teleop window"
        ]
        self._latest_image: Image | None = None
        self._camera_info: CameraInfo | None = None
        self._base_T_gripper: list[NDArray[np.float64]] = []
        self._cam_T_board: list[NDArray[np.float64]] = []
        self._optical_frame: str | None = None
        self._camera_T_optical: NDArray[np.float64] | None = None
        self._spec: BoardSpec | None = None
        self._detector: CharucoDetector | None = None
        if self.config.square_mm > 0:
            self._spec = BoardSpec.from_mm(
                self.config.squares_x,
                self.config.squares_y,
                self.config.square_mm,
                self.config.marker_mm,
                self.config.dictionary,
            )
            self._detector = CharucoDetector(self._spec)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(
            Disposable(self.color_image.subscribe(lambda msg: setattr(self, "_latest_image", msg)))
        )
        self.register_disposable(
            Disposable(self.camera_info.subscribe(lambda msg: setattr(self, "_camera_info", msg)))
        )
        # Build the buffer now so robot TF is already buffered at the first capture.
        _ = self.tfbuffer

        if self._spec is None:
            logger.warning("Hand-eye: pass the measured board square size, e.g. --square-mm 34.0")
        else:
            logger.info(f"Hand-eye board: {self._spec.describe()}")

        if self.config.show_window:
            if pygame is None:
                raise ImportError("pygame is required for the calibration window")
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._window_loop, daemon=True)
            self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        self._auto_abort.set()
        if self._thread is not None:
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        if self._auto_thread is not None:
            self._auto_thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    @rpc
    def capture(self) -> str:
        """Pair the gripper pose with the board pose in the latest frame."""
        return self._capture()[1]

    @rpc
    def auto_calibrate(self) -> str:
        """Drive the arm through calibration poses from here, capture at each, then solve.

        Start with the board in view. Progress shows in the window and the log.
        """
        if self.config.mode != "eye_in_hand":
            return "Automatic collection is eye-in-hand only -- jog and capture by hand"
        if self._detector is None:
            return "Refusing: pass the MEASURED board square size, e.g. --square-mm 34.0"
        if self._auto_thread is not None and self._auto_thread.is_alive():
            return "Automatic collection is already running (X aborts)"
        self._auto_abort.clear()
        self._auto_thread = threading.Thread(target=self._auto_run, daemon=True)
        self._auto_thread.start()
        return "Automatic collection started -- X aborts"

    @rpc
    def abort(self) -> str:
        """Stop automatic collection and the arm's current motion."""
        if self._auto_thread is None or not self._auto_thread.is_alive():
            return "Automatic collection is not running"
        self._auto_abort.set()
        self._manipulation.cancel()
        return "Aborting automatic collection"

    def _capture(self, not_before: float = 0.0, hint: bool = True) -> tuple[bool, str]:
        """Capture from the latest frame, if it was taken at or after `not_before`.

        `hint` adds advice for collecting by hand when a capture adds little.
        """
        if self._detector is None:
            return False, "Refusing: pass the MEASURED board square size, e.g. --square-mm 34.0"
        image, info = self._latest_image, self._camera_info
        if image is None or info is None:
            return False, "No camera frame or camera info yet"
        if time.time() - image.ts > self.config.max_image_age_s:
            return False, f"Latest camera frame is {time.time() - image.ts:.1f} s old"
        if image.ts < not_before:
            return False, "Waiting for a frame taken after the move"

        intrinsics = self._intrinsics(image, info)
        if isinstance(intrinsics, str):
            return False, intrinsics
        camera_matrix, dist_coeffs = intrinsics
        pose = self._detect(image, camera_matrix, dist_coeffs)
        if pose is None:
            return False, "Board not detected"
        if pose.reproj_px > self.config.max_reproj_px:
            return False, (
                f"Reprojection {pose.reproj_px:.2f} px is too high -- hold still, "
                "or face the board more squarely"
            )

        base_T_gripper, error = self._still_gripper_pose(image.ts)
        if base_T_gripper is None:
            return False, error

        optical_frame = image.frame_id or info.frame_id
        # A fixed edge, and some cameras republish it only once a second.
        camera_T_optical = self.tfbuffer.get(self.config.camera_frame, optical_frame)
        if camera_T_optical is None:
            return False, (
                f"No transform {self.config.camera_frame} -> {optical_frame} from the camera"
            )

        with self._lock:
            # Diversity measures the directions of the turns, not their size,
            # and small turns leave the camera's translation loosely determined.
            turned = min(
                (angle_between(base_T_gripper, previous) for previous in self._base_T_gripper),
                default=float("inf"),
            )
            self._optical_frame = optical_frame
            self._camera_T_optical = camera_T_optical.to_matrix()
            self._base_T_gripper.append(base_T_gripper)
            self._cam_T_board.append(pose.board_in_camera)
            count = len(self._base_T_gripper)
            spread = axis_spread(self._base_T_gripper)
        self._save_samples()
        message = (
            f"[{count:2d}] corners {pose.n_corners}  reproj {pose.reproj_px:.2f} px  "
            f"board z {pose.board_in_camera[2, 3]:.3f} m  diversity {spread:.3f}"
        )
        if turned < float("inf"):
            message += f"  {turned:.0f} deg from nearest"
            if hint and turned < MIN_TURN_DEG:
                message += " -- too close, tilt further (U to undo)"
        logger.info(f"Hand-eye capture {message}")
        return True, message

    @rpc
    def undo(self) -> str:
        """Drop the last capture."""
        with self._lock:
            if not self._base_T_gripper:
                return "Nothing to undo"
            self._base_T_gripper.pop()
            self._cam_T_board.pop()
            count = len(self._base_T_gripper)
        self._save_samples()
        return f"Undo -> {count} captures"

    @rpc
    def solve(self) -> str:
        """Solve with every method, write the best to output_path, and report it."""
        with self._lock:
            base_T_gripper = list(self._base_T_gripper)
            cam_T_board = list(self._cam_T_board)
            camera_T_optical = self._camera_T_optical
            optical_frame = self._optical_frame
        if len(base_T_gripper) < MIN_POSES or camera_T_optical is None or optical_frame is None:
            return f"{len(base_T_gripper)} captures; need at least {MIN_POSES}, want 15+"
        try:
            report, payload = solve_and_report(
                base_T_gripper,
                cam_T_board,
                camera_T_optical,
                mode=self.config.mode,
                parent_frame=self._parent_frame(),
                camera_frame=self.config.camera_frame,
                optical_frame=optical_frame,
                min_axis_spread=self.config.min_axis_spread,
            )
        except ValueError as error:
            return f"Solve refused: {error}"
        payload["board"] = self._board_dict()
        payload["intrinsics"] = self._intrinsics_dict()
        _write_json(Path(self.config.output_path), payload)
        report += f"\nwrote {self.config.output_path}"
        logger.info(f"Hand-eye result:\n{report}")
        return report

    @rpc
    def status(self) -> str:
        """Capture count and rotation diversity so far."""
        with self._lock:
            count = len(self._base_T_gripper)
            spread = axis_spread(self._base_T_gripper)
        return f"{count} captures, rotation diversity {spread:.3f}"

    def _say(self, line: str) -> None:
        """Append a progress line to the window panel and the log."""
        logger.info(f"Hand-eye auto: {line}")
        self._messages = [*self._messages, line][-(_PANEL_LINES - 1) :]

    def _auto_run(self) -> None:
        self._messages = []
        try:
            report = self._auto_collect()
        except Exception as error:
            # A thread that dies silently would leave the operator watching a still arm.
            logger.exception("Automatic hand-eye collection failed")
            report = f"Automatic collection failed: {error}"
        self._messages = report.splitlines()
        logger.info(f"Hand-eye auto:\n{report}")

    def _auto_collect(self) -> str:
        """Bootstrap turns, a coarse solve, look-at views, back to the start, solve."""
        assert self._spec is not None
        groups = [g for g in self._manipulation.list_planning_groups() if g.tip_frame is not None]
        if len(groups) != 1:
            return f"Need exactly one pose-controllable planning group, found {len(groups)}"
        group = groups[0].id

        base, gripper = self.config.base_frame, self.config.gripper_frame
        start_gripper = self.tfbuffer.get(base, gripper)
        tool_pose = self._manipulation.get_state().groups[group].end_effector_pose
        if start_gripper is None or tool_pose is None:
            return f"No {base} -> {gripper} transform or tool pose yet"
        if tool_pose.frame_id not in ("", base):
            return f"Tool pose is in {tool_pose.frame_id!r}, not {base!r}"
        base_T_gripper0 = start_gripper.to_matrix()
        base_T_tool0 = Transform(
            translation=tool_pose.position, rotation=tool_pose.orientation
        ).to_matrix()
        # Targets are planned for the tool point but chosen for the gripper frame.
        gripper_T_tool = inv_T(base_T_gripper0) @ base_T_tool0

        ok, message = self._settled_capture()
        if not ok:
            return f"Start pose: {message}. Jog until the board is in view, then press A."
        self._say(f"start: {message}")

        turns = bootstrap_turns(
            self.config.auto_bootstrap_tilt_deg, self.config.auto_bootstrap_roll_deg
        )
        for index, turn in enumerate(turns):
            target = base_T_gripper0 @ turn @ gripper_T_tool
            if self._auto_visit(group, target, f"turn {index + 1}/{len(turns)}") == "stop":
                return self._auto_stopped()

        with self._lock:
            base_T_gripper = list(self._base_T_gripper)
            cam_T_board = list(self._cam_T_board)
        try:
            coarse = compare_methods(base_T_gripper, cam_T_board, self.config.min_axis_spread)[0]
        except ValueError as error:
            return (
                f"Coarse solve failed after the bootstrap turns: {error}. Start with the "
                "board nearer the image centre, or lower --auto-bootstrap-tilt-deg."
            )
        self._say(
            f"coarse: {coarse.residual.pos_rms:.1f} mm / {coarse.residual.rot_rms:.2f} deg "
            f"spread over {coarse.n_poses} captures"
        )

        gripper_T_optical = coarse.camera_pose
        base_T_board = coarse.board_pose
        base_T_optical0 = base_T_gripper0 @ gripper_T_optical
        views = look_at_views(
            base_T_board,
            self._spec.size_m,
            base_T_optical0,
            self.config.auto_views,
            self.config.auto_max_tilt_deg,
            self.config.auto_max_roll_deg,
            self.config.auto_distance_scales,
        )
        center = board_center(base_T_board, self._spec.size_m)
        normal = board_normal_towards(base_T_board, base_T_optical0[:3, 3])
        optical_T_tool = inv_T(gripper_T_optical) @ gripper_T_tool
        for index, view in enumerate(views):
            label = f"view {index + 1}/{len(views)}"
            target = view @ optical_T_tool
            clearance = float(np.dot(target[:3, 3] - center, normal))
            if clearance < self.config.auto_min_clearance_m:
                self._say(f"{label}: tool {clearance * 100:.0f} cm above the board, skipped")
                continue
            if self._auto_visit(group, target, label) == "stop":
                return self._auto_stopped()

        if self._auto_move(group, base_T_tool0, "back to start") == "stop":
            return self._auto_stopped()
        return self.solve()

    def _auto_stopped(self) -> str:
        with self._lock:
            count = len(self._base_T_gripper)
        reason = "Aborted" if self._auto_abort.is_set() else "Stopped"
        return (
            f"{reason} with {count} captures kept. "
            + "\n".join(self._messages[-3:])
            + "\nC solves what was captured; A starts again from the current pose."
        )

    def _auto_visit(self, group: str, base_T_tool: NDArray[np.float64], label: str) -> _MoveOutcome:
        outcome = self._auto_move(group, base_T_tool, label)
        if outcome == "moved":
            _, message = self._settled_capture()
            self._say(f"{label}: {message}")
        return outcome

    def _auto_move(self, group: str, base_T_tool: NDArray[np.float64], label: str) -> _MoveOutcome:
        """Plan and execute one move; "stop" when the run must not go on."""
        if self._auto_abort.is_set():
            return "stop"
        target = PoseStamped(
            frame_id=self.config.base_frame,
            position=base_T_tool[:3, 3].tolist(),
            orientation=Quaternion.from_rotation_matrix(base_T_tool[:3, :3]),
        )
        planned = self._manipulation.plan_to_poses(
            {group: target}, speed_scale=self.config.auto_speed_scale
        )
        if not planned.succeeded or planned.plan is None:
            self._say(f"{label}: unreachable ({planned.message}), skipped")
            return "skipped"
        step = _max_joint_step_deg(planned.plan)
        if step > self.config.auto_max_joint_step_deg:
            self._say(f"{label}: plan winds a joint {step:.0f} deg, skipped")
            return "skipped"
        if self._auto_abort.is_set():
            return "stop"
        executed = self._manipulation.execute(
            blocking=True,
            timeout=self.config.auto_execution_timeout_s,
            plan_id=planned.plan.plan_id,
        )
        if executed.status is not ExecutionStatus.COMPLETED:
            self._say(f"{label}: motion {executed.status.name} {executed.message}".rstrip())
            return "stop"
        return "moved"

    def _settled_capture(self) -> tuple[bool, str]:
        """Capture once the arm has settled, retrying through a few frames."""
        settled = time.time() + self.config.auto_settle_s
        deadline = settled + 3.0
        time.sleep(self.config.auto_settle_s)
        ok, message = False, "Aborted"
        while not self._auto_abort.is_set():
            ok, message = self._capture(not_before=settled, hint=False)
            if ok or time.time() > deadline:
                break
            time.sleep(0.2)
        return ok, message

    def _parent_frame(self) -> str:
        if self.config.mode == "eye_in_hand":
            return self.config.gripper_frame
        return self.config.base_frame

    def _intrinsics(
        self, image: Image, info: CameraInfo
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | str:
        """K and distortion for this image, or why there are none."""
        if self.config.intrinsics_file is None:
            if info.K[0] <= 0 or info.width != image.width:
                return (
                    f"camera_info ({info.width} px wide, fx {info.K[0]:.1f}) does not "
                    f"describe this {image.width} px image"
                )
            return camera_info_to_cv_matrices(info)
        try:
            return load_zed_left_intrinsics(self.config.intrinsics_file, image.width)
        except ValueError as error:
            return str(error)

    def _detect(
        self,
        image: Image,
        camera_matrix: NDArray[np.float64],
        dist_coeffs: NDArray[np.float64],
    ) -> BoardPose | None:
        assert self._detector is not None
        gray = cv2.cvtColor(image.to_opencv(), cv2.COLOR_BGR2GRAY)
        return self._detector.detect(gray, camera_matrix, dist_coeffs)

    def _still_gripper_pose(self, stamp: float) -> tuple[NDArray[np.float64] | None, str]:
        """Gripper pose at `stamp`, provided the arm had been still for a window before it."""
        base, gripper = self.config.base_frame, self.config.gripper_frame
        tolerance = self.config.tf_tolerance_s
        now = self.tfbuffer.get(base, gripper, time_point=stamp, time_tolerance=tolerance)
        before = self.tfbuffer.get(
            base,
            gripper,
            time_point=stamp - self.config.still_window_s,
            time_tolerance=tolerance,
        )
        if now is None or before is None:
            return None, (
                f"No {base} -> {gripper} transform near the image stamp; is the arm "
                f"publishing TF with {gripper} in tf_extra_links?"
            )
        now_matrix, before_matrix = now.to_matrix(), before.to_matrix()
        moved_mm = float(np.linalg.norm(now_matrix[:3, 3] - before_matrix[:3, 3])) * 1000.0
        turned_deg = angle_between(now_matrix, before_matrix)
        if (
            moved_mm > self.config.max_still_motion_mm
            or turned_deg > self.config.max_still_motion_deg
        ):
            return None, f"Arm still moving ({moved_mm:.1f} mm, {turned_deg:.2f} deg) -- hold still"
        return now_matrix, ""

    def _board_dict(self) -> dict[str, Any] | None:
        if self._spec is None:
            return None
        return {
            "squares_x": self._spec.squares_x,
            "squares_y": self._spec.squares_y,
            "square_m": self._spec.square_m,
            "marker_m": self._spec.marker_m,
            "dictionary": self._spec.dictionary,
        }

    def _intrinsics_dict(self) -> dict[str, Any] | None:
        image, info = self._latest_image, self._camera_info
        if self.config.intrinsics_file is not None and image is not None:
            try:
                camera_matrix, dist_coeffs = load_zed_left_intrinsics(
                    self.config.intrinsics_file, image.width
                )
            except ValueError:
                return None
            return {
                "file": self.config.intrinsics_file,
                "K": camera_matrix.flatten().tolist(),
                "D": dist_coeffs.tolist(),
                "distortion_model": "plumb_bob",
                "width": image.width,
                "height": image.height,
            }
        if info is None:
            return None
        return {
            "K": list(info.K),
            "D": list(info.D),
            "distortion_model": info.distortion_model,
            "width": info.width,
            "height": info.height,
        }

    def _save_samples(self) -> None:
        """Keep every capture on disk so a crash or a bad solve loses nothing."""
        with self._lock:
            payload = {
                "mode": self.config.mode,
                "parent_frame": self._parent_frame(),
                "camera_frame": self.config.camera_frame,
                "optical_frame": self._optical_frame,
                "camera_T_optical": (
                    None if self._camera_T_optical is None else self._camera_T_optical.tolist()
                ),
                "base_T_gripper": [t.tolist() for t in self._base_T_gripper],
                "cam_T_board": [t.tolist() for t in self._cam_T_board],
                "board": self._board_dict(),
                "intrinsics": self._intrinsics_dict(),
            }
        _write_json(_samples_path(Path(self.config.output_path)), payload)

    def _window_loop(self) -> None:
        pygame.init()
        pygame.display.set_caption("Hand-eye calibration")
        screen = pygame.display.set_mode((848, 480 + _PANEL_PX))
        font = pygame.font.Font(None, 26)
        clock = pygame.time.Clock()

        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    if key == "space":
                        self._messages = [self.capture()]
                    elif key == "u":
                        self._messages = [self.undo()]
                    elif key == "c":
                        self._messages = self.solve().splitlines()
                    elif key == "a":
                        self._messages = [self.auto_calibrate()]
                    elif key == "x":
                        self._messages = [self.abort()]

            frame = self._render_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                if screen.get_size() != (width, height + _PANEL_PX):
                    screen = pygame.display.set_mode((width, height + _PANEL_PX))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                screen.fill((30, 30, 30))
                screen.blit(pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB"), (0, 0))
                y = height + 8
            else:
                screen.fill((30, 30, 30))
                y = 8
            for line in [self._hud_line(), *self._messages][:_PANEL_LINES]:
                screen.blit(font.render(line, True, (230, 230, 230)), (10, y))
                y += 22
            pygame.display.flip()
            clock.tick(15)
        pygame.quit()

    def _render_frame(self) -> NDArray[np.uint8] | None:
        image, info = self._latest_image, self._camera_info
        if image is None:
            return None
        bgr = np.ascontiguousarray(image.to_opencv()).copy()
        if self._detector is None or info is None:
            status = "pass --square-mm <measured>" if self._detector is None else "no camera_info"
            cv2.putText(bgr, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return bgr
        intrinsics = self._intrinsics(image, info)
        if isinstance(intrinsics, str):
            cv2.putText(bgr, intrinsics, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return bgr
        camera_matrix, dist_coeffs = intrinsics
        pose = self._detect(image, camera_matrix, dist_coeffs)
        if pose is None:
            text, color = "board NOT detected", (0, 0, 255)
        else:
            draw(bgr, pose, camera_matrix, dist_coeffs)
            assert self._spec is not None
            text = (
                f"{pose.n_corners}/{self._spec.n_corners} corners  "
                f"reproj {pose.reproj_px:.2f} px  z {pose.board_in_camera[2, 3]:.3f} m"
            )
            color = (0, 255, 0) if pose.reproj_px <= self.config.max_reproj_px else (0, 165, 255)
        cv2.putText(bgr, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return bgr

    def _hud_line(self) -> str:
        with self._lock:
            count = len(self._base_T_gripper)
            spread = axis_spread(self._base_T_gripper)
        if spread < self.config.min_axis_spread:
            verdict = f"(need > {self.config.min_axis_spread})"
        elif spread >= GOOD_AXIS_SPREAD:
            verdict = "ok"
        else:
            verdict = "thin -- turn about a new axis"
        mode = self.config.mode.replace("_", "-")
        return f"{mode}   captured {count}   rotation diversity {spread:.3f} {verdict}"


def solve_and_report(
    base_T_gripper: list[NDArray[np.float64]],
    cam_T_board: list[NDArray[np.float64]],
    camera_T_optical: NDArray[np.float64],
    *,
    mode: HandEyeMode,
    parent_frame: str,
    camera_frame: str,
    optical_frame: str,
    min_axis_spread: float = MIN_AXIS_SPREAD,
) -> tuple[str, dict[str, Any]]:
    """Run every solver, then express the best as the parent -> camera_frame edge.

    The parent is the gripper eye-in-hand and the base eye-to-hand.
    """
    results = compare_methods(base_T_gripper, cam_T_board, min_axis_spread, mode)
    best = results[0]
    parent_T_camera = best.camera_pose @ inv_T(camera_T_optical)
    translation = parent_T_camera[:3, 3]
    rotation = Quaternion.from_rotation_matrix(parent_T_camera[:3, :3])
    disagreement = method_disagreement_mm(results)
    consistent = consistent_methods(results)
    consistent_names = {r.method for r in consistent}
    failed = [r for r in results if r.method not in consistent_names]

    lines = [best.describe()]
    lines.append(f"disagreement among {len(consistent)} consistent solvers: {disagreement:.2f} mm")
    if failed:
        lines.append(
            "  left out, spread far above the best: "
            + ", ".join(f"{r.method} {r.residual.pos_rms:.1f} mm" for r in failed)
        )
    if disagreement > MAX_METHOD_DISAGREEMENT_MM:
        lines.append(
            "  the solvers disagree by more than 5 mm -- the data is thin. "
            "Collect more poses with more varied rotation."
        )
    lines.append(f"{parent_frame} -> {camera_frame}:")
    lines.append(
        "Transform(\n"
        f"    translation=Vector3(x={translation[0]:.8f}, y={translation[1]:.8f}, "
        f"z={translation[2]:.8f}),\n"
        f"    rotation=Quaternion({rotation.x:.8f}, {rotation.y:.8f}, "
        f"{rotation.z:.8f}, {rotation.w:.8f}),  # xyzw\n"
        f'    frame_id="{parent_frame}",\n'
        f'    child_frame_id="{camera_frame}",\n'
        ")"
    )

    lines.append(f"  method       {best.residual.label} spread (mm RMS, deg RMS)   camera (m)")
    for result in results:
        t = result.camera_pose[:3, 3]
        lines.append(
            f"  {result.method:11s}  {result.residual.pos_rms:6.2f}  {result.residual.rot_rms:5.2f}"
            f"                [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]"
        )
    payload: dict[str, Any] = {
        "mode": mode,
        "frame_id": parent_frame,
        "child_frame_id": camera_frame,
        "translation": translation.tolist(),
        "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        "T_parent_camera": parent_T_camera.tolist(),
        "optical_frame": optical_frame,
        "T_parent_optical": best.camera_pose.tolist(),
        # Board in base eye-in-hand, board on gripper eye-to-hand.
        "T_board": best.board_pose.tolist(),
        "method": best.method,
        "n_poses": best.n_poses,
        "residual_pos_mm_rms": best.residual.pos_rms,
        "residual_rot_deg_rms": best.residual.rot_rms,
        "residual_pos_mm_max": float(best.residual.pos_mm.max()),
        "axis_spread": best.axis_spread,
        "method_disagreement_mm": disagreement,
        "consistent_methods": sorted(consistent_names),
        "per_method": {
            r.method: {"pos_mm_rms": r.residual.pos_rms, "rot_deg_rms": r.residual.rot_rms}
            for r in results
        },
        "created": time.time(),
    }
    return "\n".join(lines), payload


def _max_joint_step_deg(plan: GeneratedPlan) -> float:
    """The furthest any joint strays from where the plan starts, in degrees."""
    if not plan.path:
        return 0.0
    positions = np.array([waypoint.position for waypoint in plan.path], dtype=np.float64)
    return float(np.degrees(np.abs(positions - positions[0]).max()))


def _samples_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_samples.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    """Re-solve a saved sample set offline, e.g. with a different diversity floor."""
    parser = argparse.ArgumentParser(description="re-solve saved hand-eye samples")
    parser.add_argument("samples", nargs="?", default=str(_samples_path(DEFAULT_OUTPUT)))
    parser.add_argument("--min-axis-spread", type=float, default=MIN_AXIS_SPREAD)
    args = parser.parse_args(argv)

    data = json.loads(Path(args.samples).read_text())
    report, _ = solve_and_report(
        [np.array(t) for t in data["base_T_gripper"]],
        [np.array(t) for t in data["cam_T_board"]],
        np.array(data["camera_T_optical"]),
        mode=data["mode"],
        parent_frame=data["parent_frame"],
        camera_frame=data["camera_frame"],
        optical_frame=data["optical_frame"],
        min_axis_spread=args.min_axis_spread,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
