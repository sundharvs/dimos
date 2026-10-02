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

"""Collect arm/board pose pairs from the live stack and solve for the wrist camera mount.

IT NEVER COMMANDS THE ARM
    It reads the gripper pose from TF and the board from the camera. Move the
    arm with teleop, then capture. There is no motion path in this module.

WHAT MAKES A GOOD SET
    Fix the board flat on the table. Then take 15-20 poses, turning the wrist
    about a genuinely different axis at each one -- roll, pitch, yaw -- while
    keeping the board in view, and vary the distance too. Translating the arm
    around constrains nothing. Rotation diversity is shown live: below 0.15 the
    solve is refused, and it should be above 0.4 before computing.

WHAT COMES OUT
    The solver finds gripper -> camera optical frame, because that is the frame
    the board pose is measured in. The camera publishes camera_link -> optical
    itself from factory extrinsics, so the mount edge a blueprint configures,
    gripper -> camera_link, is that result with the camera's own edge removed.
    Both, the per-method table and every captured sample land in `output_path`.

Keys in the calibration window: SPACE capture, U undo, C compute.
Or from `dimos shell`: app.HandEyeCalibrationModule.capture() / undo() / solve().
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray
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
from dimos.manipulation.calibration.charuco import BoardPose, BoardSpec, CharucoDetector, draw
from dimos.manipulation.calibration.hand_eye_solver import (
    GOOD_AXIS_SPREAD,
    MIN_AXIS_SPREAD,
    MIN_POSES,
    angle_between,
    axis_spread,
    compare_methods,
    inv_T,
    method_disagreement_mm,
)
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.perception.fiducial.marker_pose import camera_info_to_cv_matrices
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_OUTPUT = STATE_DIR / "calibration" / "hand_eye.json"
# Five solvers on the same data should agree. When they do not, the data is thin.
MAX_METHOD_DISAGREEMENT_MM = 5.0
# Text panel under the camera view: enough for a solve summary and the snippet.
_PANEL_LINES = 14
_PANEL_PX = 22 * _PANEL_LINES + 16


class HandEyeCalibrationConfig(ModuleConfig):
    base_frame: str = "world"
    gripper_frame: str = "link7"
    # The mount edge to report: gripper -> camera_frame.
    camera_frame: str = "camera_link"
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


class HandEyeCalibrationModule(Module):
    """Eye-in-hand calibration of a wrist camera against a fixed ChArUco board."""

    # Owns a pygame window, and SDL allows one per process.
    dedicated_worker: ClassVar[bool] = True

    config: HandEyeCalibrationConfig

    color_image: In[Image]
    camera_info: In[CameraInfo]
    tf: In[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
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
        if self._thread is not None:
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    @rpc
    def capture(self) -> str:
        """Pair the gripper pose with the board pose in the latest frame."""
        if self._detector is None:
            return "Refusing: pass the MEASURED board square size, e.g. --square-mm 34.0"
        image, info = self._latest_image, self._camera_info
        if image is None or info is None:
            return "No camera frame or camera info yet"
        if time.time() - image.ts > self.config.max_image_age_s:
            return f"Latest camera frame is {time.time() - image.ts:.1f} s old"

        camera_matrix, dist_coeffs = camera_info_to_cv_matrices(info)
        pose = self._detect(image, camera_matrix, dist_coeffs)
        if pose is None:
            return "Board not detected"
        if pose.reproj_px > self.config.max_reproj_px:
            return (
                f"Reprojection {pose.reproj_px:.2f} px is too high -- hold still, "
                "or face the board more squarely"
            )

        base_T_gripper, error = self._still_gripper_pose(image.ts)
        if base_T_gripper is None:
            return error

        optical_frame = image.frame_id or info.frame_id
        camera_T_optical = self.tfbuffer.get(
            self.config.camera_frame, optical_frame, time_point=image.ts, time_tolerance=1.0
        )
        if camera_T_optical is None:
            return f"No transform {self.config.camera_frame} -> {optical_frame} from the camera"

        with self._lock:
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
        logger.info(f"Hand-eye capture {message}")
        return message

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
                gripper_frame=self.config.gripper_frame,
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
        info = self._camera_info
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
                "gripper_frame": self.config.gripper_frame,
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
        messages = ["SPACE capture   U undo   C compute   -- jog the arm in the teleop window"]

        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    if key == "space":
                        messages = [self.capture()]
                    elif key == "u":
                        messages = [self.undo()]
                    elif key == "c":
                        messages = self.solve().splitlines()

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
            for line in [self._hud_line(), *messages][:_PANEL_LINES]:
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
        camera_matrix, dist_coeffs = camera_info_to_cv_matrices(info)
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
        return f"captured {count}   rotation diversity {spread:.3f} {verdict}"


def solve_and_report(
    base_T_gripper: list[NDArray[np.float64]],
    cam_T_board: list[NDArray[np.float64]],
    camera_T_optical: NDArray[np.float64],
    *,
    gripper_frame: str,
    camera_frame: str,
    optical_frame: str,
    min_axis_spread: float = MIN_AXIS_SPREAD,
) -> tuple[str, dict[str, Any]]:
    """Run every solver, then express the best as the gripper -> camera_frame mount."""
    results = compare_methods(base_T_gripper, cam_T_board, min_axis_spread)
    best = results[0]
    gripper_T_camera = best.camera_in_gripper @ inv_T(camera_T_optical)
    translation = gripper_T_camera[:3, 3]
    rotation = Quaternion.from_rotation_matrix(gripper_T_camera[:3, :3])
    disagreement = method_disagreement_mm(results)

    lines = [best.describe()]
    lines.append(f"best-to-worst disagreement: {disagreement:.2f} mm")
    if disagreement > MAX_METHOD_DISAGREEMENT_MM:
        lines.append(
            "  the solvers disagree by more than 5 mm -- the data is thin. "
            "Collect more poses with more varied rotation."
        )
    lines.append(f"{gripper_frame} -> {camera_frame}:")
    lines.append(
        "Transform(\n"
        f"    translation=Vector3(x={translation[0]:.8f}, y={translation[1]:.8f}, "
        f"z={translation[2]:.8f}),\n"
        f"    rotation=Quaternion({rotation.x:.8f}, {rotation.y:.8f}, "
        f"{rotation.z:.8f}, {rotation.w:.8f}),  # xyzw\n"
        f'    frame_id="{gripper_frame}",\n'
        f'    child_frame_id="{camera_frame}",\n'
        ")"
    )

    lines.append("  method       board spread (mm RMS, deg RMS)   camera in gripper (m)")
    for result in results:
        t = result.camera_in_gripper[:3, 3]
        lines.append(
            f"  {result.method:11s}  {result.residual.pos_rms:6.2f}  {result.residual.rot_rms:5.2f}"
            f"                [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]"
        )
    payload: dict[str, Any] = {
        "frame_id": gripper_frame,
        "child_frame_id": camera_frame,
        "translation": translation.tolist(),
        "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        "T_gripper_camera": gripper_T_camera.tolist(),
        "optical_frame": optical_frame,
        "T_gripper_optical": best.camera_in_gripper.tolist(),
        "T_base_board": best.board_in_base.tolist(),
        "method": best.method,
        "n_poses": best.n_poses,
        "residual_pos_mm_rms": best.residual.pos_rms,
        "residual_rot_deg_rms": best.residual.rot_rms,
        "residual_pos_mm_max": float(best.residual.pos_mm.max()),
        "axis_spread": best.axis_spread,
        "method_disagreement_mm": disagreement,
        "per_method": {
            r.method: {"pos_mm_rms": r.residual.pos_rms, "rot_deg_rms": r.residual.rot_rms}
            for r in results
        },
        "created": time.time(),
    }
    return "\n".join(lines), payload


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
        gripper_frame=data["gripper_frame"],
        camera_frame=data["camera_frame"],
        optical_frame=data["optical_frame"],
        min_axis_spread=args.min_axis_spread,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
