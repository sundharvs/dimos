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

"""XR headset full-body teleop over xr-robot-teleop-server (WebRTC).

The headset app POSTs an SDP offer to ``http://<host>:<port>/offer`` and streams
skeletons on the ``body_pose`` data channel. The right wrist flies the arm; the
left hand signs commands and (by default) drives the gripper.

Outputs match ``ArmTeleopModule`` so the module drops into the same coordinator
remappings:

- ``right_controller_output``: absolute right-wrist pose in the operator body
  frame, scaled and rotated into the robot base frame. ``TeleopIKTask``
  anchors it to the TCP when the latch engages and applies deltas about base
  axes, so a dropout can never integrate into drift.
- ``right_gripper_command``: normalized opening (1 open, 0 closed).
- ``teleop_buttons``: ``right_grip`` is the latch (the task's deadman). Gesture
  commands for ``EpisodeMonitorModule`` are pulsed on ``B`` (start/save) and
  ``Y`` (discard), its default button map.
"""

from __future__ import annotations

from enum import Enum
import threading
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation
import uvicorn
from xr_robot_teleop_server.schemas.body_pose import deserialize_pose_data
from xr_robot_teleop_server.streaming import WebRTCServer

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.imitation.collection.episode_monitor import EpisodeStatus
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.teleop.webxr.controller_types import Buttons
from dimos.teleop.xr_server.body_pose import XrBodyPose, bones_to_body_pose
from dimos.teleop.xr_server.gestures import (
    Gesture,
    GestureGate,
    classify,
    gripper_from_gesture,
    gripper_from_pinch,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

POSE_FRAME_ID = "xr_body"
# Buttons pulsed for EpisodeMonitorModule's default button map.
_TAKE_TOGGLE_BUTTON = "right_secondary"  # "B"
_TAKE_DISCARD_BUTTON = "left_secondary"  # "Y"


class _LatchRequest(Enum):
    LATCH = "latch"
    RELEASE = "release"


class XrServerTeleopConfig(ModuleConfig):
    """Configuration for XrServerTeleopModule."""

    server_host: str = "0.0.0.0"
    server_port: int = 8080
    control_loop_hz: float = Field(default=20.0, gt=0)
    # No body_pose message for this long: the pose is stale and the latch releases.
    stale_timeout_s: float = Field(default=0.5, gt=0)
    # Hand displacement -> TCP displacement. 1.0 for bring-up; the reference rig ran 2.2.
    position_scale: float = Field(default=1.0, gt=0)
    # Operator facing relative to robot +x. Stand facing +x with 0.
    body_yaw_deg: float = 0.0
    # Time constant of the EMA on the published pose (geodesic on rotation).
    # Reset on every latch so a re-latch is not smeared. 0 disables it.
    ema_tau_s: float = Field(default=0.15, ge=0)
    gestures: bool = True
    gripper_hand: Literal["left", "right"] = "left"


class XrServerTeleopModule(Module):
    """Arm teleop from xr-robot-teleop-server full-body tracking."""

    config: XrServerTeleopConfig

    right_controller_output: Out[PoseStamped]
    right_gripper_command: Out[Float32]
    teleop_buttons: Out[Buttons]
    status: In[EpisodeStatus]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest: XrBodyPose | None = None
        self._latest_at: float | None = None
        self._messages = 0

        self._latched = False
        self._latch_request: _LatchRequest | None = None
        self._gripper_closed = False
        self._last_rotation: NDArray[np.float64] | None = None
        # Set when the hand frame was missing on the first latched tick: the
        # orientation is then held for the whole latch, as no reference exists.
        self._hold_rotation = False
        # EMA state for the published pose; None until the first latched tick.
        self._ema_position: NDArray[np.float64] | None = None
        self._ema_rotation: Rotation | None = None
        self._ema_at = 0.0
        self._last_gesture: Gesture | None = None
        self._gate = GestureGate()
        self._pulse: set[str] = set()
        self._recording = False

        self._server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    @rpc
    def build(self) -> None:
        super().build()
        if self.status.connection is not None or self.status._transport is not None:
            self.register_disposable(Disposable(self.status.subscribe(self._on_episode_status)))

    @rpc
    def start(self) -> None:
        super().start()
        webrtc = WebRTCServer(
            host=self.config.server_host,
            port=self.config.server_port,
            datachannel_handlers={
                "body_pose": self._on_body_pose,
                # The headset client may open these; nothing to do with them.
                "apriltag_pose": lambda message: None,
                "haptics": lambda message: None,
            },
            state_factory=dict,
        )
        # WebRTCServer.run() calls uvicorn.run(), which cannot be stopped, so
        # drive its app with a Server we can signal.
        self._server = uvicorn.Server(
            uvicorn.Config(
                webrtc.app,
                host=self.config.server_host,
                port=self.config.server_port,
                log_level="warning",
            )
        )
        self._server_thread = threading.Thread(
            target=self._server.run, daemon=True, name="XrServerTeleopWebRTC"
        )
        self._server_thread.start()

        self._stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="XrServerTeleopControlLoop"
        )
        self._loop_thread.start()
        logger.info(
            "XR server teleop started",
            offer_url=f"http://{self.config.server_host}:{self.config.server_port}/offer",
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1.0)
            self._loop_thread = None
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
            self._server_thread = None
        self._server = None
        with self._lock:
            self._release_locked("module stopped")
        self.teleop_buttons.publish(Buttons())
        super().stop()

    # ── operator controls (also reachable from `dimos shell`) ────────────────

    @rpc
    def latch(self) -> None:
        """Anchor the current wrist pose to the TCP and go live."""
        with self._lock:
            self._latch_request = _LatchRequest.LATCH

    @rpc
    def release(self) -> None:
        """Release the latch; the arm holds where it is."""
        with self._lock:
            self._latch_request = _LatchRequest.RELEASE

    # ── inputs ───────────────────────────────────────────────────────────────

    def _on_body_pose(self, message: bytes) -> None:
        """Runs on the WebRTC server's event loop; must never raise."""
        try:
            pose = bones_to_body_pose(deserialize_pose_data(message))
        except Exception:
            logger.exception("Dropping malformed body_pose message")
            return
        with self._lock:
            self._latest = pose
            self._latest_at = time.monotonic()
            self._messages += 1
            if self._messages == 1:
                logger.info("XR body tracking acquired")

    def _on_episode_status(self, status: EpisodeStatus) -> None:
        with self._lock:
            self._recording = status.state == "recording"

    # ── control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        period = 1.0 / self.config.control_loop_hz
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            try:
                self._tick(time.monotonic())
            except Exception:
                logger.exception("Error in XR server teleop control loop")
            # Fixed rate: schedule from the previous deadline so timing does not drift.
            next_tick += period
            self._stop_event.wait(max(0.0, next_tick - time.monotonic()))

    def _tick(self, now: float) -> None:
        with self._lock:
            pose = self._fresh_pose_locked(now)
            wrist = pose.right_wrist_position if pose is not None else None
            wrist_rotation = pose.right_wrist_rotation if pose is not None else None
            gesture = classify(pose.left_fingers) if pose is not None else None
            self._log_gesture_locked(gesture)

            if self._latched and wrist is None:
                self._release_locked("lost XR wrist")

            request, self._latch_request = self._latch_request, None
            if request is _LatchRequest.LATCH:
                self._latch_locked(wrist)
            elif request is _LatchRequest.RELEASE:
                self._release_locked("released")

            if self.config.gestures:
                command = self._gate.push(gesture, now)
                if command is not None:
                    self._on_command_locked(command, wrist)

            buttons = Buttons()
            buttons.right_grip = self._latched
            for name in self._pulse:
                buttons.set_attribute(name, True)
            self._pulse.clear()

            output: PoseStamped | None = None
            gripper: Float32 | None = None
            if self._latched and wrist is not None and pose is not None:
                output = self._output_pose_locked(wrist, wrist_rotation, now)
                gripper = self._gripper_locked(pose, gesture)

        # The deadman must go out before the pose so TeleopIKTask is engaged
        # (and captures its reference) on the first latched sample.
        self.teleop_buttons.publish(buttons)
        if output is not None:
            self.right_controller_output.publish(output)
        if gripper is not None:
            self.right_gripper_command.publish(gripper)

    def _fresh_pose_locked(self, now: float) -> XrBodyPose | None:
        if self._latest_at is None or now - self._latest_at > self.config.stale_timeout_s:
            return None
        return self._latest

    def _on_command_locked(self, command: Gesture, wrist: NDArray[np.float64] | None) -> None:
        logger.info("XR gesture command", gesture=command.value)
        if command is Gesture.ONE:
            self._latch_locked(wrist)
        elif command is Gesture.TWO:
            if self._recording:
                # Ending a take: save, then go cold so the operator re-latches.
                self._pulse.add(_TAKE_TOGGLE_BUTTON)
                self._release_locked("take saved")
            elif self._latch_locked(wrist):
                # Starting a take latches first so frame 0 is the latched pose.
                self._pulse.add(_TAKE_TOGGLE_BUTTON)
        elif command is Gesture.THREE:
            # TODO: home the arm (blocking move to fixed start joints, gripper
            # open) once a home task exists. For now three fingers only stops.
            self._release_locked("stop gesture")
        elif command is Gesture.PINKY:
            self._pulse.add(_TAKE_DISCARD_BUTTON)
            self._release_locked("take discarded")

    def _latch_locked(self, wrist: NDArray[np.float64] | None) -> bool:
        if wrist is None:
            logger.warning("Cannot latch: no XR wrist")
            return False
        if self._latched:
            # Re-latch: TeleopIKTask re-captures its reference on a fresh
            # deadman edge, so drop the deadman for this tick.
            self._latched = False
            self._latch_request = _LatchRequest.LATCH
            return True
        self._latched = True
        self._last_rotation = None
        self._hold_rotation = False
        self._ema_position = self._ema_rotation = None
        logger.info("XR teleop latched")
        return True

    def _release_locked(self, reason: str) -> None:
        if self._latched:
            logger.info("XR teleop released", reason=reason)
        self._latched = False
        self._last_rotation = None
        self._ema_position = self._ema_rotation = None

    def _output_pose_locked(
        self,
        wrist: NDArray[np.float64],
        wrist_rotation: NDArray[np.float64] | None,
        now: float,
    ) -> PoseStamped:
        # On a hand-frame dropout hold the last orientation we sent: snapping
        # back to the latched one would lunge.
        if self._last_rotation is None and wrist_rotation is None:
            self._hold_rotation = True
            logger.warning("XR hand frame missing at latch; orientation held until re-latch")
        if wrist_rotation is not None and not self._hold_rotation:
            self._last_rotation = wrist_rotation
        if self._last_rotation is None:
            self._last_rotation = np.eye(3)
        rotation = self._last_rotation
        # The body -> base yaw C maps hand deltas onto base axes. TeleopIKTask
        # applies the rotation delta on the left, so publishing C R Cᵀ makes
        # its target C (R R0ᵀ) Cᵀ R_robot0.
        c = Rotation.from_euler("z", self.config.body_yaw_deg, degrees=True).as_matrix()
        position, orientation = self._smooth_locked(
            self.config.position_scale * (c @ wrist),
            Rotation.from_matrix(c @ rotation @ c.T),
            now,
        )
        quat = orientation.as_quat()  # xyzw
        return PoseStamped(
            ts=time.time(),
            frame_id=POSE_FRAME_ID,
            position=Vector3(*position),
            orientation=Quaternion(*quat),
        )

    def _smooth_locked(
        self, position: NDArray[np.float64], rotation: Rotation, now: float
    ) -> tuple[NDArray[np.float64], Rotation]:
        """EMA toward the raw pose. The first latched sample passes through
        unfiltered, so TeleopIKTask anchors to the true hand pose."""
        if self.config.ema_tau_s == 0.0 or self._ema_position is None or self._ema_rotation is None:
            self._ema_position, self._ema_rotation, self._ema_at = position, rotation, now
            return position, rotation
        alpha = 1.0 - float(np.exp(-max(0.0, now - self._ema_at) / self.config.ema_tau_s))
        self._ema_at = now
        self._ema_position = self._ema_position + alpha * (position - self._ema_position)
        # Geodesic step: move alpha of the way along the shortest rotation to the target.
        delta = (rotation * self._ema_rotation.inv()).as_rotvec()
        self._ema_rotation = Rotation.from_rotvec(alpha * delta) * self._ema_rotation
        return self._ema_position, self._ema_rotation

    def _gripper_locked(self, pose: XrBodyPose, gesture: Gesture | None) -> Float32:
        if self.config.gripper_hand == "left":
            self._gripper_closed = gripper_from_gesture(gesture, self._gripper_closed)
        else:
            self._gripper_closed = gripper_from_pinch(
                pose.right_pinch_m, pose.right_wrist_rotation is not None, self._gripper_closed
            )
        return Float32(data=0.0 if self._gripper_closed else 1.0)

    def _log_gesture_locked(self, gesture: Gesture | None) -> None:
        if gesture != self._last_gesture:
            logger.debug("XR left-hand sign", gesture=gesture.value if gesture else None)
            self._last_gesture = gesture
