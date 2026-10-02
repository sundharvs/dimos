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

"""Left-hand sign classification and the hold-to-fire gate.

A finger counts as extended when the cosine of its PIP bend is >= 0.55, which
does not depend on hand size or wrist pose. The thumb is never counted.

    sign             gesture  command
    index only       ONE      latch / re-latch
    index + middle   TWO      start / save take (auto-latches)
    three fingers    THREE    stop: release the latch
    pinky only       PINKY    discard take
    fist / open      GRIP / OPEN  gripper close / open (left-hand gripper mode)
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from dimos.teleop.xr_server.body_pose import FingerJoints

EXTENDED_COS = 0.55
# Right-hand pinch mode: thumb-index gap above this opens the gripper.
PINCH_OPEN_THRESHOLD_M = 0.05
HOLD_S = 1.0
MIN_FRAMES = 3
GLITCH_FRAMES = 2

_JOINTS: dict[str, tuple[str, str, str]] = {
    "index": ("index_finger_mcp", "index_finger_pip", "index_finger_tip"),
    "middle": ("middle_finger_mcp", "middle_finger_pip", "middle_finger_tip"),
    "ring": ("ring_finger_mcp", "ring_finger_pip", "ring_finger_tip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_tip"),
}


class Gesture(str, Enum):
    GRIP = "GRIP"
    OPEN = "OPEN"
    ONE = "ONE"
    TWO = "TWO"
    THREE = "THREE"
    PINKY = "PINKY"


COMMAND_GESTURES = frozenset({Gesture.ONE, Gesture.TWO, Gesture.THREE, Gesture.PINKY})


def _finger_cos(
    finger: dict[str, NDArray[np.float64]], names: tuple[str, str, str]
) -> float | None:
    """cos of the bend at the PIP joint: +1 straight, -1 folded back."""
    points = [finger.get(n) for n in names]
    if any(p is None or not np.all(np.isfinite(p)) for p in points):
        return None
    mcp, pip, tip = points
    a, b = pip - mcp, tip - pip  # type: ignore[operator]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


def classify(fingers: FingerJoints) -> Gesture | None:
    """Classify a hand sign, or None if any counted finger is untracked or the sign is undefined."""
    extended: dict[str, bool] = {}
    for name, joints in _JOINTS.items():
        cos = _finger_cos(fingers.get(name, {}), joints)
        if cos is None:
            return None
        extended[name] = cos >= EXTENDED_COS
    count = sum(extended.values())
    if count == 0:
        return Gesture.GRIP
    if count == 4:
        return Gesture.OPEN
    if count == 1 and extended["index"]:
        return Gesture.ONE
    if count == 1 and extended["pinky"]:
        return Gesture.PINKY
    if count == 2 and extended["index"] and extended["middle"]:
        return Gesture.TWO
    if count == 3:
        return Gesture.THREE
    return None


class GestureGate:
    """A command sign held HOLD_S seconds (and >= MIN_FRAMES) fires once."""

    def __init__(self) -> None:
        self._pose: Gesture | None = None
        self._since = 0.0
        self._frames = 0
        self._fired = False
        self._bad = 0

    def push(self, pose: Gesture | None, now: float) -> Gesture | None:
        """Feed one classification; returns the command gesture that fired, if any."""
        if pose == self._pose:
            self._bad = 0
        else:
            self._bad += 1
            if self._bad > GLITCH_FRAMES:
                self._pose, self._since = pose, now
                self._frames, self._fired, self._bad = 0, False, 0
        self._frames += 1
        if (
            self._pose in COMMAND_GESTURES
            and not self._fired
            and now - self._since >= HOLD_S
            and self._frames >= MIN_FRAMES
        ):
            self._fired = True
            return self._pose
        return None


# Gripper state: True = closed. Every function HOLDS on missing data; an
# occlusion near an object must never open the gripper.


def gripper_from_gesture(gesture: Gesture | None, closed: bool) -> bool:
    """Left-hand mode: fist closes, open hand opens, anything else holds."""
    if gesture is Gesture.GRIP:
        return True
    if gesture is Gesture.OPEN:
        return False
    return closed


def gripper_from_pinch(pinch_m: float | None, hand_frame_valid: bool, closed: bool) -> bool:
    """Right-hand mode: thumb-index gap above PINCH_OPEN_THRESHOLD_M opens."""
    if not hand_frame_valid or pinch_m is None or not np.isfinite(pinch_m):
        return closed
    return pinch_m <= PINCH_OPEN_THRESHOLD_M
