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

"""Unscrew the cap of an upright bottle with the Piper, and stop when it is off.

    python .agents/skills/piper-hardware/scripts/unscrew_cap.py 0.2695 0.017 0.098 \\
        --watch 520,330,600,400

The arguments are the cap's centre (x, y) and the height of its top, in metres,
as measured from a viewing pose high enough to see it. ``--watch`` is the bottle's
body in the scene camera's frame, as x0,y0,x1,y1 pixels.

The gripper comes straight down on the cap and turns it counter-clockwise with
joint 6, re-gripping between strokes. After each stroke it lifts with the cap
still held and compares the watched pixels before and after: a bottle that moved
is still attached to its cap, one that stayed has let go of it. Needs piper-grasp
running, timelapse.py recording, and the bottle held against turning; the cap
top must be within about 10 cm of the arm's base height for the gripper to point
straight down at it (see piper_reach.py).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from numpy.typing import NDArray
import timelapse

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.porcelain.dimos import Dimos

CONNECT_TIMEOUT_S = 15.0
# Joint 6 turns +-2.09 rad; decreasing it is counter-clockwise seen from above.
WRIST_START_RAD = 1.9
WRIST_END_RAD = -1.9
# Tool-frame heights relative to the cap top. The fingertips reach 2 cm past the
# tool frame: the approach leaves them 7 mm above the cap, the grip 12 mm below
# its top, which is the whole height of a short cap.
APPROACH_ABOVE_TOP = 0.027
GRIP_ABOVE_TOP = 0.008
LIFT_CHECK_M = 0.017
TWIST_SPEED = 0.2
MAX_STROKES = 8
GRIPPER_SETTLE_S = 1.5
# The recorder writes a frame a second; wait past one before trusting the view.
FRAME_AGE_S = 1.5
# Mean absolute grey-level change of the watched pixels. Measured: 3 with
# nothing moving, 8 when the cap came away and only the gripper rose, 25 when
# the bottle was lifted 15 mm by its neck.
MOVED_THRESHOLD = 15.0
SETTLE_TOLERANCE_RAD = 0.001
SETTLE_TIMEOUT_S = 2.0


class Arm:
    def __init__(self, app: Dimos) -> None:
        self._motion = app.ManipulationModule
        self._group = self._motion.list_planning_groups()[0].id

    def joints(self) -> NDArray[np.float64]:
        return np.asarray(self._motion.get_state().groups[self._group].joints.position)

    def tool_height(self) -> float:
        pose = self._motion.get_state().groups[self._group].end_effector_pose
        return float(pose.position.z)

    def gripper_opening(self) -> float:
        return float(self._motion.get_state().groups[self._group].gripper_position)

    def _settle(self) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        last, still = self.joints(), 0
        while still < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
            now = self.joints()
            still = still + 1 if np.abs(now - last).max() <= SETTLE_TOLERANCE_RAD else 0
            last = now

    def _run(self, plan_message: str, succeeded: bool) -> None:
        if not succeeded:
            raise RuntimeError(f"planning failed: {plan_message}")
        result = self._motion.execute(blocking=True)
        if not result.succeeded:
            raise RuntimeError(f"motion failed: {result.message}")
        self._settle()

    def to_pose_above(self, x: float, y: float, z: float) -> None:
        straight_down = Quaternion.from_euler(Vector3(-math.pi, 0.0, math.pi))
        target = PoseStamped(frame_id="world", position=Vector3(x, y, z), orientation=straight_down)
        plan = self._motion.plan_to_poses({self._group: target}, 0.5)
        self._run(plan.message, plan.succeeded)

    def wrist_to(self, angle: float, speed: float) -> None:
        joints = self.joints()
        joints[5] = angle
        names = [f"joint{index}" for index in range(1, 7)]
        target = JointState(name=names, position=[float(q) for q in joints])
        plan = self._motion.plan_to_joints({self._group: target}, speed)
        self._run(plan.message, plan.succeeded)

    def raise_by(self, dz: float) -> None:
        result = self._motion.move_linear(0.0, 0.0, dz, self._group, check_collision=False)
        if not result.plan.succeeded or result.execution is None or not result.execution.succeeded:
            raise RuntimeError(f"vertical move failed: {result.plan.message}")
        self._settle()

    def grip(self, opening: float) -> float:
        self._motion.set_gripper_position(opening, self._group)
        time.sleep(GRIPPER_SETTLE_S)
        return self.gripper_opening()


def watched_pixels(box: tuple[int, int, int, int]) -> NDArray[np.float32]:
    """The watched part of the scene camera's next frame, in grey levels."""
    recording = json.loads(timelapse.CURRENT.read_text())
    time.sleep(FRAME_AGE_S)
    frame = cv2.imread(str(Path(recording["run_dir"]) / "latest.jpg"))
    x0, y0, x1, y1 = box
    return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)


def parse_box(text: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in text.split(","))
    return x0, y0, x1, y1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("x", type=float, help="cap centre x, metres")
    parser.add_argument("y", type=float, help="cap centre y, metres")
    parser.add_argument("top", type=float, help="height of the cap's top, metres")
    parser.add_argument("--watch", type=parse_box, required=True, help="x0,y0,x1,y1 pixels")
    args = parser.parse_args()

    app = Dimos.connect(timeout=CONNECT_TIMEOUT_S)
    try:
        arm = Arm(app)
        arm.grip(1.0)
        arm.to_pose_above(args.x, args.y, args.top + APPROACH_ABOVE_TOP)
        arm.wrist_to(WRIST_START_RAD, speed=0.5)
        arm.raise_by(args.top + GRIP_ABOVE_TOP - arm.tool_height())

        for stroke in range(1, MAX_STROKES + 1):
            held = arm.grip(0.0)
            arm.wrist_to(WRIST_END_RAD, speed=TWIST_SPEED)
            before = watched_pixels(args.watch)
            arm.raise_by(LIFT_CHECK_M)
            change = float(np.abs(watched_pixels(args.watch) - before).mean())
            print(f"stroke {stroke}: jaws at {held:.3f}, bottle changed by {change:.1f} on lifting")
            if change < MOVED_THRESHOLD:
                print(f"cap is off after {stroke} stroke(s); the gripper is holding it")
                return
            arm.raise_by(-LIFT_CHECK_M)
            arm.grip(1.0)
            arm.wrist_to(WRIST_START_RAD, speed=0.5)
        raise SystemExit(f"cap still on after {MAX_STROKES} strokes")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
