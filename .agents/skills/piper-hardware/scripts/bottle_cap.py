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

"""Unscrew or screw on the cap of an upright, clamped bottle with the Piper.

    python .agents/skills/piper-hardware/scripts/bottle_cap.py unscrew 0.2835 0.0115 0.1236 \\
        --watch 525,310,570,400
    python .agents/skills/piper-hardware/scripts/bottle_cap.py screw 0.2835 0.0115 0.1236 \\
        --watch 525,310,570,400

The arguments are the cap's centre (x, y) and the height of its top when screwed
on, in metres, measured from the viewing pose. ``--watch`` is the bottle's body
in the scene camera's frame, as x0,y0,x1,y1 pixels.

``unscrew`` grips the cap from above, turns it counter-clockwise with joint 6 and
ends holding it. ``screw`` takes a cap that is in the gripper or resting on the
spout, turns it clockwise while easing down, and ends with the gripper clear.
Neither trusts a proxy for where the cap is. After every stroke the arm lifts,
retracts to the viewing pose with the jaws still closed, and looks: the cap is on
the bottle only if the detector finds it at the bottle's top and at the cap's
height (an open spout is found lower down). A grip narrower than the first one
is not on the cap, so it is not twisted. The arm never goes back down holding a
cap it has not accounted for. Each run leaves a step log and a contact sheet of
scene-camera frames, one per phase, in its own folder: read the sheet before
believing the result. Exit status is 0 on success, 2 when the cap came off but
was dropped, 1 otherwise.

Needs piper-grasp running, timelapse.py recording, and the bottle clamped so it
can neither turn nor lift. The gripper leans a few degrees off vertical, which
lets it reach a cap up to about 15 cm above the arm's base.
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
from scipy.spatial.transform import Rotation
import timelapse

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.porcelain.dimos import Dimos

Joints = NDArray[np.float64]

CONNECT_TIMEOUT_S = 15.0
# Joint 6 turns +-2.09 rad; increasing it is clockwise seen from above.
WRIST_LIMIT_RAD = 1.9
# How far the gripper leans outward from vertical. Straight down, the wrist
# pitch limit stops the tool about 13 cm above the base; a few degrees of lean
# reach higher and barely tilt the jaws on the cap.
GRIP_TILT_DEG = 4.0
CLEAR_TILT_DEG = 8.0
# Tool-frame heights relative to the cap top. The fingertips reach 2 cm past the
# tool frame, so the grip puts them 12 mm below the top of the cap and the
# approach leaves them 7 mm above it.
GRIP_ABOVE_TOP = 0.008
CLEAR_ABOVE_TOP = 0.027
LIFT_CHECK_M = 0.017
# A cap resting on its thread start sits about this far above its seat.
THREAD_TRAVEL_M = 0.003
TWIST_SPEED = 0.2
SCREW_STEPS = 13
# A wrist this far behind its target after settling is being held by the cap.
WRIST_LAG_RAD = 0.05
MAX_STROKES = 6
GRIPPER_SETTLE_S = 1.5
# The recorder writes a frame a second; wait past one before using its view.
FRAME_AGE_S = 1.5
# Margin around the watched box in the frames kept for the contact sheet.
SHEET_MARGIN_PX = 90
SETTLE_TOLERANCE_RAD = 0.001
SETTLE_TIMEOUT_S = 2.0
# Jaws closed further than this normalized opening are holding nothing.
EMPTY_JAWS = 0.1
# Where the wrist camera sees the top of an object standing in front of the arm.
VIEW_JOINTS = (0.0, 0.5, -0.9, 0.0, 1.2, 0.0)
VIEW_SETTLE_S = 1.5
# A detected cap this close to where the cap belongs is on the bottle, provided
# its points are this near the cap's top: an open spout sits about 15 mm lower.
ON_BOTTLE_RADIUS_M = 0.03
ON_BOTTLE_DEPTH_M = 0.008
# A grip this much narrower than the first is on something other than the cap.
NARROWER_GRIP = 0.03
RUNS = Path(timelapse.ROOT).parent / "bottle_cap"


class Arm:
    def __init__(self, app: Dimos) -> None:
        self._motion = app.ManipulationModule
        self._group = self._motion.list_planning_groups()[0].id

    def joints(self) -> Joints:
        return np.asarray(self._motion.get_state().groups[self._group].joints.position)

    def jaws(self) -> float:
        return float(self._motion.get_state().groups[self._group].gripper_position)

    def _settle(self) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        last, still = self.joints(), 0
        while still < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
            now = self.joints()
            still = still + 1 if np.abs(now - last).max() <= SETTLE_TOLERANCE_RAD else 0
            last = now

    def solve(self, x: float, y: float, z: float, tilt_deg: float) -> Joints:
        """Joints that put the tool at a point, leaning outward, jaws tangential."""
        bearing, lean = math.atan2(y, x), math.radians(tilt_deg)
        approach = np.array(
            [
                math.sin(lean) * math.cos(bearing),
                math.sin(lean) * math.sin(bearing),
                -math.cos(lean),
            ]
        )
        jaw = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
        frame = np.column_stack([np.cross(jaw, approach), jaw, approach])
        qx, qy, qz, qw = Rotation.from_matrix(frame).as_quat()
        target = PoseStamped(
            frame_id="world", position=Vector3(x, y, z), orientation=Quaternion(qx, qy, qz, qw)
        )
        plan = self._motion.plan_to_poses({self._group: target}, 0.5)
        if not plan.succeeded:
            raise RuntimeError(f"cannot reach ({x:.3f}, {y:.3f}, {z:.3f}): {plan.message}")
        solution = np.asarray(plan.plan.trajectory.points[-1].positions[:6], dtype=np.float64)
        self._motion.clear_planned_path()
        return solution

    def move(self, joints: Joints, wrist: float, speed: float) -> None:
        names = [f"joint{index}" for index in range(1, 7)]
        positions = [*(float(q) for q in joints[:5]), float(wrist)]
        plan = self._motion.plan_to_joints(
            {self._group: JointState(name=names, position=positions)}, speed
        )
        if not plan.succeeded:
            raise RuntimeError(f"planning failed: {plan.message}")
        result = self._motion.execute(blocking=True)
        if not result.succeeded:
            raise RuntimeError(f"motion failed: {result.message}")
        self._settle()

    def grip(self, opening: float) -> float:
        self._motion.set_gripper_position(opening, self._group)
        time.sleep(GRIPPER_SETTLE_S)
        return self.jaws()


class Bottle:
    """One clamped bottle: where its cap belongs, and a record of what was seen."""

    def __init__(self, app: Dimos, args: argparse.Namespace) -> None:
        self._app = app
        self.arm = Arm(app)
        self._cap = (args.x, args.y, args.top)
        self._watch = args.watch
        self._prompt = args.prompt
        self._run = RUNS / time.strftime("%Y%m%d-%H%M%S")
        self._run.mkdir(parents=True)
        self._frames: list[NDArray[np.uint8]] = []
        self.cap_width: float | None = None
        self.clear = self.arm.solve(args.x, args.y, args.top + CLEAR_ABOVE_TOP, CLEAR_TILT_DEG)
        self.seated = self.arm.solve(args.x, args.y, args.top + GRIP_ABOVE_TOP, GRIP_TILT_DEG)
        self.resting = self.arm.solve(
            args.x, args.y, args.top + GRIP_ABOVE_TOP + THREAD_TRAVEL_M, GRIP_TILT_DEG
        )
        self.lifted = self.arm.solve(
            args.x, args.y, args.top + GRIP_ABOVE_TOP + LIFT_CHECK_M, CLEAR_TILT_DEG
        )

    def log(self, message: str) -> None:
        """Say it, write it down, and keep the scene camera's view of it."""
        print(message, flush=True)
        with (self._run / "steps.log").open("a") as steps:
            steps.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        recording = json.loads(timelapse.CURRENT.read_text())
        time.sleep(FRAME_AGE_S)
        frame = cv2.imread(str(Path(recording["run_dir"]) / "latest.jpg"))
        x0, y0, x1, y1 = self._watch
        tile = frame[
            max(0, y0 - 2 * SHEET_MARGIN_PX) : y1 + SHEET_MARGIN_PX // 2,
            max(0, x0 - SHEET_MARGIN_PX) : x1 + SHEET_MARGIN_PX,
        ].copy()
        cv2.putText(tile, message[:34], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        self._frames.append(tile)

    def save_sheet(self) -> None:
        if not self._frames:
            return
        columns = 4
        blank = np.zeros_like(self._frames[0])
        tiles = self._frames + [blank] * (-len(self._frames) % columns)
        rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
        cv2.imwrite(str(self._run / "sheet.jpg"), np.vstack(rows))
        print(f"contact sheet: {self._run / 'sheet.jpg'}")

    def grip_is_on_cap(self, held: float) -> bool:
        if held < EMPTY_JAWS:
            return False
        if self.cap_width is None:
            self.cap_width = held
        return held > self.cap_width - NARROWER_GRIP

    def look_for_cap_on_top(self) -> bool:
        """Lift, retract with the jaws as they are, and look for a cap where it belongs."""
        wrist = float(self.arm.joints()[5])
        self.arm.move(self.lifted, wrist, speed=0.15)
        self.arm.move(np.asarray(VIEW_JOINTS), VIEW_JOINTS[5], speed=0.3)
        time.sleep(VIEW_SETTLE_S)
        x, y, top = self._cap
        on_top = False
        for _ in range(2):
            scan = self._app.PickAndPlaceModule.scan_objects([self._prompt])
            for detected in scan.metadata.get("objects", []):
                cloud = self._app.ObjectSceneRegistrationModule.get_object_pointcloud_by_object_id(
                    detected["object_id"]
                )
                cx, cy, cz = np.median(cloud.points_f32(), axis=0)
                near = math.hypot(cx - x, cy - y) < ON_BOTTLE_RADIUS_M
                on_top = on_top or bool(near and cz > top - ON_BOTTLE_DEPTH_M)
        self.log(f"looked: cap {'is' if on_top else 'is not'} on the bottle")
        return on_top


def unscrew(bottle: Bottle) -> int:
    arm = bottle.arm
    for stroke in range(1, MAX_STROKES + 1):
        arm.grip(1.0)
        arm.move(bottle.clear, WRIST_LIMIT_RAD, speed=0.4)
        arm.move(bottle.seated, WRIST_LIMIT_RAD, speed=0.2)
        held = arm.grip(0.0)
        bottle.log(f"stroke {stroke}: grip {held:.3f}")
        if bottle.grip_is_on_cap(held):
            arm.move(bottle.seated, -WRIST_LIMIT_RAD, speed=TWIST_SPEED)
            bottle.log(f"stroke {stroke}: twisted")
        else:
            arm.grip(1.0)
            bottle.log(f"stroke {stroke}: not the cap, no twist")
        if not bottle.look_for_cap_on_top():
            in_jaws = arm.jaws() >= EMPTY_JAWS
            bottle.log(f"cap is off after {stroke} stroke(s), {'held' if in_jaws else 'NOT held'}")
            return 0 if in_jaws else 2
    bottle.log(f"cap still on after {MAX_STROKES} strokes")
    return 1


def screw(bottle: Bottle) -> int:
    arm = bottle.arm
    start = bottle.resting
    for stroke in range(1, MAX_STROKES + 1):
        if arm.jaws() < EMPTY_JAWS or stroke > 1:
            arm.grip(1.0)
        arm.move(bottle.clear, -WRIST_LIMIT_RAD, speed=0.4)
        arm.move(start, -WRIST_LIMIT_RAD, speed=0.2)
        held = arm.grip(0.0)
        bottle.log(f"stroke {stroke}: grip {held:.3f}")
        if not bottle.grip_is_on_cap(held):
            break
        tight = False
        for step in range(1, SCREW_STEPS + 1):
            share = step / SCREW_STEPS
            target = -WRIST_LIMIT_RAD + 2.0 * WRIST_LIMIT_RAD * share
            arm.move(start + (bottle.seated - start) * share, target, speed=0.3)
            if abs(arm.joints()[5] - target) > WRIST_LAG_RAD or arm.jaws() < EMPTY_JAWS:
                tight = True
                break
        bottle.log(f"stroke {stroke}: {'cap resisted' if tight else 'full turn'}")
        start = bottle.seated
        if tight:
            break

    # Pull on it, then look: a threaded cap stays behind when the jaws leave.
    arm.grip(1.0)
    arm.move(bottle.clear, 0.0, speed=0.3)
    arm.move(bottle.seated, 0.0, speed=0.2)
    bottle.log(f"tug: grip {arm.grip(0.0):.3f}")
    threaded = bottle.look_for_cap_on_top()
    bottle.log("cap is screwed on" if threaded else "cap did not stay on the bottle")
    return 0 if threaded else 1


def parse_box(text: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in text.split(","))
    return x0, y0, x1, y1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["unscrew", "screw"])
    parser.add_argument("x", type=float, help="cap centre x, metres")
    parser.add_argument("y", type=float, help="cap centre y, metres")
    parser.add_argument("top", type=float, help="height of the cap's top when on, metres")
    parser.add_argument("--watch", type=parse_box, required=True, help="x0,y0,x1,y1 pixels")
    parser.add_argument("--prompt", default="bottle cap", help="what the detector calls the cap")
    args = parser.parse_args()

    app = Dimos.connect(timeout=CONNECT_TIMEOUT_S)
    try:
        bottle = Bottle(app, args)
        try:
            status = unscrew(bottle) if args.action == "unscrew" else screw(bottle)
        finally:
            bottle.save_sheet()
    finally:
        app.stop()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
