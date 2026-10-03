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

"""Start, stop and inspect a dimOS stack on the Piper rig from a script.

    python .agents/skills/piper-hardware/scripts/stack.py start
    python .agents/skills/piper-hardware/scripts/stack.py restart piper-grasp
    python .agents/skills/piper-hardware/scripts/stack.py status
    python .agents/skills/piper-hardware/scripts/stack.py stop
    python .agents/skills/piper-hardware/scripts/stack.py park

``start`` launches the blueprint detached, waits until it reports ready or dies,
and prints why when it dies. ``stop`` and ``restart`` first park the arm in its
rest posture with a planned move and refuse to go on when that fails
(``--no-park`` skips it): a stack killed with the arm stretched out leaves the
adapter's own homing half done, and the next connect drops the limp arm onto
whatever is under it. ``stop`` also removes what a killed stack leaves behind. Settings that belong to one rig (PIPER_JOINT_OFFSETS_DEG,
PIPER_JUDGE_CAN) come from ``rig.ignore.env`` in the skill folder, a git-ignored
file of KEY=VALUE lines.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import piper_joints
import psutil

from dimos.constants import DIMOS_PROJECT_ROOT, STATE_DIR
from dimos.core.run_registry import get_most_recent, stop_entry
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.porcelain.dimos import Dimos

RIG_ENV = Path(__file__).resolve().parent.parent / "rig.ignore.env"
LAUNCH_LOG = STATE_DIR / "piper-hardware" / "launch.log"
READY_MARKER = "DimOS running in background"
DEFAULT_BLUEPRINT = "piper-grasp"
# The first start of a blueprint downloads its model weights, several GB.
START_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 1.0
LEFTOVER_GRACE_S = 3.0
ERROR_LINES_SHOWN = 15
# The rest posture, a little inside the limits joints 2 and 3 rest against.
PARK_JOINTS = [0.0, 0.03, -0.03, 0.0, 0.08, 0.0]
PARKED_TOLERANCE_RAD = 0.12
# Below this the tool may be inside or beside an object: go straight up first.
PARK_CLEAR_Z = 0.20
PARK_SPEED_SCALE = 0.3
CONNECT_TIMEOUT_S = 15.0


def rig_environment() -> dict[str, str]:
    """This rig's settings from ``rig.ignore.env``, empty when there is none."""
    if not RIG_ENV.exists():
        return {}
    settings = {}
    for line in RIG_ENV.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and not key.startswith("#"):
            settings[key.strip()] = value.strip()
    return settings


def _leftovers() -> list[psutil.Process]:
    """Processes of this checkout's stack that outlived it."""
    found = []
    venv_bin = Path(sys.executable).parent
    for process in psutil.process_iter(["cmdline"]):
        command = process.info["cmdline"] or []
        line = " ".join(command)
        worker = (
            bool(command) and Path(command[0]).parent == venv_bin and ("multiprocessing." in line)
        )
        native = str(DIMOS_PROJECT_ROOT) in line and "realsense_native" in line
        viewer = "dimos-viewer" in line and "--connect" in line
        if worker or native or viewer:
            found.append(process)
    return found


def _remove_leftovers() -> None:
    leftovers = _leftovers()
    for process in leftovers:
        process.terminate()
    _, alive = psutil.wait_procs(leftovers, timeout=LEFTOVER_GRACE_S)
    for process in alive:
        process.kill()
    if leftovers:
        print(f"removed {len(leftovers)} leftover process(es)")


def _print_arm_status(can_port: str) -> None:
    try:
        print(piper_joints.format_status(piper_joints.read_status(can_port)))
    except piper_joints.CanBusError as error:
        print(f"arm: {error}")


def start(blueprint: str, can_port: str) -> bool:
    running = get_most_recent()
    if running is not None:
        print(f"already running: {running.blueprint} ({running.run_id}); use restart")
        return False

    LAUNCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    dimos = Path(sys.executable).parent / "dimos"
    with LAUNCH_LOG.open("w") as log:
        launcher = subprocess.Popen(
            [str(dimos), "--can-port", can_port, "run", blueprint, "--daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=DIMOS_PROJECT_ROOT,
            env={**os.environ, **rig_environment()},
        )

    started = time.monotonic()
    while time.monotonic() - started < START_TIMEOUT_S:
        output = LAUNCH_LOG.read_text(errors="replace")
        if READY_MARKER in output:
            entry = get_most_recent()
            run = entry.run_id if entry is not None else "unregistered"
            print(f"{blueprint} ready after {time.monotonic() - started:.0f}s ({run})")
            _print_arm_status(can_port)
            return True
        if launcher.poll() is not None:
            break
        time.sleep(POLL_INTERVAL_S)

    print(f"{blueprint} did not start; see {LAUNCH_LOG}")
    errors = [line for line in output.replace("\r", "\n").splitlines() if "rror" in line]
    print("\n".join(errors[-ERROR_LINES_SHOWN:]))
    _remove_leftovers()
    return False


def park() -> bool:
    """Bring the arm to its rest posture through the running stack. True when it is there."""
    try:
        app = Dimos.connect(timeout=CONNECT_TIMEOUT_S)
    except Exception as error:
        print(f"park: cannot reach the stack ({error})")
        return False
    try:
        manipulation = app.ManipulationModule

        def group() -> tuple[object, object]:
            return next(iter(manipulation.get_state().groups.items()))

        def away(state: object) -> float:
            return max(abs(a - b) for a, b in zip(state.joints.position, PARK_JOINTS, strict=True))

        group_id, state = group()
        if away(state) > PARKED_TOLERANCE_RAD:
            rise = PARK_CLEAR_Z - float(state.end_effector_pose.position.z)
            if rise > 0.005:
                # Best effort: the tool cannot always go straight up this far.
                manipulation.move_linear(0.0, 0.0, rise, speed_scale=PARK_SPEED_SCALE)
            names = list(group()[1].joints.name)
            plan = manipulation.plan_to_joints(
                {group_id: JointState(name=names, position=PARK_JOINTS)},
                speed_scale=PARK_SPEED_SCALE,
            )
            if plan.succeeded:
                manipulation.execute(blocking=True)
            time.sleep(1.0)
        distance = away(group()[1])
        if distance > PARKED_TOLERANCE_RAD:
            print(f"park: arm is {distance:.2f} rad from its rest posture")
            return False
        print("parked")
        return True
    except Exception as error:
        print(f"park: failed ({error})")
        return False
    finally:
        app.stop()


def stop(park_first: bool = True) -> bool:
    running = get_most_recent()
    if running is None:
        print("no stack running")
    else:
        if park_first and not park():
            print("not stopping: park the arm by hand or with RPC, or pass --no-park")
            return False
        message, _ = stop_entry(running)
        print(f"{running.blueprint}: {message}")
    _remove_leftovers()
    return True


def status(can_port: str) -> None:
    running = get_most_recent()
    if running is None:
        print("no stack running")
    else:
        print(f"{running.blueprint} running as {running.run_id}, pid {running.pid}")
    _print_arm_status(can_port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "park"])
    parser.add_argument("blueprint", nargs="?", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--can-port", default="can0")
    parser.add_argument(
        "--no-park",
        action="store_true",
        help="stop without parking first (only when the arm is already at rest or held)",
    )
    args = parser.parse_args()

    if args.command == "status":
        status(args.can_port)
        return
    if args.command == "park":
        if not park():
            raise SystemExit(1)
        return
    if args.command in ("stop", "restart") and not stop(park_first=not args.no_park):
        raise SystemExit(1)
    if args.command in ("start", "restart") and not start(args.blueprint, args.can_port):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
