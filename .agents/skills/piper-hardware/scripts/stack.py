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

``start`` launches the blueprint detached, waits until it reports ready or dies,
and prints why when it dies. ``stop`` also removes what a killed stack leaves
behind. Settings that belong to one rig (PIPER_JOINT_OFFSETS_DEG,
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

RIG_ENV = Path(__file__).resolve().parent.parent / "rig.ignore.env"
LAUNCH_LOG = STATE_DIR / "piper-hardware" / "launch.log"
READY_MARKER = "DimOS running in background"
DEFAULT_BLUEPRINT = "piper-grasp"
# The first start of a blueprint downloads its model weights, several GB.
START_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 1.0
LEFTOVER_GRACE_S = 3.0
ERROR_LINES_SHOWN = 15


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


def stop() -> None:
    running = get_most_recent()
    if running is None:
        print("no stack running")
    else:
        message, _ = stop_entry(running)
        print(f"{running.blueprint}: {message}")
    _remove_leftovers()


def status(can_port: str) -> None:
    running = get_most_recent()
    if running is None:
        print("no stack running")
    else:
        print(f"{running.blueprint} running as {running.run_id}, pid {running.pid}")
    _print_arm_status(can_port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["start", "stop", "restart", "status"])
    parser.add_argument("blueprint", nargs="?", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--can-port", default="can0")
    args = parser.parse_args()

    if args.command == "status":
        status(args.can_port)
        return
    if args.command in ("stop", "restart"):
        stop()
    if args.command in ("start", "restart") and not start(args.blueprint, args.can_port):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
