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

"""Is the Piper rig ready to run? One line per thing that has to be true.

    python .agents/skills/piper-hardware/scripts/preflight.py

Run it first in a session and whenever something stops responding. Each line is
``ok``, ``warn`` (works, with a consequence worth knowing) or ``FAIL`` with what
to do about it. Exits non-zero when anything failed. It only looks: nothing is
sent to the arm.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub import constants as hf_constants
import piper_joints
import psutil
import stack
import torch

from dimos.core.run_registry import get_most_recent
from dimos.hardware.sensors.camera.realsense import camera as realsense_camera

Check = tuple[str, str, str]

IFF_UP = 0x1
# Motors run at 25-45 C in normal use; a driver latched an overheat fault near 50.
WARM_MOTOR_C = 48
MOONDREAM_CACHE = "models--vikhyatk--moondream2"


def _can_checks(can_port: str) -> Iterator[Check]:
    interface = Path("/sys/class/net") / can_port
    if not interface.exists():
        serial_adapters = sorted(Path("/dev").glob("ttyACM*"))
        if not serial_adapters:
            yield "FAIL", "CAN adapter", "not on USB. Is the hub plugged in?"
            return
        stale = any(process.name() == "slcand" for process in psutil.process_iter())
        yield (
            "FAIL",
            "CAN adapter",
            f"{serial_adapters[0]} has no {can_port}. Operator: "
            + ("sudo pkill slcand; " if stale else "")
            + f"sudo slcand -o -c -s8 {serial_adapters[0]} {can_port} "
            f"&& sudo ip link set {can_port} up",
        )
        return
    if not int((interface / "flags").read_text(), 16) & IFF_UP:
        yield "FAIL", "CAN adapter", f"{can_port} is down. Operator: sudo ip link set {can_port} up"
        return
    yield "ok", "CAN adapter", f"{can_port} is up"

    try:
        arm = piper_joints.read_status(can_port)
    except piper_joints.CanBusError as error:
        yield "FAIL", "arm", str(error)
        return
    if arm.mode == "teaching":
        yield "FAIL", "arm", "in teaching mode: press the mode button on its base to leave it"
    elif arm.faulted_joints:
        joints = ", ".join(str(joint.joint) for joint in arm.faulted_joints)
        yield (
            "FAIL",
            "arm",
            f"joint {joints} faulted: stop the stack, then piper_joints.py --clear <joint>",
        )
    else:
        enabled = sum(joint.enabled for joint in arm.joints)
        yield "ok", "arm", f"powered, mode {arm.mode}, {enabled}/6 joints enabled"
    hottest = max(arm.joints, key=lambda joint: joint.motor_c)
    if hottest.motor_c >= WARM_MOTOR_C:
        yield (
            "warn",
            "motors",
            f"joint {hottest.joint} at {hottest.motor_c} C: park the arm (go_init) to cool",
        )


def _camera_checks() -> Iterator[Check]:
    products = [path.read_text() for path in Path("/sys/bus/usb/devices").glob("*/product")]
    if any("RealSense" in product for product in products):
        yield "ok", "wrist camera", "RealSense on USB"
    else:
        yield "FAIL", "wrist camera", "no RealSense on USB. Is the hub plugged in?"

    fields = realsense_camera.RealSenseCameraConfig.model_fields
    crate = Path(realsense_camera.__file__).parent / fields["cwd"].default
    if (crate / fields["executable"].default).exists():
        yield "ok", "camera driver", "native module is built"
    else:
        yield (
            "FAIL",
            "camera driver",
            f"not built: (cd {crate} && {fields['build_command'].default})",
        )


def _software_checks() -> Iterator[Check]:
    if torch.cuda.is_available():
        yield "ok", "GPU", torch.cuda.get_device_name(0)
    else:
        yield "warn", "GPU", "no CUDA: moondream and EdgeTAM fall back to the CPU, seconds per scan"

    if (Path(hf_constants.HF_HUB_CACHE) / MOONDREAM_CACHE).exists():
        yield "ok", "models", "moondream weights are cached"
    else:
        yield "warn", "models", "moondream not cached: the first start downloads 3.5 GB silently"

    settings = stack.rig_environment()
    if "PIPER_JOINT_OFFSETS_DEG" in settings:
        yield "ok", "rig settings", f"{stack.RIG_ENV.name}: {', '.join(sorted(settings))}"
    else:
        yield (
            "warn",
            "rig settings",
            f"no PIPER_JOINT_OFFSETS_DEG in {stack.RIG_ENV}: the wrist camera edge in "
            "piper/blueprints/grasp.py was calibrated with this arm's offsets applied",
        )

    running = get_most_recent()
    if running is None:
        yield "ok", "stack", "none running"
    else:
        yield "ok", "stack", f"{running.blueprint} running as {running.run_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--can-port", default="can0")
    args = parser.parse_args()

    checks = [*_can_checks(args.can_port), *_camera_checks(), *_software_checks()]
    for level, subject, detail in checks:
        print(f"[{level:^4}] {subject}: {detail}")
    if any(level == "FAIL" for level, _, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
