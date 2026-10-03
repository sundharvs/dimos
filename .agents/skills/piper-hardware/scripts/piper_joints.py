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

"""Piper joint driver status, and clearing a latched driver fault.

    python .agents/skills/piper-hardware/scripts/piper_joints.py
    python .agents/skills/piper-hardware/scripts/piper_joints.py --clear 3

A joint whose driver has latched a fault (a collision, an overheat) is disabled,
and the arm then ignores every joint command while still reporting moves as
completed. Status only listens on the bus, so it is safe next to a running
stack. ``--clear`` sends one configuration frame; stop the stack first and
restart it afterwards so the adapter re-enables the joint.
"""

from __future__ import annotations

import argparse
import time

import can
from piper_sdk import C_PiperInterface_V2

ARM_STATUS_ID = 0x2A1
JOINT_STATUS_IDS = range(0x261, 0x267)
LISTEN_TIMEOUT_S = 2.0
# Bit positions in the driver status byte of a low-speed feedback frame.
ENABLED_BIT = 6
FAULT_BITS = {
    0: "low voltage",
    1: "motor overheat",
    2: "driver overcurrent",
    3: "driver overheat",
    4: "collision",
    5: "driver error",
    7: "stall",
}
CTRL_MODES = {0x00: "standby", 0x01: "CAN control", 0x02: "teaching"}
# piper_sdk's "this field is not being set" value for the joint acceleration.
ACCELERATION_UNCHANGED = 0x7FFF
CLEAR_ERROR = 0xAE
ALL_JOINTS = 7


def read_frames(channel: str) -> dict[int, bytes]:
    """The latest arm-status and per-joint driver frames seen on the bus."""
    wanted = {ARM_STATUS_ID, *JOINT_STATUS_IDS}
    frames: dict[int, bytes] = {}
    deadline = time.monotonic() + LISTEN_TIMEOUT_S
    try:
        bus = can.Bus(channel=channel, interface="socketcan")
    except OSError as error:
        raise SystemExit(f"cannot open {channel}: {error}. Is the CAN interface up?") from None
    with bus:
        while frames.keys() != wanted and time.monotonic() < deadline:
            message = bus.recv(timeout=0.1)
            if message is not None and message.arbitration_id in wanted:
                frames[message.arbitration_id] = bytes(message.data)
    return frames


def print_status(channel: str) -> None:
    frames = read_frames(channel)
    if not frames:
        raise SystemExit(f"no Piper feedback on {channel}: is the arm powered and the bus up?")
    arm = frames.get(ARM_STATUS_ID)
    if arm is not None:
        mode = CTRL_MODES.get(arm[0], f"0x{arm[0]:02x}")
        print(f"arm: mode={mode} status=0x{arm[1]:02x} error=0x{arm[6]:02x}{arm[7]:02x}")
    for frame_id in JOINT_STATUS_IDS:
        data = frames.get(frame_id)
        joint = frame_id - JOINT_STATUS_IDS.start + 1
        if data is None:
            print(f"joint {joint}: no feedback")
            continue
        status = data[5]
        faults = [name for bit, name in FAULT_BITS.items() if status >> bit & 1]
        enabled = "enabled " if status >> ENABLED_BIT & 1 else "DISABLED"
        motor_c = int.from_bytes(data[4:5], "big", signed=True)
        driver_c = int.from_bytes(data[2:4], "big", signed=True)
        print(
            f"joint {joint}: {enabled} motor {motor_c:3d} C  driver {driver_c:3d} C  "
            f"{', '.join(faults) or 'ok'}"
        )


def clear_fault(channel: str, joint: int) -> None:
    sdk = C_PiperInterface_V2(can_name=channel, judge_flag=False, can_auto_init=True)
    sdk.ConnectPort(piper_init=True, start_thread=True)
    try:
        sdk.JointConfig(
            joint_num=joint,
            set_zero=0x00,
            acc_param_is_effective=0x00,
            max_joint_acc=ACCELERATION_UNCHANGED,
            clear_err=CLEAR_ERROR,
        )
        # The next low-speed feedback frame carries the cleared status.
        time.sleep(1.0)
    finally:
        sdk.DisconnectPort()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--can-port", default="can0")
    parser.add_argument(
        "--clear",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="clear the latched fault on this joint before printing status",
    )
    args = parser.parse_args()

    if args.clear is not None:
        clear_fault(args.can_port, ALL_JOINTS if args.clear == "all" else int(args.clear))
    print_status(args.can_port)


if __name__ == "__main__":
    main()
