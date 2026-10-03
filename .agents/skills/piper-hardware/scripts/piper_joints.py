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
from dataclasses import dataclass
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
# What every driver reports between power-up and its first enable.
POWER_UP_FAULTS = ("collision", "driver error")
CTRL_MODES = {0x00: "standby", 0x01: "CAN control", 0x02: "teaching"}
# piper_sdk's "this field is not being set" value for the joint acceleration.
ACCELERATION_UNCHANGED = 0x7FFF
CLEAR_ERROR = 0xAE
ALL_JOINTS = 7


class CanBusError(RuntimeError):
    """The CAN interface cannot be opened, or carries no Piper feedback."""


@dataclass(frozen=True)
class JointStatus:
    joint: int
    enabled: bool
    faults: tuple[str, ...]
    motor_c: int
    driver_c: int


@dataclass(frozen=True)
class ArmStatus:
    mode: str
    joints: tuple[JointStatus, ...]

    @property
    def awaiting_first_enable(self) -> bool:
        """Every driver disabled with the flags they all carry after power-up."""
        return all(not joint.enabled and joint.faults == POWER_UP_FAULTS for joint in self.joints)

    @property
    def faulted_joints(self) -> tuple[JointStatus, ...]:
        if self.awaiting_first_enable:
            return ()
        return tuple(joint for joint in self.joints if joint.faults)


def read_status(channel: str) -> ArmStatus:
    """Listen for one arm-status frame and one driver frame per joint."""
    wanted = {ARM_STATUS_ID, *JOINT_STATUS_IDS}
    frames: dict[int, bytes] = {}
    deadline = time.monotonic() + LISTEN_TIMEOUT_S
    try:
        bus = can.Bus(channel=channel, interface="socketcan")
    except OSError as error:
        raise CanBusError(f"cannot open {channel}: {error}") from None
    with bus:
        while frames.keys() != wanted and time.monotonic() < deadline:
            message = bus.recv(timeout=0.1)
            if message is not None and message.arbitration_id in wanted:
                frames[message.arbitration_id] = bytes(message.data)
    if frames.keys() != wanted:
        raise CanBusError(f"no Piper feedback on {channel}: is the arm powered?")

    joints = []
    for frame_id in JOINT_STATUS_IDS:
        data = frames[frame_id]
        status = data[5]
        joints.append(
            JointStatus(
                joint=frame_id - JOINT_STATUS_IDS.start + 1,
                enabled=bool(status >> ENABLED_BIT & 1),
                faults=tuple(name for bit, name in FAULT_BITS.items() if status >> bit & 1),
                motor_c=int.from_bytes(data[4:5], "big", signed=True),
                driver_c=int.from_bytes(data[2:4], "big", signed=True),
            )
        )
    mode_byte = frames[ARM_STATUS_ID][0]
    return ArmStatus(CTRL_MODES.get(mode_byte, f"0x{mode_byte:02x}"), tuple(joints))


def format_status(status: ArmStatus) -> str:
    lines = [f"arm: mode={status.mode}"]
    for joint in status.joints:
        enabled = "enabled " if joint.enabled else "DISABLED"
        lines.append(
            f"joint {joint.joint}: {enabled} motor {joint.motor_c:3d} C  "
            f"driver {joint.driver_c:3d} C  {', '.join(joint.faults) or 'ok'}"
        )
    if status.awaiting_first_enable:
        lines.append("(all drivers idle since power-up; a stack's connect enables them)")
    return "\n".join(lines)


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

    try:
        if args.clear is not None:
            clear_fault(args.can_port, ALL_JOINTS if args.clear == "all" else int(args.clear))
        print(format_status(read_status(args.can_port)))
    except CanBusError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
