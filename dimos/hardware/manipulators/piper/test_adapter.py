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

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

piper_sdk_module = ModuleType("piper_sdk")
piper_sdk_module.__dict__["C_PiperInterface_V2"] = lambda **_: None
sys.modules.setdefault("piper_sdk", piper_sdk_module)

from dimos.hardware.manipulators.piper import adapter as piper_adapter
from dimos.hardware.manipulators.piper.adapter import PiperAdapter


def test_connect_continues_when_gripper_startup_fails(
    mocker: Any,
) -> None:
    sdk = mocker.Mock()
    sdk.GetArmStatus.return_value = object()
    sdk.GripperCtrl.side_effect = RuntimeError("gripper unavailable")
    mocker.patch.object(piper_adapter, "C_PiperInterface_V2", lambda **_: sdk)
    mocker.patch("dimos.hardware.manipulators.piper.adapter.time.sleep")

    adapter = PiperAdapter(dof=7)

    assert adapter.connect()
    assert adapter.is_connected()
    assert sdk.GripperCtrl.called


def test_joint_offsets_shift_reads_and_writes(mocker: Any) -> None:
    sdk = mocker.Mock()
    sdk.GetArmJointMsgs.return_value.joint_state = mocker.Mock(
        joint_1=0, joint_2=1000, joint_3=0, joint_4=0, joint_5=-5000, joint_6=0
    )
    offsets = [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    adapter = PiperAdapter(joint_offsets=offsets)
    adapter._sdk = sdk

    positions = adapter.read_joint_positions()
    assert positions[1] == pytest.approx(1000 * piper_adapter.MILLIDEG_TO_RAD)
    assert positions[4] == pytest.approx(-5000 * piper_adapter.MILLIDEG_TO_RAD + 0.1)

    assert adapter.write_joint_positions(positions)
    assert sdk.JointCtrl.call_args.args == (0, 1000, 0, 0, -5000, 0)


def test_joint_offsets_need_six_values() -> None:
    with pytest.raises(ValueError, match="6 values"):
        PiperAdapter(joint_offsets=[0.1])
