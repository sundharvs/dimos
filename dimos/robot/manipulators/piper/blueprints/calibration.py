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

"""Hand-eye calibration of the Piper wrist RealSense used by piper-grasp.

``dimos --can-port can0 run piper-hand-eye-calibration --square-mm 34.0``

Fix a ChArUco board flat on the table, jog the arm with the Keyboard Teleop
window, and capture in the Hand-eye calibration window. The result is the
link6 -> camera_link edge that PIPER_WRIST_CAMERA_TRANSFORM in grasp.py holds.
Set PIPER_JUDGE_CAN and PIPER_JOINT_OFFSETS_DEG as for piper-grasp: the edge is
only valid with the joint offsets it was measured under.
"""

from __future__ import annotations

import os

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.calibration.hand_eye_module import HandEyeCalibrationModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.common.blueprints import eef_twist_task
from dimos.robot.manipulators.common.coordinators import ArmTwistCoordinator
from dimos.robot.manipulators.piper.config import (
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
)
from dimos.teleop.keyboard.keyboard_teleop_module import KeyboardTeleopModule

_JUDGE_CAN = os.getenv("PIPER_JUDGE_CAN", "1").strip().lower() not in ("0", "false", "no", "off")
# A mock arm cannot be calibrated against, so a missing --can-port means can0.
_hardware = piper_hardware(
    "arm",
    mock_without_address=False,
    judge_can=_JUDGE_CAN,
    joint_offsets=piper_joint_offsets_from_env(),
)
# Publishes world -> link6, the flange the camera is mounted on and the pose each
# capture is paired with. No camera edge: that is what is being measured.
_model = make_piper_model_config().model_copy(update={"tf_extra_links": ["link6"]})

piper_hand_eye_calibration = autoconnect(
    KeyboardTeleopModule.blueprint(),
    ArmTwistCoordinator.blueprint(
        instance_name="ControlCoordinator",
        tick_rate=100.0,
        publish_joint_state=True,
        joint_state_frame_id="coordinator",
        hardware=[_hardware],
        tasks=[
            eef_twist_task(
                _hardware,
                robot_model=_model,
                target_frame="gripper_base",
                timeout=0.0,
            ),
            TaskConfig(
                name="arm_gripper",
                type="gripper",
                joint_names=["arm/gripper"],
                priority=20,
            ),
        ],
    ),
    ManipulationModule.blueprint(model=_model),
    # Corner error is in pixels, so calibrate at the highest colour resolution.
    RealSenseCamera.blueprint(width=1280, height=720),
    HandEyeCalibrationModule.blueprint(base_frame="world", gripper_frame="link6"),
)
