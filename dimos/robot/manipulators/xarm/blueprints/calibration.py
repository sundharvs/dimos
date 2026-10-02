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

"""Hand-eye calibration of the xArm7 wrist RealSense used by xarm-grasp.

``dimos run xarm7-hand-eye-calibration --xarm7-ip 192.168.1.x --square-mm 34.0``

Fix a ChArUco board flat on the table, jog the arm with the Keyboard Teleop
window, and capture in the Hand-eye calibration window. The result is the
link7 -> camera_link edge that XARM_WRIST_CAMERA_TRANSFORM in grasp.py holds.
"""

from __future__ import annotations

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.calibration.hand_eye_module import HandEyeCalibrationModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.common.blueprints import eef_twist_task
from dimos.robot.manipulators.common.coordinators import ArmTwistCoordinator
from dimos.robot.manipulators.xarm.config import make_xarm7_model_config, xarm7_hardware
from dimos.teleop.keyboard.keyboard_teleop_module import KeyboardTeleopModule

_hardware = xarm7_hardware("arm", gripper=True)

xarm7_hand_eye_calibration = autoconnect(
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
                robot_model=make_xarm7_model_config(add_gripper=False),
                target_frame="link7",
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
    # Publishes world -> link7, the gripper pose each capture is paired with.
    # No camera mount here: that edge is what is being measured.
    ManipulationModule.blueprint(
        model=make_xarm7_model_config(
            add_gripper=True,
            gripper_hardware_id="arm",
            tf_extra_links=["link7"],
        ),
    ),
    # Corner error is in pixels, so calibrate at the highest colour resolution.
    RealSenseCamera.blueprint(width=1280, height=720),
    HandEyeCalibrationModule.blueprint(base_frame="world", gripper_frame="link7"),
)
