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

"""Hand-eye calibration of the Piper's wrist RealSense.

``dimos run piper-hand-eye-calibration --can-port can0 --square-mm 34.0``
    Eye-in-hand. Fix the ChArUco board flat on the table, jog until it is in
    view in the Hand-eye calibration window, and press A to let the module
    drive the arm through the poses, or capture by hand with SPACE. The result
    is the link6 -> camera_link edge for the wrist camera.
"""

from __future__ import annotations

from dimos.constants import STATE_DIR
from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.calibration.hand_eye_module import HandEyeCalibrationModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.common.blueprints import eef_twist_task, trajectory_task
from dimos.robot.manipulators.common.coordinators import ArmTwistCoordinator
from dimos.robot.manipulators.piper.config import make_piper_model_config, piper_hardware
from dimos.teleop.keyboard.keyboard_teleop_module import KeyboardTeleopModule

# A mock arm cannot be calibrated against, so a missing --can-port means can0.
_hardware = piper_hardware("arm", mock_without_address=False)
# Publishes world -> link6, the flange the camera is mounted on and the pose
# each capture is paired with. No camera edge: that is what is being measured.
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
            eef_twist_task(_hardware, robot_model=_model, target_frame="gripper_base"),
            TaskConfig(
                name="arm_gripper",
                type="gripper",
                joint_names=["arm/gripper"],
                priority=20,
            ),
            # Automatic collection's planned moves. The twist task holds the
            # arm whenever it is idle, so these must outrank it; a finished
            # trajectory releases the joints back to the keyboard.
            trajectory_task(_hardware, priority=15),
        ],
    ),
    ManipulationModule.blueprint(model=_model),
    # Corner error is in pixels, so calibrate at the highest colour resolution.
    RealSenseCamera.blueprint(width=1280, height=720),
    HandEyeCalibrationModule.blueprint(
        mode="eye_in_hand",
        base_frame="world",
        gripper_frame="link6",
        output_path=str(STATE_DIR / "calibration" / "piper_wrist_realsense.json"),
        # The planning tip is gripper_base and the fingertips are 13.8 cm past
        # it; keep them roughly 6 cm above the board.
        auto_min_clearance_m=0.20,
    ),
)
