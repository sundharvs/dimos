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

"""Hand-eye calibration of the cameras around the xArm7.

``dimos run xarm7-hand-eye-calibration --xarm7-ip 192.168.1.x --square-mm 34.0``
    Eye-in-hand: the wrist RealSense used by xarm-grasp. Fix the ChArUco board
    flat on the table. The result is the link7 -> camera_link edge that
    XARM_WRIST_CAMERA_TRANSFORM in grasp.py holds.

``dimos run xarm7-side-camera-calibration --xarm7-ip 192.168.1.x --square-mm 34.0``
    Eye-to-hand: the fixed side ZED 2i, read as a plain webcam. Mount the board
    rigidly on the gripper. The result is world -> camera_link for that camera.

Jog the arm in the Keyboard Teleop window and capture in the Hand-eye
calibration window -- or, for the wrist camera, jog until the board is in view
and press A to let the module drive the arm through the poses itself.
"""

from __future__ import annotations

from dimos.constants import STATE_DIR
from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.hardware.sensors.camera.webcam import Webcam
from dimos.manipulation.calibration.hand_eye_module import HandEyeCalibrationModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.robot.manipulators.common.blueprints import eef_twist_task, trajectory_task
from dimos.robot.manipulators.common.coordinators import ArmTwistCoordinator
from dimos.robot.manipulators.xarm.config import make_xarm7_model_config, xarm7_hardware
from dimos.teleop.keyboard.keyboard_teleop_module import KeyboardTeleopModule

CALIBRATION_DIR = STATE_DIR / "calibration"
# The side ZED 2i (serial 33805648) on its V4L2 node, and the factory
# calibration the ZED SDK downloads for it.
SIDE_ZED_DEVICE = "/dev/v4l/by-id/usb-Technologies__Inc._ZED_2i_OV0001-video-index0"
SIDE_ZED_INTRINSICS = CALIBRATION_DIR / "zed" / "SN33805648.conf"

_hardware = xarm7_hardware("arm", gripper=True)


def _arm_with_teleop() -> tuple[Blueprint, ...]:
    """Keyboard-jogged, plannable xArm7 that publishes world -> link7 and nothing about cameras."""
    return (
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
                # Automatic collection's planned moves. The twist task holds the
                # arm whenever it is idle, so these must outrank it; a finished
                # trajectory releases the joints back to the keyboard.
                trajectory_task(_hardware, priority=15),
            ],
        ),
        # Publishes world -> link7, the gripper pose each capture is paired
        # with. No camera edge here: that is what is being measured.
        ManipulationModule.blueprint(
            model=make_xarm7_model_config(
                add_gripper=True,
                gripper_hardware_id="arm",
                tf_extra_links=["link7"],
            ),
        ),
    )


xarm7_hand_eye_calibration = autoconnect(
    *_arm_with_teleop(),
    # Corner error is in pixels, so calibrate at the highest colour resolution.
    RealSenseCamera.blueprint(width=1280, height=720),
    HandEyeCalibrationModule.blueprint(
        mode="eye_in_hand",
        base_frame="world",
        gripper_frame="link7",
        output_path=str(CALIBRATION_DIR / "xarm7_wrist_realsense.json"),
    ),
)


def _side_zed() -> Webcam:
    # 2K side-by-side; the left half is 2208x1242, the factory file's LEFT_CAM_2K.
    return Webcam(
        camera_index=SIDE_ZED_DEVICE,
        width=4416,
        height=1242,
        fps=15,
        stereo_slice="left",
    )


xarm7_side_camera_calibration = autoconnect(
    *_arm_with_teleop(),
    CameraModule.blueprint(
        hardware=_side_zed,
        # CameraModule publishes camera_link -> camera_optical only alongside a
        # parent edge. Hang it off a frame of its own so the camera_link ->
        # optical step is known while its place in the world is still unknown.
        transform=Transform(frame_id="side_zed_uncalibrated", child_frame_id="camera_link"),
    ),
    HandEyeCalibrationModule.blueprint(
        mode="eye_to_hand",
        base_frame="world",
        gripper_frame="link7",
        intrinsics_file=str(SIDE_ZED_INTRINSICS),
        output_path=str(CALIBRATION_DIR / "xarm7_side_zed.json"),
    ),
)
