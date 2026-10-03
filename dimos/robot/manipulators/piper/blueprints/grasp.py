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

"""Heuristic top-down grasping with the Piper's wrist RealSense.

``dimos run piper-grasp --can-port can0``

The xArm grasp stack on a Piper, without the voxel map: its ray tracer is a
native module this machine cannot build, so the planner knows only the robot.
Scene registration finds prompted objects and the heuristic provider grasps them
straight down. Detection runs YOLO-E, which needs no GPU; it knows shape words
better than object names ("rectangular block" finds an eraser, "eraser" does
not). Set PIPER_JUDGE_CAN and PIPER_JOINT_OFFSETS_DEG as for the other Piper
blueprints; the camera edge below was calibrated with the joint offsets applied.
"""

from __future__ import annotations

import math
import os

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasp_verification import GraspVerificationConfig
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.detection.detectors.yoloe import YoloePromptMode
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.piper.config import (
    PIPER_COLLISION_LINKS,
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule

# The hand-eye calibration start pose: the wrist camera looks down at the table
# from about 22 cm, with joint 5 clear of its limit. go_home returns here.
PIPER_GRASP_SCAN_JOINTS = [-0.0406, 0.9366, -0.5184, 0.0, 0.805, 0.0]

# The planning tip is gripper_base; the fingertips are 13.8 cm along its +Z (the
# finger meshes span 6.2-13.8 cm). Grasp 2 cm in from the tips, and keep the
# tips 1 cm above the table.
PIPER_GRASP_TIP_OFFSET = 0.118
PIPER_FINGERTIP_DEPTH = 0.138
# The table top in world, measured from the wrist camera's scene cloud on
# 2026-10-02 with the camera edge below. Re-measure if the arm or table moves.
PIPER_GRASP_TABLE_Z = -0.011

# piper-hand-eye-calibration, 2026-10-02, with PIPER_JOINT_OFFSETS_DEG=0,0,0,0,4.42,0:
# 16 poses, board-in-base spread 6.9 mm / 1.36 deg RMS. Re-measure whenever the
# camera mount moves or the joint offsets change.
PIPER_WRIST_CAMERA_TRANSFORM = Transform(
    translation=Vector3(x=-0.13631749, y=0.00055216, z=0.08518030),
    rotation=Quaternion(-0.00054318, -0.50776572, 0.06781325, 0.85882189),  # xyzw
    frame_id="link6",
    child_frame_id="camera_link",
)

_JUDGE_CAN = os.getenv("PIPER_JUDGE_CAN", "1").strip().lower() not in ("0", "false", "no", "off")
_hardware = piper_hardware(
    "arm",
    mock_without_address=False,
    judge_can=_JUDGE_CAN,
    joint_offsets=piper_joint_offsets_from_env(),
)
_model = make_piper_model_config(home_joints=PIPER_GRASP_SCAN_JOINTS).model_copy(
    update={
        "base_pose": PoseStamped(frame_id="world"),
        # link6 carries the camera.
        "tf_extra_links": PIPER_COLLISION_LINKS,
    }
)

piper_grasp = autoconnect(
    ManipulationModule.blueprint(
        model=_model,
        static_transforms=[PIPER_WRIST_CAMERA_TRANSFORM],
        planning_timeout=10.0,
        visualization={"backend": "viser"},
        world_frame="world",
    ),
    ManipulationSkills.blueprint(),
    PickAndPlaceModule.blueprint(
        planning_frame="world",
        # Commanded fully open (0.08 m) the Piper's jaws stop at 0.84, about
        # 67 mm, which the default 0.15 tolerance just rejects.
        grasp_verification=GraspVerificationConfig(open_tolerance=0.2),
    ),
    HeuristicGraspModule.blueprint(
        tip_offset=PIPER_GRASP_TIP_OFFSET,
        fingertip_depth=PIPER_FINGERTIP_DEPTH,
        support_z=PIPER_GRASP_TABLE_Z,
        # Top-down with yaw 0 needs joint 6 at +-180 deg, past its +-120 deg range.
        yaw_offset=math.pi,
    ),
    RealSenseCamera.blueprint(enable_pointcloud=True),
    ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="yoloe",
        # scan_objects passes text prompts, which the prompt-free model rejects.
        prompt_mode=YoloePromptMode.PROMPT,
        # Text-prompted YOLO-E scores real tabletop objects well below the 0.6
        # default; at 0.6 a plainly visible eraser was not detected.
        detector_confidence=0.25,
        # The wrist camera is a D405, whose depth unit is 0.1 mm.
        depth_unit_m=0.0001,
        segmentation_backend="yolo",
        detect_on_request=True,
        distance_threshold=0.05,
        # One explicit scan must promote what it sees.
        min_detections_for_permanent=1,
        max_distance=1.0,
        use_aabb=True,
        max_obstacle_width=0.06,
    ),
    RerunBridgeModule.blueprint(),
    coordinator(
        hardware=[_hardware],
        tasks=[
            trajectory_task(_hardware),
            TaskConfig(
                name="arm_gripper",
                type="gripper",
                joint_names=["arm/gripper"],
                priority=20,
            ),
        ],
    ),
)
