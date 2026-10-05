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

"""The xArm grasping stack on a Piper, with the wrist RealSense (a D405).

``dimos --can-port can0 run piper-grasp``

The same modules as ``xarm-grasp`` -- planner, skills, pick-and-place, scene
registration with moondream + EdgeTAM, the heuristic grasp provider -- with the
Piper's measured rig constants. The voxel map is left out: its ray tracer is a
native cargo build this rig has not had, so the planner knows only the robot.

Set PIPER_JUDGE_CAN=0 for a slcan adapter and PIPER_JOINT_OFFSETS_DEG as for the
other Piper blueprints; the camera edge below was calibrated with the joint
offsets applied. With a second RealSense connected (a scene camera), set
PIPER_WRIST_CAMERA_SERIAL to the wrist camera's serial number, or the driver may
open the other one.
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
from dimos.manipulation.rim_grasp_module import RimGraspModule
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.piper.config import (
    PIPER_COLLISION_LINKS,
    PIPER_FINGERTIP_DEPTH,
    PIPER_TCP_DEPTH,
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
    piper_wrist_camera_serial_from_env,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule

# Home preset for go_home: the hand-eye calibration start pose, with the wrist
# camera looking down at the table from about 22 cm and joint 5 clear of its
# limit.
PIPER_GRASP_HOME_JOINTS = [-0.0406, 0.9366, -0.5184, 0.0, 0.805, 0.0]

# Poses are planned to the tool frame between the jaws; the fingertips reach
# this far past it.
PIPER_FINGERTIPS_PAST_TCP = PIPER_FINGERTIP_DEPTH - PIPER_TCP_DEPTH
# The table top in world, measured from the wrist camera's scene cloud on
# 2026-10-02 with the camera edge below. Re-measure if the arm or table moves.
PIPER_GRASP_TABLE_Z = -0.011
# World-frame tool height with the gripper pointing down and the fingertips on
# the table: the table top plus the fingertips' reach past the tool frame.
# move_near never plans below it.
PIPER_GRASP_NEAR_MIN_Z = PIPER_GRASP_TABLE_Z + PIPER_FINGERTIPS_PAST_TCP
# How far above a grasp the arm stops before the straight final leg. The wrist
# pitch range bounds where the tool can point straight down, and the band
# narrows with height: 10 cm up it spans radii of 0.16-0.33 m from the base,
# 6 cm up 0.08-0.41 m.
PIPER_GRASP_PREGRASP_OFFSET = 0.06

# Hand-eye calibration for the eye-in-hand RealSense, measured 2026-10-02 with
# PIPER_JOINT_OFFSETS_DEG=0,0,0,0,4.42,0: 16 poses, board-in-base spread
# 6.9 mm / 1.36 deg RMS. Re-measure whenever the camera mount moves or the joint
# offsets change.
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
_model = make_piper_model_config(home_joints=PIPER_GRASP_HOME_JOINTS, tcp=True).model_copy(
    update={
        "base_pose": PoseStamped(frame_id="world"),
        # The self filter needs a capture-time transform for every collision link
        # and drops the whole cloud when one is missing; link6 carries the camera.
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
        pregrasp_offset=PIPER_GRASP_PREGRASP_OFFSET,
        near_min_z=PIPER_GRASP_NEAR_MIN_Z,
        # Commanded fully open (0.08 m) the Piper's jaws stop at 0.84, about
        # 67 mm, which the default 0.15 tolerance just rejects.
        grasp_verification=GraspVerificationConfig(open_tolerance=0.2),
    ),
    # Rim pinch for containers wider than the jaws (autoresearch "pick up the
    # yellow bin", 2026-10-04). Held-out validation through this blueprint:
    # RIM_VALIDATION_PLACEHOLDER
    RimGraspModule.blueprint(
        planning_frame="world",
        table_z=PIPER_GRASP_TABLE_Z,
        fingertips_past_tcp=PIPER_FINGERTIPS_PAST_TCP,
        # Straight down, the planner rejects poses more than 12 cm above the
        # table at 0.315 m from the base (JOINT_LIMITS); leaning 15 deg outward
        # it plans from the grasp height up to 18 cm. Swept offline 2026-10-04.
        lean=math.radians(15.0),
        pregrasp_offset=PIPER_GRASP_PREGRASP_OFFSET,
        # Fingertips 1.7-2.5 cm below the rim of the yellow shelf bin dropped it
        # in two of three lifts; 4.5 cm held. Measured 2026-10-04.
        grasp_depth=0.045,
        gripper=GraspVerificationConfig(open_tolerance=0.2),
    ),
    # Top-down with yaw 0 needs joint 6 at +-180 deg, past its +-120 deg range.
    HeuristicGraspModule.blueprint(yaw_offset=math.pi),
    # enable_pointcloud is off by default.
    RealSenseCamera.blueprint(
        enable_pointcloud=True, serial_number=piper_wrist_camera_serial_from_env()
    ),
    ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="moondream",
        segmentation_backend="edgetam",
        # The wrist camera is a D405, whose depth unit is 0.1 mm.
        depth_unit_m=0.0001,
        detect_on_request=True,
        distance_threshold=0.08,
        min_detections_for_permanent=3,
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
