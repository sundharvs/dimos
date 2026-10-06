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

"""Heuristic top-down pick and place with the Piper's wrist RealSense.

``dimos run piper-grasp --can-port can0``

The xArm grasp stack on a Piper, without the voxel map: its ray tracer is a
native cargo build this rig has not had, so the planner knows only the robot.
Scene registration finds prompted objects by name, moondream boxes refined into
masks by EdgeTAM, and the heuristic provider grasps them straight down. Both
models want a GPU. Set PIPER_JUDGE_CAN and PIPER_JOINT_OFFSETS_DEG as for the
other Piper blueprints; the camera edge below was calibrated with the joint
offsets applied. With a second RealSense attached, set PIPER_WRIST_CAMERA_SERIAL
to the wrist camera's serial number, or the stack may open the other one.

Poses are planned to the tool frame between the jaws, so a ``place_at`` height
is where the grasp point ends up; leave z out to set an object back down on the
table. Objects must be in the wrist camera's view from the scan pose, about
x 0.17-0.38 m and y +-0.19 m: one cut off by the edge of the frame is grasped
off centre.
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
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.piper.config import (
    PIPER_COLLISION_LINKS,
    PIPER_FINGERTIP_DEPTH,
    PIPER_TCP_DEPTH,
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule

# The hand-eye calibration start pose: the wrist camera looks down at the table
# from about 22 cm, with joint 5 clear of its limit. go_home returns here.
PIPER_GRASP_SCAN_JOINTS = [-0.0406, 0.9366, -0.5184, 0.0, 0.805, 0.0]

# Poses are planned to the tool frame between the jaws; the fingertips reach
# this far past it, and are kept 1 cm above the table when grasping.
PIPER_FINGERTIPS_PAST_TCP = PIPER_FINGERTIP_DEPTH - PIPER_TCP_DEPTH
# The table top in world, measured from the wrist camera's scene cloud on
# 2026-10-02 with the camera edge below. Re-measure if the arm or table moves.
PIPER_GRASP_TABLE_Z = -0.011
# How far above a grasp or a place the arm stops before the straight final leg.
# The wrist pitch range bounds where the tool can point straight down, and the
# band narrows with height: 10 cm up it spans radii of 0.16-0.33 m from the
# base, 6 cm up 0.08-0.41 m (see the piper-hardware skill's piper_reach.py). The
# fingers still clear an object up to about 7 cm tall on the way in.
PIPER_GRASP_PREGRASP_OFFSET = 0.06

# piper-hand-eye-calibration, 2026-10-02, with PIPER_JOINT_OFFSETS_DEG=0,0,0,0,4.42,0,
# then refined against three objects left where they were and seen from 28
# poses with the wrist rolled and pitched, last on 2026-10-03 15:35: the views
# disagreed about where one object was by 5.6 mm RMS before and 2.2 mm after.
# The camera turns on its mount when the arm is knocked or handled, by 2 to 3
# degrees each of the three times it was measured that day, which puts an
# object 1 to 2 cm off depending on where in the image it is. Park the arm
# before stopping the stack, and re-measure whenever views stop agreeing, the
# mount is touched, or the joint offsets change.
PIPER_WRIST_CAMERA_TRANSFORM = Transform(
    translation=Vector3(x=-0.12844295, y=0.01598712, z=0.09041254),
    rotation=Quaternion(-0.01760351, -0.51145408, 0.02674281, 0.85871396),  # xyzw
    frame_id="link6",
    child_frame_id="camera_link",
)

_JUDGE_CAN = os.getenv("PIPER_JUDGE_CAN", "1").strip().lower() not in ("0", "false", "no", "off")
# None lets librealsense take the first camera it enumerates.
PIPER_WRIST_CAMERA_SERIAL = os.getenv("PIPER_WRIST_CAMERA_SERIAL", "").strip() or None
_hardware = piper_hardware(
    "arm",
    mock_without_address=False,
    judge_can=_JUDGE_CAN,
    joint_offsets=piper_joint_offsets_from_env(),
)
_model = make_piper_model_config(home_joints=PIPER_GRASP_SCAN_JOINTS, tcp=True).model_copy(
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
        pregrasp_offset=PIPER_GRASP_PREGRASP_OFFSET,
        # A place whose fingertips would be under the table top.
        min_place_z=PIPER_GRASP_TABLE_Z + PIPER_FINGERTIPS_PAST_TCP,
        # Commanded fully open (0.08 m) the Piper's jaws stop at 0.84, about
        # 67 mm, which the default 0.15 tolerance just rejects.
        grasp_verification=GraspVerificationConfig(open_tolerance=0.2),
    ),
    HeuristicGraspModule.blueprint(
        fingertip_depth=PIPER_FINGERTIPS_PAST_TCP,
        support_z=PIPER_GRASP_TABLE_Z,
        # The jaws open 67 mm, so they must straddle the object's real middle;
        # anything off the camera's axis shows a side wall as well as its top.
        centering="extent",
        # Top-down with yaw 0 needs joint 6 at +-180 deg, past its +-120 deg range.
        yaw_offset=math.pi,
    ),
    RealSenseCamera.blueprint(enable_pointcloud=True, serial_number=PIPER_WRIST_CAMERA_SERIAL),
    ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="moondream",
        segmentation_backend="edgetam",
        # The wrist camera is a D405, whose depth unit is 0.1 mm.
        depth_unit_m=0.0001,
        detect_on_request=True,
        # Objects get moved here, and a moved object must not keep the outline
        # it had before.
        accumulate_pointclouds=False,
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
