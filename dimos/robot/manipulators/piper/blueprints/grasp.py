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
offsets applied.

Poses are planned to the tool frame between the jaws, so a ``place_at`` height
is where the grasp point ends up; leave z out to set an object back down on the
table. Objects must be in the wrist camera's view from the scan pose, about
x 0.17-0.38 m and y +-0.19 m: one cut off by the edge of the frame is grasped
off centre.

``dimos run piper-grasp-bin --can-port can0``

The same stack with rim grasps and the ContainerPickModule skills
(``pick_up_container``, ``set_down_container``, ...) for an open container the
jaws cannot span: the yellow shelf bin, 29 x 11 x 7.5 cm, found by colour from a
higher survey pose that sees the whole table in front of the arm.

Origin: autoresearch task "pick up the yellow bin", 2026-10-04. First success by
script (straddle the arm-side long wall, close, lift, lean back); the skill is
the xArm's container pick with the Piper's limits in configuration.
Validation on the arm: see PIPER_BIN_VALIDATION below.
"""

from __future__ import annotations

import math
import os

from dimos.control.components import HardwareComponent
from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.container_pick_module import ContainerPickModule
from dimos.manipulation.grasp_verification import GraspVerificationConfig
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.grasping.rim_grasp import RimGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.manipulation.wrist_tabletop_module import WristTabletopModule
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
_model = make_piper_model_config(home_joints=PIPER_GRASP_SCAN_JOINTS, tcp=True).model_copy(
    update={
        "base_pose": PoseStamped(frame_id="world"),
        # link6 carries the camera.
        "tf_extra_links": PIPER_COLLISION_LINKS,
    }
)


def _piper_hardware(gripper_effort: int | None = None) -> HardwareComponent:
    return piper_hardware(
        "arm",
        mock_without_address=False,
        judge_can=_JUDGE_CAN,
        joint_offsets=piper_joint_offsets_from_env(),
        gripper_effort=gripper_effort,
    )


# Everything but the grasp provider: PickAndPlaceModule resolves its generator by
# spec, so exactly one may be composed in.
def _piper_grasp_stack(hardware: HardwareComponent) -> tuple[Blueprint, ...]:
    return (
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
        RealSenseCamera.blueprint(
            enable_pointcloud=True,
            # With a second RealSense on the rig (a scene camera), name the wrist one.
            serial_number=os.getenv("PIPER_WRIST_CAMERA_SERIAL") or None,
        ),
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
            hardware=[hardware],
            tasks=[
                trajectory_task(hardware),
                TaskConfig(
                    name="arm_gripper",
                    type="gripper",
                    joint_names=["arm/gripper"],
                    priority=20,
                ),
            ],
        ),
    )


piper_grasp = autoconnect(
    *_piper_grasp_stack(_piper_hardware()),
    HeuristicGraspModule.blueprint(
        fingertip_depth=PIPER_FINGERTIPS_PAST_TCP,
        support_z=PIPER_GRASP_TABLE_Z,
        # The jaws open 67 mm, so they must straddle the object's real middle;
        # anything off the camera's axis shows a side wall as well as its top.
        centering="extent",
        # Top-down with yaw 0 needs joint 6 at +-180 deg, past its +-120 deg range.
        yaw_offset=math.pi,
    ),
)

# --- Container (bin) pick -----------------------------------------------------

# Held-out result of pick_up_container through this blueprint, each trial judged
# from a scene-camera frame (bin hanging from the jaws, clear of the table).
PIPER_BIN_VALIDATION = (
    "2026-10-04: 4 of 5 held-out bin poses lifted clear (centres x 0.26-0.31 m, y -0.08-0.08 m, "
    "long-axis yaw 50, -81, -3 and 33 deg); the skill's verdict matched the frame on all five. "
    "The fifth (bin lying along x at 0.31, 0.07) was refused: no IK for the pre-grasp. "
    "Untested: other containers, a loaded bin, anything outside the survey view."
)

# Wrist camera 35 cm above the table, looking down at x = 0.29 m: the view spans
# about x 0.10-0.45 m, y +-0.25 m, the whole shelf bin with a margin. Found with
# forward kinematics and checked with a frame on 2026-10-04. The tool cannot
# point straight down from this height (wrist pitch range), so it is a joint pose.
PIPER_BIN_SURVEY_JOINTS = [0.0, 0.45, -0.7, 0.0, 1.15, 0.0]
# Lowest TCP height for a wall grasp: the fingertips (2 cm past the TCP) stop
# 1.5 cm above the table, clear of the bin's floor.
PIPER_BIN_MIN_Z = PIPER_GRASP_TABLE_Z + PIPER_FINGERTIPS_PAST_TCP + 0.015
# Guard boxes in world (the arm's base frame). Forward kinematics of the rest,
# survey, grasp and leaned-back poses puts link6 and the TCP within x 0.05-0.30,
# z 0.03-0.33 and link3/link4 within x -0.29-0.30, z 0.16-0.42; the boxes add the
# table in front of the arm and a margin.
PIPER_BIN_HAND_BOX = ((-0.05, 0.50), (-0.40, 0.40), (0.0, 0.60))
PIPER_BIN_ELBOW_BOX = ((-0.40, 0.45), (-0.40, 0.40), (0.05, 0.70))

# Closing torque of the jaws for the bin, mN*m. At the adapter's 1000 (1 N*m of
# the gripper's rated 5) the pinch on the bin's 2 mm wall let it pivot and slide
# out in 4 of 7 lifts on 2026-10-04, and at 2500 once in three; the operator
# allowed up to 4000.
PIPER_BIN_GRIPPER_EFFORT = 4000

piper_grasp_bin = autoconnect(
    *_piper_grasp_stack(_piper_hardware(PIPER_BIN_GRIPPER_EFFORT)),
    RimGraspModule.blueprint(
        min_z=PIPER_BIN_MIN_Z,
        # The bin's walls are 10 cm tall (top at world z 0.09) with a stacking
        # step 2.5 cm below the top. TCP 3.3 cm below the top puts the pads
        # (fingertips 2 cm past the TCP) across the step; that depth held on
        # 2026-10-04, while 5 cm or more below the top slipped at 1 N*m.
        insertion_depth=0.033,
        # A long wall keeps the hanging bin's lever arm at half its width; an end
        # wall would hang it by 14 cm.
        sides="long",
    ),
    WristTabletopModule.blueprint(
        planning_frame="world",
        # The yellow shelf bin, as on the xArm.
        object_hsv_low=(15, 120, 100),
        object_hsv_high=(40, 255, 255),
        # The wrist camera is a D405: 0.1 mm depth units, usable from 7 cm, which
        # is where a held bin is.
        depth_unit_m=0.0001,
        depth_range=(0.07, 1.5),
        # A bin leaned back in the jaws reaches 0.35 m.
        object_z_range=(-0.04, 0.45),
    ),
    ContainerPickModule.blueprint(
        model=_model.model,
        planning_frame="world",
        min_z=PIPER_BIN_MIN_Z,
        hand_links=["link6", "gripper_tcp"],
        elbow_links=["link3", "link4"],
        workspace_box=PIPER_BIN_HAND_BOX,
        elbow_box=PIPER_BIN_ELBOW_BOX,
        reach_max=0.45,
        # Survey to a pre-grasp over the far wall sums to 3.5 rad over six joints.
        path_max_length=4.5,
        wrist_joint="joint6",
        # Tool down, yaw = joint1 - joint6 + pi (forward kinematics, 2026-10-04).
        wrist_joint_sign=-1.0,
        survey_joints=PIPER_BIN_SURVEY_JOINTS,
        planned_approach=True,
        # The tool points straight down only up to a TCP height of about 0.125 m,
        # and there only at radii of 0.16-0.32 m (piper_reach.py): pre-grasp and
        # straight lift stay under it, the lean adds the rest. Leaning joint 2
        # back 0.35 rad raised the TCP from 0.117 to 0.217 m with the bin held.
        # Pre-grasp at the top of that band: fingertips 1 cm above the bin's rim.
        pregrasp_offset=0.062,
        carry_height=0.04,
        lift_height=0.04,
        lift_joint_offsets={"joint2": -0.35},
        # With the far long wall in the jaws the leaned-back bin hangs straight
        # down from it; that wall is preferred while it is inside the top-down band.
        preferred_reach=0.30,
        # The jaws read 0.002 with the bin hanging in them, as on air: look instead.
        hold_check="camera",
        release_exit="up",
        prompts=["yellow bin", "bin"],
        container_long_min=0.24,
        container_short_range=(0.07, 0.14),
        grasp_verification=GraspVerificationConfig(open_tolerance=0.2),
    ),
)
