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

"""Container (bin) pick by the rim with the Piper's wrist RealSense.

``dimos run piper-grasp-bin --can-port can0``

The ``piper-grasp`` stack with rim grasps in place of the heuristic provider and
the guarded ContainerPickModule skills on top (``pick_up_container``,
``set_down_container``). What differs from the xArm7 the skill was learned on:

- The Piper cannot hold its tool straight down more than about 11 cm above its
  base plane, so the tool leans within the grasped wall's plane, fingertips away
  from the base. That only buys height on walls that run roughly radially from
  the base; the module checks each wall by inverse kinematics and takes the
  first one it can approach, grasp and lift.
- The wrist camera sits beside the fingers, 13 cm along the tool X and barely
  above the tool point. Leaning the tool lifts it clear of the rim; straight
  down it would land on a long wall.
- The camera does not look along the tool, so the survey is a joint posture
  with the camera looking down from about 40 cm, not a top-down tool pose.
- There is no voxel map on this rig, so nothing to pause while holding, and the
  planner knows only the robot: the path guard's boxes are the protection.

Set PIPER_JUDGE_CAN, PIPER_JOINT_OFFSETS_DEG and PIPER_WRIST_CAMERA_SERIAL as for
``piper-grasp``.
"""

from __future__ import annotations

import os

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.container_pick_module import ContainerPickModule
from dimos.manipulation.grasp_verification import GraspVerificationConfig
from dimos.manipulation.grasping.rim_grasp import RimGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.piper.blueprints.grasp import (
    PIPER_FINGERTIPS_PAST_TCP,
    PIPER_GRASP_PREGRASP_OFFSET,
    PIPER_GRASP_SCAN_JOINTS,
    PIPER_GRASP_TABLE_Z,
    PIPER_WRIST_CAMERA_TRANSFORM,
)
from dimos.robot.manipulators.piper.config import (
    PIPER_COLLISION_LINKS,
    PIPER_TCP_FRAME,
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule

# The wrist camera looks down from about 39 cm above the base plane, 35 cm out,
# which shows a 28 cm bin whole. Joint 1 is turned toward the container.
PIPER_BIN_SURVEY_JOINTS = [0.0, 1.05, -1.18, 0.0, 1.10, 0.0]

# The lowest tool point: straight down, fingertips 1 cm above the table.
PIPER_BIN_MIN_Z = PIPER_GRASP_TABLE_Z + PIPER_FINGERTIPS_PAST_TCP + 0.01

# Leans of the tool within the wall's plane, tried in order. Negative lifts the
# camera side. From inverse kinematics of the model: at 30 deg a radial wall is
# reachable from 5 to 25 cm up at radii of 0.15 to 0.50 m, at 40 deg up to
# 30 cm, and straight down only below 11 cm.
PIPER_BIN_TOOL_TILTS = [-0.52, -0.70, -0.35, 0.0]

# The wrist camera's centre in the tool frame (PIPER_WRIST_CAMERA_TRANSFORM
# carried from link6 to gripper_tcp). It must clear the rim at the grasp.
PIPER_BIN_CAMERA_IN_TOOL = (-0.128, 0.016, -0.032)

# Where the path guard lets the hand (link6 to the tool point) and the elbow
# (link3, link4) go: over the table in front of the base, with room behind for
# the elbow at the rest and survey postures. piper_sdk has no TCP boundary of
# its own, so these boxes and the reduced speeds are all there is.
PIPER_BIN_HAND_BOX = ((-0.05, 0.60), (-0.40, 0.40), (0.01, 0.55))
PIPER_BIN_ELBOW_BOX = ((-0.35, 0.50), (-0.35, 0.35), (0.05, 0.60))

_JUDGE_CAN = os.getenv("PIPER_JUDGE_CAN", "1").strip().lower() not in ("0", "false", "no", "off")
# None lets librealsense take the first camera it enumerates.
_WRIST_CAMERA_SERIAL = os.getenv("PIPER_WRIST_CAMERA_SERIAL", "").strip() or None
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
# A thin wall leaves the jaws almost closed; see GraspVerificationConfig for
# how to place empty_epsilon between an empty close and a close on the wall.
_grasp_verification = GraspVerificationConfig(open_tolerance=0.2, empty_epsilon=0.012)

piper_grasp_bin = autoconnect(
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
        min_place_z=PIPER_GRASP_TABLE_Z + PIPER_FINGERTIPS_PAST_TCP,
        grasp_verification=_grasp_verification,
    ),
    RimGraspModule.blueprint(min_z=PIPER_BIN_MIN_Z),
    ContainerPickModule.blueprint(
        model=_model.model,
        planning_frame="world",
        min_z=PIPER_BIN_MIN_Z,
        hand_links=["link6", "gripper_base", PIPER_TCP_FRAME],
        elbow_links=["link3", "link4"],
        workspace_box=PIPER_BIN_HAND_BOX,
        elbow_box=PIPER_BIN_ELBOW_BOX,
        reach_max=0.50,
        base_joint="joint1",
        # Turning to face a container at the side of the table is a wide but
        # slow base move; the rest and survey postures are 3.3 rad apart.
        base_joint_max_excursion=1.6,
        path_max_length=5.0,
        wrist_joint="joint6",
        plan_speed_scale=0.2,
        cartesian_speed_scale=0.2,
        survey_joints=PIPER_BIN_SURVEY_JOINTS,
        approach="plan",
        tool_tilts=PIPER_BIN_TOOL_TILTS,
        check_reachability=True,
        tool_keepout_points=[PIPER_BIN_CAMERA_IN_TOOL],
        max_candidates=4,
        # Fingertips 3 cm above the rim before the descent, then 8 cm more.
        pregrasp_offset=0.08,
        lift_height=0.08,
        prompts=["yellow bin", "bin"],
        grasp_verification=_grasp_verification,
        pause_mapping_while_holding=False,
    ),
    RealSenseCamera.blueprint(enable_pointcloud=True, serial_number=_WRIST_CAMERA_SERIAL),
    ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="moondream",
        segmentation_backend="edgetam",
        # The wrist camera is a D405, whose depth unit is 0.1 mm.
        depth_unit_m=0.0001,
        detect_on_request=True,
        accumulate_pointclouds=False,
        distance_threshold=0.05,
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
