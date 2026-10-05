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

"""Stack toy rings onto an upright rod with the Piper's wrist RealSense.

``dimos run piper-ring-stack --can-port can0``

The perception and planning stack of ``piper-grasp`` with ``RingStackModule`` in
place of pick and place: ``scan_rings`` finds the rod and the rings by name,
``stack_ring`` puts one on, ``stack_all_rings`` puts every one on in whatever
order it finds them. Set PIPER_JUDGE_CAN, PIPER_JOINT_OFFSETS_DEG and
PIPER_WRIST_CAMERA_SERIAL as for ``piper-grasp``; PIPER_GRIPPER_EFFORT changes how
hard the jaws close.

The rod and the rings stand on the table in front of the arm, rings 0.2-0.47 m
from the base and the rod 0.25-0.45 m. A ring is gripped with the fingers
upright and carried over the rod with the gripper leaning 25 degrees outward,
which is what the wrist pitch range needs at that height, so it arrives tilted
by that much and levels out as it drops.
"""

from __future__ import annotations

import math
import os

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasp_verification import GraspVerificationConfig
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.ring_stack_module import RingStackModule
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.piper.blueprints.grasp import (
    _JUDGE_CAN,
    PIPER_FINGERTIPS_PAST_TCP,
    PIPER_GRASP_SCAN_JOINTS,
    PIPER_WRIST_CAMERA_SERIAL,
    PIPER_WRIST_CAMERA_TRANSFORM,
)
from dimos.robot.manipulators.piper.config import (
    PIPER_COLLISION_LINKS,
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule

# The table top in world: where the fingertips touch down, and what the wrist
# camera's depth reads beside a ring, both measured on 2026-10-03.
PIPER_RING_TABLE_Z = -0.019
# The whole table in view: the camera 43 cm up, looking forward and down.
PIPER_RING_OVERVIEW_JOINTS = [0.0, 0.5, -0.9, 0.0, 1.2, 0.0]
# The camera 36 cm above the table and looking straight down at a spot 0.445 m
# from the base, which puts it about 19 cm above the top of a 19 cm rod.
PIPER_ROD_VIEW_JOINTS = [0.0, 1.41, -1.41, 0.0, 1.15, 0.0]
PIPER_ROD_VIEW_DISTANCE = 0.445
# From the scan pose the camera looks straight down at a spot this far out.
PIPER_RING_VIEW_DISTANCE = 0.29
# Upright fingers hold a ring by their whole width: a ring gripped that way
# did not move in the jaws through ten seconds and a turn, where one gripped
# with the fingers leaning 30 degrees crept out of them in about fifteen.
PIPER_RING_PICK_LEAN = 0.0
# The wrist can hold the tool 24 cm up with 22 degrees of lean; three more keep
# the carrying poses off the joint limit, where the planner cannot connect them.
PIPER_RING_CARRY_LEAN = math.radians(25.0)
# Where the tool can point straight down at the table, and where it can lean
# by the carry lean at carrying height.
PIPER_RING_REACH = (0.2, 0.47)
PIPER_RING_CARRY_REACH = (0.25, 0.45)
# The toy's rod is 37 mm across at its top.
PIPER_RING_ROD_RADIUS = 0.0185
# The jaws' torque limit in mN.m; unset leaves the adapter's default, 1000. A
# ring is held by the last centimetre of the fingers, and a harder squeeze does
# not hold it better: at 3000 a soft ring creeps out of the jaws within seconds.
_GRIPPER_EFFORT = os.getenv("PIPER_GRIPPER_EFFORT", "").strip()
PIPER_RING_GRIPPER_EFFORT = int(_GRIPPER_EFFORT) if _GRIPPER_EFFORT else None

_hardware = piper_hardware(
    "arm",
    mock_without_address=False,
    judge_can=_JUDGE_CAN,
    joint_offsets=piper_joint_offsets_from_env(),
    gripper_effort=PIPER_RING_GRIPPER_EFFORT,
)
_model = make_piper_model_config(home_joints=PIPER_GRASP_SCAN_JOINTS, tcp=True).model_copy(
    update={
        "base_pose": PoseStamped(frame_id="world"),
        # link6 carries the camera.
        "tf_extra_links": PIPER_COLLISION_LINKS,
    }
)

piper_ring_stack = autoconnect(
    ManipulationModule.blueprint(
        model=_model,
        static_transforms=[PIPER_WRIST_CAMERA_TRANSFORM],
        planning_timeout=10.0,
        visualization={"backend": "viser"},
        world_frame="world",
    ),
    ManipulationSkills.blueprint(),
    RingStackModule.blueprint(
        planning_frame="world",
        support_z=PIPER_RING_TABLE_Z,
        overview_joints=PIPER_RING_OVERVIEW_JOINTS,
        ring_view_joints=PIPER_GRASP_SCAN_JOINTS,
        ring_view_distance=PIPER_RING_VIEW_DISTANCE,
        rod_view_joints=PIPER_ROD_VIEW_JOINTS,
        rod_view_distance=PIPER_ROD_VIEW_DISTANCE,
        pick_lean=PIPER_RING_PICK_LEAN,
        carry_lean=PIPER_RING_CARRY_LEAN,
        reach=PIPER_RING_REACH,
        carry_reach=PIPER_RING_CARRY_REACH,
        rod_radius=PIPER_RING_ROD_RADIUS,
        fingertip_depth=PIPER_FINGERTIPS_PAST_TCP,
        # Commanded fully open (0.08 m) the Piper's jaws stop at 0.84.
        grasp_verification=GraspVerificationConfig(open_tolerance=0.2),
    ),
    RealSenseCamera.blueprint(enable_pointcloud=True, serial_number=PIPER_WRIST_CAMERA_SERIAL),
    ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="moondream",
        segmentation_backend="edgetam",
        # The wrist camera is a D405, whose depth unit is 0.1 mm.
        depth_unit_m=0.0001,
        detect_on_request=True,
        # Rings get moved here, and a moved ring must not keep the outline it
        # had before.
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
