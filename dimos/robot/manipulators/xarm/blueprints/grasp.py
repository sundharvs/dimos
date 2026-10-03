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

"""The xArm grasping stack, on hardware by default.

``dimos run xarm-grasp --xarm7-ip 192.168.1.x``   heuristic grasps
``dimos run xarm-grasp-graspgenx --xarm7-ip ...`` learned grasps
``dimos run xarm-grasp --simulation mujoco``      the same stack in MuJoCo

Only the grasp provider separates the two blueprints. The arm-versus-sim split is
decided here at import time, because composition runs before module config is
applied. What it swaps is the hardware adapter, the base pose, the camera and the
engine behind it, the detector backends and the home pose. Everything else -- the
coordinator, pick-and-place, scene registration -- is the same stack either way.
"""

from __future__ import annotations

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.container_pick_module import ContainerPickModule
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXModule
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.grasping.rim_grasp import RimGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.manipulation.planning.utils.point_cloud_self_filter import PointCloudSelfFilter
from dimos.manipulation.wrist_tabletop_module import WristTabletopModule
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.xarm.config import (
    XARM7_COLLISION_LINKS,
    make_xarm7_model_config,
    make_xarm7_sim_hardware,
    make_xarm7_sim_module_kwargs,
    make_xarm7_sim_robot_config,
    xarm7_hardware,
)
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
from dimos.utils.data import LfsPath
from dimos.visualization.rerun.bridge import RerunBridgeModule

SIMULATED = bool(global_config.simulation)

XARM_GRASP_SCENE_PATH = LfsPath("xarm_grasp_sim/scene.xml")
# The stock xArm home points the narrow wrist-camera frustum between the widely
# spaced targets. This collision-free top-down pose raises the camera enough to
# put every mesh in one frame, without changing the configured base pose.
XARM_GRASP_SCAN_JOINTS = [0.0, -0.04609, 0.0, 1.83940, 0.0, 1.87106, 0.0]
# One resolution for the whole mapping chain. The self filter's clear mask, the
# mapper's cells and the planner's octree must all agree: a mismatched mask names
# cells the map does not hold, and a mismatched octree does not line up with what
# was mapped.
XARM_GRASP_VOXEL_SIZE = 0.025

# TCP floor for the container pick: the operator hand-guided the fingertips to a
# safe height above the table on 2026-10-03 and this is where the model put them.
XARM_CONTAINER_MIN_Z = -0.0085
# Workspace the container pick may use on this table (planning frame = link_base).
# The hand box ends 10 cm behind the base; the elbow box allows the fold-back of
# the upper arm when the hand is out over the table. Set the xArm's own TCP
# boundary in the controller (SDK set_reduced_tcp_boundary) as the first line.
XARM_CONTAINER_HAND_BOX = ((-0.42, 0.42), (-0.78, 0.10), (-0.03, 0.80))
XARM_CONTAINER_ELBOW_BOX = ((-0.45, 0.45), (-0.78, 0.30), (-0.03, 0.88))

XARM_GRASP_PROMPTS = [
    "black bottle",
    "gray can",
    "red cup",
    "green tape roll",
    "blue marker",
    "brown box",
    # The wrist camera sees the tape almost directly from above, where it reads
    # as a ring instead of a roll. Keep a shape-word fallback for that view.
    "green ring",
]

# Measured off data/xarm_grasp_sim (mj_forward at the driver joint limits) and the
# gripper URDF, expressed in GraspGenX's convention -- approach along +Z, jaws
# closing along X -- with the origin on xarm_gripper_base_link, the frame
# GraspGenX predicts into. The geometry is the real gripper's, so it holds on
# hardware too.
XARM_GRIPPER_SWEEP_VOLUME = {
    "extents_open": (0.0889, 0.030, 0.0370),
    "offset_open": (0.0, 0.0, 0.1421),
    "extents_half_open": (0.0479, 0.030, 0.0370),
    "offset_half_open": (0.0, 0.0, 0.1530),
    "fingertip_depth": 0.1606,
}
# xarm_gripper_base_link -> link_tcp, the planning tip frame: +0.172 m along the
# approach axis (xarm_gripper.urdf.xacro joint_tcp) plus the quarter turn that
# takes GraspGenX's X closing axis onto the xArm gripper's Y.
XARM_GRASP_FRAME_TO_TCP = (
    (0.0, 1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.172),
    (0.0, 0.0, 0.0, 1.0),
)

# Hand-eye calibration for the eye-in-hand RealSense. RealSenseCamera publishes
# only its own subtree, so without this edge camera_link has no parent, nothing
# resolves into world, and every cloud the camera produces is silently unusable.
# link7 is a frame the model already publishes, so ManipulationModule emits the
# whole chain from one loop at one rate.
#
# Measured 2026-10-03 on the real arm (192.168.1.197): 58 still-arm captures of a
# 5x7 ChArUco board (33.0 mm printed squares) at 1280x720, AX=XB over all five
# OpenCV solvers against URDF forward kinematics of link7, board-in-base spread
# 2.7 mm / 0.6 deg RMS (the previous value gave 14 mm). The offset along the
# camera's optical axis is the least constrained direction, about +-4 mm.
XARM_WRIST_CAMERA_TRANSFORM = Transform(
    translation=Vector3(x=0.06865142, y=-0.01899432, z=0.02597990),
    rotation=Quaternion(0.71878005, 0.00373539, 0.69521878, 0.00348488),  # xyzw
    frame_id="link7",
    child_frame_id="camera_link",
)


if SIMULATED:
    # data/xarm_grasp_sim/xarm7.xml bolts link_base to the world origin instead
    # of the 12 cm pedestal data/xarm7 uses. Inheriting that offset would put the
    # planning model 12 cm above the arm MuJoCo simulates.
    _model = make_xarm7_sim_robot_config(
        base_pose=PoseStamped(frame_id="world"),
        # The self filter needs a capture-time transform for every collision link
        # and drops the whole cloud when one is missing.
        tf_extra_links=XARM7_COLLISION_LINKS,
    )
    _hardware = make_xarm7_sim_hardware(XARM_GRASP_SCENE_PATH, home_joints=XARM_GRASP_SCAN_JOINTS)
else:
    _model = make_xarm7_model_config(
        add_gripper=True,
        gripper_hardware_id="arm",
        base_pose=PoseStamped(frame_id="world"),
        # The self filter needs a capture-time transform for every collision link
        # and drops the whole cloud when one is missing.
        tf_extra_links=XARM7_COLLISION_LINKS,
    )
    _hardware = xarm7_hardware("arm", gripper=True)


def _sensing() -> tuple[Blueprint, ...]:
    """The camera, and in sim the engine that produces its frames."""
    if SIMULATED:
        return (
            MujocoSimModule.blueprint(
                **{
                    **make_xarm7_sim_module_kwargs(XARM_GRASP_SCENE_PATH),
                    "headless": True,
                    # Publish the simulated camera pose directly in the planning
                    # frame. A wrist-relative TF would need a second
                    # asynchronously stamped world->link7 edge and can make an
                    # otherwise valid scan unregistrable.
                    "base_frame_id": "world",
                    "reset_joint_positions": XARM_GRASP_SCAN_JOINTS,
                    # Off by default, and the voxel chain has nothing to map
                    # without it. Scene registration reads colour and depth.
                    "enable_pointcloud": True,
                }
            ),
        )
    return (
        # enable_pointcloud is off by default here too, and the voxel chain has
        # nothing to map without it.
        RealSenseCamera.blueprint(enable_pointcloud=True),
    )


def _scene_registration() -> Blueprint:
    """Detector settings differ: synthetic renders score far below real images."""
    if SIMULATED:
        return ObjectSceneRegistrationModule.blueprint(
            target_frame="world",
            detector_backend="owlv2",
            # OWLv2 is box-only; YOLO-E visual prompts refine its boxes into masks.
            segmentation_backend="yolo",
            detector_confidence=0.07,
            segmentation_confidence=0.05,
            # Keep adjacent tabletop targets distinct instead of merging by label.
            distance_threshold=0.05,
            detect_on_request=True,
            # The obstacle stream carries permanent objects only, so one explicit
            # scan must promote its first sightings immediately.
            min_detections_for_permanent=1,
        )
    return ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="moondream",
        segmentation_backend="edgetam",
        detect_on_request=True,
        distance_threshold=0.08,
        min_detections_for_permanent=3,
        max_distance=1.0,
        use_aabb=True,
        max_obstacle_width=0.06,
    )


def _voxel_mapping() -> tuple[Blueprint, ...]:
    """Wrist camera -> self filter -> mapper -> the planner's octree obstacle."""
    return (
        # The wrist camera sees the arm itself, so the arm's returns must be
        # dropped before mapping and the volume it occupies erased from the map:
        # ray tracing cannot clear what the arm permanently occludes.
        PointCloudSelfFilter.blueprint(
            model=_model.model,
            voxel_size=XARM_GRASP_VOXEL_SIZE,
            world_frame="world",
            # ManipulationModule publishes robot TF at 10Hz, so the stock 20ms
            # tolerance cannot bracket a ~92ms publish period and drops most
            # clouds. One full period admits them all, and the arm holds still
            # while scanning, so a transform a period old describes the same pose.
            tf_tolerance_s=0.1,
            tf_forward_tolerance_s=0.1,
            # The base plate sits on the table: cells mapped around it collide
            # with link_base and leave every start configuration "in collision".
            base_exclusion_radius=0.17,
        ),
        # Tabletop reach, not a room-scale lidar sweep.
        RayTracingVoxelMap.blueprint(
            voxel_size=XARM_GRASP_VOXEL_SIZE,
            world_frame="world",
            max_range=2.0,
        ),
    )


# The mapping chain is wired by stream name, so the two edges whose names differ
# are bridged here: self filter -> mapper, and mapper -> the planner's octree.
_REMAPPINGS = [
    (RayTracingVoxelMap, "lidar", "filtered_pointcloud"),
    (ManipulationModule, "voxel_map", "global_map"),
]

# Everything but the grasp provider. Exactly one provider may be composed in:
# PickAndPlaceModule resolves its generator by spec, so two would be ambiguous.
_XARM_GRASP_MODULES = (
    ManipulationModule.blueprint(
        model=_model,
        static_transforms=[] if SIMULATED else [XARM_WRIST_CAMERA_TRANSFORM],
        planning_timeout=10.0,
        visualization={"backend": "viser"},
        world_frame="world",
        voxel_map_resolution=XARM_GRASP_VOXEL_SIZE,
    ),
    ManipulationSkills.blueprint(),
    PickAndPlaceModule.blueprint(planning_frame="world"),
    *_sensing(),
    _scene_registration(),
    *_voxel_mapping(),
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

xarm_grasp = autoconnect(*_XARM_GRASP_MODULES, HeuristicGraspModule.blueprint()).remappings(
    _REMAPPINGS
)

# ``xarm-grasp-bin``: rim grasps for open containers (bins, boxes) that a centroid
# grasp cannot hold, the guarded ContainerPickModule skills on top of them
# (pick_up_container / place_container / place_container_in_slot / map_slots /
# check_container_pose / rotate_held_container / set_down_container) and the
# wrist-camera colour scan + tape-slot mapper they use. All parameters are
# RPC-settable so an outer research loop can tune them live.
xarm_grasp_bin = autoconnect(
    *_XARM_GRASP_MODULES,
    RimGraspModule.blueprint(min_z=None if SIMULATED else XARM_CONTAINER_MIN_Z),
    WristTabletopModule.blueprint(
        planning_frame="world",
        # The yellow shelf bin (H 23-25, S > 190 in the wrist camera) and beige
        # masking tape on beech (bluish-grey next to the orange wood).
        object_hsv_low=(15, 120, 100),
        object_hsv_high=(40, 255, 255),
        tape_hsv_low=(80, 20, 120),
        tape_hsv_high=(125, 110, 255),
        slot_names=[
            "left",
            "middle",
            "right",
        ],  # from -X to +X: left as seen from the table's front
    ),
    ContainerPickModule.blueprint(
        model=_model.model,
        planning_frame="world",
        min_z=None if SIMULATED else XARM_CONTAINER_MIN_Z,
        hand_links=["link7", "link_tcp"],
        elbow_links=["link4", "link5"],
        workspace_box=XARM_CONTAINER_HAND_BOX,
        elbow_box=XARM_CONTAINER_ELBOW_BOX,
        wrist_joint="joint7",
        wrist_joint_sign=-1.0,
        prompts=["yellow bin", "bin"],
        # The shelf bin the skill was learned on: 28 x 10 cm rim, so a fit much
        # shorter than that is a partial segmentation, not a smaller bin.
        container_long_min=0.24,
        container_short_range=(0.07, 0.14),
        # Wrist RealSense optical centre relative to the TCP at the survey yaw
        # (measured from TF), so surveys put the camera, not the TCP, above the target.
        survey_camera_offset=(0.032, -0.070),
        # Learned 2026-10-03: the hanging bin lands 4.0 cm from the jaws, not the
        # 5.3 cm rim half width.
        landing_offset=0.040,
        preferred_reach=0.42,
        # Camera positions that see the whole three-slot tape grid in front of the arm.
        slot_survey_points=[(-0.02, -0.37), (-0.02, -0.55), (-0.02, -0.59)],
    ),
).remappings(_REMAPPINGS)

xarm_grasp_graspgenx = autoconnect(
    *_XARM_GRASP_MODULES,
    GraspGenXModule.blueprint(
        gripper=XARM_GRIPPER_SWEEP_VOLUME,
        grasp_frame_to_tcp=XARM_GRASP_FRAME_TO_TCP,
    ),
).remappings(_REMAPPINGS)
