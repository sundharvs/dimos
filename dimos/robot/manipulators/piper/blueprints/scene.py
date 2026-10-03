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
"""Piper arm with an RGB scene camera.

The camera is any V4L2 device opened through OpenCV, so an Intel RealSense can
be used for its colour stream alone without librealsense or pyrealsense2.

Environment overrides (read at import time):

* ``PIPER_SCENE_CAMERA`` - V4L2 node or index of the RGB stream (default ``/dev/video6``).
* ``PIPER_SCENE_CAMERA_WIDTH`` / ``PIPER_SCENE_CAMERA_HEIGHT`` / ``PIPER_SCENE_CAMERA_FPS``
  - capture size and software frame-rate cap (default 1280x720 @ 15 Hz).
* ``PIPER_JUDGE_CAN`` - set to ``0`` to skip piper_sdk's CAN bitrate self-check,
  which slcan (serial CAN) interfaces cannot pass.
* ``PIPER_JOINT_OFFSETS_DEG`` - six comma-separated degrees added to the encoder
  readings, for an arm whose joint zeros disagree with the URDF (e.g. ``0,0,0,0,4.42,0``).

Run on hardware with ``dimos --can-port can0 run piper-scene``; without
``--can-port`` the arm is a mock adapter and only the camera is real.
"""

from __future__ import annotations

import os
from typing import Any

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.hardware.sensors.camera.webcam import Webcam
from dimos.robot.manipulators.common.blueprints import (
    coordinator,
    planner,
    trajectory_task,
)
from dimos.robot.manipulators.common.sim import mujoco_if_sim
from dimos.robot.manipulators.piper.config import (
    PIPER_SIM_PATH,
    make_piper_model_config,
    piper_hardware,
    piper_joint_offsets_from_env,
)
from dimos.visualization.vis_module import vis_module

# The RealSense exposes several V4L2 nodes; only one carries RGB (YUYV). The
# by-id ``video-index0`` link points at the depth node, so a videoN / by-path
# node is used here.
_CAMERA_DEVICE = os.getenv("PIPER_SCENE_CAMERA", "/dev/video6")
_CAMERA_WIDTH = int(os.getenv("PIPER_SCENE_CAMERA_WIDTH", "1280"))
_CAMERA_HEIGHT = int(os.getenv("PIPER_SCENE_CAMERA_HEIGHT", "720"))
_CAMERA_FPS = float(os.getenv("PIPER_SCENE_CAMERA_FPS", "15"))
_JUDGE_CAN = os.getenv("PIPER_JUDGE_CAN", "1").strip().lower() not in ("0", "false", "no", "off")


def _scene_camera() -> Webcam:
    return Webcam(
        camera_index=_CAMERA_DEVICE,
        width=_CAMERA_WIDTH,
        height=_CAMERA_HEIGHT,
        fps=_CAMERA_FPS,
    )


def _gripper_task() -> TaskConfig:
    return TaskConfig(
        name="arm_gripper",
        type="gripper",
        joint_names=["arm/gripper"],
        priority=20,
    )


def _rerun_blueprint() -> Any:
    """Scene camera beside the 3D view."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="Scene camera"),
            rrb.Spatial3DView(origin="world", name="3D"),
            column_shares=[1, 1],
        ),
    )


_piper_hw = piper_hardware(
    "arm", judge_can=_JUDGE_CAN, joint_offsets=piper_joint_offsets_from_env()
)
_piper_model = make_piper_model_config()

_scene_camera_module = CameraModule.blueprint(hardware=_scene_camera)
_vis = vis_module(global_config.viewer, rerun_config={"blueprint": _rerun_blueprint})

# Joint trajectories + gripper, camera, and the viewer. No motion planner.
piper_scene_coordinator = autoconnect(
    coordinator(
        hardware=[_piper_hw],
        tasks=[trajectory_task(_piper_hw), _gripper_task()],
    ),
    _scene_camera_module,
    _vis,
    *mujoco_if_sim(PIPER_SIM_PATH, len(_piper_hw.joints)),
)

# The same plus the Drake planner, which adds plan_to_joints / plan_to_poses /
# move_linear / execute RPCs and a viser model view.
piper_scene = autoconnect(
    planner(model=_piper_model, visualization={"backend": "viser"}),
    coordinator(
        hardware=[_piper_hw],
        tasks=[trajectory_task(_piper_hw), _gripper_task()],
    ),
    _scene_camera_module,
    _vis,
    *mujoco_if_sim(PIPER_SIM_PATH, len(_piper_hw.joints)),
)
