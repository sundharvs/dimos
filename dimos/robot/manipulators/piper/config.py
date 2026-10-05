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

"""Piper planning model configuration helpers."""

from __future__ import annotations

import math
import os
from pathlib import Path

from dimos.control.components import HardwareComponent, HardwareType
from dimos.core.global_config import global_config
from dimos.hardware.spec import JointLimits
from dimos.manipulation.planning.groups.models import PlanningGroupDefinition
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.robot.assets.model import RobotModel
from dimos.robot.assets.source import RobotDescriptionSource
from dimos.robot.manipulators._modeling import (
    joint_names,
)
from dimos.utils.data import LfsPath

# Every Piper link with collision geometry: the wrist camera's self filter needs a
# capture-time transform for each and drops the whole cloud when one is missing.
PIPER_COLLISION_LINKS = [
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "flange_link",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
]

PIPER_GRIPPER_COLLISION_EXCLUSIONS: list[tuple[str, str]] = [
    ("gripper_base", "gripper_link1"),
    ("gripper_base", "gripper_link2"),
    ("gripper_link1", "gripper_link2"),
    ("link6", "gripper_base"),
]

# The fingers run along gripper_base's +Z; their meshes span 6.2-13.8 cm of it.
PIPER_FINGERTIP_DEPTH = 0.138
# The tool frame: between the jaws, 2 cm in from the fingertips.
PIPER_TCP_FRAME = "gripper_tcp"
PIPER_TCP_DEPTH = 0.118

PIPER_DESCRIPTION_REPO = "https://github.com/agilexrobotics/agx_arm_urdf"
PIPER_DESCRIPTION_REF = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"

_PIPER_REPO = RobotDescriptionSource(
    url=PIPER_DESCRIPTION_REPO,
    ref=PIPER_DESCRIPTION_REF,
)

PIPER_MODEL_PATH = _PIPER_REPO / "piper" / "urdf" / "piper_with_gripper_description.xacro"

PIPER_PACKAGE_PATHS: dict[str, Path] = {
    # Upstream URIs are package://agx_arm_description/agx_arm_urdf/...
    # so the package root is the parent of the preserved agx_arm_urdf checkout.
    "agx_arm_description": _PIPER_REPO.parent,
}

PIPER_FK_MODEL = _PIPER_REPO / "piper" / "urdf" / "piper_description.urdf"

PIPER_SIM_PATH = LfsPath("piper/scene.xml")
PIPER_HOME_JOINTS = [
    0.793,
    1.568186214614724,
    -1.0290351975897356,
    0.0008456548489068756,
    0.9771515619106422,
    -0.13286819850920156,
]


def _adapter_kwargs(home_joints: list[float] | None = None) -> dict[str, object]:
    if home_joints is None:
        return {}
    return {"initial_positions": home_joints}


def make_piper_hardware(
    hw_id: str = "arm",
    *,
    adapter_type: str = "mock",
    address: str | None = None,
    gripper: bool = True,
    auto_enable: bool = True,
    adapter_kwargs: dict[str, object] | None = None,
    home_joints: list[float] | None = None,
    canonical_joint_names: list[str] | None = None,
) -> HardwareComponent:
    kwargs = _adapter_kwargs(home_joints)
    if adapter_kwargs:
        kwargs.update(adapter_kwargs)
    gripper_joints = [f"{hw_id}/gripper"] if gripper else []
    initial_positions = kwargs.get("initial_positions")
    if gripper and isinstance(initial_positions, list):
        kwargs["initial_positions"] = [*initial_positions, 0.0]
    limits: JointLimits | None = None
    if adapter_type == "mock":
        limits = JointLimits(
            position_lower=[*([-math.pi] * 6), *([0.0] * len(gripper_joints))],
            position_upper=[*([math.pi] * 6), *([0.08] * len(gripper_joints))],
            velocity_max=[*([math.pi] * 6), *([0.0] * len(gripper_joints))],
        )
    return HardwareComponent(
        hardware_id=hw_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=[*(canonical_joint_names or joint_names(6)), *gripper_joints],
        adapter_type=adapter_type,
        address=address,
        auto_enable=auto_enable,
        limits=limits,
        adapter_kwargs=kwargs,
    )


def piper_hardware(
    hw_id: str = "arm",
    *,
    gripper: bool = True,
    mock_without_address: bool = True,
    home_joints: list[float] | None = None,
    canonical_joint_names: list[str] | None = None,
    judge_can: bool = True,
    joint_offsets: list[float] | None = None,
) -> HardwareComponent:
    if global_config.simulation:
        return make_piper_hardware(
            hw_id,
            adapter_type="sim_mujoco",
            address=str(PIPER_SIM_PATH),
            gripper=gripper,
            home_joints=home_joints,
            canonical_joint_names=canonical_joint_names,
        )
    address = global_config.can_port or "can0"
    if mock_without_address and not global_config.can_port:
        return make_piper_hardware(
            hw_id,
            gripper=gripper,
            home_joints=home_joints,
            canonical_joint_names=canonical_joint_names,
        )
    return make_piper_hardware(
        hw_id,
        adapter_type="piper",
        address=address,
        gripper=gripper,
        home_joints=home_joints,
        canonical_joint_names=canonical_joint_names,
        adapter_kwargs={"judge_can": judge_can, "joint_offsets": joint_offsets},
    )


def piper_joint_offsets_from_env() -> list[float] | None:
    """Per-joint encoder offsets from ``PIPER_JOINT_OFFSETS_DEG``, in radians.

    Six comma-separated degrees, e.g. ``0,0,0,0,4.42,0``: what to add to each
    encoder reading so the URDF sees the arm's true angles. Unset means none.
    """
    raw = os.getenv("PIPER_JOINT_OFFSETS_DEG", "").strip()
    if not raw:
        return None
    degrees = [float(value) for value in raw.split(",")]
    if len(degrees) != 6:
        raise ValueError(f"PIPER_JOINT_OFFSETS_DEG needs 6 values (got {len(degrees)})")
    return [math.radians(value) for value in degrees]


def piper_wrist_camera_serial_from_env() -> str | None:
    """The wrist RealSense's serial number from ``PIPER_WRIST_CAMERA_SERIAL``.

    Without one the driver opens whichever RealSense it finds first, which on a
    rig with a second (scene) RealSense may not be the wrist camera. This is
    librealsense's serial, not the USB serial in the ``/dev/v4l/by-id`` name.
    Unset means the first found.
    """
    return os.getenv("PIPER_WRIST_CAMERA_SERIAL", "").strip() or None


def make_piper_model_config(
    *,
    home_joints: list[float] | None = None,
    tcp: bool = False,
) -> RobotModelConfig:
    """The Piper planning model.

    With ``tcp`` the planning tip is the tool frame between the jaws rather
    than gripper_base, so a pose target names where the grasp happens, as it
    does on an arm whose description ships a TCP link.
    """
    dof = 6
    model_joint_names = joint_names(dof)
    model_home_joints = list(home_joints) if home_joints is not None else list(PIPER_HOME_JOINTS)
    model = RobotModel.from_file(
        PIPER_MODEL_PATH, package_paths=PIPER_PACKAGE_PATHS
    ).with_default_joint_acceleration_limit(2.0)
    if tcp:
        model = model.with_fixed_frame(
            PIPER_TCP_FRAME, "gripper_base", xyz=(0.0, 0.0, PIPER_TCP_DEPTH)
        )
    return RobotModelConfig(
        model=model,
        joint_names=model_joint_names,
        base_link="base_link",
        planning_groups=[
            PlanningGroupDefinition(
                name="manipulator",
                joint_names=tuple(model_joint_names),
                base_link="base_link",
                tip_link=PIPER_TCP_FRAME if tcp else "gripper_base",
            )
        ],
        auto_convert_meshes=True,
        collision_exclusion_pairs=PIPER_GRIPPER_COLLISION_EXCLUSIONS,
        gripper_hardware_id="arm",
        home_joints=model_home_joints,
    )
