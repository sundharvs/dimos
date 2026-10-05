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

"""Host contract for isolated LeRobot policy rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict

from pydantic import Field, ValidationInfo, field_validator

from dimos.constants import STATE_DIR
from dimos.control.tasks.trajectory_task.trajectory_task import (
    TrajectoryCancellationResult,
    TrajectoryExecutionResult,
)
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.experimental.isolated_python.module import (
    IsolatedPythonModule,
    IsolatedPythonModuleConfig,
)
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.spec.utils import Spec
from dimos.teleop.webxr.controller_types import BUTTON_ALIASES, Buttons


class PolicyControlSpec(Spec, Protocol):
    """Coordinator operations used by policy rollout."""

    def execute_trajectory(
        self,
        trajectory: JointTrajectory,
    ) -> TrajectoryExecutionResult: ...

    def cancel_trajectory(self) -> TrajectoryCancellationResult: ...

    def list_tasks(self) -> list[str]: ...


class RolloutStatus(TypedDict):
    """Operator-facing state of the configured policy rollout."""

    active: bool
    policy_path: str
    task: str
    device: str | None
    policy_ready: bool
    observations_ready: bool
    chunks_accepted: int
    last_error: str | None
    rollout_log: str | None


class RolloutControlSpec(Spec, Protocol):
    """The existing policy's operator controls, discoverable by attached clients."""

    def preflight_rollout(self) -> RolloutStatus: ...
    def start_rollout(self) -> RolloutStatus: ...
    def stop_rollout(self) -> RolloutStatus: ...
    def rollout_status(self) -> RolloutStatus: ...


class LeRobotPolicyModuleConfig(IsolatedPythonModuleConfig):
    """Configuration for one checkpoint shared with the isolated runtime."""

    policy_path: str = Field(min_length=1)
    task: str = Field(min_length=1)
    device: str | None = None
    joint_names: list[str] = Field(min_length=1)
    fps: float = Field(default=30.0, gt=0)
    robot_type: str = ""
    image_width: int = Field(default=640, gt=0)
    image_height: int = Field(default=480, gt=0)
    max_observation_age_s: float = Field(default=0.5, gt=0)
    rollout_button: str = "A"
    # Name of the checkpoint's single camera feature (LeRobot dataset key).
    image_feature: str = Field(default="observation.images.wrist", min_length=1)
    # A joint in ``joint_names`` whose action is a normalized opening (0 closed ..
    # 1 open) rather than a native target: it is kept out of the trajectory and
    # published on ``gripper_command`` for the coordinator's gripper task.
    gripper_joint: str | None = None
    # Executed steps between inferences (1 = predict every step). Each
    # submission carries the checkpoint's ``n_action_steps`` targets and lands
    # while the previous one is still running, so the coordinator continues from
    # its commanded position instead of stopping at every chunk boundary. Must
    # not exceed ``n_action_steps``; None predicts once per ``n_action_steps``.
    replan_steps: int | None = Field(default=1, ge=1)
    # Temporal ensembling (ACT, Algorithm 2): every step's target is the
    # exp(-coeff * i)-weighted mean of all chunks that predicted it, i = 0 for
    # the oldest. 0 weighs them uniformly, None executes the newest chunk only.
    temporal_ensemble_coeff: float | None = 0.01
    # Keep only the newest N overlapping chunks in the ensemble, so a target is
    # averaged over at most N predictions instead of every chunk that reaches
    # it (up to chunk_size). None keeps them all.
    ensemble_window: int | None = Field(default=None, ge=1)
    # Free-text tag written to every rollout log header, e.g. the execution
    # mode under study, so logs can be grouped afterwards.
    label: str = ""
    # Directory for per-rollout JSONL logs (every predicted action chunk plus the
    # live joint states), plotted by ``tool_plot_rollout.py``. None disables.
    rollout_log_dir: str | None = str(STATE_DIR / "policy_rollouts")

    @field_validator("policy_path")
    @classmethod
    def policy_path_must_not_be_blank(cls, policy_path: str) -> str:
        if not policy_path.strip():
            raise ValueError("policy_path must not be blank")
        path = Path(policy_path).expanduser()
        return str(path.resolve()) if path.exists() else policy_path

    @field_validator("joint_names")
    @classmethod
    def joint_names_must_be_unique(cls, joint_names: list[str]) -> list[str]:
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must not contain duplicates")
        return joint_names

    @field_validator("gripper_joint")
    @classmethod
    def gripper_joint_must_be_configured(
        cls, gripper_joint: str | None, info: ValidationInfo
    ) -> str | None:
        joint_names = info.data.get("joint_names") or []
        if gripper_joint is not None and gripper_joint not in joint_names:
            raise ValueError(f"gripper_joint {gripper_joint!r} is not in joint_names")
        return gripper_joint

    @field_validator("rollout_button")
    @classmethod
    def rollout_button_must_be_digital(cls, name: str) -> str:
        if BUTTON_ALIASES.get(name, name) not in Buttons.BITS:
            raise ValueError(f"unknown Quest button {name!r}")
        return name


class LeRobotPolicyModule(IsolatedPythonModule):
    """Convert live image and joint-state observations into joint targets."""

    project_dir = "native/python/lerobot"
    implementation = "dimos_lerobot.runtime:LeRobotPolicyRuntime"
    config: LeRobotPolicyModuleConfig

    color_image: In[Image]
    coordinator_joint_state: In[JointState]
    button_pressed: In[Buttons]
    teleop_buttons: In[Buttons]
    gripper_command: Out[Float32]

    _control: PolicyControlSpec

    @rpc
    def preflight_rollout(self) -> RolloutStatus:
        """Load and validate the policy and live inputs without moving the robot."""
        raise NotImplementedError

    @rpc
    def start_rollout(self) -> RolloutStatus:
        """Start the configured policy until explicitly stopped or it fails."""
        raise NotImplementedError

    @rpc
    def stop_rollout(self) -> RolloutStatus:
        """Stop rollout publication and clear the policy action queue."""
        raise NotImplementedError

    @rpc
    def rollout_status(self) -> RolloutStatus:
        """Return the lifecycle and observation state of the configured policy."""
        raise NotImplementedError
