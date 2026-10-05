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

"""Pick up an open container that is too wide for the jaws, by one of its walls.

The gripper comes straight down over the middle of the long wall nearest the
arm, pinches the wall between the jaws, lifts, and then looks: the pick counts
only when the depth camera sees something risen with the gripper.

Origin: autoresearch task "pick up the yellow bin" on the AgileX Piper,
2026-10-04 (a 29 x 11.5 x 10 cm plastic shelf bin with 2 mm walls). See
``VALIDATION`` below for what has and has not been run on the arm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.skill_result import SkillResult
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.grasp_verification import GraspVerificationConfig, await_gripper_settle
from dimos.manipulation.manipulation_spec import ManipulationSpec, PlanningGroupInfo
from dimos.manipulation.pick_and_place_module import await_arm_settle
from dimos.manipulation.skill_errors import ManipulationSkillError
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.perception.experimental.object_scene_registration_spec import ObjectSceneRegistrationSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Filled in from the held-out trials run through the piper-container blueprint.
VALIDATION = "work in progress: not yet validated through its blueprint"

Points = NDArray[np.floating[Any]]
Failure = SkillResult[ManipulationSkillError]


@dataclass(frozen=True)
class WallPinch:
    """Where to pinch a container's wall, in the planning frame."""

    x: float
    y: float
    # Direction the wall runs in; the jaws open across it.
    yaw: float
    rim_z: float
    length: float
    width: float


@dataclass(frozen=True)
class Held:
    """What put_down_container needs to undo a pick."""

    grasp_z: float
    raised_joints: tuple[float, ...]
    lowered_joints: tuple[float, ...]


def wall_pinch(
    points: Points,
    base_xy: tuple[float, float] = (0.0, 0.0),
    slice_half_width: float = 0.03,
    inset: float = 0.003,
) -> WallPinch | None:
    """The middle of a container's long wall nearest ``base_xy``.

    ``points`` is the container's cloud seen from above. The long wall nearest a
    camera on the arm is seen edge-on, so it shows up as the edge of the
    footprint rather than as tall points; the footprint's edge is what is used.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 50:
        return None
    xy = points[:, :2]
    centre = xy.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((xy - centre).T))
    long_axis = vectors[:, 1]
    if long_axis[1] < 0:
        long_axis = -long_axis
    short_axis = np.array([long_axis[1], -long_axis[0]])
    along = (xy - centre) @ long_axis
    across = (xy - centre) @ short_axis
    along_lo, along_hi = np.percentile(along, [2, 98])
    middle = (along_lo + along_hi) / 2.0
    in_slice = np.abs(along - middle) < slice_half_width
    if in_slice.sum() < 10:
        return None
    across_lo, across_hi = np.percentile(across[in_slice], [2, 98])
    base = np.asarray(base_xy)
    walls = [
        centre + middle * long_axis + (across_lo + inset) * short_axis,
        centre + middle * long_axis + (across_hi - inset) * short_axis,
    ]
    near = min(walls, key=lambda wall: float(np.linalg.norm(wall - base)))
    return WallPinch(
        x=float(near[0]),
        y=float(near[1]),
        yaw=math.atan2(long_axis[1], long_axis[0]),
        rim_z=float(np.percentile(points[:, 2], 99)),
        length=float(along_hi - along_lo),
        width=float(across_hi - across_lo),
    )


def rim_offset(
    points: Points,
    x: float,
    y: float,
    yaw: float,
    above_z: float,
    toward_base: float,
    along_half_width: float = 0.14,
    across_half_width: float = 0.08,
    gap: float = 0.015,
    min_points: int = 20,
) -> float | None:
    """How far across the wall direction the nearest rim is from (x, y).

    ``points`` is a cloud taken from straight above. Points higher than
    ``above_z`` beside (x, y) are rims; they fall into lines across the wall
    direction ``yaw``, and the line furthest toward the base (the sign
    ``toward_base`` along the across axis) is the near wall. Returns its signed
    distance along the across axis, or None when no rim is in the window. The
    window is long because a wrist camera sits beside the gripper and sees the
    rim beside the fingers, not under them.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > above_z)]
    along_axis = np.array([math.cos(yaw), math.sin(yaw)])
    across_axis = np.array([along_axis[1], -along_axis[0]])
    offset = points[:, :2] - np.array([x, y])
    along = offset @ along_axis
    across = offset @ across_axis
    across = np.sort(
        across[(np.abs(along) < along_half_width) & (np.abs(across) < across_half_width)]
    )
    if len(across) < min_points:
        return None
    lines = [
        line
        for line in np.split(across, np.where(np.diff(across) > gap)[0] + 1)
        if len(line) >= min_points
    ]
    if not lines:
        return None
    line = lines[0] if toward_base < 0 else lines[-1]
    return float(np.median(line))


def height_under(points: Points, x: float, y: float, radius: float) -> tuple[float, int] | None:
    """The median height of the points within ``radius`` of (x, y), and how many."""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    near = points[np.hypot(points[:, 0] - x, points[:, 1] - y) < radius]
    if len(near) == 0:
        return None
    return float(np.median(near[:, 2])), len(near)


class ContainerGraspModuleConfig(ModuleConfig):
    planning_frame: str = "world"
    # A joint pose whose camera sees the whole work area from above.
    survey_joints: list[float] = Field(default_factory=list)
    # Seconds to wait at a viewing pose for a frame taken from it.
    view_settle: float = 1.5
    # The tool point's height above the wall before the straight descent.
    hover_z: float = 0.12
    # How far the fingertips reach past the tool point.
    fingertip_depth: float = 0.02
    # How far below the rim the fingertips go.
    pinch_depth: float = 0.03
    # Containers the pinch is known to hold: rim height and footprint, metres.
    rim_z_range: tuple[float, float] = (0.05, 0.12)
    min_length: float = 0.10
    # Straight lift with the wall in the jaws, then joint steps that raise it further.
    lift: float = 0.04
    lift_speed: float = 0.5
    raise_step: list[float] = Field(default_factory=list)
    raise_steps: int = 3
    # The close look from the hover pose: rim points are those within rim_band
    # of the rim height; a wall further than max_shift from where the overview
    # put it is not the wall that was seen.
    rim_band: float = 0.02
    max_shift: float = 0.07
    # Further than this off the wall, slide over it before going down.
    align_tolerance: float = 0.004
    # The table top's height in the planning frame.
    support_z: float = 0.0
    # The hold check: what the camera sees within held_radius under the raised
    # gripper is, in the median, at least held_rise above the table. A held
    # container's floor is; the table, or a container left on it, is not.
    held_radius: float = 0.1
    held_rise: float = 0.05
    held_min_points: int = 100
    held_voxel: float = 0.005
    settle_tolerance: float = 0.002
    settle_timeout: float = 3.0
    grasp_verification: GraspVerificationConfig = Field(default_factory=GraspVerificationConfig)


class ContainerGraspModule(Module):
    """Pick up and put down open containers by a wall."""

    config: ContainerGraspModuleConfig
    _scene: ObjectSceneRegistrationSpec
    _manipulation: ManipulationSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._held: Held | None = None

    @skill(uses=[CAP_MOVEMENT])
    def pick_up_container(self, prompt: str) -> SkillResult[ManipulationSkillError]:
        """Pick up an open bin, box or tray by pinching one of its long walls.

        Use for a container wider than the jaws that stands upright on the
        table with its long wall within straight-down reach. Looks from the
        overview pose first, and after lifting checks with the depth camera
        that the container rose; if it did not, puts it back and fails.

        Args:
            prompt: What to look for, e.g. "yellow bin".
        """
        cfg = self.config
        group = self._group()
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "Gripper-capable planning group is missing")
        if self._held is not None:
            return SkillResult.fail("INVALID_STATE", "Already holding a container; put it down")
        if failure := self._set_gripper(group, cfg.grasp_verification.open_position):
            return failure
        if failure := self._move_joints(group, cfg.survey_joints):
            return failure
        time.sleep(cfg.view_settle)
        cloud = self._container_cloud(prompt)
        if cloud is None:
            return SkillResult.fail("OBJECT_NOT_DETECTED", f"No '{prompt}' in view")
        pinch = wall_pinch(cloud)
        if pinch is None:
            return SkillResult.fail("PERCEPTION_FAILED", "Too few points on the container")
        seen = (
            f"wall at ({pinch.x:.3f}, {pinch.y:.3f}), rim {pinch.rim_z:.3f}, "
            f"{pinch.length:.2f} x {pinch.width:.2f} m"
        )
        if not cfg.rim_z_range[0] <= pinch.rim_z <= cfg.rim_z_range[1]:
            return SkillResult.fail(
                "INVALID_INPUT", f"Rim height outside what a pinch holds: {seen}"
            )
        if pinch.length < cfg.min_length:
            return SkillResult.fail("INVALID_INPUT", f"Too small for a wall pinch: {seen}")

        if failure := self._hover(group, pinch):
            return SkillResult.fail("PLANNING_FAILED", f"No approach above the {seen}: {failure}")
        shift = self._align(group, prompt, pinch)
        if isinstance(shift, SkillResult):
            return shift
        seen += f", rim found {shift * 1000:+.0f} mm from there"
        grasp_z = pinch.rim_z - cfg.pinch_depth + cfg.fingertip_depth
        if failure := self._linear_to_z(group, grasp_z):
            self._linear_to_z(group, cfg.hover_z)
            return failure
        readback = self._close(group)
        if isinstance(readback, SkillResult):
            self._release(group, grasp_z)
            return readback
        lowered = self._joints(group)
        failure = self._linear_to_z(group, grasp_z + cfg.lift, cfg.lift_speed)
        if failure is None:
            failure = self._raise(group)
        raised = self._joints(group)
        if failure is not None or lowered is None or raised is None:
            self._release(group, grasp_z)
            return failure or SkillResult.fail("INVALID_STATE", "Arm state is unavailable")
        held = Held(grasp_z, tuple(raised), tuple(lowered))

        under = self._under_gripper(group, prompt)
        rise = None if under is None else under[0] - cfg.support_z
        if under is None or rise is None or under[1] < cfg.held_min_points or rise < cfg.held_rise:
            self._lower(group, held)
            saw = (
                "nothing" if rise is None or under is None else f"{under[1]} points {rise:.3f} m up"
            )
            return SkillResult.fail(
                "GRASP_VERIFICATION_FAILED",
                f"Pinched the {seen} (jaws read {readback:.4f}) and lifted, but under the "
                f"gripper the camera saw {saw} (held needs {cfg.held_min_points} points "
                f"{cfg.held_rise:.2f} m up); let go",
            )
        self._held = held
        return SkillResult.ok(
            f"Holding the {prompt} by its wall, clear of the table: {seen}",
            wall={"x": pinch.x, "y": pinch.y, "rim_z": pinch.rim_z},
            rise=rise,
            gripper_readback=readback,
        )

    @skill(uses=[CAP_MOVEMENT])
    def put_down_container(self) -> SkillResult[ManipulationSkillError]:
        """Put the container picked up with pick_up_container back where it was taken from."""
        group = self._group()
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "Gripper-capable planning group is missing")
        if self._held is None:
            return SkillResult.fail("INVALID_STATE", "Not holding a container")
        held, self._held = self._held, None
        if failure := self._lower(group, held):
            return failure
        return SkillResult.ok("Put the container down and let go")

    def _container_cloud(self, prompt: str) -> Points | None:
        """One detection pass: the points of the largest thing the prompt found."""
        best: Points | None = None
        for detection in self._scene.scan_scene(text=[prompt]).detections:
            if not detection.id:
                continue
            cloud = self._scene.get_object_pointcloud_by_object_id(str(detection.id))
            if cloud is None or cloud.frame_id != self.config.planning_frame:
                continue
            points = cloud.points_f32()
            if best is None or len(points) > len(best):
                best = points
        return best

    def _fresh_cloud(self, prompt: str, voxel: float) -> Points | None:
        """The whole depth frame from where the arm is now, in the planning frame."""
        time.sleep(self.config.view_settle)
        # A detection pass is what takes a fresh depth frame; what it finds is not used.
        self._scene.scan_scene(text=[prompt])
        cloud = self._scene.get_full_scene_pointcloud(None, 1.0, voxel)
        if cloud is None or cloud.frame_id != self.config.planning_frame:
            return None
        return cloud.points_f32()

    def _under_gripper(self, group: PlanningGroupInfo, prompt: str) -> tuple[float, int] | None:
        """Look from where the arm is: how high is what lies under the gripper."""
        cfg = self.config
        tip = self._tip(group)
        cloud = self._fresh_cloud(prompt, cfg.held_voxel)
        if tip is None or cloud is None:
            return None
        return height_under(cloud, tip.position.x, tip.position.y, cfg.held_radius)

    def _align(self, group: PlanningGroupInfo, prompt: str, pinch: WallPinch) -> float | Failure:
        """From the hover pose, find the near rim under the camera and slide over it.

        Returns how far the gripper was moved across the wall.

        The overview sees the near wall edge-on and puts it up to 4 cm too far
        out; from straight above its rim is a line of high points.
        """
        cfg = self.config
        across = (math.sin(pinch.yaw), -math.cos(pinch.yaw))
        toward_base = -(pinch.x * across[0] + pinch.y * across[1])
        moved = 0.0
        for _ in range(2):
            tip = self._tip(group)
            cloud = self._fresh_cloud(prompt, 0.003)
            if tip is None or cloud is None:
                return SkillResult.fail("PERCEPTION_FAILED", "No depth frame from above the wall")
            offset = rim_offset(
                cloud,
                tip.position.x,
                tip.position.y,
                pinch.yaw,
                pinch.rim_z - cfg.rim_band,
                toward_base,
                across_half_width=cfg.max_shift,
            )
            if offset is None:
                self._linear_to_z(group, cfg.hover_z)
                return SkillResult.fail(
                    "PERCEPTION_FAILED",
                    "From above, no rim within reach of where the wall was seen",
                )
            if abs(offset) <= cfg.align_tolerance:
                break
            if failure := self._linear(group, offset * across[0], offset * across[1]):
                return failure
            moved += offset
        return moved

    def _hover(self, group: PlanningGroupInfo, pinch: WallPinch) -> Failure | None:
        """Go above the wall, fingers down, jaws across it; either way round."""
        failure: Failure | None = None
        for yaw in (pinch.yaw, pinch.yaw - math.pi):
            pose = PoseStamped(
                frame_id=self.config.planning_frame,
                position=Vector3(pinch.x, pinch.y, self.config.hover_z),
                orientation=Quaternion.from_euler(Vector3(math.pi, 0.0, yaw)),
            )
            failure = self._move_pose(group, pose)
            if failure is None:
                return None
        return failure

    def _raise(self, group: PlanningGroupInfo) -> Failure | None:
        for _ in range(self.config.raise_steps):
            joints = self._joints(group)
            if joints is None:
                return SkillResult.fail("INVALID_STATE", "Arm state is unavailable")
            target = [q + step for q, step in zip(joints, self.config.raise_step, strict=True)]
            if failure := self._move_joints(group, target):
                return failure
        return None

    def _lower(self, group: PlanningGroupInfo, held: Held) -> Failure | None:
        """Back down the way it came up, then let go."""
        if failure := self._move_joints(group, held.lowered_joints):
            return failure
        return self._release(group, held.grasp_z)

    def _release(self, group: PlanningGroupInfo, grasp_z: float) -> Failure | None:
        cfg = self.config
        self._linear_to_z(group, grasp_z, cfg.lift_speed)
        if failure := self._set_gripper(group, cfg.grasp_verification.open_position):
            return failure
        return self._linear_to_z(group, cfg.hover_z)

    def _group(self) -> PlanningGroupInfo | None:
        groups = [
            group
            for group in self._manipulation.list_planning_groups()
            if group.has_gripper and group.tip_frame is not None
        ]
        return groups[0] if len(groups) == 1 else None

    def _joints(self, group: PlanningGroupInfo) -> Sequence[float] | None:
        state = self._manipulation.get_state().groups.get(group.id)
        if state is None or state.joints is None:
            return None
        return state.joints.position

    def _settle(self, group: PlanningGroupInfo) -> None:
        await_arm_settle(
            lambda: self._joints(group), self.config.settle_tolerance, self.config.settle_timeout
        )

    def _tip(self, group: PlanningGroupInfo) -> PoseStamped | None:
        """Where the tool point is once the arm has stopped."""
        self._settle(group)
        state = self._manipulation.get_state().groups.get(group.id)
        return state.end_effector_pose if state is not None else None

    def _execute(self, group: PlanningGroupInfo) -> Failure | None:
        execution = self._manipulation.execute(blocking=True)
        if not execution.succeeded:
            self._manipulation.reset()
            return SkillResult.fail("EXECUTION_FAILED", execution.message)
        self._settle(group)
        return None

    def _move_joints(self, group: PlanningGroupInfo, joints: Sequence[float]) -> Failure | None:
        target = JointState(name=list(group.joint_names), position=[float(q) for q in joints])
        try:
            plan = self._manipulation.plan_to_joints({group.id: target})
        except RuntimeError as error:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", str(error))
        if not plan.succeeded:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", plan.message)
        return self._execute(group)

    def _move_pose(self, group: PlanningGroupInfo, pose: PoseStamped) -> Failure | None:
        try:
            plan = self._manipulation.plan_to_poses({group.id: pose})
        except RuntimeError as error:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", str(error))
        if not plan.succeeded:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", plan.message)
        return self._execute(group)

    def _linear_to_z(
        self, group: PlanningGroupInfo, z: float, speed: float | None = None
    ) -> Failure | None:
        tip = self._tip(group)
        if tip is None:
            return SkillResult.fail("INVALID_STATE", "End-effector pose is unavailable")
        return self._linear(group, dz=z - tip.position.z, speed=speed)

    def _linear(
        self,
        group: PlanningGroupInfo,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        speed: float | None = None,
    ) -> Failure | None:
        """A straight move of the tool point, not collision checked."""
        if math.hypot(dx, dy, dz) < 1e-4:
            return None
        result = self._manipulation.move_linear(dx, dy, dz, group.id, False, speed)
        if not result.plan.succeeded:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", result.plan.message)
        if result.execution is None or not result.execution.succeeded:
            message = "" if result.execution is None else result.execution.message
            self._manipulation.reset()
            return SkillResult.fail("EXECUTION_FAILED", message)
        self._settle(group)
        return None

    def _gripper(self, group: PlanningGroupInfo) -> float | None:
        state = self._manipulation.get_state().groups.get(group.id)
        return state.gripper_position if state is not None else None

    def _set_gripper(self, group: PlanningGroupInfo, opening: float) -> Failure | None:
        verification = self.config.grasp_verification
        result = self._manipulation.set_gripper_position(opening, group.id)
        if not result.succeeded:
            return SkillResult.fail("GRIPPER_FAILED", result.message or "Gripper command rejected")
        await_gripper_settle(
            lambda: self._gripper(group),
            opening,
            verification,
            arrival_tolerance=verification.open_tolerance,
        )
        return None

    def _close(self, group: PlanningGroupInfo) -> float | Failure:
        """Close on the wall. The readback is reported, not judged: a thin wall
        reads the same as empty jaws, so the hold is checked by looking."""
        verification = self.config.grasp_verification
        result = self._manipulation.set_gripper_position(verification.closed_position, group.id)
        if not result.succeeded:
            return SkillResult.fail("GRIPPER_FAILED", result.message or "Gripper command rejected")
        settle = await_gripper_settle(
            lambda: self._gripper(group), verification.closed_position, verification
        )
        return float(settle.position) if settle.position is not None else 0.0
