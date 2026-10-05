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

"""Pick up an open container by pinching one wall at its rim.

For containers wider than the gripper (bins, trays, boxes): the object detector
and the centroid grasp of ``PickAndPlaceModule`` have nothing to offer when the
container fills the wrist camera's view and no two faces fit between the jaws.
The rim is found from depth alone, as the highest straight edge above the table,
and pinched top-down with one finger inside the container and one outside.

Origin: autoresearch task "pick up the yellow bin" on the Piper, 2026-10-04.
Validation: see the ``piper-grasp`` blueprint, which carries the rig's numbers
and the held-out result. Untested: containers other than that bin, rims that are
not straight, walls thicker than the jaw opening, a cluttered table (the
highest edge in view is taken to be the rim, whatever it belongs to).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.spatial.transform import Rotation

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.skill_result import SkillResult
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.grasp_verification import GraspVerificationConfig, await_gripper_settle
from dimos.manipulation.manipulation_spec import ManipulationSpec
from dimos.manipulation.planning.spec.models import PlanningGroupID
from dimos.manipulation.skill_errors import ManipulationSkillError
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.experimental.object_scene_registration_spec import ObjectSceneRegistrationSpec

Points = NDArray[np.float64]


class RimGraspModuleConfig(ModuleConfig):
    planning_frame: str = "world"
    # Table top in the planning frame. Rig-specific: measure it from the cloud.
    table_z: float = 0.0
    # How far the fingertips reach past the planned tool frame.
    fingertips_past_tcp: float = Field(default=0.02, ge=0.0)
    # Points this far above the table, and no further than max_height, can be a
    # container; lower ones are table noise.
    min_height: float = Field(default=0.03, gt=0.0)
    max_height: float = Field(default=0.30, gt=0.0)
    # Horizontal range from the planning-frame origin inside which a rim is looked for.
    max_range: float = Field(default=0.60, gt=0.0)
    # Rim candidates are straight lines through the points within rim_band of
    # the highest raised point; the longest one is taken, so that a long wall
    # wins over a slightly taller short one and the container hangs level.
    rim_band: float = Field(default=0.025, gt=0.0)
    # A line's own top edge: its points within this much of its highest.
    edge_band: float = Field(default=0.015, gt=0.0)
    line_tolerance: float = Field(default=0.008, gt=0.0)
    min_rim_points: int = Field(default=25, gt=0)
    min_rim_length: float = Field(default=0.04, gt=0.0)
    # How far below the rim top the fingertips close on the wall. Deep enough
    # to get under a rolled or stepped lip: a pinch on the lip alone lifts the
    # container and then lets it slip.
    grasp_depth: float = Field(default=0.045, gt=0.0)
    # The tool leans away from the robot base by this much, so that a rim near
    # the edge of the straight-down workspace stays reachable through the lift.
    lean: float = Field(default=math.radians(15.0), ge=0.0)
    pregrasp_offset: float = Field(default=0.06, gt=0.0)
    lift_height: float = Field(default=0.10, gt=0.0)
    # Seconds to let the camera settle after a motion before measuring.
    settle_time: float = Field(default=0.7, ge=0.0)
    # The scene module rebuilds its depth snapshot on a scan; the prompt only
    # has to be something for the detector to look for.
    scan_prompt: str = "container"
    cloud_range: float = Field(default=1.0, gt=0.0)
    cloud_voxel: float = Field(default=0.003, gt=0.0)
    # Held check after the lift: at least min_held_points depth points within
    # hold_radius of the grasp point, between hold_below under and hold_above
    # over where the rim now is. Nothing else is in the air there, and the
    # camera does not see the fingers.
    hold_radius: float = Field(default=0.10, gt=0.0)
    hold_below: float = Field(default=0.07, gt=0.0)
    hold_above: float = Field(default=0.03, gt=0.0)
    min_held_points: int = Field(default=20, gt=0)
    # Tries at getting a depth cloud; the first scan after start-up has no
    # camera transform yet.
    cloud_attempts: int = Field(default=3, gt=0)
    gripper: GraspVerificationConfig = Field(default_factory=GraspVerificationConfig)


@dataclass(frozen=True)
class RimEstimate:
    """A straight rim segment in the planning frame."""

    x: float
    y: float
    rim_z: float
    wall_yaw: float
    length: float
    points: int


@dataclass(frozen=True)
class HoldEstimate:
    """What the depth cloud shows under the gripper after a lift."""

    held: bool
    lifted_points: int


def find_rim(points: Points, config: RimGraspModuleConfig) -> RimEstimate | None:
    """Fit the highest straight edge above the table and return its midpoint."""
    if len(points) == 0:
        return None
    height = points[:, 2] - config.table_z
    near = np.hypot(points[:, 0], points[:, 1]) < config.max_range
    raised = points[near & (height > config.min_height) & (height < config.max_height)]
    if len(raised) < config.min_rim_points:
        return None
    top = float(np.percentile(raised[:, 2], 98))
    band = raised[raised[:, 2] > top - config.rim_band]

    best: RimEstimate | None = None
    for _ in range(3):
        if len(band) < config.min_rim_points:
            break
        inliers = _longest_line(band[:, :2], config.line_tolerance)
        if inliers is None or int(inliers.sum()) < config.min_rim_points:
            break
        rim = _rim_from_line(band[inliers], config)
        if rim is not None and (best is None or rim.length > best.length):
            best = rim
        band = band[~inliers]
    return best


def _longest_line(xy: Points, tolerance: float) -> NDArray[np.bool_] | None:
    """Inliers of the best-supported line through ``xy`` (RANSAC, fixed seed)."""
    rng = np.random.default_rng(0)
    best: NDArray[np.bool_] | None = None
    for _ in range(200):
        a, b = xy[rng.choice(len(xy), size=2, replace=False)]
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm < 0.02:
            continue
        normal = np.array([-direction[1], direction[0]]) / norm
        inliers: NDArray[np.bool_] = np.abs((xy - a) @ normal) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    return best


def _rim_from_line(line: Points, config: RimGraspModuleConfig) -> RimEstimate | None:
    edge = line[line[:, 2] > float(np.percentile(line[:, 2], 95)) - config.edge_band]
    centre = edge[:, :2].mean(axis=0)
    _, _, vt = np.linalg.svd(edge[:, :2] - centre)
    direction = vt[0]
    along = (edge[:, :2] - centre) @ direction
    low, high = np.percentile(along, [5, 95])
    if high - low < config.min_rim_length:
        return None
    middle = centre + direction * (low + high) / 2.0
    # A line has no sense of direction; report its yaw in (-pi/2, pi/2].
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
    return RimEstimate(
        x=float(middle[0]),
        y=float(middle[1]),
        # Depth noise scatters points above the real edge; the median of the
        # edge band sits on it where a high percentile overshoots by most of a
        # centimetre.
        rim_z=float(np.median(edge[:, 2])),
        wall_yaw=yaw,
        length=float(high - low),
        points=len(edge),
    )


def check_held(
    points: Points, x: float, y: float, rim_z: float, config: RimGraspModuleConfig
) -> HoldEstimate:
    """Decide from a cloud taken after the lift whether the container came up.

    A held wall hangs in the air around the lifted grasp point; a container
    left behind leaves that space empty.
    """
    if len(points) == 0:
        return HoldEstimate(False, 0)
    lifted_rim = rim_z + config.lift_height
    near = np.hypot(points[:, 0] - x, points[:, 1] - y) < config.hold_radius
    z = points[near, 2]
    lifted = int(
        ((z > lifted_rim - config.hold_below) & (z < lifted_rim + config.hold_above)).sum()
    )
    return HoldEstimate(lifted >= config.min_held_points, lifted)


def rim_grasp_pose(
    x: float, y: float, rim_z: float, jaw_yaw: float, config: RimGraspModuleConfig
) -> PoseStamped:
    """Tool pose that puts the fingertips grasp_depth below the rim at (x, y).

    The jaws close across a wall running along ``jaw_yaw``; the tool axis leans
    away from the planning-frame origin by ``config.lean``.
    """
    bearing = math.atan2(y, x)
    lean = Rotation.from_rotvec(
        -config.lean * np.array([-math.sin(bearing), math.cos(bearing), 0.0])
    )
    rotation = lean * Rotation.from_euler("z", jaw_yaw) * Rotation.from_euler("x", math.pi)
    tool_axis = rotation.apply([0.0, 0.0, 1.0])
    tips = np.array([x, y, rim_z - config.grasp_depth])
    tcp = tips - config.fingertips_past_tcp * tool_axis
    qx, qy, qz, qw = rotation.as_quat()
    return PoseStamped(
        frame_id=config.planning_frame,
        position=Vector3(float(tcp[0]), float(tcp[1]), float(tcp[2])),
        orientation=Quaternion(float(qx), float(qy), float(qz), float(qw)),
    )


class RimGraspModule(Module):
    """Find, pinch, lift and release an open container by one wall's rim."""

    config: RimGraspModuleConfig
    _scene: ObjectSceneRegistrationSpec
    _manipulation: ManipulationSpec

    _held_at: tuple[float, float] | None = None

    @skill
    def find_rim(self) -> SkillResult[ManipulationSkillError]:
        """Measure the rim of an open container under the wrist camera.

        Use from a pose where the camera looks down at the container (go_home).
        Takes the highest straight edge above the table in view; it does not
        know what the edge belongs to. Returns x, y, rim_z and wall_yaw for
        grasp_rim.
        """
        rim = self._measure_rim()
        if isinstance(rim, SkillResult):
            return rim
        return SkillResult.ok(
            f"Rim at ({rim.x:.3f}, {rim.y:.3f}), top z {rim.rim_z:.3f}, "
            f"running at yaw {rim.wall_yaw:.2f} rad for {rim.length:.2f} m",
            x=rim.x,
            y=rim.y,
            rim_z=rim.rim_z,
            wall_yaw=rim.wall_yaw,
            length=rim.length,
            points=rim.points,
        )

    @skill(uses=[CAP_MOVEMENT])
    def grasp_rim(
        self,
        x: float,
        y: float,
        rim_z: float,
        wall_yaw: float,
        planning_group: PlanningGroupID | None = None,
    ) -> SkillResult[ManipulationSkillError]:
        """Pinch a container wall at its rim, lift it, and check that it came up.

        Use with the values from find_rim, for a wall thinner than the jaw
        opening with free space on both sides. Fails with
        GRASP_VERIFICATION_FAILED, gripper still closed and raised, when depth
        does not show the container off the table.

        Args:
            x: Rim point in the planning frame, metres.
            y: Rim point in the planning frame, metres.
            rim_z: Height of the rim top in the planning frame, metres.
            wall_yaw: Direction the rim runs in, radians.
            planning_group: Gripper-capable pose group; omit when there is only one.
        """
        group = self._resolve_group(planning_group)
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "Gripper pose group is missing or ambiguous")
        if self._holding(group):
            return SkillResult.fail("INVALID_STATE", "Release the held container first")
        if failure := self._gripper(self.config.gripper.open_position, group):
            return failure

        # The jaws are symmetric, so either half turn is the same grasp; the
        # wrist roll range usually admits only one.
        failure = None
        for jaw_yaw in (wall_yaw, wall_yaw + math.pi, wall_yaw - math.pi):
            grasp = rim_grasp_pose(x, y, rim_z, jaw_yaw, self.config)
            pregrasp = PoseStamped(
                frame_id=grasp.frame_id,
                position=grasp.position + Vector3(0.0, 0.0, self.config.pregrasp_offset),
                orientation=grasp.orientation,
            )
            plan = self._manipulation.plan_to_poses({group: pregrasp})
            if plan.succeeded:
                break
            failure = SkillResult.fail("PLANNING_FAILED", f"pre-grasp: {plan.message}")
        else:
            return failure or SkillResult.fail("PLANNING_FAILED", "pre-grasp")
        execution = self._manipulation.execute(blocking=True)
        if not execution.succeeded:
            return SkillResult.fail("EXECUTION_FAILED", f"pre-grasp: {execution.message}")

        if failure := self._linear(-self.config.pregrasp_offset, group, "descent"):
            return failure
        if failure := self._gripper(self.config.gripper.closed_position, group):
            return failure
        if failure := self._linear(self.config.lift_height, group, "lift"):
            return failure

        cloud = self._cloud()
        if isinstance(cloud, SkillResult):
            return SkillResult.fail(
                "GRASP_VERIFICATION_FAILED", f"No depth after the lift: {cloud.message}"
            )
        hold = check_held(cloud, x, y, rim_z, self.config)
        if not hold.held:
            return SkillResult.fail(
                "GRASP_VERIFICATION_FAILED",
                f"Container not seen at the gripper after the lift: {hold.lifted_points} depth "
                f"points around the lifted rim, {self.config.min_held_points} needed",
            )
        self._held_at = (x, y)
        return SkillResult.ok(
            f"Container held {self.config.lift_height:.2f} m up by its rim",
            lifted_points=hold.lifted_points,
        )

    @skill(uses=[CAP_MOVEMENT])
    def pick_up_by_rim(
        self, planning_group: PlanningGroupID | None = None
    ) -> SkillResult[ManipulationSkillError]:
        """Pick up the open container under the wrist camera by one wall's rim.

        Use for bins, trays and boxes too wide for the gripper, from a pose
        where the camera looks down at one of the walls (go_home). Succeeds only
        when depth shows the container off the table after the lift.

        Args:
            planning_group: Gripper-capable pose group; omit when there is only one.
        """
        rim = self._measure_rim()
        if isinstance(rim, SkillResult):
            return rim
        return self.grasp_rim(rim.x, rim.y, rim.rim_z, rim.wall_yaw, planning_group)

    @skill(uses=[CAP_MOVEMENT])
    def release_rim(
        self, planning_group: PlanningGroupID | None = None
    ) -> SkillResult[ManipulationSkillError]:
        """Lower the container held by grasp_rim back to the table and let go.

        Sets it down under the gripper's current position, opens, and retracts
        upward. Move the arm first to put the container somewhere else.

        Args:
            planning_group: Gripper-capable pose group; omit when there is only one.
        """
        group = self._resolve_group(planning_group)
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "Gripper pose group is missing or ambiguous")
        if not self._holding(group):
            return SkillResult.fail("INVALID_STATE", "No container is held")
        if failure := self._linear(-self.config.lift_height, group, "lowering"):
            return failure
        if failure := self._gripper(self.config.gripper.open_position, group):
            return failure
        self._held_at = None
        if failure := self._linear(self.config.pregrasp_offset, group, "retract"):
            return failure
        return SkillResult.ok("Container released on the table")

    def _holding(self, group: PlanningGroupID) -> bool:
        """Whether a grasp_rim hold is still on: jaws opened by anyone else end it."""
        if self._held_at is None:
            return False
        position = self._manipulation.get_state().groups[group].gripper_position
        gripper = self.config.gripper
        if position is not None and position > gripper.open_position - gripper.open_tolerance:
            self._held_at = None
        return self._held_at is not None

    def _measure_rim(self) -> RimEstimate | SkillResult[ManipulationSkillError]:
        cloud = self._cloud()
        if isinstance(cloud, SkillResult):
            return cloud
        rim = find_rim(cloud, self.config)
        if rim is None:
            return SkillResult.fail(
                "OBJECT_NOT_DETECTED",
                f"No straight edge more than {self.config.min_height:.2f} m above the table "
                f"in {len(cloud)} depth points",
            )
        return rim

    def _cloud(self) -> Points | SkillResult[ManipulationSkillError]:
        failure: SkillResult[ManipulationSkillError] = SkillResult.fail(
            "PERCEPTION_FAILED", "No depth cloud from the scene module"
        )
        for _ in range(self.config.cloud_attempts):
            time.sleep(self.config.settle_time)
            try:
                self._scene.scan_scene(text=[self.config.scan_prompt])
            except RuntimeError as exc:
                failure = SkillResult.fail("PERCEPTION_FAILED", str(exc))
                continue
            cloud = self._scene.get_full_scene_pointcloud(
                None, self.config.cloud_range, self.config.cloud_voxel
            )
            if cloud is None:
                continue
            if cloud.frame_id != self.config.planning_frame:
                return SkillResult.fail(
                    "PERCEPTION_FAILED",
                    f"Depth cloud is in {cloud.frame_id!r}, not {self.config.planning_frame!r}",
                )
            return np.asarray(cloud.points_f32(), dtype=np.float64)
        return failure

    def _resolve_group(self, planning_group: PlanningGroupID | None) -> PlanningGroupID | None:
        groups = [
            group
            for group in self._manipulation.list_planning_groups()
            if group.has_gripper and group.tip_frame is not None
        ]
        if planning_group is not None:
            return planning_group if any(group.id == planning_group for group in groups) else None
        return groups[0].id if len(groups) == 1 else None

    def _linear(
        self, dz: float, group: PlanningGroupID, step: str
    ) -> SkillResult[ManipulationSkillError] | None:
        result = self._manipulation.move_linear(0.0, 0.0, dz, group, check_collision=False)
        if not result.plan.succeeded:
            return SkillResult.fail("PLANNING_FAILED", f"{step}: {result.plan.message}")
        if result.execution is None or not result.execution.succeeded:
            message = "" if result.execution is None else result.execution.message
            return SkillResult.fail("EXECUTION_FAILED", f"{step}: {message}")
        return None

    def _gripper(
        self, position: float, group: PlanningGroupID
    ) -> SkillResult[ManipulationSkillError] | None:
        result = self._manipulation.set_gripper_position(position, group)
        if not result.succeeded:
            return SkillResult.fail(
                "GRIPPER_FAILED", result.message or "Gripper command was rejected"
            )
        # A thin wall leaves the jaws reading closed, so the readback only says
        # that they stopped; whether anything is held is decided from depth.
        await_gripper_settle(
            lambda: self._manipulation.get_state().groups[group].gripper_position,
            position,
            self.config.gripper,
            arrival_tolerance=self.config.gripper.open_tolerance,
        )
        return None
