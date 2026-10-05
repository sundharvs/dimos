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

"""Stack loose rings onto an upright rod, in whatever order they are found.

A ring lying flat is wider than the jaws, so it is held by its rim: one finger
goes down through the hole, the other outside, and the jaws close across the
tube. The ring then sticks out beside the gripper, its centre one ring radius
along the jaw axis from the tool point.

The rod is taller than an arm's wrist can point straight down over, so above it
the gripper leans outward, in the vertical plane through the arm's base. The jaw
axis stays horizontal and square to that plane, which makes the lean a rotation
about the line the jaws close along. A ring picked up with the fingers already
leaning that much stays level over the rod; one picked up with upright fingers
tilts by the lean. Leaning fingers hold a ring by less of their width, and it
creeps out of them, so the pick lean is a setting of its own: the ring is
gripped as upright as it needs to be, and arrives over the rod tilted by the
difference.

Every measurement comes from a view taken for it. The overview finds the rings
and the rod; a ring is looked at again from above before it is picked; the top
of the rod is looked at from straight above before a ring is carried to it, and
again afterwards, when the pile around the rod must have grown by a ring.

No planned move is executed unless its path stays between where the arm is and
where it is going. A sampling planner that cannot connect the two directly
returns a detour instead, which can swing the arm across the whole workspace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.skill_result import SkillResult
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.grasp_verification import (
    GraspVerificationConfig,
    await_gripper_settle,
    grasp_failure,
)
from dimos.manipulation.manipulation_spec import (
    ManipulationSpec,
    PlanningGroupInfo,
    PlanResult,
)
from dimos.manipulation.pick_and_place_module import await_arm_settle
from dimos.manipulation.skill_errors import ManipulationSkillError
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.perception.experimental.object_scene_registration_spec import ObjectSceneRegistrationSpec
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

Points = NDArray[np.float32]
Failure = SkillResult[ManipulationSkillError]

# Share of points ignored at each end when measuring an extent, so that a few
# stray points do not move an edge.
_TRIM = 0.01
# Points this close under the highest ones are taken as an object's top surface.
_TOP_BAND = 0.004
# A cloud with fewer points than this is not measured.
_MIN_POINTS = 30
_ROD_OBSTACLE = "ring_stack_rod"


class PlanningWorldSpec(Spec, Protocol):
    """The planner's obstacle list, which the rod is added to."""

    def add_obstacle(
        self,
        name: str,
        pose: Pose,
        shape: str,
        dimensions: list[float] | None = None,
        mesh_path: str | None = None,
    ) -> str: ...

    def remove_obstacle(self, obstacle_id: str) -> bool: ...


@dataclass(frozen=True)
class Ring:
    """A ring lying flat: its centre, the height of its top, and two radii."""

    x: float
    y: float
    top_z: float
    # Radius of the circle the tube runs along, which is where it is gripped.
    grip_radius: float
    outer_radius: float


@dataclass(frozen=True)
class Rod:
    """An upright rod: where its top is."""

    x: float
    y: float
    top_z: float


@dataclass(frozen=True)
class RimHold:
    """Where the tool point goes to hold a ring by its rim."""

    x: float
    y: float
    # Direction of the tool point from the base, which the gripper leans along.
    bearing: float


@dataclass(frozen=True)
class Held:
    """How a ring sits in the jaws, which is all that carrying and placing it needs."""

    # The ring's centre, and the direction its hole points, in the tool frame.
    offset: tuple[float, float, float]
    normal: tuple[float, float, float]
    outer_radius: float
    height: float
    # Whether the wrist is the half turn away, with the fingers swapped.
    half_turn: bool
    # Jaw opening that lets the ring go without pushing it, and the opening
    # the jaws settled at on the tube.
    opening: float
    jaws: float


@dataclass(frozen=True)
class CarryPose:
    """A tool pose that holds a ring's centre over a point."""

    x: float
    y: float
    z: float
    bearing: float
    # How far the ring is from level there, radians.
    tilt: float


def fit_ring(points: Points) -> Ring | None:
    """Measure a ring from the points seen on it, or None when it has no hole."""
    if len(points) < _MIN_POINTS:
        return None
    xy = points[:, :2]
    low, high = np.quantile(xy, [_TRIM, 1.0 - _TRIM], axis=0)
    centre = (low + high) / 2.0
    outer = float(np.mean(high - low)) / 2.0
    top_z = float(np.quantile(points[:, 2], 0.95))
    # The highest points of a torus lie on the circle its tube runs along. A
    # segmentation mask may take in the table seen through the hole, which
    # this height band leaves out.
    crest = xy[points[:, 2] > top_z - _TOP_BAND]
    if len(crest) < _MIN_POINTS // 2:
        return None
    radial = np.hypot(*(crest - centre).T)
    grip = float(np.median(radial))
    # A lid or a disc has its highest points all the way in to the middle.
    if not 0.5 * outer < grip < outer or np.quantile(radial, 0.1) < 0.4 * outer:
        return None
    return Ring(float(centre[0]), float(centre[1]), top_z, grip, outer)


def fit_rod_top(points: Points) -> Rod | None:
    """The middle and height of the top face among points on and around a rod."""
    if len(points) < _MIN_POINTS:
        return None
    top_z = float(np.quantile(points[:, 2], 0.98))
    face = points[points[:, 2] > top_z - 2.0 * _TOP_BAND]
    if len(face) < _MIN_POINTS // 2:
        return None
    # The face's rim is ragged in a depth image, and a median does not follow it.
    x, y, z = np.median(face, axis=0)
    return Rod(float(x), float(y), float(z))


def rim_hold(
    centre: tuple[float, float], grip_radius: float, lead: float, side: int
) -> RimHold | None:
    """Tool point that holds a ring by its rim with the jaws square to the arm.

    The jaw axis is horizontal and square to the vertical plane through the base
    and the tool point, so the tool point lies where a line from the base
    touches the circle the tube runs along. ``side`` is +1 when the ring's
    centre is to the left of the tool point seen from the base, -1 when to the
    right. ``lead`` moves the tool point back toward the base along that line:
    a leaning finger meets the tube above its tip, so its tip goes further out.
    The base is the origin. None when the ring surrounds the base.
    """
    distance = math.hypot(*centre)
    if distance <= grip_radius:
        return None
    bearing = math.atan2(centre[1], centre[0]) - side * math.asin(grip_radius / distance)
    reach = math.sqrt(distance**2 - grip_radius**2) - lead
    return RimHold(reach * math.cos(bearing), reach * math.sin(bearing), bearing)


def leaning_orientation(bearing: float, lean: float, half_turn: bool = False) -> Quaternion:
    """Tool orientation that leans outward along a bearing, jaws square to it.

    The approach axis (tool z) points down and outward by ``lean`` from
    vertical; the jaw axis (tool y) is horizontal. ``half_turn`` gives the same
    grasp with the fingers swapped, for a wrist whose range is centred there.
    """
    outward = np.array([math.cos(bearing), math.sin(bearing), 0.0])
    approach = math.sin(lean) * outward - math.cos(lean) * np.array([0.0, 0.0, 1.0])
    jaw = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
    if half_turn:
        jaw = -jaw
    return Quaternion.from_rotation_matrix(
        np.column_stack([np.cross(jaw, approach), jaw, approach])
    )


def carry_pose(held: Held, x: float, y: float, top_z: float, gap: float, lean: float) -> CarryPose:
    """Tool pose that holds a ring's centre over (x, y), clear of what is under it.

    The lowest point of the ring, which hangs down on one side when it is
    tilted, ends ``gap`` above ``top_z``. The gripper leans along the bearing of
    the tool point, which the ring's offset moves away from the bearing of the
    target, so the bearing is found by repetition. The base is the origin.
    """
    offset = np.asarray(held.offset)
    bearing = math.atan2(y, x)
    for _ in range(8):
        rotation = leaning_orientation(bearing, lean, held.half_turn).to_rotation_matrix()
        reach = rotation @ offset
        bearing = math.atan2(y - reach[1], x - reach[0])
    rotation = leaning_orientation(bearing, lean, held.half_turn).to_rotation_matrix()
    reach = rotation @ offset
    upright = float(np.clip((rotation @ np.asarray(held.normal))[2], -1.0, 1.0))
    tilt = math.acos(abs(upright))
    drop = held.outer_radius * math.sin(tilt) + held.height / 2.0 * math.cos(tilt)
    return CarryPose(
        float(x - reach[0]),
        float(y - reach[1]),
        float(top_z + gap + drop - reach[2]),
        bearing,
        tilt,
    )


def is_direct(plan: PlanResult, slack: float) -> bool:
    """Whether a planned path stays between its start and its goal in every joint."""
    if plan.plan is None or not plan.plan.trajectory.points:
        return False
    path = np.asarray([point.positions for point in plan.plan.trajectory.points], dtype=float)
    low = np.minimum(path[0], path[-1]) - slack
    high = np.maximum(path[0], path[-1]) + slack
    return bool(np.all(path >= low) and np.all(path <= high))


class RingStackModuleConfig(ModuleConfig):
    planning_frame: str = "world"
    ring_prompt: str = "ring"
    rod_prompt: str = "rod"
    # Height of the surface the rings and the rod stand on.
    support_z: float = 0.0
    # Joint positions the whole scene is looked at from.
    overview_joints: list[float]
    # Joint positions that put the camera above the table, looking straight
    # down, and how far from the base the spot it looks at then is. The arm
    # turns to the bearing of a ring and moves in or out until it is over it.
    ring_view_joints: list[float]
    ring_view_distance: float
    # The same for looking straight down at the top of the rod.
    rod_view_joints: list[float]
    rod_view_distance: float
    # Wait after the arm stops before a frame is trusted to show the scene.
    view_settle: float = Field(default=1.5, ge=0.0)

    # How far the gripper leans outward from vertical, radians: when it grips a
    # ring on the table, and when it carries one over the rod. The carry lean is
    # what the arm needs to hold its tool above the rod; the ring arrives tilted
    # by the difference between the two.
    pick_lean: float = Field(default=0.0, ge=0.0, lt=math.pi / 2)
    carry_lean: float = Field(default=math.radians(20.0), ge=0.0, lt=math.pi / 2)
    # How far the fingertips reach past the tool point, and how thick one is.
    fingertip_depth: float = Field(default=0.02, gt=0.0)
    finger_thickness: float = Field(default=0.008, gt=0.0)
    # Fingertip height above the support when gripping.
    fingertip_clearance: float = Field(default=0.003, ge=0.0)
    # Widest the jaws open, metres, at a commanded opening of 1.0.
    jaw_stroke: float = Field(default=0.08, gt=0.0)
    pregrasp_offset: float = Field(default=0.06, gt=0.0)
    # Distance from the base at which the gripper can work with the pick lean
    # at the table, and with the carry lean above the rod.
    reach: tuple[float, float] = (0.2, 0.5)
    carry_reach: tuple[float, float] = (0.25, 0.45)

    # Outer radius of something that can be one of the rings, and how tall.
    ring_radius: tuple[float, float] = (0.025, 0.06)
    max_ring_height: float = Field(default=0.05, gt=0.0)
    # A rod stands at least this far above the support.
    min_rod_height: float = Field(default=0.08, gt=0.0)
    rod_radius: float = Field(default=0.018, gt=0.0)
    # A ring whose centre is this close to the rod is on it.
    on_rod_distance: float = Field(default=0.025, gt=0.0)
    # The rod with rings on it, as the planner should avoid it.
    rod_obstacle_radius: float = Field(default=0.06, gt=0.0)

    # Gap between the lowest point of the ring and the top of the rod on
    # release, and how much higher than that the ring is carried.
    release_gap: float = Field(default=0.004)
    carry_clearance: float = Field(default=0.02, gt=0.0)
    # How far a gripped ring is raised straight up before the arm turns it.
    lift: float = Field(default=0.05, gt=0.0)
    # How far the open gripper rises off the rod before a planned move.
    retreat: float = Field(default=0.025, gt=0.0)
    # The pile around the rod grows by at least this when a ring lands on it,
    # and by no more than a ring that hangs up near the top would show.
    min_stack_rise: float = Field(default=0.01, gt=0.0)
    max_stack_rise: float = Field(default=0.05, gt=0.0)
    max_attempts_per_ring: int = Field(default=2, gt=0)

    # A ring sliding out of the jaws lets them close: more than this much
    # closing since the grip, in normalized opening, and it is set down.
    slip_tolerance: float = Field(default=0.04, gt=0.0)
    # How far outside the span between its start and its goal a joint may go
    # on a planned path, radians. A path that leaves it is not executed.
    path_slack: float = Field(default=0.2, gt=0.0)
    settle_timeout: float = Field(default=1.5, ge=0.0)
    settle_tolerance: float = Field(default=0.001, gt=0.0)
    carry_speed: float = Field(default=0.3, gt=0.0, le=1.0)
    grasp_verification: GraspVerificationConfig = Field(default_factory=GraspVerificationConfig)


class RingStackModule(Module):
    """Find rings and a rod, and put the rings on the rod."""

    config: RingStackModuleConfig
    _scene: ObjectSceneRegistrationSpec
    _manipulation: ManipulationSpec
    _world: PlanningWorldSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rings: dict[str, Ring] = {}
        self._rod: Rod | None = None
        # Where the camera really is relative to where its mount edge says, as a
        # transform in the tool frame.
        self._camera_correction: NDArray[np.float64] = np.eye(4)

    @skill(uses=[CAP_MOVEMENT])
    def scan_rings(self) -> SkillResult[ManipulationSkillError]:
        """Look over the table for the rod and the rings.

        Moves the arm to its overview pose. Returns each ring's ID, its position
        and whether it is already on the rod; use an ID with stack_ring.
        """
        group = self._group()
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "Gripper-capable planning group is missing")
        if failure := self._look(group, self.config.overview_joints):
            return failure
        rod = self._find_rod()
        if rod is None:
            return SkillResult.fail("OBJECT_NOT_DETECTED", "No upright rod in view")
        self._set_rod(rod)
        self._rings = self._find_rings()
        return SkillResult.ok(
            f"Found a rod and {len(self._rings)} ring(s), {len(self._free_rings())} not on the rod",
            rod={"x": rod.x, "y": rod.y, "top_z": rod.top_z},
            rings=self.get_rings(),
        )

    @rpc
    def set_camera_correction(self, translation: list[float], rotation: list[float]) -> bool:
        """Correct the wrist camera's pose without restarting the stack.

        A camera that has turned on its mount puts everything it sees in the
        wrong place by an amount that depends on where the arm is. The
        correction is the rigid transform, in the tool frame, from where things
        are measured to where they are: a translation in metres and a
        quaternion (x, y, z, w). It lasts until the module stops.
        """
        if len(translation) != 3 or len(rotation) != 4:
            return False
        correction = np.eye(4)
        correction[:3, :3] = Quaternion(*rotation).to_rotation_matrix()
        correction[:3, 3] = translation
        self._camera_correction = correction
        return True

    @rpc
    def get_rings(self) -> list[dict[str, Any]]:
        """The rings of the latest scan."""
        return [
            {
                "object_id": object_id,
                "x": ring.x,
                "y": ring.y,
                "on_rod": self._on_rod(ring),
            }
            for object_id, ring in self._rings.items()
        ]

    @skill(uses=[CAP_MOVEMENT])
    def stack_ring(self, object_id: str) -> SkillResult[ManipulationSkillError]:
        """Pick one ring up by its rim and put it on the rod.

        Args:
            object_id: Exact ring ID returned by the latest scan_rings call.
        """
        ring = self._rings.get(object_id)
        if ring is None or self._rod is None:
            return SkillResult.fail("OBJECT_NOT_DETECTED", f"Unknown ring: {object_id}")
        if self._on_rod(ring):
            return SkillResult.fail("INVALID_STATE", "That ring is already on the rod")
        group = self._group()
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "Gripper-capable planning group is missing")
        return self._stack(group, ring)

    @skill(uses=[CAP_MOVEMENT])
    def stack_all_rings(self) -> SkillResult[ManipulationSkillError]:
        """Put every ring on the table onto the rod, nearest ring first.

        Scans, stacks one ring, and scans again until no ring is left off the
        rod. Stops at the first ring that fails more than once.
        """
        stacked = 0
        attempts: dict[tuple[int, int], int] = {}
        while True:
            scan = self.scan_rings()
            if not scan.is_success():
                return scan
            free = self._free_rings()
            if not free:
                return SkillResult.ok(
                    f"All rings are on the rod; stacked {stacked}", stacked=stacked
                )
            if sum(attempts.values()) >= self.config.max_attempts_per_ring * len(self._rings):
                return SkillResult.fail(
                    "EXECUTION_FAILED", f"Stacked {stacked}; {len(free)} ring(s) keep failing"
                )
            object_id, ring = min(free.items(), key=lambda item: math.hypot(item[1].x, item[1].y))
            # A ring that was dropped is found again near where it was.
            place = (round(ring.x / 0.05), round(ring.y / 0.05))
            attempts[place] = attempts.get(place, 0) + 1
            if attempts[place] > self.config.max_attempts_per_ring:
                return SkillResult.fail(
                    "EXECUTION_FAILED",
                    f"Stacked {stacked}; the ring at ({ring.x:.3f}, {ring.y:.3f}) keeps failing",
                )
            result = self.stack_ring(object_id)
            logger.info("Ring stack attempt", ring=object_id, result=result.message)
            if result.is_success():
                stacked += 1

    def _stack(self, group: PlanningGroupInfo, seen: Ring) -> SkillResult[ManipulationSkillError]:
        cfg = self.config
        # The rod first, while the jaws are empty: where its top is, and how
        # high the pile around it stands.
        rod = self._rod
        assert rod is not None
        if failure := self._look_at_rod(group, rod):
            return failure
        measured = self._measure_rod(rod)
        if measured is None:
            return SkillResult.fail("OBJECT_NOT_DETECTED", "The rod's top is not where it was")
        rod, pile_before = measured
        self._set_rod(rod)

        held = self._pick(group, seen)
        if isinstance(held, SkillResult):
            return held
        if failure := self._release_over(group, held, rod.x, rod.y, rod.top_z):
            return failure

        if failure := self._look_at_rod(group, rod):
            return failure
        measured = self._measure_rod(rod)
        if measured is None:
            return SkillResult.fail("EXECUTION_FAILED", "The rod is gone after the ring's release")
        rod, pile_after = measured
        self._set_rod(rod)
        rise = pile_after - pile_before
        if rise < cfg.min_stack_rise:
            return SkillResult.fail(
                "EXECUTION_FAILED",
                f"The ring is not on the rod: its pile rose {rise * 1000:.0f} mm",
            )
        if rise > cfg.max_stack_rise:
            return SkillResult.fail(
                "EXECUTION_FAILED",
                f"The ring is on the rod but has not slid down: pile up {rise * 1000:.0f} mm",
            )
        return SkillResult.ok(
            f"Ring is on the rod; the pile rose {rise * 1000:.0f} mm",
            jaws=held.jaws,
            rise=rise,
            pile_top=pile_after,
        )

    def _pick(self, group: PlanningGroupInfo, seen: Ring) -> Held | Failure:
        """Look at a ring from above, then take it by its rim and stay there."""
        cfg = self.config
        if failure := self._look_down(
            group, cfg.ring_view_joints, cfg.ring_view_distance, seen.x, seen.y
        ):
            return failure
        nearby = [
            ring
            for ring in self._find_rings().values()
            if math.hypot(ring.x - seen.x, ring.y - seen.y) < seen.outer_radius
        ]
        if not nearby:
            return SkillResult.fail("OBJECT_NOT_DETECTED", "The ring is not where it was")
        ring = nearby[0]

        # A finger leaning along the tube meets it above its tip: put the tip
        # further out, so that the finger crosses the tube's middle height at
        # the point where the line from the base touches the tube's circle.
        lean = cfg.pick_lean
        tip_z = cfg.support_z + cfg.fingertip_clearance
        tool_z = tip_z + cfg.fingertip_depth * math.cos(lean)
        middle_z = (cfg.support_z + ring.top_z) / 2.0
        lead = (cfg.fingertip_depth - (middle_z - tip_z) / math.cos(lean)) * math.sin(lean)
        side = self._clearer_side(ring, lead)
        hold = rim_hold((ring.x, ring.y), ring.grip_radius, lead, side)
        if hold is None or not cfg.reach[0] <= math.hypot(hold.x, hold.y) <= cfg.reach[1]:
            return SkillResult.fail(
                "PLANNING_FAILED",
                f"The ring at ({ring.x:.2f}, {ring.y:.2f}) is outside the reach of "
                f"{cfg.reach[0]:.2f}-{cfg.reach[1]:.2f} m",
            )
        # Jaws open far enough to put the inner finger in the middle of the hole.
        opening = (2.0 * ring.grip_radius - cfg.finger_thickness) / cfg.jaw_stroke
        opening = min(1.0, max(0.0, opening))
        if failure := self._set_gripper(group, opening):
            return failure
        grasp = self._approach(group, hold, tool_z)
        if isinstance(grasp, SkillResult):
            return grasp
        orientation, half_turn = grasp
        jaws = self._close(group)
        if isinstance(jaws, SkillResult):
            self._set_gripper(group, opening)
            self._linear(group, dz=cfg.pregrasp_offset)
            return jaws
        logger.info("Ring gripped", jaws=jaws, side=side, x=ring.x, y=ring.y)

        # The ring lies level with its centre where it was seen, whatever the
        # fingers lean; in the tool frame that is all there is to know of it.
        to_tool = orientation.to_rotation_matrix().T
        centre = np.array([ring.x - hold.x, ring.y - hold.y, middle_z - tool_z])
        offset = to_tool @ centre
        normal = to_tool @ np.array([0.0, 0.0, 1.0])
        return Held(
            (float(offset[0]), float(offset[1]), float(offset[2])),
            (float(normal[0]), float(normal[1]), float(normal[2])),
            ring.outer_radius,
            ring.top_z - cfg.support_z,
            half_turn,
            opening,
            jaws,
        )

    def _release_over(
        self, group: PlanningGroupInfo, held: Held, x: float, y: float, top_z: float
    ) -> Failure | None:
        """Carry the held ring until its centre is over a point, and let go.

        The ring is released with its lowest point ``release_gap`` above
        ``top_z``: the top of the rod, or the table. On the way it stays above
        the rod, and it is set down where it is if it starts to slip.
        """
        cfg = self.config
        # The planner does not know about the ring in the jaws, so the ring
        # travels above everything that stands on the table.
        tallest = top_z if self._rod is None else max(top_z, self._rod.top_z)
        above = tallest + cfg.carry_clearance
        release = carry_pose(held, x, y, top_z, cfg.release_gap, cfg.carry_lean)
        over = carry_pose(held, x, y, above, cfg.release_gap, cfg.carry_lean)
        if not cfg.carry_reach[0] <= math.hypot(over.x, over.y) <= cfg.carry_reach[1]:
            self._put_down(group, held)
            return SkillResult.fail(
                "PLANNING_FAILED",
                f"({x:.2f}, {y:.2f}) is outside the carrying reach of "
                f"{cfg.carry_reach[0]:.2f}-{cfg.carry_reach[1]:.2f} m",
            )

        failure = self._linear(group, dz=cfg.lift, speed=cfg.carry_speed) or self._slipping(
            group, held
        )
        # Up to carrying height with the carry lean, about where the ring is,
        # pulled in or pushed out to a distance the arm can hold that lean at.
        tip = None if failure else self._tip(group)
        if tip is not None:
            centre = tip.position + tip.orientation.rotate_vector(Vector3(*held.offset))
            distance = math.hypot(centre.x, centre.y)
            staged = min(max(distance, cfg.carry_reach[0] + 0.03), cfg.carry_reach[1] - 0.03)
            stage = carry_pose(
                held,
                centre.x * staged / distance,
                centre.y * staged / distance,
                above,
                cfg.release_gap,
                cfg.carry_lean,
            )
            failure = (
                self._move_pose(group, self._pose(stage, held), cfg.carry_speed)
                or self._slipping(group, held)
                or self._turn_to(group, over.bearing)
                or self._slipping(group, held)
                or self._linear_to(group, over.x, over.y, over.z)
                or self._linear_to(group, release.x, release.y, release.z)
                or self._slipping(group, held)
            )
        elif failure is None:
            failure = SkillResult.fail("INVALID_STATE", "End-effector pose is unavailable")
        if failure is not None:
            self._put_down(group, held)
            return failure
        logger.info("Ring released", tilt=release.tilt, x=x, y=y)
        if failure := self._set_gripper(group, held.opening):
            return failure
        return self._linear(group, dz=cfg.retreat)

    def _pose(self, carry: CarryPose, held: Held) -> PoseStamped:
        return PoseStamped(
            frame_id=self.config.planning_frame,
            position=Vector3(carry.x, carry.y, carry.z),
            orientation=leaning_orientation(carry.bearing, self.config.carry_lean, held.half_turn),
        )

    def _slipping(self, group: PlanningGroupInfo, held: Held) -> Failure | None:
        """A ring leaving the jaws lets them close further than they gripped."""
        jaws = self._gripper(group)
        if jaws is None or jaws >= held.jaws - self.config.slip_tolerance:
            return None
        return SkillResult.fail(
            "GRASP_VERIFICATION_FAILED",
            f"The ring is slipping out of the jaws: opening {jaws:.3f}, was {held.jaws:.3f}",
        )

    # Perception

    def _corrected(self, points: Points) -> Points:
        """Points seen from where the arm is now, with the camera correction applied."""
        if np.array_equal(self._camera_correction, np.eye(4)):
            return points
        group = self._group()
        state = None if group is None else self._manipulation.get_state().groups.get(group.id)
        if state is None or state.end_effector_pose is None:
            return points
        tool = np.eye(4)
        tool[:3, :3] = state.end_effector_pose.orientation.to_rotation_matrix()
        tool[:3, 3] = [
            state.end_effector_pose.position.x,
            state.end_effector_pose.position.y,
            state.end_effector_pose.position.z,
        ]
        fix = tool @ self._camera_correction @ np.linalg.inv(tool)
        corrected: Points = (points @ fix[:3, :3].T + fix[:3, 3]).astype(np.float32)
        return corrected

    def _clouds(self, prompt: str) -> list[tuple[str, Points]]:
        """One detection pass: the points of everything the prompt found."""
        detections = self._scene.scan_scene(text=[prompt])
        found = []
        for detection in detections.detections:
            if not detection.id:
                continue
            cloud = self._scene.get_object_pointcloud_by_object_id(str(detection.id))
            if cloud is None or cloud.frame_id != self.config.planning_frame:
                continue
            found.append((str(detection.id), self._corrected(cloud.points_f32())))
        return found

    def _find_rings(self) -> dict[str, Ring]:
        cfg = self.config
        rings = {}
        for object_id, points in self._clouds(cfg.ring_prompt):
            ring = fit_ring(points)
            if ring is None:
                continue
            flat = ring.top_z - cfg.support_z < cfg.max_ring_height or (
                self._rod is not None and self._on_rod(ring)
            )
            if flat and cfg.ring_radius[0] <= ring.outer_radius <= cfg.ring_radius[1]:
                rings[object_id] = ring
        return rings

    def _find_rod(self) -> Rod | None:
        """The tallest thing the rod prompt finds, if it is tall enough."""
        rods = [fit_rod_top(points) for _, points in self._clouds(self.config.rod_prompt)]
        tallest = max((rod for rod in rods if rod is not None), key=lambda r: r.top_z, default=None)
        if tallest is None or tallest.top_z - self.config.support_z < self.config.min_rod_height:
            return None
        return tallest

    def _measure_rod(self, rod: Rod) -> tuple[Rod, float] | None:
        """From above the rod: its top, and the height of the pile around it."""
        cfg = self.config
        # The scene cloud is cut from the depth frame of the latest detection pass.
        self._scene.scan_scene(text=[cfg.ring_prompt])
        cloud = self._scene.get_full_scene_pointcloud(None, 1.0, 0.002)
        if cloud is None or cloud.frame_id != cfg.planning_frame:
            return None
        points = self._corrected(cloud.points_f32())
        distance = np.hypot(points[:, 0] - rod.x, points[:, 1] - rod.y)
        near_top = points[(distance < 2.0 * cfg.rod_radius) & (points[:, 2] > rod.top_z - 0.03)]
        top = fit_rod_top(near_top)
        if top is None or math.hypot(top.x - rod.x, top.y - rod.y) > 2.0 * cfg.rod_radius:
            return None
        # A ring on the rod shows the top of its tube in this band; the rod's
        # own ragged edge and whatever lies beyond the ring stay out of it.
        distance = np.hypot(points[:, 0] - top.x, points[:, 1] - top.y)
        around = points[
            (distance > cfg.rod_radius + 0.01)
            & (distance < cfg.rod_radius + 0.028)
            & (points[:, 2] < top.top_z - 0.01)
        ]
        pile = cfg.support_z
        if len(around) >= _MIN_POINTS:
            pile = max(pile, float(np.median(around[:, 2])))
        return top, pile

    def _set_rod(self, rod: Rod) -> None:
        """Remember the rod, and keep the planner's copy of it where it is."""
        cfg = self.config
        self._rod = rod
        # An obstacle is known by its name; two that overlap put every start
        # configuration in collision.
        self._world.remove_obstacle(_ROD_OBSTACLE)
        height = rod.top_z - cfg.support_z
        self._world.add_obstacle(
            _ROD_OBSTACLE,
            Pose(
                Vector3(rod.x, rod.y, cfg.support_z + height / 2.0), Quaternion(0.0, 0.0, 0.0, 1.0)
            ),
            "cylinder",
            [cfg.rod_obstacle_radius, height],
        )

    def _on_rod(self, ring: Ring) -> bool:
        rod = self._rod
        return rod is not None and (
            math.hypot(ring.x - rod.x, ring.y - rod.y) < self.config.on_rod_distance
        )

    def _free_rings(self) -> dict[str, Ring]:
        return {key: ring for key, ring in self._rings.items() if not self._on_rod(ring)}

    def _clearer_side(self, ring: Ring, lead: float) -> int:
        """The side of the ring whose outer finger lands furthest from anything else."""
        others = [(other.x, other.y) for other in self._rings.values() if other != ring]
        if self._rod is not None:
            others.append((self._rod.x, self._rod.y))

        def room(side: int) -> float:
            hold = rim_hold((ring.x, ring.y), ring.grip_radius, lead, side)
            if hold is None:
                return -math.inf
            # The outer finger lands about one grip radius beyond the tool point.
            x = 2.0 * hold.x - ring.x
            y = 2.0 * hold.y - ring.y
            gaps = [math.hypot(x - ox, y - oy) for ox, oy in others if (ox, oy) != (ring.x, ring.y)]
            return min(gaps, default=math.inf)

        return max((1, -1), key=room)

    # Motion

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

    def _run(self, group: PlanningGroupInfo, plan: PlanResult) -> Failure | None:
        """Execute a plan, but only one that goes straight to its goal."""
        if not plan.succeeded:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", plan.message)
        if not is_direct(plan, self.config.path_slack):
            self._manipulation.clear_planned_path()
            return SkillResult.fail("PLANNING_FAILED", "The planner's path is a detour")
        return self._execute(group)

    def _move_joints(
        self, group: PlanningGroupInfo, joints: Sequence[float], speed: float | None = None
    ) -> Failure | None:
        target = JointState(name=list(group.joint_names), position=[float(q) for q in joints])
        try:
            plan = self._manipulation.plan_to_joints({group.id: target}, speed)
        except RuntimeError as error:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", str(error))
        return self._run(group, plan)

    def _move_pose(
        self, group: PlanningGroupInfo, pose: PoseStamped, speed: float | None = None
    ) -> Failure | None:
        try:
            plan = self._manipulation.plan_to_poses({group.id: pose}, speed)
        except RuntimeError as error:
            self._manipulation.reset()
            return SkillResult.fail("PLANNING_FAILED", str(error))
        return self._run(group, plan)

    def _look(self, group: PlanningGroupInfo, joints: Sequence[float]) -> Failure | None:
        """Go to a viewing pose and wait for a frame taken from it."""
        if failure := self._move_joints(group, joints):
            return failure
        time.sleep(self.config.view_settle)
        return None

    def _look_down(
        self, group: PlanningGroupInfo, joints: Sequence[float], distance: float, x: float, y: float
    ) -> Failure | None:
        """Put the camera above a point: turn a viewing pose toward it, then slide over it."""
        bearing = math.atan2(y, x)
        target = [bearing, *joints[1:]]
        if self._move_joints(group, target) is not None:
            # From the overview every viewing pose is a short, direct move.
            if failure := self._move_joints(group, self.config.overview_joints):
                return failure
            if failure := self._move_joints(group, target):
                return failure
        slide = math.hypot(x, y) - distance
        # Out of reach, the point is still in view from where the pose looks.
        self._linear(group, slide * math.cos(bearing), slide * math.sin(bearing))
        time.sleep(self.config.view_settle)
        return None

    def _look_at_rod(self, group: PlanningGroupInfo, rod: Rod) -> Failure | None:
        cfg = self.config
        return self._look_down(group, cfg.rod_view_joints, cfg.rod_view_distance, rod.x, rod.y)

    def _approach(
        self, group: PlanningGroupInfo, hold: RimHold, tool_z: float
    ) -> tuple[Quaternion, bool] | Failure:
        """Plan to a point above the hold, then slide down the gripper's own axis.

        Returns the orientation the wrist could reach, and whether it is the
        half turn.
        """
        cfg = self.config
        failure: Failure = SkillResult.fail("PLANNING_FAILED", "No wrist orientation was tried")
        for half_turn in (False, True):
            orientation = leaning_orientation(hold.bearing, cfg.pick_lean, half_turn)
            back = orientation.rotate_vector(Vector3(0.0, 0.0, -cfg.pregrasp_offset))
            pregrasp = PoseStamped(
                frame_id=cfg.planning_frame,
                position=Vector3(hold.x, hold.y, tool_z) + back,
                orientation=orientation,
            )
            planned = self._move_pose(group, pregrasp)
            if planned is not None:
                failure = planned
                continue
            if slid := self._linear_to(group, hold.x, hold.y, tool_z):
                return slid
            return orientation, half_turn
        return failure

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

    def _linear_to(self, group: PlanningGroupInfo, x: float, y: float, z: float) -> Failure | None:
        """A straight move to a point, measured from where the arm settled."""
        tip = self._tip(group)
        if tip is None:
            return SkillResult.fail("INVALID_STATE", "End-effector pose is unavailable")
        return self._linear(
            group,
            x - tip.position.x,
            y - tip.position.y,
            z - tip.position.z,
            self.config.carry_speed,
        )

    def _turn_to(self, group: PlanningGroupInfo, bearing: float) -> Failure | None:
        """Turn the base joint so the tool point lies along a bearing.

        The held ring sweeps an arc at the height and distance it is at.
        """
        tip = self._tip(group)
        joints = self._joints(group)
        if tip is None or joints is None:
            return SkillResult.fail("INVALID_STATE", "Arm state is unavailable")
        turn = bearing - math.atan2(tip.position.y, tip.position.x)
        turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
        return self._move_joints(group, [joints[0] + turn, *joints[1:]], self.config.carry_speed)

    def _put_down(self, group: PlanningGroupInfo, held: Held) -> None:
        """Best effort: lower the held ring to the table where it is, and let go."""
        cfg = self.config
        tip = self._tip(group)
        if tip is not None:
            rotation = tip.orientation.to_rotation_matrix()
            upright = float(np.clip((rotation @ np.asarray(held.normal))[2], -1.0, 1.0))
            tilt = math.acos(abs(upright))
            drop = held.outer_radius * math.sin(tilt) + held.height / 2.0 * math.cos(tilt)
            centre_z = tip.position.z + float((rotation @ np.asarray(held.offset))[2])
            self._linear(
                group, dz=cfg.support_z + cfg.release_gap + drop - centre_z, speed=cfg.carry_speed
            )
        self._set_gripper(group, held.opening)
        self._linear(group, dz=cfg.pregrasp_offset)

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
        """Close on the tube; the settled jaw opening, or why it holds nothing."""
        verification = self.config.grasp_verification
        result = self._manipulation.set_gripper_position(verification.closed_position, group.id)
        if not result.succeeded:
            return SkillResult.fail("GRIPPER_FAILED", result.message or "Gripper command rejected")
        settle = await_gripper_settle(
            lambda: self._gripper(group), verification.closed_position, verification
        )
        if failure := grasp_failure(settle, verification):
            return SkillResult.fail("GRASP_VERIFICATION_FAILED", failure)
        assert settle.position is not None
        return settle.position
