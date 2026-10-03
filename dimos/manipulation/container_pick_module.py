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

"""Pick an open container (bin, box, tray) by its rim, carry it and place it.

The skill a parallel-jaw arm needs for a container it cannot span: scan, fit the
rim, straddle one wall from above, close, lift, turn the hanging container so its
opening faces where it should, carry it over the target, lower it, open, and back
the open jaws out over the container's LOW end before lifting away.

LEARNED ON THE xArm7 (2026-10-03, bin into tape slots, 9/9 placements)
    * The scene registry accumulates clouds of anything re-seen within its
      distance threshold, so repeated placements corrupt its clouds; a per-frame
      colour segmentation from the wrist camera (WristTabletopModule) has no
      memory and is preferred when present.
    * A scoop-front container's opening is its lower end wall. Measure each end's
      height in the MIDDLE of the end: the corners belong to the full-height side
      walls and hide the drop.
    * The container lands closer to the jaws than the rim says: it hangs tilted
      from one wall. The learned ``landing_offset`` (4.0 cm for a 10.7 cm wide bin,
      rim half width 5.3 cm) puts it within 1 cm.
    * After opening, the finger INSIDE the container cannot pass a full-height end
      wall: sliding out over it pushes the container along (9.5 cm seen). Exit
      over the low opening end, 2 cm up, past the end plus a margin.
    * The wrist joint has a finite range (+-3.1 rad on the xArm7). The grasp yaw
      equivalent (a parallel jaw is symmetric under a half turn) is chosen so the
      turn that follows stays inside it.

MOTION SAFETY
    Every planned motion is checked before it is executed: forward kinematics
    of every waypoint must keep the hand inside ``workspace_box`` and the elbow
    inside ``elbow_box``, the base joint may turn at most
    ``base_joint_max_excursion`` along one path, and the joint-space length is
    bounded (a pure wrist turn is exempt from the length bound). A rejected plan
    is cleared, never run; a planner exception clears the pending plan so the
    manipulation module does not stay in PLANNING. Every translation is a
    straight-line Cartesian move whose target is box- and reach-checked first;
    the planner is only asked to rotate the wrist in place. Put a TCP box in the
    robot controller as well where the SDK offers one; this module is the second
    line of defence, not the first.

WHAT IT NEEDS IN THE BLUEPRINT
    A ``ManipulationSpec`` (ManipulationModule), an ``ObjectSceneRegistrationSpec``
    for prompted scans, a ``GraspGenSpec`` that proposes rim grasps
    (RimGraspModule) and, optionally, a ``WristTabletopSpec`` (colour scans and
    tape slots) and a map-pause capable self filter so the carried container is
    not mapped as an obstacle along the carry.
"""

from __future__ import annotations

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
from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.manipulation.manipulation_spec import ManipulationSpec, PlanResult
from dimos.manipulation.planning.spec.models import PlanningGroupID
from dimos.manipulation.skill_errors import ManipulationSkillError
from dimos.manipulation.wrist_tabletop_spec import WristTabletopSpec
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.perception.experimental.object_scene_registration_spec import ObjectSceneRegistrationSpec
from dimos.robot.assets.model import RobotModel
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

Box = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


class MapPauseSpec(Spec, Protocol):
    """A mapper front end that can stop feeding the planner's obstacle map."""

    def set_paused(self, paused: bool) -> bool: ...


class ContainerPickConfig(ModuleConfig):
    # The robot model; forward kinematics of planned waypoints is checked with it.
    model: RobotModel
    planning_frame: str = "world"
    # Links that must stay inside workspace_box (the hand) and elbow_box (the arm).
    hand_links: list[str] = Field(default_factory=lambda: ["link7", "link_tcp"])
    elbow_links: list[str] = Field(default_factory=lambda: ["link4", "link5"])
    workspace_box: Box = ((-0.42, 0.42), (-0.78, 0.10), (-0.03, 0.80))
    elbow_box: Box = ((-0.45, 0.45), (-0.78, 0.30), (-0.03, 0.88))
    # Horizontal TCP radius from the base the arm may reach.
    reach_max: float = Field(default=0.66, gt=0.0)
    base_joint: str = "joint1"
    base_joint_max_excursion: float = Field(default=1.0, gt=0.0)
    path_max_length: float = Field(default=3.0, gt=0.0)
    wrist_joint: str = "joint7"
    # Planning-frame yaw change per radian of wrist joint motion with the tool
    # pointing down (-1 on the xArm7: joint 7 positive turns the tool clockwise).
    wrist_joint_sign: float = -1.0
    # Keep the wrist joint this far inside its model limits.
    wrist_joint_margin: float = 0.15
    plan_speed_scale: float = Field(default=0.3, gt=0.0, le=1.0)
    cartesian_speed_scale: float = Field(default=0.3, gt=0.0, le=1.0)
    # TCP never goes below this planning-frame height (None disables).
    min_z: float | None = None
    # Survey pose: the camera looks straight down from here. With
    # survey_camera_offset (optical centre minus TCP, planning XY, at survey_yaw)
    # the camera, not the TCP, is put above the target and the nearer yaw
    # equivalent is used; without it survey_offset_xy is added to the target.
    survey_height: float = 0.40
    survey_yaw: float = -1.6
    survey_offset_xy: tuple[float, float] = (0.0, 0.07)
    survey_camera_offset: tuple[float, float] | None = None
    survey_reach: float = 0.54
    survey_x_range: tuple[float, float] = (-0.30, 0.30)
    survey_y_range: tuple[float, float] = (-0.56, -0.18)
    pregrasp_offset: float = Field(default=0.10, gt=0.0)
    lift_height: float = Field(default=0.15, ge=0.0)
    # Carry height above the grasp height (lift stages add up to this).
    carry_height: float = Field(default=0.25, ge=0.0)
    # Prefer the wall whose horizontal radius is nearest this (the arm is neither
    # folded under itself nor stretched).
    preferred_reach: float = 0.42
    hold_seconds: float = 1.5
    gripper_settle_seconds: float = 2.5
    prompts: list[str] = Field(default_factory=lambda: ["bin"])
    scan_attempts: int = 3
    # Rim-fit plausibility: the fitted rectangle's long side must be at least
    # container_long_min and its short side within container_short_range, and
    # its centre within rim_center_tolerance of the footprint centroid. Zero /
    # wide ranges disable the check.
    container_long_min: float = 0.0
    container_short_range: tuple[float, float] = (0.0, 10.0)
    rim_center_tolerance: float = 0.03
    # Opening (scoop) detection: the lower end wall, if it is lower by at least this.
    opening_min_drop: float = 0.012
    # Distance from the TCP to where the container's centre lands, across the
    # grasped wall (None: the rim half width). The container hangs tilted from
    # one wall and lands closer to the jaws than the rim geometry says.
    landing_offset: float | None = 0.040
    # Set-down: raise before backing out over the low end, and go this far past
    # the container's half length.
    exit_raise: float = 0.02
    exit_margin: float = 0.05
    # Clearance added past the container's half length when backing out along a
    # wall (fallback when the opening is unknown).
    slide_margin: float = 0.06
    # Slot placement: wrist-camera positions (planning XY) to view the tape from,
    # and how many place/verify/correct rounds to allow.
    slot_survey_points: list[tuple[float, float]] = Field(default_factory=list)
    max_place_rounds: int = Field(default=3, ge=1)
    pause_mapping_while_holding: bool = True
    grasp_verification: GraspVerificationConfig = Field(
        default_factory=lambda: GraspVerificationConfig(empty_epsilon=0.012)
    )


def inside(point: NDArray[np.float64], box: Box) -> bool:
    return bool(
        box[0][0] <= point[0] <= box[0][1]
        and box[1][0] <= point[1] <= box[1][1]
        and box[2][0] <= point[2] <= box[2][1]
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_xy(vector: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def nearest_equivalent_yaw(yaw: float, reference: float) -> float:
    """A parallel jaw is symmetric under a half turn: the yaw nearest ``reference``."""
    return wrap_angle(
        min(
            (yaw + k * math.pi for k in (-2, -1, 0, 1, 2)),
            key=lambda y: abs(wrap_angle(y - reference)),
        )
    )


def wrist_feasible_yaw(
    yaw: float,
    current_yaw: float,
    wrist_position: float,
    turn_after: float,
    wrist_limits: tuple[float, float],
    wrist_sign: float = -1.0,
) -> float | None:
    """The yaw equivalent (``yaw`` or ``yaw + pi``) that keeps the wrist joint inside
    its limits both at the grasp and after turning the held object by ``turn_after``
    (planning-frame radians); the one ending nearer the middle of the range wins.
    None when neither fits."""
    low, high = wrist_limits
    middle = (low + high) / 2.0
    best: tuple[float, float] | None = None
    for candidate in (wrap_angle(yaw), wrap_angle(yaw + math.pi)):
        shortest = wrap_angle(candidate - current_yaw)
        # the wrist may reach the same tool yaw by turning either way round
        for world_turn in (shortest, shortest - math.copysign(2.0 * math.pi, shortest)):
            at_grasp = wrist_position + world_turn / wrist_sign
            after = at_grasp + turn_after / wrist_sign
            if low <= at_grasp <= high and low <= after <= high:
                score = abs(after - middle)
                if best is None or score < best[0]:
                    best = (score, candidate)
    return None if best is None else best[1]


def describe_container(
    points: NDArray[np.floating], rim: dict[str, Any], opening_min_drop: float = 0.012
) -> dict[str, Any]:
    """Centre, axes and opening of a container from its cloud and rim rectangle.

    The opening is the lower of the two end walls, each measured in the middle
    of the end (the corners belong to the full-height side walls). ``opening_dir``
    is a planning-frame XY unit vector from the centre toward the opening; it is
    only trusted when ``opening_known``.
    """
    center = np.asarray(rim["rect_center"], dtype=float)
    axes = np.asarray(rim["rect_axes"], dtype=float)
    extents = np.asarray(rim["rect_extents"], dtype=float)
    long_index = int(np.argmax(extents))
    along = axes[long_index]
    across = axes[1 - long_index]
    half_length = float(extents[long_index]) / 2.0
    width = float(extents[1 - long_index])
    top = float(rim["rim_top_z"])
    xy = points[:, :2].astype(np.float64) - center
    z = points[:, 2].astype(np.float64)
    p_along = xy @ along
    q_across = xy @ across
    band = z >= top - 0.06
    heights: dict[str, float | None] = {}
    for sign, key in ((1.0, "plus"), (-1.0, "minus")):
        selected = (
            band
            & (sign * p_along > half_length - 0.03)
            & (sign * p_along < half_length + 0.015)
            & (np.abs(q_across) < 0.3 * width)
        )
        heights[key] = (
            float(np.quantile(z[selected], 0.9)) if np.count_nonzero(selected) >= 30 else None
        )
    plus, minus = heights["plus"], heights["minus"]
    if plus is not None and minus is not None:
        drop = abs(plus - minus)
        opening = along if plus < minus else -along
    else:
        drop = 0.0
        opening = along
    corners = [
        (center + sx * across * width / 2.0 + sy * along * half_length).tolist()
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    return {
        "center": center.tolist(),
        "long_axis": along.tolist(),
        "yaw": float(math.atan2(along[1], along[0])),
        "half_length": half_length,
        "width": width,
        "top_z": top,
        "end_heights": heights,
        "opening_drop": drop,
        "opening_known": drop >= opening_min_drop,
        "opening_dir": opening.tolist(),
        "corners": corners,
        "n": len(points),
    }


class PathGuard:
    """Forward-kinematic checks of a joint path against the workspace boxes."""

    def __init__(self, config: ContainerPickConfig) -> None:
        import pinocchio

        self._pin = pinocchio
        self.config = config
        self.model = pinocchio.buildModelFromXML(config.model.load().xml)
        self.data = self.model.createData()
        self.q_index = {
            self.model.names[j]: self.model.joints[j].idx_q for j in range(1, self.model.njoints)
        }
        self.frames = {
            name: self.model.getFrameId(name)
            for name in [*config.hand_links, *config.elbow_links]
            if self.model.existFrame(name)
        }
        missing = [n for n in [*config.hand_links, *config.elbow_links] if n not in self.frames]
        if missing:
            raise ValueError(f"links not in the robot model: {missing}")

    def joint_limits(self, name: str) -> tuple[float, float] | None:
        index = self.q_index.get(name)
        if index is None:
            return None
        return (
            float(self.model.lowerPositionLimit[index]),
            float(self.model.upperPositionLimit[index]),
        )

    def link_positions(
        self, names: list[str], positions: list[float]
    ) -> dict[str, NDArray[np.float64]]:
        q = np.zeros(self.model.nq)
        for name, value in zip(names, positions, strict=False):
            if name in self.q_index:
                q[self.q_index[name]] = value
        self._pin.framesForwardKinematics(self.model, self.data, q)
        return {name: np.array(self.data.oMf[fid].translation) for name, fid in self.frames.items()}

    def check_path(self, path: list[JointState]) -> str | None:
        """None when the path is acceptable, otherwise why it is not."""
        if not path:
            return "empty path"
        config = self.config
        names = list(path[0].name)
        positions = np.array([[float(v) for v in w.position] for w in path])
        excursion = positions.max(axis=0) - positions.min(axis=0)
        if config.base_joint in names:
            base = excursion[names.index(config.base_joint)]
            if base > config.base_joint_max_excursion:
                return f"{config.base_joint} excursion {base:.2f} rad"
        length = 0.0
        previous: NDArray[np.float64] | None = None
        for waypoint in path:
            q = np.array([float(v) for v in waypoint.position])
            if previous is not None:
                length += float(np.abs(q - previous).sum())
            previous = q
            links = self.link_positions(names, list(q))
            for name, point in links.items():
                box = config.elbow_box if name in config.elbow_links else config.workspace_box
                if not inside(point, box):
                    return f"{name} leaves its box at {np.round(point, 3).tolist()}"
            tip = links.get(config.hand_links[-1])
            if tip is not None and math.hypot(tip[0], tip[1]) > config.reach_max + 0.05:
                return f"tip radius {math.hypot(tip[0], tip[1]):.2f} m"
        # A pure wrist turn may be a half turn or more; anything else stays short.
        others = [e for n, e in zip(names, excursion, strict=False) if n != config.wrist_joint]
        wrist_only = not others or max(others) < 0.1
        limit = 2.0 * math.pi + 0.5 if wrist_only else config.path_max_length
        if length > limit:
            return f"path length {length:.2f} rad (limit {limit:.2f})"
        return None


class ContainerPickModule(Module):
    """Rim-grasp pick, carry and placement of an open container with guarded motions."""

    config: ContainerPickConfig

    _manipulation: ManipulationSpec
    _scene: ObjectSceneRegistrationSpec
    _grasps: GraspGenSpec
    _tabletop: WristTabletopSpec | None = None
    _map_pause: MapPauseSpec | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._guard: PathGuard | None = None
        self._group: PlanningGroupID | None = None
        self._last: dict[str, Any] = {}
        self._holding: dict[str, Any] | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._guard = PathGuard(self.config)

    @rpc
    def stop(self) -> None:
        super().stop()

    @rpc
    def status(self) -> dict[str, Any]:
        """Last pick's details and whether a container is believed held."""
        return {"holding": self._holding, "last": self._last}

    @rpc
    def set_params(self, **params: Any) -> dict[str, Any]:
        """Tune speeds, heights, prompts or boxes in place for a research loop."""
        for key, value in params.items():
            if not hasattr(self.config, key):
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self.config, key, value)
        self._guard = PathGuard(self.config)
        return self.config.model_dump(exclude={"model"})

    # skills

    @skill(uses=[CAP_MOVEMENT])
    def survey(self) -> SkillResult[ManipulationSkillError]:
        """Move the wrist camera to look straight down over the workspace.

        Goes above the last seen container when there is one, by straight-line
        moves at reduced speed, then restores a top-down wrist orientation.
        """
        failure = self._survey_over(self._survey_target_xy())
        if failure is not None:
            return failure
        return SkillResult.ok("At survey pose", tcp=self._tcp().tolist())

    @skill(uses=[CAP_MOVEMENT])
    def pick_up_container(
        self, prompt: str = "", turn_after_degrees: float = 0.0
    ) -> SkillResult[ManipulationSkillError]:
        """Scan for the container, grasp one of its side walls by the rim and lift it.

        Args:
            prompt: Object label for the detector; empty uses the configured prompts.
            turn_after_degrees: Turn the held container will get next (counter-clockwise from above); the grasp is chosen so the wrist can make it.
        """
        if self._holding is not None:
            return SkillResult.fail("INVALID_STATE", "Set the held container down first")
        group = self._resolve_group()
        if group is None:
            return SkillResult.fail("ROBOT_NOT_FOUND", "No gripper-capable planning group")
        opening = self._gripper()
        if opening is not None and opening < 0.8:
            self._gripper_to(1.0)
        survey = self.survey()
        if not survey.success:
            return survey
        scan = self._scan([prompt] if prompt.strip() else None)
        if isinstance(scan, SkillResult):
            return scan
        object_id, points, cloud, container = scan
        return self._pick(object_id, points, cloud, container, math.radians(turn_after_degrees))

    @skill(uses=[CAP_MOVEMENT])
    def rotate_held_container(self, yaw_degrees: float) -> SkillResult[ManipulationSkillError]:
        """Turn the held container about the vertical axis with a guarded wrist move.

        Args:
            yaw_degrees: Rotation to apply, positive counter-clockwise seen from above.
        """
        if self._holding is None:
            return SkillResult.fail("INVALID_STATE", "Nothing is held")
        ok, applied = self._turn_wrist(math.radians(yaw_degrees))
        if not ok:
            return SkillResult.fail("PLANNING_FAILED", "Wrist rotation rejected by the guards")
        return SkillResult.ok("Rotated", applied_degrees=math.degrees(applied))

    @skill(uses=[CAP_MOVEMENT])
    def place_container(
        self, x: float, y: float, opening_yaw_degrees: float | None = None
    ) -> SkillResult[ManipulationSkillError]:
        """Carry the held container to a spot, turn it so its opening faces a direction, set it down.

        Args:
            x: Planning-frame X of where the container's centre should land.
            y: Planning-frame Y of where the container's centre should land.
            opening_yaw_degrees: Direction the opening (low end) should face, degrees counter-clockwise from +X; omit to keep the current heading.
        """
        if self._holding is None:
            return SkillResult.fail("INVALID_STATE", "Nothing is held")
        held = self._holding
        turned = 0.0
        if opening_yaw_degrees is not None:
            if not held["opening_known"]:
                return SkillResult.fail(
                    "PERCEPTION_FAILED", "The container's opening was not identified"
                )
            opening_now = rotate_xy(np.array(held["opening_dir"]), held["turned"])
            wanted = wrap_angle(
                math.radians(opening_yaw_degrees) - math.atan2(opening_now[1], opening_now[0])
            )
            ok, turned = self._turn_wrist(wanted)
            if not ok:
                return SkillResult.fail("PLANNING_FAILED", "Could not turn the container")
            if abs(wrap_angle(turned - wanted)) > math.radians(10.0):
                return SkillResult.fail(
                    "EXECUTION_FAILED",
                    f"Turned {math.degrees(turned):.0f} of {math.degrees(wanted):.0f} deg",
                )
        held["turned"] = wrap_angle(held["turned"] + turned)
        offset_now = (
            rotate_xy(np.array(held["offset_dir"]), held["turned"]) * self._landing_offset()
        )
        tcp_xy = np.array([x, y]) - offset_now
        tcp = self._tcp()
        failure = self._cartesian_to(np.array([tcp_xy[0], tcp_xy[1], tcp[2]]))
        if failure is not None:
            return failure
        tcp_before_lower = self._tcp().tolist()
        release = self._release_and_lift()
        self._last = {
            **self._last,
            "place": {
                "target": [x, y],
                "turned": held["turned"],
                "tcp_before_lower": tcp_before_lower,
                "offset_used": offset_now.tolist(),
                "release": release,
            },
        }
        self._holding = None
        return SkillResult.ok(
            "Container placed",
            target=[x, y],
            turned_degrees=math.degrees(held["turned"]),
            **release,
        )

    @skill(uses=[CAP_MOVEMENT])
    def set_down_container(
        self, x: float | None = None, y: float | None = None
    ) -> SkillResult[ManipulationSkillError]:
        """Lower the held container onto the surface it came from and let go cleanly.

        Args:
            x: Planning-frame X for the container's centre; default is where it was picked.
            y: Planning-frame Y for the centre; default is where it was picked.
        """
        if self._holding is None:
            return SkillResult.fail("INVALID_STATE", "Nothing is held")
        center = self._holding["container"]["center"]
        return self.place_container(center[0] if x is None else x, center[1] if y is None else y)

    @skill(uses=[CAP_MOVEMENT])
    def check_container_pose(self, prompt: str = "") -> SkillResult[ManipulationSkillError]:
        """Look down at the container and report its centre, heading and opening direction.

        Args:
            prompt: Object label for the detector; empty uses the configured prompts.
        """
        failure = self._survey_over(self._survey_target_xy())
        if failure is not None:
            return failure
        scan = self._scan([prompt] if prompt.strip() else None)
        if isinstance(scan, SkillResult):
            return scan
        _, _, _, container = scan
        return SkillResult.ok("Container seen", **container)

    @skill(uses=[CAP_MOVEMENT])
    def map_slots(self) -> SkillResult[ManipulationSkillError]:
        """Look at the tape slots on the table from the configured viewpoints and fit the grid."""
        if self._tabletop is None:
            return SkillResult.fail("INVALID_STATE", "No wrist tabletop module in the blueprint")
        if not self.config.slot_survey_points:
            return SkillResult.fail("INVALID_STATE", "No slot_survey_points configured")
        self._tabletop.clear_tape_views()
        views = []
        for point in self.config.slot_survey_points:
            failure = self._survey_over(np.array(point, dtype=float))
            if failure is not None:
                return failure
            time.sleep(0.8)
            views.append(self._tabletop.add_tape_view())
        slots = self._tabletop.fit_slots()
        if not slots.get("slots"):
            return SkillResult.fail(
                "PERCEPTION_FAILED", f"No slot grid found ({views} tape points per view)"
            )
        return SkillResult.ok(
            f"Mapped {len(slots['slots'])} slots",
            slots=slots["slots"],
            x_lines=slots["x_lines"],
            y_lines=slots["y_lines"],
        )

    @skill(uses=[CAP_MOVEMENT])
    def place_container_in_slot(
        self, slot: str, prompt: str = ""
    ) -> SkillResult[ManipulationSkillError]:
        """Pick up the container and place it inside a mapped tape slot, opening toward the arm.

        Verifies from above that the rim is inside the tape and the opening faces
        the base, and re-picks to correct up to max_place_rounds times.

        Args:
            slot: Slot name from map_slots, e.g. "left", "middle" or "right".
            prompt: Object label for the detector; empty uses the configured prompts.
        """
        if self._tabletop is None:
            return SkillResult.fail("INVALID_STATE", "No wrist tabletop module in the blueprint")
        grid = self._tabletop.get_slots()
        if not grid.get("slots"):
            mapped = self.map_slots()
            if not mapped.success:
                return mapped
            grid = self._tabletop.get_slots()
        if slot not in grid["slots"]:
            return SkillResult.fail(
                "INVALID_STATE", f"Unknown slot {slot!r}; have {list(grid['slots'])}"
            )
        geometry = grid["slots"][slot]
        tape_half = float(grid.get("tape_half_width", 0.012))
        y_far, y_near = geometry["y"]
        # the opening faces the slot end nearer the base (the planning-frame origin)
        toward_base = 1.0 if abs(y_near) < abs(y_far) else -1.0
        opening_yaw = 90.0 * toward_base
        target = np.array(geometry["center"], dtype=float)
        bias = np.zeros(2)
        rounds: list[dict[str, Any]] = []
        prompts = [prompt] if prompt.strip() else None
        for round_index in range(self.config.max_place_rounds):
            if self._holding is None:
                survey = self._survey_over(self._survey_target_xy())
                if survey is not None:
                    return survey
                scan = self._scan(prompts)
                if isinstance(scan, SkillResult):
                    return scan
                object_id, points, cloud, container = scan
                turn = 0.0
                if container["opening_known"]:
                    opening = container["opening_dir"]
                    turn = wrap_angle(
                        math.radians(opening_yaw) - math.atan2(opening[1], opening[0])
                    )
                picked = self._pick(object_id, points, cloud, container, turn)
                if not picked.success:
                    return picked
            placed = self.place_container(
                float(target[0] - bias[0]), float(target[1] - bias[1]), opening_yaw
            )
            if not placed.success:
                return placed
            # verify from above
            failure = self._survey_over(target)
            if failure is not None:
                return failure
            scan = self._scan(prompts)
            if isinstance(scan, SkillResult):
                return scan
            _, _, _, seen = scan
            verdict = self._slot_verdict(seen, geometry, tape_half, toward_base)
            rounds.append(verdict)
            self._last = {**self._last, "slot_rounds": rounds}
            if verdict["success"]:
                return SkillResult.ok(
                    f"Container placed in the {slot} slot", slot=slot, rounds=len(rounds), **verdict
                )
            error = np.array(verdict["center_error"])
            bias = bias + np.array([0.4 * error[0], 0.8 * error[1]])
            logger.info(
                f"Container pick: {slot} slot round {round_index} off by {np.round(error * 100, 1).tolist()} cm; "
                f"re-picking with bias {np.round(bias * 100, 1).tolist()} cm"
            )
        return SkillResult.fail(
            "EXECUTION_FAILED",
            f"Container still outside the {slot} slot after {len(rounds)} rounds: {rounds[-1]}",
        )

    # internals: perception

    def _scan(
        self, prompts: list[str] | None
    ) -> tuple[str, NDArray[np.float32], Any, dict[str, Any]] | SkillResult[ManipulationSkillError]:
        last_error = "nothing detected"
        for _attempt in range(self.config.scan_attempts):
            for object_id, cloud in self._candidate_clouds(prompts):
                points = cloud.points_f32()
                if len(points) < 50:
                    continue
                plausible, why, rim = self._plausible(cloud)
                if plausible and rim is not None:
                    container = describe_container(points, rim, self.config.opening_min_drop)
                    self._last = {**self._last, "container": container, "rim": rim}
                    return object_id, points, cloud, container
                last_error = why
                if rim is not None:
                    # re-centre the survey over what was seen and look again
                    self._survey_over(np.asarray(rim["footprint_centroid"], dtype=float))
            time.sleep(0.5)
        return SkillResult.fail("OBJECT_NOT_DETECTED", f"No usable container: {last_error}")

    def _candidate_clouds(self, prompts: list[str] | None) -> list[tuple[str, Any]]:
        if self._tabletop is not None:
            cloud = self._tabletop.scan_object_cloud()
            return [("colour-blob", cloud)] if cloud is not None else []
        prompts = prompts or list(self.config.prompts)
        try:
            detections = self._scene.scan_scene(text=prompts)
        except RuntimeError as exc:
            logger.warning(f"Container pick: scan failed: {exc}")
            return []
        clouds = []
        for detection in detections.detections[: detections.detections_length]:
            cloud = self._scene.get_object_pointcloud_by_object_id(str(detection.id))
            if cloud is not None:
                clouds.append((str(detection.id), cloud))
        return clouds

    def _plausible(self, cloud: Any) -> tuple[bool, str, dict[str, Any] | None]:
        describe = getattr(self._grasps, "describe_rim", None)
        if describe is None:
            return True, "", None
        try:
            rim = describe(cloud)
        except Exception as exc:
            return False, f"rim fit failed: {exc}", None
        rim = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rim.items()}
        extents = sorted(float(v) for v in rim["rect_extents"])
        center = np.asarray(rim["rect_center"], dtype=float)
        centroid = np.asarray(rim["footprint_centroid"], dtype=float)
        offset = float(np.linalg.norm(center - centroid))
        config = self.config
        if extents[1] < config.container_long_min:
            return False, f"rim long side {extents[1] * 100:.0f} cm is shorter than expected", rim
        if not config.container_short_range[0] <= extents[0] <= config.container_short_range[1]:
            return False, f"rim short side {extents[0] * 100:.0f} cm is out of range", rim
        if offset > config.rim_center_tolerance:
            return False, f"rim centre is {offset * 100:.1f} cm off the footprint centroid", rim
        return True, "", rim

    def _slot_verdict(
        self, seen: dict[str, Any], geometry: dict[str, Any], tape_half: float, toward_base: float
    ) -> dict[str, Any]:
        inner_x = (geometry["x"][0] + tape_half, geometry["x"][1] - tape_half)
        inner_y = (geometry["y"][0] + tape_half, geometry["y"][1] - tape_half)
        corners = np.asarray(seen["corners"], dtype=float)
        inside_flags = [
            bool(inner_x[0] <= c[0] <= inner_x[1] and inner_y[0] <= c[1] <= inner_y[1])
            for c in corners
        ]
        opening = np.asarray(seen["opening_dir"], dtype=float)
        opening_ok = bool(seen["opening_known"] and opening[1] * toward_base > 0.7)
        center_error = (
            np.asarray(seen["center"], dtype=float) - np.asarray(geometry["center"], dtype=float)
        ).tolist()
        yaw_error = math.degrees(wrap_angle(2.0 * (seen["yaw"] - math.pi / 2.0)) / 2.0)
        return {
            "success": all(inside_flags) and opening_ok,
            "corners_inside": inside_flags,
            "opening_toward_base": opening_ok,
            "center_error": center_error,
            "yaw_error_degrees": yaw_error,
            "margin_x": float(min(min(c[0] - inner_x[0], inner_x[1] - c[0]) for c in corners)),
            "margin_y": float(min(min(c[1] - inner_y[0], inner_y[1] - c[1]) for c in corners)),
        }

    # internals: grasp and release

    def _pick(
        self,
        object_id: str,
        points: NDArray[np.float32],
        cloud: Any,
        container: dict[str, Any],
        turn_after: float,
    ) -> SkillResult[ManipulationSkillError]:
        try:
            candidates = self._grasps.propose_grasps(cloud)
        except (RuntimeError, ValueError) as exc:
            return SkillResult.fail("GRASP_GENERATION_FAILED", str(exc))
        if candidates.header.frame_id != self.config.planning_frame or not candidates.candidates:
            return SkillResult.fail("GRASP_GENERATION_FAILED", "No rim grasp in the planning frame")
        current_yaw = self._tcp_yaw()
        wrist_now = self._wrist_position()
        limits = self._wrist_limits()
        center = np.asarray(container["center"], dtype=float)
        ordered = sorted(
            candidates.candidates[:4],
            key=lambda c: abs(
                math.hypot(c.pose.position.x, c.pose.position.y) - self.config.preferred_reach
            ),
        )
        attempts: list[dict[str, Any]] = []
        for candidate in ordered:
            position = candidate.pose.position
            grasp_z = float(position.z)
            if self.config.min_z is not None:
                grasp_z = max(grasp_z, self.config.min_z)
            raw_yaw = float(candidate.pose.orientation.to_euler().z)
            if wrist_now is not None and limits is not None:
                yaw = wrist_feasible_yaw(
                    raw_yaw,
                    current_yaw,
                    wrist_now,
                    turn_after,
                    limits,
                    self.config.wrist_joint_sign,
                )
                if yaw is None:
                    attempts.append(
                        {
                            "grasp": [position.x, position.y, grasp_z],
                            "step": "wrist",
                            "error": "no wrist-feasible yaw",
                        }
                    )
                    continue
            else:
                yaw = nearest_equivalent_yaw(raw_yaw, current_yaw)
            grasp = np.array([float(position.x), float(position.y), grasp_z])
            pregrasp = grasp + np.array([0.0, 0.0, self.config.pregrasp_offset])
            failure = self._cartesian_to(pregrasp)
            if failure is not None:
                attempts.append(
                    {
                        "grasp": grasp.tolist(),
                        "yaw": yaw,
                        "step": "approach",
                        "error": failure.message,
                    }
                )
                continue
            if not self._rotate_to(yaw, pregrasp):
                attempts.append(
                    {
                        "grasp": grasp.tolist(),
                        "yaw": yaw,
                        "step": "rotate",
                        "error": "plan rejected",
                    }
                )
                continue
            failure = self._cartesian_to(grasp)
            if failure is not None:
                attempts.append(
                    {
                        "grasp": grasp.tolist(),
                        "yaw": yaw,
                        "step": "descend",
                        "error": failure.message,
                    }
                )
                self._cartesian_to(pregrasp)
                continue
            reading = self._gripper_to(self.config.grasp_verification.closed_position)
            why = grasp_failure(reading, self.config.grasp_verification)
            if why is not None:
                attempts.append(
                    {"grasp": grasp.tolist(), "yaw": yaw, "step": "close", "error": why}
                )
                self._gripper_to(self.config.grasp_verification.open_position)
                self._cartesian_to(pregrasp)
                continue
            if self.config.pause_mapping_while_holding and self._map_pause is not None:
                self._map_pause.set_paused(True)
            self._cartesian_to(pregrasp)
            lifted = grasp + np.array([0.0, 0.0, self.config.carry_height])
            self._cartesian_to(lifted)
            time.sleep(self.config.hold_seconds)
            held = self._gripper()
            still_held = (
                held is not None
                and grasp_failure(_settled(held), self.config.grasp_verification) is None
            )
            offset_dir = center - grasp[:2]
            offset_dir = offset_dir / max(float(np.linalg.norm(offset_dir)), 1e-6)
            self._holding = {
                "object_id": object_id,
                "grasp": grasp.tolist(),
                "grasp_yaw": self._tcp_yaw(),
                "offset_dir": offset_dir.tolist(),
                "rim_half_width": float(np.linalg.norm(center - grasp[:2])),
                "opening_dir": container["opening_dir"],
                "opening_known": container["opening_known"],
                "half_length": container["half_length"],
                "turned": 0.0,
                "container": container,
                "closed_reading": reading.position,
            }
            self._last = {
                **self._last,
                **self._holding,
                "held_after_lift": held,
                "attempts": attempts,
            }
            if not still_held:
                self._holding = None
                self._resume_mapping()
                return SkillResult.fail(
                    "GRASP_VERIFICATION_FAILED",
                    f"Container slipped during the lift (readback {held})",
                )
            return SkillResult.ok(
                "Container lifted",
                object_id=object_id,
                grasp=grasp.tolist(),
                yaw=yaw,
                gripper=held,
                opening_known=container["opening_known"],
                attempts=attempts,
            )
        self._last = {**self._last, "attempts": attempts, "container": container}
        return SkillResult.fail("PLANNING_FAILED", f"No wall could be grasped: {attempts}")

    def _landing_offset(self) -> float:
        assert self._holding is not None
        if self.config.landing_offset is not None:
            return float(self.config.landing_offset)
        return float(self._holding["rim_half_width"])

    def _release_and_lift(self) -> dict[str, Any]:
        """Lower to the grasp height, open, back out, lift, resume mapping."""
        assert self._holding is not None
        held = self._holding
        tcp = self._tcp()
        # The jaws may have slid up to the rim lip, so the container hangs lower
        # than it was grasped; lowering back to the grasp height lets it touch
        # down early and the jaws slide back down the wall.
        self._cartesian_to(np.array([tcp[0], tcp[1], held["grasp"][2] + 0.005]))
        self._gripper_to(self.config.grasp_verification.open_position)
        result: dict[str, Any] = {"exit": "none"}
        if held["opening_known"]:
            # Out over the low opening end: the finger inside the container cannot
            # pass a full-height end wall without pushing the container along.
            direction = rotate_xy(np.array(held["opening_dir"]), held["turned"])
            self._cartesian_to(self._tcp() + np.array([0.0, 0.0, self.config.exit_raise]))
            tcp = self._tcp()
            for distance in (
                held["half_length"] + self.config.exit_margin,
                held["half_length"] + self.config.exit_margin / 2.0,
            ):
                end = tcp[:2] + direction * distance
                if self._cartesian_to(np.array([end[0], end[1], tcp[2]])) is None:
                    result = {"exit": "opening", "distance": distance}
                    break
        else:
            # Unknown opening: along the grasped wall toward the base, as learned
            # for lifts (the container end there may push back; the caller sees it).
            yaw = self._tcp_yaw()
            along = np.array([math.cos(yaw), math.sin(yaw)])
            distance = held["half_length"] + self.config.slide_margin
            tcp = self._tcp()
            options = []
            for sign in (1.0, -1.0):
                end = tcp[:2] + sign * along * distance
                if (
                    inside(np.array([end[0], end[1], tcp[2]]), self.config.workspace_box)
                    and math.hypot(end[0], end[1]) <= self.config.reach_max - 0.06
                ):
                    options.append((math.hypot(end[0], end[1]), sign))
            if options:
                sign = min(options)[1]
                end = tcp[:2] + sign * along * distance
                if self._cartesian_to(np.array([end[0], end[1], tcp[2]])) is None:
                    result = {"exit": "wall", "distance": sign * distance}
        tcp = self._tcp()
        self._cartesian_to(tcp + np.array([0.0, 0.0, self.config.lift_height]))
        self._resume_mapping()
        return result

    # internals: robot state and motion

    def _resolve_group(self) -> PlanningGroupID | None:
        if self._group is None:
            groups = [
                g
                for g in self._manipulation.list_planning_groups()
                if g.has_gripper and g.tip_frame
            ]
            self._group = groups[0].id if len(groups) == 1 else None
        return self._group

    def _require_group(self) -> PlanningGroupID:
        group = self._resolve_group()
        if group is None:
            raise RuntimeError("No gripper-capable planning group")
        return group

    def _state(self) -> Any:
        return self._manipulation.get_state().groups[self._require_group()]

    def _tcp(self) -> NDArray[np.float64]:
        p = self._state().end_effector_pose.position
        return np.array([p.x, p.y, p.z], dtype=float)

    def _tcp_yaw(self) -> float:
        return float(self._state().end_effector_pose.orientation.to_euler().z)

    def _joints(self) -> tuple[list[str], list[float]]:
        js = self._state().joints
        return list(js.name), [float(v) for v in js.position]

    def _wrist_position(self) -> float | None:
        names, positions = self._joints()
        if self.config.wrist_joint not in names:
            return None
        return positions[names.index(self.config.wrist_joint)]

    def _wrist_limits(self) -> tuple[float, float] | None:
        if self._guard is None:
            return None
        limits = self._guard.joint_limits(self.config.wrist_joint)
        if limits is None:
            return None
        margin = self.config.wrist_joint_margin
        return (limits[0] + margin, limits[1] - margin)

    def _gripper(self) -> float | None:
        value = self._state().gripper_position
        return None if value is None else float(value)

    def _gripper_to(self, position: float) -> Any:
        self._manipulation.set_gripper_position(position, self._require_group())
        return await_gripper_settle(self._gripper, position, self.config.grasp_verification)

    def _resume_mapping(self) -> None:
        if self._map_pause is not None:
            self._map_pause.set_paused(False)

    def _survey_target_xy(self) -> NDArray[np.float64]:
        container = (self._holding or self._last or {}).get("container")
        if container:
            return np.asarray(container["center"], dtype=float)
        tcp = self._tcp()
        return np.array([tcp[0], tcp[1]])

    def _survey_yaw(self) -> float:
        yaw = self.config.survey_yaw
        if self.config.survey_camera_offset is None:
            return yaw
        current = self._tcp_yaw()
        return min((yaw, wrap_angle(yaw + math.pi)), key=lambda y: abs(wrap_angle(y - current)))

    def _camera_offset(self, yaw: float) -> NDArray[np.float64]:
        if self.config.survey_camera_offset is None:
            return -np.asarray(self.config.survey_offset_xy, dtype=float)
        return rotate_xy(
            np.asarray(self.config.survey_camera_offset, dtype=float), yaw - self.config.survey_yaw
        )

    def _survey_over(
        self, target_xy: NDArray[np.float64]
    ) -> SkillResult[ManipulationSkillError] | None:
        """Put the wrist camera above target_xy at the survey height, tool pointing down."""
        config = self.config
        yaw = self._survey_yaw()
        tcp_xy = np.asarray(target_xy, dtype=float) - self._camera_offset(yaw)
        radius = math.hypot(tcp_xy[0], tcp_xy[1])
        if radius > config.survey_reach:
            tcp_xy = tcp_xy * config.survey_reach / radius
        tcp_xy = np.array(
            [
                np.clip(tcp_xy[0], *config.survey_x_range),
                np.clip(tcp_xy[1], *config.survey_y_range),
            ]
        )
        failure = self._cartesian_to(np.array([tcp_xy[0], tcp_xy[1], config.survey_height]))
        if failure is not None:
            return failure
        if not self._rotate_to(yaw, self._tcp()):
            logger.warning("Container pick: survey wrist yaw not restored; scanning as is")
        return None

    def _cartesian_to(
        self, target: NDArray[np.float64]
    ) -> SkillResult[ManipulationSkillError] | None:
        """Straight-line move, vertical leg first when going up, last when going down."""
        config = self.config
        if not inside(target, config.workspace_box):
            return SkillResult.fail(
                "PLANNING_FAILED",
                f"target {np.round(target, 3).tolist()} is outside the workspace box",
            )
        if math.hypot(target[0], target[1]) > config.reach_max:
            return SkillResult.fail(
                "PLANNING_FAILED", f"target {np.round(target, 3).tolist()} is beyond reach"
            )
        if config.min_z is not None and target[2] < config.min_z:
            return SkillResult.fail("PLANNING_FAILED", f"target z {target[2]:.3f} is below min_z")
        current = self._tcp()
        delta = target - current
        legs: list[NDArray[np.float64]]
        if delta[2] > 0.0:
            legs = [np.array([0.0, 0.0, delta[2]]), np.array([delta[0], delta[1], 0.0])]
        else:
            legs = [np.array([delta[0], delta[1], 0.0]), np.array([0.0, 0.0, delta[2]])]
        for leg in legs:
            if np.linalg.norm(leg) < 1e-4:
                continue
            result = self._manipulation.move_linear(
                float(leg[0]),
                float(leg[1]),
                float(leg[2]),
                self._require_group(),
                check_collision=False,
                speed_scale=config.cartesian_speed_scale,
                blocking=True,
            )
            if result.execution is None or not result.execution.succeeded:
                status = result.execution.status if result.execution else result.plan.status
                return SkillResult.fail("EXECUTION_FAILED", f"linear move failed: {status}")
        return None

    def _plan(self, kind: str, target: Any) -> PlanResult | None:
        """Plan to a pose or joint target; a planner exception clears the pending plan
        so the manipulation module leaves its PLANNING state."""
        try:
            if kind == "pose":
                return self._manipulation.plan_to_poses(
                    {self._require_group(): target}, speed_scale=self.config.plan_speed_scale
                )
            return self._manipulation.plan_to_joints(
                {self._require_group(): target}, speed_scale=self.config.plan_speed_scale
            )
        except Exception as exc:
            logger.warning(
                f"Container pick: planner raised {str(exc)[:120]}; clearing the pending plan"
            )
            self._manipulation.clear_planned_path()
            return None

    def _execute_checked(self, plan: PlanResult | None) -> bool:
        if plan is None or not plan.succeeded or plan.plan is None:
            if plan is not None:
                logger.info(f"Container pick: plan failed: {plan.message}")
            return False
        assert self._guard is not None
        why = self._guard.check_path(plan.plan.path)
        if why is not None:
            self._manipulation.clear_planned_path()
            logger.warning(
                f"Container pick: REJECTED plan ({len(plan.plan.path)} waypoints): {why}"
            )
            return False
        execution = self._manipulation.execute(blocking=True)
        return bool(execution.succeeded)

    def _rotate_to(self, yaw: float, position: NDArray[np.float64]) -> bool:
        """Rotate the wrist in place to ``yaw`` (tool pointing down) with a checked plan."""
        if abs(wrap_angle(yaw - self._tcp_yaw())) < 0.05:
            return True
        pose = PoseStamped(
            frame_id=self.config.planning_frame,
            position=Vector3(*position.tolist()),
            orientation=Quaternion.from_euler(Vector3(-math.pi, 0.0, yaw)),
        )
        return self._execute_checked(self._plan("pose", pose))

    def _turn_wrist(self, delta: float) -> tuple[bool, float]:
        """Turn the tool about the vertical axis by ``delta`` (planning-frame radians)
        with a guarded wrist-joint move. Returns (ok, applied)."""
        before = self._tcp_yaw()
        if abs(delta) < 0.02:
            return True, 0.0
        names, positions = self._joints()
        if self.config.wrist_joint not in names:
            return False, 0.0
        index = names.index(self.config.wrist_joint)
        limits = self._wrist_limits()
        wanted = delta
        sign = self.config.wrist_joint_sign
        for _attempt in range(2):
            joint_delta = wanted / sign
            trial = list(positions)
            trial[index] = positions[index] + joint_delta
            if limits is not None and not limits[0] <= trial[index] <= limits[1]:
                logger.warning(
                    f"Container pick: wrist turn to {trial[index]:.2f} rad is outside {limits}; refused"
                )
                return False, wrap_angle(self._tcp_yaw() - before)
            if not self._execute_checked(
                self._plan("joints", JointState(name=names, position=trial))
            ):
                return False, wrap_angle(self._tcp_yaw() - before)
            applied = wrap_angle(self._tcp_yaw() - before)
            if abs(wrap_angle(applied - delta)) < math.radians(10.0):
                return True, applied
            # turned the other way: the sign assumption is wrong for this tool pose
            logger.warning("Container pick: wrist turned the other way; correcting")
            positions = self._joints()[1]
            sign = -sign
            wanted = wrap_angle(delta - applied)
        return False, wrap_angle(self._tcp_yaw() - before)


def _settled(position: float) -> Any:
    from dimos.manipulation.grasp_verification import GripperSettle

    return GripperSettle(settled=True, position=position, moved=True, elapsed=0.0)


container_pick = ContainerPickModule.blueprint
