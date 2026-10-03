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

"""Pick an open container (bin, box, tray) by its rim and set it down again.

The skill a parallel-jaw arm needs for a container it cannot span: scan, fit the
rim, straddle one wall from above, close, lift, and later lower, open and back the
open jaws out along the wall before lifting away (an open finger left inside the
container hooks the rim lip and takes the container along).

MOTION SAFETY
    Learned the hard way on 2026-10-03, when a sampling planner returned a path
    that swung an xArm7 up and over its base: every planned motion is checked
    before it is executed. Forward kinematics of every waypoint must keep the
    hand inside ``workspace_box`` and the elbow inside ``elbow_box``, the base
    joint may turn at most ``base_joint_max_excursion`` along one path, and the
    joint-space length is bounded. A rejected plan is cleared, never run. Every
    translation (approach, descent, lift, carry, return) is a straight-line
    Cartesian move whose target is box- and reach-checked first; the planner is
    only asked to rotate the wrist in place. Speeds are scaled down. Put a TCP
    box in the robot controller as well where the SDK offers one; this module
    is the second line of defence, not the first.

TILTED GRASPS
    A small 6-DoF arm cannot hold its tool straight down far above the table
    (the AgileX Piper's wrist pitch range ends about 11 cm up), which leaves no
    room to clear a rim and lift. The jaws still straddle a wall when the tool
    leans within the wall's own plane, so ``tool_tilts`` lists lean angles to
    try and ``check_reachability`` picks, per wall, the first lean and half-turn
    for which the pre-grasp, the grasp and the lifted pose all have an inverse
    kinematics solution. ``survey_joints`` replaces the top-down survey pose
    with a joint posture for a wrist camera that does not look along the tool.

WHAT IT NEEDS IN THE BLUEPRINT
    A ``ManipulationSpec`` (ManipulationModule), an ``ObjectSceneRegistrationSpec``
    for prompted scans, a ``GraspGenSpec`` that proposes rim grasps
    (RimGraspModule) and, optionally, a map-pause capable self filter so the
    carried container is not mapped as an obstacle along the carry.
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal, Protocol

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
    plan_speed_scale: float = Field(default=0.3, gt=0.0, le=1.0)
    cartesian_speed_scale: float = Field(default=0.3, gt=0.0, le=1.0)
    # TCP never goes below this planning-frame height (None disables).
    min_z: float | None = None
    # Survey pose: the camera looks straight down from here. survey_offset_xy is
    # where the TCP sits relative to the object so the wrist camera is above it.
    survey_height: float = 0.40
    survey_yaw: float = -1.6
    survey_offset_xy: tuple[float, float] = (0.0, 0.07)
    survey_reach: float = 0.52
    # A joint-space survey posture, in the planning group's joint order, for an
    # arm whose wrist camera does not look along the tool or that cannot hold
    # the tool straight down at survey height. The base joint is turned toward
    # the last seen container. None keeps the Cartesian top-down survey.
    survey_joints: list[float] | None = None
    # More joint postures to look from after the survey, for a container that
    # does not fit in one view: what each sees of the container is merged
    # before the rim is fitted. The base joint follows the container here too.
    survey_extra_views: list[list[float]] = Field(default_factory=list)
    # "cartesian": straight-line move to the pre-grasp, then rotate in place.
    # "plan": one guarded plan straight to the pre-grasp pose, for arms that
    # cannot translate at the survey orientation.
    approach: Literal["cartesian", "plan"] = "cartesian"
    # Lean of the tool within the grasped wall's plane, radians about the jaw
    # closing axis (the tool Y), tried in order. 0 is straight down.
    tool_tilts: list[float] = Field(default_factory=lambda: [0.0])
    # Choose the lean and the half-turn of the yaw by inverse kinematics of the
    # pre-grasp, grasp and lifted poses instead of taking the first of each.
    check_reachability: bool = False
    # Points fixed to the tool (tip frame, metres), e.g. a wrist camera, that
    # must stay keepout_clearance above the rim or outside it at the grasp pose.
    tool_keepout_points: list[tuple[float, float, float]] = Field(default_factory=list)
    keepout_clearance: float = Field(default=0.03, ge=0.0)
    # How many of the proposed walls to try, best first.
    max_candidates: int = Field(default=3, ge=1)
    pregrasp_offset: float = Field(default=0.10, gt=0.0)
    lift_height: float = Field(default=0.15, ge=0.0)
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
    # Clearance added past the container's half length when backing out.
    slide_margin: float = 0.06
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


def nearest_equivalent_yaw(yaw: float, reference: float) -> float:
    """A parallel jaw is symmetric under a half turn: the yaw nearest ``reference``."""
    return wrap_angle(
        min(
            (yaw + k * math.pi for k in (-2, -1, 0, 1, 2)),
            key=lambda y: abs(wrap_angle(y - reference)),
        )
    )


def tool_rotation(yaw: float, tilt: float = 0.0) -> NDArray[np.float64]:
    """Tool pointing down with its X axis at ``yaw``, then leaned ``tilt`` about its Y.

    The jaws close along the tool Y, which stays horizontal, so a lean keeps
    them straddling a wall that runs along the tool X.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    ct, st = math.cos(tilt), math.sin(tilt)
    down = np.array([[cy, sy, 0.0], [sy, -cy, 0.0], [0.0, 0.0, -1.0]])
    lean = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return np.asarray(down @ lean, dtype=np.float64)


def rotation_angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """The angle of the rotation taking orientation ``a`` to ``b``."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(a.T @ b)) - 1.0) / 2.0)))


def keepout_violated(
    rim: dict[str, Any],
    points: list[tuple[float, float, float]],
    clearance: float,
    grasp: NDArray[np.float64],
    rotation: NDArray[np.float64],
) -> bool:
    """Whether a tool-mounted point would sit on or inside the rim at the grasp pose.

    ``rim`` is RimGraspModule.describe_rim's rectangle; ``points`` are in the
    tool frame. A point is fine ``clearance`` above the rim top, or outside the
    rim rectangle by the same margin.
    """
    center = np.asarray(rim["rect_center"], dtype=float)
    axes = np.asarray(rim["rect_axes"], dtype=float)
    half = np.asarray(rim["rect_extents"], dtype=float) / 2.0 + clearance
    for point in points:
        world = grasp + rotation @ np.asarray(point, dtype=float)
        if world[2] >= float(rim["rim_top_z"]) + clearance:
            continue
        if np.all(np.abs(axes @ (world[:2] - center)) <= half):
            return True
    return False


class PathGuard:
    """Forward-kinematic checks of a joint path against the workspace boxes."""

    # Inverse kinematics for reachable(): damped least squares from a few seeds.
    IK_SEEDS = 8
    IK_ITERATIONS = 150
    IK_TOLERANCE = 1e-3
    IK_LIMIT_MARGIN = 0.03

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
        self.v_index = {
            self.model.names[j]: self.model.joints[j].idx_v for j in range(1, self.model.njoints)
        }

    def reachable(
        self,
        names: list[str],
        seed: list[float],
        position: NDArray[np.float64],
        rotation: NDArray[np.float64],
    ) -> bool:
        """Whether the tip link (the last hand link) can take this pose inside the joint limits.

        Only the joints in ``names`` move. An answer, not a plan: it says a
        configuration exists, not that a straight line to it does.
        """
        pin = self._pin
        tip = self.frames[self.config.hand_links[-1]]
        q_cols = [self.q_index[n] for n in names if n in self.q_index]
        v_cols = [self.v_index[n] for n in names if n in self.v_index]
        lower = self.model.lowerPositionLimit[q_cols] + self.IK_LIMIT_MARGIN
        upper = self.model.upperPositionLimit[q_cols] - self.IK_LIMIT_MARGIN
        target = pin.SE3(rotation, np.asarray(position, dtype=float))
        rng = np.random.default_rng(0)
        start = np.clip(
            np.array([v for n, v in zip(names, seed, strict=False) if n in self.q_index]),
            lower,
            upper,
        )
        for attempt in range(self.IK_SEEDS):
            q = np.zeros(self.model.nq)
            q[q_cols] = start if attempt == 0 else rng.uniform(lower, upper)
            for _ in range(self.IK_ITERATIONS):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                error = pin.log(self.data.oMf[tip].actInv(target)).vector
                if float(np.linalg.norm(error)) < self.IK_TOLERANCE:
                    return True
                jacobian = pin.computeFrameJacobian(self.model, self.data, q, tip, pin.LOCAL)[
                    :, v_cols
                ]
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
                q[q_cols] = np.clip(q[q_cols] + 0.5 * step, lower, upper)
        return False

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
        if config.base_joint in names:
            base = [float(w.position[names.index(config.base_joint)]) for w in path]
            if max(base) - min(base) > config.base_joint_max_excursion:
                return f"{config.base_joint} excursion {max(base) - min(base):.2f} rad"
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
        if length > config.path_max_length:
            return f"path length {length:.2f} rad"
        return None


class ContainerPickModule(Module):
    """Rim-grasp pick and set-down of an open container with guarded motions."""

    config: ContainerPickConfig

    _manipulation: ManipulationSpec
    _scene: ObjectSceneRegistrationSpec
    _grasps: GraspGenSpec
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

    @rpc
    def preview(self, prompt: str = "") -> dict[str, Any]:
        """Survey and scan, then report the rim and how each wall would be grasped.

        Moves only to the survey postures. For bring-up: check the rim size and
        the chosen yaw and lean per wall before the first pick.
        """
        survey = self.survey()
        if not survey.success:
            return {"error": survey.message}
        scan = self._scan([prompt] if prompt.strip() else None)
        if isinstance(scan, SkillResult):
            return {"error": scan.message}
        object_id, points, cloud = scan
        current_yaw = self._tcp_yaw()
        walls = []
        for candidate in self._grasps.propose_grasps(cloud).candidates:
            position = candidate.pose.position
            grasp = np.array([float(position.x), float(position.y), float(position.z)])
            if self.config.min_z is not None:
                grasp[2] = max(grasp[2], self.config.min_z)
            wall_yaw = float(candidate.pose.orientation.to_euler().z)
            walls.append(
                {
                    "grasp": np.round(grasp, 4).tolist(),
                    "wall_yaw": wall_yaw,
                    "score": float(candidate.score),
                    "tool": self._choose_tool_pose(grasp, wall_yaw, current_yaw),
                }
            )
        return {
            "object_id": object_id,
            "footprint": self._footprint(points),
            "rim": self._last.get("rim"),
            "walls": walls,
        }

    # skills

    @skill(uses=[CAP_MOVEMENT])
    def survey(self) -> SkillResult[ManipulationSkillError]:
        """Move the wrist camera to look straight down over the workspace.

        Goes above the last seen container when there is one, by straight-line
        moves at reduced speed, then restores the top-down wrist orientation.
        """
        if self.config.survey_joints is not None:
            return self._survey_by_joints(list(self.config.survey_joints))
        target = self._survey_target()
        failure = self._cartesian_to(target)
        if failure is not None:
            return failure
        if not self._face_down():
            logger.warning("Container pick: wrist yaw not restored; scanning as is")
        return SkillResult.ok("At survey pose", target=target.tolist())

    @skill(uses=[CAP_MOVEMENT])
    def pick_up_container(self, prompt: str = "") -> SkillResult[ManipulationSkillError]:
        """Scan for the container, grasp one of its walls by the rim and lift it.

        Args:
            prompt: Object label for the detector; empty uses the configured prompts.
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
        object_id, points, cloud = scan
        footprint = self._footprint(points)
        try:
            candidates = self._grasps.propose_grasps(cloud)
        except (RuntimeError, ValueError) as exc:
            return SkillResult.fail("GRASP_GENERATION_FAILED", str(exc))
        if candidates.header.frame_id != self.config.planning_frame or not candidates.candidates:
            return SkillResult.fail("GRASP_GENERATION_FAILED", "No rim grasp in the planning frame")
        current_yaw = self._tcp_yaw()
        attempts: list[dict[str, Any]] = []
        for candidate in candidates.candidates[: self.config.max_candidates]:
            position = candidate.pose.position
            grasp_z = float(position.z)
            if self.config.min_z is not None:
                grasp_z = max(grasp_z, self.config.min_z)
            grasp = np.array([float(position.x), float(position.y), grasp_z])
            pregrasp = grasp + np.array([0.0, 0.0, self.config.pregrasp_offset])
            wall_yaw = float(candidate.pose.orientation.to_euler().z)
            choice = self._choose_tool_pose(grasp, wall_yaw, current_yaw)
            if choice is None:
                attempts.append(
                    {
                        "grasp": grasp.tolist(),
                        "yaw": wall_yaw,
                        "step": "reach",
                        "error": "no reachable tool orientation clears the rim",
                    }
                )
                continue
            yaw, tilt = choice
            if self.config.approach == "plan":
                if not self._plan_to(pregrasp, yaw, tilt):
                    attempts.append(
                        {
                            "grasp": grasp.tolist(),
                            "yaw": yaw,
                            "tilt": tilt,
                            "step": "approach",
                            "error": "plan rejected",
                        }
                    )
                    continue
            else:
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
                if not self._plan_to(pregrasp, yaw, tilt):
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
            lifted = pregrasp + np.array([0.0, 0.0, self.config.lift_height])
            self._cartesian_to(lifted)
            time.sleep(self.config.hold_seconds)
            held = self._gripper()
            still_held = (
                held is not None
                and grasp_failure(_settled(held), self.config.grasp_verification) is None
            )
            self._holding = {
                "object_id": object_id,
                "grasp": grasp.tolist(),
                "grasp_yaw": yaw,
                "tilt": tilt,
                "footprint": footprint,
                "closed_reading": reading.position,
            }
            self._last = {**self._holding, "held_after_lift": held, "attempts": attempts}
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
                tilt=tilt,
                gripper=held,
                footprint=footprint,
                attempts=attempts,
            )
        self._last = {"attempts": attempts, "footprint": footprint}
        return SkillResult.fail("PLANNING_FAILED", f"No wall could be grasped: {attempts}")

    @skill(uses=[CAP_MOVEMENT])
    def set_down_container(
        self, x: float | None = None, y: float | None = None
    ) -> SkillResult[ManipulationSkillError]:
        """Lower the held container onto the surface it came from and let go cleanly.

        Args:
            x: Planning-frame X for the container's footprint centre; default is where it was picked.
            y: Planning-frame Y for the footprint centre; default is where it was picked.
        """
        if self._holding is None:
            return SkillResult.fail("INVALID_STATE", "Nothing is held")
        held = self._holding
        grasp = np.array(held["grasp"])
        footprint = held["footprint"]
        tcp = self._tcp()
        target_xy = np.array(
            [footprint["cx"] if x is None else x, footprint["cy"] if y is None else y]
        )
        carry = target_xy - np.array([footprint["cx"], footprint["cy"]])
        carry = np.clip(carry, -0.15, 0.15)
        if np.linalg.norm(carry) > 0.01:
            failure = self._cartesian_to(tcp + np.array([carry[0], carry[1], 0.0]))
            if failure is not None:
                return failure
        # The jaws may have slid up to the rim lip, so the container hangs lower
        # than it was grasped; lowering back to the grasp height lets it touch
        # down early and the jaws slide back down the wall.
        tcp = self._tcp()
        lowered = np.array([tcp[0], tcp[1], grasp[2] + 0.005])
        failure = self._cartesian_to(lowered)
        if failure is not None:
            return failure
        self._gripper_to(self.config.grasp_verification.open_position)
        # Back out along the wall past the container's end, then lift.
        along = self._tool_x_horizontal()
        length = float(max(footprint.get("extents", [0.3, 0.1])))
        distance = length / 2.0 + self.config.slide_margin
        tcp = self._tcp()
        options = []
        for sign in (1.0, -1.0):
            end = tcp[:2] + sign * along * distance
            if (
                inside(np.array([end[0], end[1], tcp[2]]), self.config.workspace_box)
                and math.hypot(end[0], end[1]) <= self.config.reach_max - 0.06
                and self._reachable_here(np.array([end[0], end[1], tcp[2]]))
            ):
                options.append((math.hypot(end[0], end[1]), sign))
        slid = False
        if options:
            sign = min(options)[1]
            slid = self._cartesian_to(tcp + np.array([*(sign * along * distance), 0.0])) is None
        tcp = self._tcp()
        self._cartesian_to(tcp + np.array([0.0, 0.0, self.config.lift_height]))
        self._resume_mapping()
        self._last = {**self._last, "set_down": {"target_xy": target_xy.tolist(), "slid": slid}}
        self._holding = None
        return SkillResult.ok("Container set down", target_xy=target_xy.tolist(), slid_out=slid)

    @skill(uses=[CAP_MOVEMENT])
    def rotate_held_container(self, yaw_degrees: float) -> SkillResult[ManipulationSkillError]:
        """Turn the held container about the vertical axis with a guarded wrist move.

        Args:
            yaw_degrees: Rotation to apply, positive counter-clockwise seen from above.
        """
        if self._holding is None:
            return SkillResult.fail("INVALID_STATE", "Nothing is held")
        if abs(float(self._holding.get("tilt", 0.0))) > 0.05:
            # The wrist axis is not vertical when the tool leans.
            return SkillResult.fail(
                "INVALID_STATE",
                "A container held with a leaning tool cannot be turned by the wrist",
            )
        before = self._tcp_yaw()
        names, positions = self._joints()
        if self.config.wrist_joint not in names:
            return SkillResult.fail("ROBOT_NOT_FOUND", f"No joint {self.config.wrist_joint!r}")
        index = names.index(self.config.wrist_joint)
        delta = math.radians(yaw_degrees)
        for sign in (1.0, -1.0):  # the wrist axis points down with the tool facing the table
            trial = list(positions)
            trial[index] = positions[index] + sign * delta
            plan = self._manipulation.plan_to_joints(
                {self._require_group(): JointState(name=names, position=trial)},
                speed_scale=self.config.plan_speed_scale,
            )
            if self._execute_checked(plan):
                applied = wrap_angle(self._tcp_yaw() - before)
                if abs(wrap_angle(applied - delta)) < math.radians(10.0):
                    return SkillResult.ok("Rotated", applied_degrees=math.degrees(applied))
                # turned the other way; a second move of twice the delta brings it round
                delta = -delta
                positions = self._joints()[1]
                continue
        return SkillResult.fail("PLANNING_FAILED", "Wrist rotation rejected by the guards")

    # internals

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

    def _gripper(self) -> float | None:
        value = self._state().gripper_position
        return None if value is None else float(value)

    def _gripper_to(self, position: float) -> Any:
        self._manipulation.set_gripper_position(position, self._require_group())
        return await_gripper_settle(self._gripper, position, self.config.grasp_verification)

    def _resume_mapping(self) -> None:
        if self._map_pause is not None:
            self._map_pause.set_paused(False)

    def _survey_target(self) -> NDArray[np.float64]:
        footprint = (self._holding or self._last or {}).get("footprint")
        if footprint:
            x = footprint["cx"] + self.config.survey_offset_xy[0]
            y = footprint["cy"] + self.config.survey_offset_xy[1]
        else:
            tcp = self._tcp()
            x, y = float(tcp[0]), float(tcp[1])
        radius = math.hypot(x, y)
        if radius > self.config.survey_reach:
            x, y = x * self.config.survey_reach / radius, y * self.config.survey_reach / radius
        return np.array([x, y, self.config.survey_height])

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

    def _execute_checked(self, plan: PlanResult) -> bool:
        if not plan.succeeded or plan.plan is None:
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

    def _plan_to(self, position: NDArray[np.float64], yaw: float, tilt: float = 0.0) -> bool:
        """Move the tool to a pose (pointing down at ``yaw``, leaned ``tilt``) with a checked plan."""
        rotation = tool_rotation(yaw, tilt)
        pose_now = self._state().end_effector_pose
        turned = rotation_angle(pose_now.orientation.to_rotation_matrix(), rotation)
        if turned < 0.05 and float(np.linalg.norm(self._tcp() - position)) < 0.02:
            return True
        pose = PoseStamped(
            frame_id=self.config.planning_frame,
            position=Vector3(*position.tolist()),
            orientation=Quaternion.from_rotation_matrix(rotation),
        )
        plan = self._manipulation.plan_to_poses(
            {self._require_group(): pose}, speed_scale=self.config.plan_speed_scale
        )
        return self._execute_checked(plan)

    def _face_down(self) -> bool:
        return self._plan_to(self._tcp(), self.config.survey_yaw)

    def _survey_by_joints(self, target: list[float]) -> SkillResult[ManipulationSkillError]:
        names, positions = self._joints()
        if len(target) != len(names):
            return SkillResult.fail(
                "INVALID_STATE", f"survey_joints needs {len(names)} values (got {len(target)})"
            )
        footprint = (self._holding or self._last or {}).get("footprint")
        if footprint and self.config.base_joint in names:
            target[names.index(self.config.base_joint)] = math.atan2(
                footprint["cy"], footprint["cx"]
            )
        if max(abs(a - b) for a, b in zip(target, positions, strict=True)) < 0.01:
            return SkillResult.ok("At survey pose", joints=target)
        plan = self._manipulation.plan_to_joints(
            {self._require_group(): JointState(name=names, position=target)},
            speed_scale=self.config.plan_speed_scale,
        )
        if not self._execute_checked(plan):
            return SkillResult.fail("PLANNING_FAILED", "Survey move rejected by the guards")
        return SkillResult.ok("At survey pose", joints=target)

    def _choose_tool_pose(
        self, grasp: NDArray[np.float64], wall_yaw: float, current_yaw: float
    ) -> tuple[float, float] | None:
        """The (yaw, tilt) to grasp this wall with, or None when no option works."""
        config = self.config
        if not config.check_reachability:
            return nearest_equivalent_yaw(wall_yaw, current_yaw), config.tool_tilts[0]
        assert self._guard is not None
        names, seed = self._joints()
        up = np.array([0.0, 0.0, 1.0])
        poses = [
            grasp + up * config.pregrasp_offset,
            grasp,
            grasp + up * (config.pregrasp_offset + config.lift_height),
        ]
        radial = grasp[:2] / max(float(np.linalg.norm(grasp[:2])), 1e-9)
        for tilt in config.tool_tilts:
            yaws = [wrap_angle(wall_yaw), wrap_angle(wall_yaw + math.pi)]
            if abs(tilt) > 1e-6:
                # The tool tip leans toward sin(tilt) * X: prefer leaning away
                # from the base, which is where the reach is.
                yaws.sort(
                    key=lambda y: -math.sin(tilt)
                    * float(np.dot([math.cos(y), math.sin(y)], radial))
                )
            else:
                yaws.sort(key=lambda y: abs(wrap_angle(y - current_yaw)))
            for yaw in yaws:
                rotation = tool_rotation(yaw, tilt)
                if self._keepout_violated(grasp, rotation):
                    continue
                if all(self._guard.reachable(names, seed, p, rotation) for p in poses):
                    return yaw, tilt
        return None

    def _keepout_violated(self, grasp: NDArray[np.float64], rotation: NDArray[np.float64]) -> bool:
        rim = self._last.get("rim")
        if not rim:
            return False
        return keepout_violated(
            rim,
            self.config.tool_keepout_points,
            self.config.keepout_clearance,
            grasp,
            rotation,
        )

    def _tool_x_horizontal(self) -> NDArray[np.float64]:
        """Unit horizontal direction of the tool X axis: along the grasped wall."""
        x_axis = self._state().end_effector_pose.orientation.to_rotation_matrix()[:2, 0]
        return np.asarray(x_axis / max(float(np.linalg.norm(x_axis)), 1e-9), dtype=float)

    def _reachable_here(self, position: NDArray[np.float64]) -> bool:
        """Whether the tool can be at ``position`` in its present orientation."""
        if not self.config.check_reachability:
            return True
        assert self._guard is not None
        names, seed = self._joints()
        rotation = self._state().end_effector_pose.orientation.to_rotation_matrix()
        return self._guard.reachable(names, seed, position, rotation)

    def _scan(
        self, prompts: list[str] | None
    ) -> tuple[str, NDArray[np.float32], Any] | SkillResult[ManipulationSkillError]:
        prompts = prompts or list(self.config.prompts)
        last_error = "nothing detected"
        for _attempt in range(self.config.scan_attempts):
            try:
                detections = self._scene.scan_scene(text=prompts)
            except RuntimeError as exc:
                return SkillResult.fail("PERCEPTION_FAILED", str(exc))
            for detection in detections.detections[: detections.detections_length]:
                object_id = str(detection.id)
                cloud = self._scene.get_object_pointcloud_by_object_id(object_id)
                if cloud is None:
                    continue
                points = cloud.points_f32()
                if len(points) < 50:
                    continue
                footprint = self._footprint(points)
                self._last = {**self._last, "footprint": footprint}
                if self.config.survey_extra_views:
                    cloud = self._add_views(cloud, prompts)
                    points = cloud.points_f32()
                    footprint = self._footprint(points)
                    self._last = {**self._last, "footprint": footprint}
                plausible, why = self._plausible(cloud)
                if plausible:
                    return object_id, points, cloud
                last_error = why
                # re-centre the survey over what was seen and look again
                self.survey()
            time.sleep(0.5)
        return SkillResult.fail("OBJECT_NOT_DETECTED", f"No usable container: {last_error}")

    def _add_views(self, cloud: Any, prompts: list[str]) -> Any:
        """Look from each extra posture and merge what it shows of the same container."""
        center = np.median(cloud.points_f32()[:, :2], axis=0)
        for view in self.config.survey_extra_views:
            if not self._survey_by_joints(list(view)).success:
                logger.warning("Container pick: extra survey view rejected; skipping it")
                continue
            try:
                detections = self._scene.scan_scene(text=prompts)
            except RuntimeError as exc:
                logger.warning(f"Container pick: extra view scan failed: {exc}")
                continue
            best: Any = None
            for detection in detections.detections[: detections.detections_length]:
                seen = self._scene.get_object_pointcloud_by_object_id(str(detection.id))
                if seen is None or len(seen.points_f32()) < 50:
                    continue
                # The same container, not something else that matched the prompt.
                offset = np.linalg.norm(np.median(seen.points_f32()[:, :2], axis=0) - center)
                if offset < 0.25 and (best is None or len(seen) > len(best)):
                    best = seen
            if best is not None:
                cloud = cloud + best
        return cloud

    def _plausible(self, cloud: Any) -> tuple[bool, str]:
        describe = getattr(self._grasps, "describe_rim", None)
        if describe is None:
            return True, ""
        try:
            rim = describe(cloud)
        except Exception as exc:
            return False, f"rim fit failed: {exc}"
        extents = sorted(float(v) for v in rim["rect_extents"])
        center = np.asarray(rim["rect_center"], dtype=float)
        centroid = np.asarray(rim["footprint_centroid"], dtype=float)
        offset = float(np.linalg.norm(center - centroid))
        config = self.config
        if extents[1] < config.container_long_min:
            return False, f"rim long side {extents[1] * 100:.0f} cm is shorter than expected"
        if not config.container_short_range[0] <= extents[0] <= config.container_short_range[1]:
            return False, f"rim short side {extents[0] * 100:.0f} cm is out of range"
        if offset > config.rim_center_tolerance:
            return False, f"rim centre is {offset * 100:.1f} cm off the footprint centroid"
        self._last["rim"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rim.items()}
        return True, ""

    def _footprint(self, points: NDArray[np.float32]) -> dict[str, Any]:
        xy = points[:, :2].astype(np.float64)
        centered = xy - xy.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / max(len(xy), 1))
        long_axis = vectors[:, int(np.argmax(values))]
        proj = centered @ vectors
        extents = (proj.max(axis=0) - proj.min(axis=0)).tolist()
        return {
            "cx": float(np.median(points[:, 0])),
            "cy": float(np.median(points[:, 1])),
            "top_z": float(np.quantile(points[:, 2], 0.98)),
            "n": len(points),
            "yaw_deg": float(math.degrees(math.atan2(long_axis[1], long_axis[0])) % 180.0),
            "extents": sorted(extents, reverse=True),
        }


def _settled(position: float) -> Any:
    from dimos.manipulation.grasp_verification import GripperSettle

    return GripperSettle(settled=True, position=position, moved=True, elapsed=0.0)


container_pick = ContainerPickModule.blueprint
