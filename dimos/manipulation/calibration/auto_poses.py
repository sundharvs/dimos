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

"""Where to put a wrist camera for an automatic eye-in-hand calibration.

Two stages, because the second needs a rough answer to the question being asked.

BOOTSTRAP
    Small turns of the gripper about its own axes. They need nothing but the
    current pose, keep a board that is already in view in view, and turn about
    three orthogonal axes, which is enough for a coarse solve.

VIEWS
    With that coarse camera mount and board pose, look-at poses on a cone about
    the board's normal: every view points the optical axis at the board centre,
    so the board stays framed, and the views differ in tilt, direction, distance
    and roll about the optical axis. Those differences are what constrain the
    solve; plain translation constrains nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray

from dimos.manipulation.calibration.hand_eye_solver import make_T

Matrix4 = NDArray[np.float64]


def rotation_about(axis: NDArray[np.float64], angle_rad: float) -> NDArray[np.float64]:
    """Rotation matrix for a turn of `angle_rad` about `axis` (Rodrigues)."""
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    skew = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    rotation: NDArray[np.float64] = (
        np.eye(3) + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * skew @ skew
    )
    return rotation


def _align(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """The smallest rotation taking unit vector `source` onto unit vector `target`."""
    axis = np.asarray(np.cross(source, target), dtype=np.float64)
    sin = float(np.linalg.norm(axis))
    cos = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sin < 1e-9:
        if cos > 0:
            return np.eye(3)
        # Opposite: any perpendicular axis does.
        perpendicular = np.asarray(np.cross(source, [1.0, 0.0, 0.0]), dtype=np.float64)
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.asarray(np.cross(source, [0.0, 1.0, 0.0]), dtype=np.float64)
        return rotation_about(perpendicular, math.pi)
    return rotation_about(axis, math.atan2(sin, cos))


def bootstrap_turns(tilt_deg: float, roll_deg: float) -> list[Matrix4]:
    """Gripper-frame turns about x, y and z, both ways: right-multiply onto a gripper pose.

    The turns are about the gripper's own origin, so a wrist camera near it
    swings its view by roughly range * tan(angle); keep them small.
    """
    turns: list[Matrix4] = []
    for axis, degrees in (((1.0, 0.0, 0.0), tilt_deg), ((0.0, 1.0, 0.0), tilt_deg)):
        for sign in (1.0, -1.0):
            turns.append(
                make_T(rotation_about(np.array(axis), sign * math.radians(degrees)), np.zeros(3))
            )
    for sign in (1.0, -1.0):
        turns.append(
            make_T(
                rotation_about(np.array([0.0, 0.0, 1.0]), sign * math.radians(roll_deg)),
                np.zeros(3),
            )
        )
    return turns


def board_center(base_T_board: Matrix4, size_m: tuple[float, float]) -> NDArray[np.float64]:
    """The board centre in the base frame; ChArUco's origin is a corner."""
    local = np.array([size_m[0] / 2.0, size_m[1] / 2.0, 0.0, 1.0])
    center: NDArray[np.float64] = (base_T_board @ local)[:3]
    return center


def board_normal_towards(base_T_board: Matrix4, point: NDArray[np.float64]) -> NDArray[np.float64]:
    """The board's normal, on the side `point` (the camera) is on."""
    normal = base_T_board[:3, 2].copy()
    if np.dot(normal, point - base_T_board[:3, 3]) < 0:
        normal = -normal
    return normal


def look_at_views(
    base_T_board: Matrix4,
    size_m: tuple[float, float],
    base_T_optical_start: Matrix4,
    count: int,
    max_tilt_deg: float,
    max_roll_deg: float,
    distance_scales: Sequence[float] = (1.0, 0.85, 1.15),
) -> list[Matrix4]:
    """Optical-frame poses looking at the board centre from a cone about its normal.

    The range is the start pose's range, scaled. One ring of views at the full
    tilt, then one at half of it offset by half a step, each stepping evenly
    around the normal so consecutive moves stay short. Roll about the optical
    axis cycles through a third, two thirds and all of max_roll, one way on the
    outer ring and the other way on the inner. Each view's orientation is the start orientation
    turned the least amount that points it at the centre, then rolled, so the
    image stays upright-ish and the wrist does not wind up.

    Flipping tilt or roll sign view by view would make every consecutive turn
    mostly a roll about the normal, and the solver would see one axis.
    """
    if count < 1:
        return []
    center = board_center(base_T_board, size_m)
    start_position = base_T_optical_start[:3, 3]
    start_rotation = base_T_optical_start[:3, :3]
    normal = board_normal_towards(base_T_board, start_position)
    distance = float(np.linalg.norm(start_position - center))

    # An in-plane basis, anchored to where the camera starts so view 0 sits on
    # the start side of the cone.
    toward_start = start_position - center
    in_plane = toward_start - np.dot(toward_start, normal) * normal
    if np.linalg.norm(in_plane) < 1e-6:
        in_plane = np.cross(normal, start_rotation[:, 0])
    first = in_plane / np.linalg.norm(in_plane)
    second = np.cross(normal, first)

    outer = (count + 1) // 2
    views: list[Matrix4] = []
    for index in range(count):
        ring, step = (0, index) if index < outer else (1, index - outer)
        ring_size = outer if ring == 0 else count - outer
        tilt = math.radians(max_tilt_deg) * (1.0 if ring == 0 else 0.5)
        azimuth = 2.0 * math.pi * (step + 0.5 * ring) / ring_size
        roll_fraction = ((step % 3) + 1) / 3.0
        roll = math.radians(max_roll_deg) * roll_fraction * (1.0 if ring == 0 else -1.0)
        scale = distance_scales[index % len(distance_scales)]

        out = math.cos(azimuth) * first + math.sin(azimuth) * second
        direction = math.cos(tilt) * normal + math.sin(tilt) * out
        position = center + distance * scale * direction
        optical_z = -direction
        rotation = _align(start_rotation[:, 2], optical_z) @ start_rotation
        rotation = rotation_about(optical_z, roll) @ rotation
        views.append(make_T(rotation, position))
    return views
