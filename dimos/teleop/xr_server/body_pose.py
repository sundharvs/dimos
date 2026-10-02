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

"""Full-body skeleton (xr-robot-teleop-server bones) -> operator body-frame poses.

Ported from SafeMimic's ``XRRTCBodyPoseDevice`` via bilinear_flow_toy's
``xr_receiver.py``. Bones arrive in right-handed Z-up FLU (``z_up=True``).

Body frame, re-estimated every message, at the shoulder midpoint:
    y = right -> left shoulder, x = cross(-(shoulders - SpineMiddle), y), z = x cross y.

Hand frame, from the wrist and the four proximal finger joints:
    x = wrist -> palm centre, y = palm normal (SVD of the knuckle vectors) signed
    by cross(wrist->index, wrist->little), z = x cross y.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from xr_robot_teleop_server.schemas.body_pose import Bone
from xr_robot_teleop_server.schemas.openxr_skeletons import FullBodyBoneId

Vec3: TypeAlias = NDArray[np.float64]
Mat3: TypeAlias = NDArray[np.float64]
Side: TypeAlias = Literal["left", "right"]
# {finger: {joint_name: xyz}}; untracked joints are absent, never zeros.
FingerJoints: TypeAlias = dict[str, dict[str, Vec3]]

# SafeMimic pushes each fingertip 1 cm out along the bone's local y; the
# gesture thresholds were tuned with this offset in place.
_TIP_OFFSET_M = 0.01

# Our joint names -> OpenXR bone suffix.
_FINGERS: dict[str, dict[str, str]] = {
    "thumb": {"thumb_mcp": "ThumbProximal", "thumb_pip": "ThumbDistal", "thumb_tip": "ThumbTip"},
    "index": {
        "index_finger_mcp": "IndexProximal",
        "index_finger_pip": "IndexIntermediate",
        "index_finger_tip": "IndexTip",
    },
    "middle": {
        "middle_finger_mcp": "MiddleProximal",
        "middle_finger_pip": "MiddleIntermediate",
        "middle_finger_tip": "MiddleTip",
    },
    "ring": {
        "ring_finger_mcp": "RingProximal",
        "ring_finger_pip": "RingIntermediate",
        "ring_finger_tip": "RingTip",
    },
    "pinky": {
        "pinky_mcp": "LittleProximal",
        "pinky_pip": "LittleIntermediate",
        "pinky_tip": "LittleTip",
    },
}


@dataclass(frozen=True)
class XrBodyPose:
    """One decoded skeleton message, in the operator's body frame."""

    right_wrist_position: Vec3 | None = None
    # None when the hand frame could not be built; position is still usable.
    right_wrist_rotation: Mat3 | None = None
    # Thumb-tip <-> index-tip distance in metres; None when a tip is untracked.
    right_pinch_m: float | None = None
    left_fingers: FingerJoints = field(default_factory=dict)


def _bone(name: str) -> int:
    return int(FullBodyBoneId[f"FullBody_{name}"])


def _unit(v: NDArray[np.floating[Any]]) -> Vec3:
    return np.asarray(v / (np.linalg.norm(v) + 1e-8), dtype=np.float64)


def _project_so3(m: NDArray[np.floating[Any]]) -> Mat3:
    u, _, vt = np.linalg.svd(m)
    return np.asarray(u @ vt, dtype=np.float64)


def body_frame(positions: Mapping[int, Vec3]) -> tuple[Vec3, Mat3] | None:
    """(origin, R_world_body), or None if a shoulder or the spine is untracked."""
    ids = [_bone(n) for n in ("LeftArmUpper", "RightArmUpper", "SpineMiddle")]
    if any(i not in positions for i in ids):
        return None
    left_shoulder, right_shoulder, spine = (positions[i] for i in ids)
    center = (left_shoulder + right_shoulder) / 2.0
    y = _unit(left_shoulder - right_shoulder)
    x = _unit(np.cross(-(center - spine), y))
    z = _unit(np.cross(x, y))
    return center, _project_so3(np.column_stack([x, y, z]))


def hand_frame(positions: Mapping[int, Vec3], side: Side) -> Mat3 | None:
    """R_world_hand from the wrist and four proximal finger joints, or None."""
    prefix = side.capitalize()
    names = (
        "HandWrist",
        "HandIndexProximal",
        "HandMiddleProximal",
        "HandRingProximal",
        "HandLittleProximal",
    )
    ids = [_bone(prefix + n) for n in names]
    if any(i not in positions for i in ids):
        return None
    wrist, index, middle, ring, little = (positions[i] for i in ids)
    x = _unit((index + middle + ring + little) / 4.0 - wrist)
    knuckles = np.stack([index - wrist, middle - wrist, ring - wrist, little - wrist])
    normal = _unit(np.linalg.svd(knuckles, full_matrices=False)[2][-1])
    if np.dot(normal, np.cross(index - wrist, little - wrist)) < 0:
        normal = -normal
    z = _unit(np.cross(x, normal))
    return _project_so3(np.column_stack([x, normal, z]))


def finger_joints(
    positions: Mapping[int, Vec3], rotations: Mapping[int, NDArray[np.float64]], side: Side
) -> FingerJoints:
    """World-frame finger joints for the gesture classifier.

    PIP angles are frame-invariant, so world frame classifies the same as
    SafeMimic's hand-centric frame.
    """
    prefix = side.capitalize()
    sign = 1.0 if side == "left" else -1.0
    out: FingerJoints = {}
    for finger, joints in _FINGERS.items():
        found: dict[str, Vec3] = {}
        for joint_name, suffix in joints.items():
            bone_id = _bone(f"{prefix}Hand{suffix}")
            if bone_id not in positions:
                continue
            p = positions[bone_id]
            if suffix.endswith("Tip") and bone_id in rotations:
                offset = np.array([0.0, sign * _TIP_OFFSET_M, 0.0])
                p = p + Rotation.from_quat(rotations[bone_id]).as_matrix() @ offset
            found[joint_name] = p
        if found:
            out[finger] = found
    return out


def bones_to_body_pose(bones: Iterable[Bone]) -> XrBodyPose | None:
    """Decode one skeleton into body-frame poses, or None without a body frame."""
    positions: dict[int, Vec3] = {}
    rotations: dict[int, NDArray[np.float64]] = {}
    for bone in bones:
        positions[int(bone.id)] = np.asarray(bone.position, dtype=np.float64)
        rotations[int(bone.id)] = np.asarray(bone.rotation, dtype=np.float64)  # xyzw

    frame = body_frame(positions)
    if frame is None:
        return None
    origin, r_world_body = frame

    wrist_id = _bone("RightHandWrist")
    wrist_position = None
    wrist_rotation = None
    if wrist_id in positions:
        wrist_position = r_world_body.T @ (positions[wrist_id] - origin)
        r_world_hand = hand_frame(positions, "right")
        if r_world_hand is not None:
            wrist_rotation = r_world_body.T @ r_world_hand

    thumb, index = _bone("RightHandThumbTip"), _bone("RightHandIndexTip")
    pinch = (
        float(np.linalg.norm(positions[thumb] - positions[index]))
        if thumb in positions and index in positions
        else None
    )

    return XrBodyPose(
        right_wrist_position=wrist_position,
        right_wrist_rotation=wrist_rotation,
        right_pinch_m=pinch,
        left_fingers=finger_joints(positions, rotations, "left"),
    )
