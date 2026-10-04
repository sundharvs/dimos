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

"""Deterministic grasp proposals for segmented object point clouds."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.manipulation_msgs.GraspCandidate import GraspCandidate
from dimos.msgs.manipulation_msgs.GraspCandidateArray import GraspCandidateArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Header import Header

# Share of points ignored at each end of a footprint axis when measuring its
# extent, so a few stray points do not move an edge.
_EXTENT_TRIM = 0.02


class HeuristicGraspModuleConfig(ModuleConfig):
    # How far the fingertips reach past the planning tip along the approach. When
    # set, a short object is grasped higher so the fingertips stop above its
    # lowest point (the table).
    fingertip_depth: float | None = Field(default=None, gt=0.0)
    fingertip_clearance: float = Field(default=0.01, ge=0.0)
    # Height of the surface objects rest on. A camera looking down sees little
    # below an object's top, so with this set the object is taken to reach down
    # to the surface, and the fingertip clearance is kept above it.
    support_z: float | None = None
    # Added to the jaw yaw. A parallel-jaw grasp is unchanged by a half turn, so
    # pi picks the equivalent grasp for a wrist whose range is centred there.
    yaw_offset: float = 0.0
    # What the jaws centre on: where the points are densest, or the middle of
    # the footprint measured along its own axes. The two agree for a cloud seen
    # from straight above. A camera off to one side also sees the near side wall,
    # a dense line of points that pulls the median toward that edge by more than
    # a narrow gripper's clearance.
    centering: Literal["median", "extent"] = "median"


class HeuristicGraspModule(Module, GraspGenSpec):
    """Generate one top-down parallel-jaw grasp from a gravity-aligned point cloud.

    The input frame's XY plane must be horizontal and its -Z axis must point down.
    """

    config: HeuristicGraspModuleConfig

    @rpc
    def propose_grasps(self, object_pointcloud: PointCloud2) -> GraspCandidateArray:
        if object_pointcloud.ts is None or not math.isfinite(float(object_pointcloud.ts)):
            raise ValueError("object pointcloud must have a finite timestamp")
        if not object_pointcloud.frame_id:
            raise ValueError("object pointcloud frame_id must not be empty")
        points = object_pointcloud.points_f32()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("object pointcloud must contain at least three XYZ points")
        if not np.all(np.isfinite(points)):
            raise ValueError("object pointcloud XYZ values must be finite floats in metres")

        xy = points[:, :2]
        if self.config.centering == "extent":
            center_xy = self._extent_center(xy)
        else:
            center_xy = np.median(xy, axis=0)
        low_z, high_z = np.quantile(points[:, 2], [0.05, 0.95])
        if self.config.support_z is not None:
            low_z = min(self.config.support_z, high_z)
        grasp_z = float((low_z + high_z) / 2.0)
        if self.config.fingertip_depth is not None:
            grasp_z = max(
                grasp_z,
                float(low_z) + self.config.fingertip_clearance + self.config.fingertip_depth,
            )
        pose = Pose(
            Vector3(float(center_xy[0]), float(center_xy[1]), grasp_z),
            Quaternion.from_euler(
                Vector3(-math.pi, 0.0, self._narrow_axis_yaw(xy) + self.config.yaw_offset)
            ),
        )
        return GraspCandidateArray(
            Header(float(object_pointcloud.ts), object_pointcloud.frame_id),
            [GraspCandidate(pose, score=1.0)],
        )

    @staticmethod
    def _extent_center(xy: NDArray[np.float32]) -> NDArray[np.float64]:
        mean = np.mean(xy, axis=0, dtype=np.float64)
        centered = xy - mean
        _, axes = np.linalg.eigh(centered.T @ centered)
        along_axes = centered @ axes
        low, high = np.quantile(along_axes, [_EXTENT_TRIM, 1.0 - _EXTENT_TRIM], axis=0)
        center: NDArray[np.float64] = mean + axes @ ((low + high) / 2.0)
        return center

    @staticmethod
    def _narrow_axis_yaw(xy: NDArray[np.float32]) -> float:
        centered = xy - np.mean(xy, axis=0)
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        if values[1] <= 0.0 or np.isclose(values[0], values[1], rtol=0.05):
            return 0.0
        narrow_axis = vectors[:, 0]
        yaw = math.atan2(float(narrow_axis[1]), float(narrow_axis[0])) - math.pi / 2.0
        # A parallel-jaw grasp is unchanged by a 180-degree wrist rotation.
        return (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
