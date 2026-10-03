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


class HeuristicGraspModuleConfig(ModuleConfig):
    # How far the planning tip sits behind the point between the jaws that should
    # meet the object's centre, along the approach. Zero when the tip is already
    # there, as for a TCP frame between the fingertips.
    tip_offset: float = Field(default=0.0, ge=0.0)
    # Planning tip to fingertips along the approach. When set, a short object is
    # grasped higher so the fingertips stop above its lowest point (the table).
    fingertip_depth: float | None = Field(default=None, gt=0.0)
    fingertip_clearance: float = Field(default=0.01, ge=0.0)
    # Height of the surface objects rest on. A camera looking down sees little
    # below an object's top, so with this set the object is taken to reach down
    # to the surface, and the fingertip clearance is kept above it.
    support_z: float | None = None
    # Added to the jaw yaw. A parallel-jaw grasp is unchanged by a half turn, so
    # pi picks the equivalent grasp for a wrist whose range is centred there.
    yaw_offset: float = 0.0


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
        center_xy = np.median(xy, axis=0)
        low_z, high_z = np.quantile(points[:, 2], [0.05, 0.95])
        if self.config.support_z is not None:
            low_z = min(self.config.support_z, high_z)
        grasp_z = float((low_z + high_z) / 2.0)
        tip_offset = self.config.tip_offset
        if self.config.fingertip_depth is not None:
            fingertips_past_grasp = self.config.fingertip_depth - tip_offset
            grasp_z = max(
                grasp_z, float(low_z) + self.config.fingertip_clearance + fingertips_past_grasp
            )
        # The approach is straight down, so the tip sits directly above the grasp.
        pose = Pose(
            Vector3(float(center_xy[0]), float(center_xy[1]), grasp_z + tip_offset),
            Quaternion.from_euler(
                Vector3(-math.pi, 0.0, self._narrow_axis_yaw(xy) + self.config.yaw_offset)
            ),
        )
        return GraspCandidateArray(
            Header(float(object_pointcloud.ts), object_pointcloud.frame_id),
            [GraspCandidate(pose, score=1.0)],
        )

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
