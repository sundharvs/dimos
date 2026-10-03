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

from collections.abc import Callable, Iterator
import math
from typing import Any

import numpy as np
import pytest

from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec.utils import spec_annotation_compliance


def _cloud(
    points: np.ndarray, *, frame_id: str = "world", timestamp: float | None = 1.0
) -> PointCloud2:
    return PointCloud2.from_numpy(points.astype(np.float32), frame_id=frame_id, timestamp=timestamp)


@pytest.fixture
def module() -> Iterator[HeuristicGraspModule]:
    instance = HeuristicGraspModule()
    yield instance
    instance.stop()


@pytest.fixture
def make_module() -> Iterator[Callable[..., HeuristicGraspModule]]:
    instances: list[HeuristicGraspModule] = []

    def make(**config: Any) -> HeuristicGraspModule:
        instance = HeuristicGraspModule(**config)
        instances.append(instance)
        return instance

    yield make
    for instance in instances:
        instance.stop()


def test_heuristic_grasp_implements_grasp_provider_spec(module: HeuristicGraspModule) -> None:
    assert spec_annotation_compliance(module, GraspGenSpec)


def test_heuristic_grasp_proposes_centered_top_down_pose(module: HeuristicGraspModule) -> None:
    proposals = module.propose_grasps(
        _cloud(
            np.asarray(
                [
                    [-0.10, -0.02, 0.10],
                    [-0.10, 0.02, 0.10],
                    [0.10, -0.02, 0.20],
                    [0.10, 0.02, 0.20],
                ]
            )
        )
    )

    assert len(proposals.candidates) == 1
    assert proposals.header.frame_id == "world"
    assert proposals.header.timestamp == pytest.approx(1.0)
    pose = proposals.candidates[0].pose
    assert pose.position.x == pytest.approx(0.0)
    assert pose.position.y == pytest.approx(0.0)
    assert pose.position.z == pytest.approx(0.15)
    approach = pose.orientation.rotate_vector(Vector3(0.0, 0.0, 1.0))
    assert approach.z == pytest.approx(-1.0)
    assert proposals.candidates[0].score == pytest.approx(1.0)


def test_heuristic_grasp_aligns_jaw_axis_with_narrow_axis(module: HeuristicGraspModule) -> None:
    proposals = module.propose_grasps(
        _cloud(
            np.asarray(
                [
                    [-0.02, -0.10, 0.10],
                    [0.02, -0.10, 0.10],
                    [-0.02, 0.10, 0.20],
                    [0.02, 0.10, 0.20],
                ]
            )
        )
    )

    jaw_axis = proposals.candidates[0].pose.orientation.rotate_vector(Vector3(0.0, 1.0, 0.0))
    assert abs(jaw_axis.x) == pytest.approx(1.0)
    assert jaw_axis.y == pytest.approx(0.0, abs=1e-6)
    assert jaw_axis.z == pytest.approx(0.0, abs=1e-6)


def test_heuristic_grasp_canonicalizes_pca_eigenvector_sign(
    module: HeuristicGraspModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.asarray([1.0, 4.0])
    same_axis = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    opposite_axis = np.asarray([[-1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(np.linalg, "eigh", lambda _: (values, same_axis))
    positive_yaw = module._narrow_axis_yaw(np.zeros((3, 2), dtype=np.float32))
    monkeypatch.setattr(np.linalg, "eigh", lambda _: (values, opposite_axis))
    negative_yaw = module._narrow_axis_yaw(np.zeros((3, 2), dtype=np.float32))

    assert negative_yaw == pytest.approx(positive_yaw)


@pytest.mark.parametrize(
    "points, frame_id, timestamp, error",
    [
        (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "world",
            1.0,
            "at least three",
        ),
        (np.asarray([[0.0, 0.0, math.nan]] * 3), "world", 1.0, "finite"),
        (np.zeros((3, 3)), "", 1.0, "frame_id"),
        (np.zeros((3, 3)), "world", None, "timestamp"),
        (np.zeros((3, 3)), "world", math.nan, "finite timestamp"),
        (np.zeros((3, 3)), "world", math.inf, "finite timestamp"),
    ],
)
def test_heuristic_grasp_rejects_invalid_pointclouds(
    module: HeuristicGraspModule,
    points: np.ndarray,
    frame_id: str,
    timestamp: float | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        module.propose_grasps(_cloud(points, frame_id=frame_id, timestamp=timestamp))


def test_heuristic_grasp_keeps_fingertips_above_short_object_base(
    make_module: Callable[..., HeuristicGraspModule],
) -> None:
    module = make_module(fingertip_depth=0.02, fingertip_clearance=0.01)
    flat = np.asarray([[-0.02, -0.02, 0.0], [0.02, 0.02, 0.0], [0.0, 0.0, 0.02]])

    pose = module.propose_grasps(_cloud(flat)).candidates[0].pose

    fingertips_z = pose.position.z - 0.02
    assert fingertips_z == pytest.approx(0.01, abs=1e-3)


def test_heuristic_grasp_measures_a_top_only_cloud_down_to_its_support(
    make_module: Callable[..., HeuristicGraspModule],
) -> None:
    """A camera overhead sees an object's top face and nothing of its sides."""
    module = make_module(support_z=0.0)
    top_face = np.asarray(
        [[-0.02, -0.01, 0.10], [0.02, -0.01, 0.10], [0.02, 0.01, 0.10], [-0.02, 0.01, 0.10]]
    )

    pose = module.propose_grasps(_cloud(top_face)).candidates[0].pose

    assert pose.position.z == pytest.approx(0.05)


def test_heuristic_grasp_yaw_offset_turns_the_jaws_half_way_round(
    make_module: Callable[..., HeuristicGraspModule],
) -> None:
    long_along_y = _cloud(
        np.asarray(
            [[-0.02, -0.10, 0.10], [0.02, -0.10, 0.10], [-0.02, 0.10, 0.10], [0.02, 0.10, 0.10]]
        )
    )

    plain = make_module().propose_grasps(long_along_y).candidates[0].pose
    turned = make_module(yaw_offset=math.pi).propose_grasps(long_along_y).candidates[0].pose

    plain_jaw = plain.orientation.rotate_vector(Vector3(0.0, 1.0, 0.0))
    turned_jaw = turned.orientation.rotate_vector(Vector3(0.0, 1.0, 0.0))
    assert turned_jaw.x == pytest.approx(-plain_jaw.x)
    assert abs(turned_jaw.x) == pytest.approx(1.0)


def test_heuristic_grasp_extent_centering_ignores_point_density(
    make_module: Callable[..., HeuristicGraspModule],
) -> None:
    """A side wall seen at an angle piles points onto one edge of the footprint."""
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 11), np.linspace(-0.02, 0.02, 5))
    top_face = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.03)])
    side_wall = np.column_stack(
        [np.linspace(-0.05, 0.05, 200), np.full(200, -0.02), np.linspace(0.0, 0.03, 200)]
    )
    cloud = _cloud(np.vstack([top_face, side_wall]))

    by_median = make_module().propose_grasps(cloud).candidates[0].pose
    by_extent = make_module(centering="extent").propose_grasps(cloud).candidates[0].pose

    assert by_median.position.y == pytest.approx(-0.02)
    assert by_extent.position.x == pytest.approx(0.0, abs=1e-6)
    assert by_extent.position.y == pytest.approx(0.0, abs=1e-6)
