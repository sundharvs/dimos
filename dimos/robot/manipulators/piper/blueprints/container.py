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

"""Piper picking up open containers by a wall, on real hardware.

Usage:
    dimos run piper-container

``piper-grasp`` plus ``ContainerGraspModule``: ``pick_up_container("yellow bin")``
pinches the bin's near long wall, lifts it and checks that it rose;
``put_down_container`` puts it back. Set the environment as for ``piper-grasp``,
and PIPER_GRIPPER_EFFORT=3000: at the adapter's default 1000 mN.m a pinched wall
hinges in the jaws and slips out as the arm leans back (2026-10-04).
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.manipulation.container_grasp_module import ContainerGraspModule
from dimos.manipulation.grasp_verification import GraspVerificationConfig
from dimos.robot.manipulators.piper.blueprints.grasp import (
    PIPER_FINGERTIPS_PAST_TCP,
    PIPER_GRASP_TABLE_Z,
    piper_grasp,
)

# Camera 0.25 m up, looking 42 degrees forward of straight down: a 30 cm bin
# standing 0.2-0.4 m in front of the base is whole in the frame (2026-10-04).
PIPER_CONTAINER_SURVEY_JOINTS = [0.0, 0.45, -0.52, 0.0, 1.0, 0.0]
# piper_reach.py, 2026-10-04: fingers straight down, the tool point reaches
# 0.12 m high between 0.16 and 0.32 m from the base, and no higher than 0.13.
PIPER_CONTAINER_HOVER_Z = 0.122
# Leaning the shoulder back 0.36 rad in three steps takes the tool point from
# 0.125 to 0.21 m, which is what clears a 10 cm bin from the table.
PIPER_CONTAINER_RAISE_STEP = [0.0, -0.12, 0.0, 0.0, 0.0, 0.0]

piper_container = autoconnect(
    piper_grasp,
    ContainerGraspModule.blueprint(
        planning_frame="world",
        survey_joints=PIPER_CONTAINER_SURVEY_JOINTS,
        hover_z=PIPER_CONTAINER_HOVER_Z,
        fingertip_depth=PIPER_FINGERTIPS_PAST_TCP,
        support_z=PIPER_GRASP_TABLE_Z,
        raise_step=PIPER_CONTAINER_RAISE_STEP,
        # As piper-grasp: fully open, the jaws stop at 0.84.
        grasp_verification=GraspVerificationConfig(open_tolerance=0.2),
    ),
)
