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

"""Where can the Piper hold its tool straight down?

    python .agents/skills/piper-hardware/scripts/piper_reach.py 0.019 0.079 0.119

Each argument is a tool-frame height above the base in metres. For each, a
planar sweep (joints 1, 4 and 6 at zero) reports the band of radii from the base
axis at which the tool points straight down inside the joint limits. The wrist
pitch range is what bounds a top-down pose on this arm, so the band narrows
quickly with height. No hardware needed.
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray
import pinocchio as pin

from dimos.robot.manipulators.piper.config import PIPER_TCP_FRAME, make_piper_model_config

# How close a sample's height must be to a requested height to count toward it.
HEIGHT_BIN_M = 0.004
SWEEP_STEP_RAD = 0.01
# Probe spacing for the tool tilt's dependence on wrist pitch, which is linear.
TILT_PROBE_RAD = 0.1
STRAIGHT_DOWN_TOLERANCE = 1e-3


def top_down_samples(frame: str) -> NDArray[np.float64]:
    """Rows of (radius, height, joint5) for every straight-down pose in the sweep."""
    model = pin.buildModelFromXML(make_piper_model_config(tcp=True).model.load().xml)
    data = model.createData()
    frame_id = model.getFrameId(frame)
    index = {
        name: model.joints[model.getJointId(name)].idx_q for name in ("joint2", "joint3", "joint5")
    }
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit

    def tool(q: NDArray[np.float64]) -> pin.SE3:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[frame_id]

    def tilt(q: NDArray[np.float64]) -> float:
        approach = tool(q).rotation[:, 2]
        return float(np.arctan2(approach[0], -approach[2]))

    rows = []
    q = pin.neutral(model)
    for joint2 in np.arange(lower[index["joint2"]], upper[index["joint2"]], SWEEP_STEP_RAD):
        for joint3 in np.arange(lower[index["joint3"]], upper[index["joint3"]], SWEEP_STEP_RAD):
            q[index["joint2"]], q[index["joint3"]] = joint2, joint3
            q[index["joint5"]] = 0.0
            tilt_at_zero = tilt(q)
            q[index["joint5"]] = TILT_PROBE_RAD
            slope = (tilt(q) - tilt_at_zero) / TILT_PROBE_RAD
            joint5 = -tilt_at_zero / slope
            if not lower[index["joint5"]] <= joint5 <= upper[index["joint5"]]:
                continue
            q[index["joint5"]] = joint5
            pose = tool(q)
            if abs(pose.rotation[2, 2] + 1.0) > STRAIGHT_DOWN_TOLERANCE:
                continue
            rows.append((pose.translation[0], pose.translation[2], joint5))
    return np.asarray(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("heights", type=float, nargs="+", help="tool heights in metres")
    parser.add_argument(
        "--frame",
        default=PIPER_TCP_FRAME,
        help=f"tool frame to hold straight down (default {PIPER_TCP_FRAME})",
    )
    args = parser.parse_args()

    samples = top_down_samples(args.frame)
    for height in args.heights:
        at_height = samples[np.abs(samples[:, 1] - height) < HEIGHT_BIN_M]
        if not len(at_height):
            print(f"{args.frame} z={height:.3f}: unreachable straight down")
            continue
        print(
            f"{args.frame} z={height:.3f}: radius "
            f"{at_height[:, 0].min():.3f} to {at_height[:, 0].max():.3f} m"
        )


if __name__ == "__main__":
    main()
