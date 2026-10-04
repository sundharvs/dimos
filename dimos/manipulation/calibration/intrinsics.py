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

"""Factory intrinsics for cameras read without their vendor SDK.

A ZED read as a plain UVC webcam delivers raw, unrectified side-by-side frames,
so the rectified K the ZED SDK reports does not describe them. The per-device
factory file the SDK downloads (`/usr/local/zed/settings/SN<serial>.conf`)
holds the raw left-camera K and plumb_bob distortion for every resolution.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Width of one (left) image -> the resolution's section suffix in a ZED .conf.
ZED_RESOLUTIONS = {2208: "2K", 1920: "FHD", 1280: "HD", 672: "VGA"}


def load_zed_left_intrinsics(
    path: str | Path, width: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Raw left-camera K and plumb_bob distortion for left images `width` wide.

    Raises ValueError when the width is not a ZED resolution or the file lacks
    the section, because the wrong K places the board at a plausible wrong
    distance and nothing downstream notices.
    """
    resolution = ZED_RESOLUTIONS.get(width)
    if resolution is None:
        raise ValueError(
            f"{width} px wide is not a ZED left-image width ({sorted(ZED_RESOLUTIONS)})"
        )
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise ValueError(f"Cannot read ZED calibration file {path}")
    section = f"LEFT_CAM_{resolution}"
    if section not in parser:
        raise ValueError(f"{path} has no [{section}] section")
    values = parser[section]
    camera_matrix = np.array(
        [
            [float(values["fx"]), 0.0, float(values["cx"])],
            [0.0, float(values["fy"]), float(values["cy"])],
            [0.0, 0.0, 1.0],
        ]
    )
    # OpenCV's plumb_bob order.
    dist_coeffs = np.array([float(values[key]) for key in ("k1", "k2", "p1", "p2", "k3")])
    return camera_matrix, dist_coeffs
