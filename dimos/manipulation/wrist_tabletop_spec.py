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

from typing import Any, Protocol

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec.utils import Spec


class WristTabletopSpec(Spec, Protocol):
    """A wrist camera looking down at a tabletop: colour-segmented object clouds and tape slots."""

    def scan_object_cloud(
        self,
        hsv_low: tuple[int, int, int] | None = None,
        hsv_high: tuple[int, int, int] | None = None,
    ) -> PointCloud2 | None: ...
    def object_view_fraction(self) -> float | None: ...
    def add_tape_view(self) -> int: ...
    def clear_tape_views(self) -> None: ...
    def fit_slots(self) -> dict[str, Any]: ...
    def get_slots(self) -> dict[str, Any]: ...
