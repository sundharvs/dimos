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

"""Run pick-and-place cycles on the Piper and measure where each one lands.

    python .agents/skills/piper-hardware/scripts/pick_place_bench.py eraser 0.27,0.0 0.22,0.15
    python .agents/skills/piper-hardware/scripts/pick_place_bench.py eraser 0.27,0.0 --rounds 3

Needs piper-grasp running. Each cycle finds the object, picks it, places it at
the next target and measures the landing with the wrist camera turned to face
it. The landing is compared with where the object should be: the target, shifted
by how far the grasp point was from the object's middle. Every cycle is one line
of a JSON-lines log with a camera frame beside it, and the closing summary is
the number to compare before and after a change.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import time

from grab_frame import save_frame
import numpy as np
from numpy.typing import NDArray

from dimos.constants import STATE_DIR
from dimos.porcelain.dimos import Dimos

LOG_ROOT = STATE_DIR / "piper-hardware" / "bench"
CONNECT_TIMEOUT_S = 15.0
# Azimuths the scan pose is turned to, in order, when the object is not in view.
SEARCH_AZIMUTHS = (0.0, 0.6, -0.6, 1.2, -1.2)
# An object further than this off the camera's heading is faced and scanned
# again: off to one side it is cut off by the frame and reads up to 1 cm off.
RECENTER_RAD = 0.1
# The camera and the transform buffer need a moment after the arm stops.
VIEW_SETTLE_S = 1.0
# Share of points ignored at each end of a footprint axis.
EXTENT_TRIM = 0.02


@dataclass(frozen=True)
class Sighting:
    object_id: str
    centre: tuple[float, float]
    size: tuple[float, float]


@dataclass
class Cycle:
    index: int
    target: tuple[float, float]
    found_at: tuple[float, float] | None = None
    pick: str = "not attempted"
    place: str = "not attempted"
    landed_at: tuple[float, float] | None = None
    error_mm: tuple[float, float] | None = None
    seconds: float = 0.0
    frame: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_mm is not None

    @property
    def miss_mm(self) -> float:
        return math.hypot(*self.error_mm) if self.error_mm is not None else math.nan


def footprint(points: NDArray[np.float32]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Middle and (long, short) extent of a cloud's outline on the table."""
    xy = points[:, :2].astype(np.float64)
    mean = xy.mean(axis=0)
    centered = xy - mean
    _, axes = np.linalg.eigh(centered.T @ centered)
    low, high = np.quantile(centered @ axes, [EXTENT_TRIM, 1.0 - EXTENT_TRIM], axis=0)
    centre = mean + axes @ ((low + high) / 2.0)
    short, long = sorted(high - low)
    return (float(centre[0]), float(centre[1])), (float(long), float(short))


class Rig:
    """The running stack, with the scan pose turned to wherever the object is."""

    def __init__(self, app: Dimos) -> None:
        self._app = app
        self._group = app.ManipulationModule.list_planning_groups()[0].id
        home = app.ManipulationModule.get_state().groups[self._group].joint_presets["home"]
        self._scan_joints = list(home.position)
        self._heading = math.nan

    def face(self, azimuth: float) -> None:
        joints = [azimuth, *self._scan_joints[1:]]
        result = self._app.ManipulationSkills.move_to_joints(", ".join(f"{q:.4f}" for q in joints))
        if not result.is_success():
            raise RuntimeError(f"could not turn the scan pose to {azimuth:.2f} rad: {result}")
        self._heading = azimuth
        time.sleep(VIEW_SETTLE_S)

    def _scan(self, prompt: str) -> Sighting | None:
        scan = self._app.PickAndPlaceModule.scan_objects([prompt])
        objects = scan.metadata.get("objects", [])
        if not objects:
            return None
        object_id = objects[0]["object_id"]
        cloud = self._app.ObjectSceneRegistrationModule.get_object_pointcloud_by_object_id(
            object_id
        )
        centre, size = footprint(cloud.points_f32())
        return Sighting(object_id, centre, size)

    def locate(self, prompt: str, near: tuple[float, float] | None) -> Sighting | None:
        """Find the object, looking first where it should be, and face it."""
        expected = () if near is None else (math.atan2(near[1], near[0]),)
        for heading in (*expected, *SEARCH_AZIMUTHS):
            self.face(heading)
            sighting = self._scan(prompt)
            if sighting is None:
                continue
            bearing = math.atan2(sighting.centre[1], sighting.centre[0])
            if abs(bearing - self._heading) > RECENTER_RAD:
                self.face(bearing)
                sighting = self._scan(prompt) or sighting
            return sighting
        return None

    def grasp_point(self, rank: int) -> tuple[float, float]:
        candidates = self._app.PickAndPlaceModule.get_grasp_candidates().candidates
        position = candidates[rank].pose.position
        return float(position.x), float(position.y)

    def pick(self, sighting: Sighting) -> tuple[str, tuple[float, float] | None]:
        result = self._app.PickAndPlaceModule.pick_object(sighting.object_id)
        if not result.is_success():
            return str(result), None
        return "ok", self.grasp_point(int(result.metadata["rank"]))

    def place(self, target: tuple[float, float]) -> str:
        result = self._app.PickAndPlaceModule.place_at(*target)
        return "ok" if result.is_success() else str(result)

    def go_home(self) -> None:
        self._app.ManipulationSkills.go_home()


def run_cycle(rig: Rig, cycle: Cycle, prompt: str, near: tuple[float, float] | None) -> None:
    before = rig.locate(prompt, near)
    if before is None:
        cycle.pick = "object not found"
        return
    cycle.found_at = before.centre
    cycle.pick, grasp = rig.pick(before)
    if grasp is None:
        return
    cycle.place = rig.place(cycle.target)
    if cycle.place != "ok":
        return
    after = rig.locate(prompt, cycle.target)
    if after is None:
        cycle.place = "placed, then not found"
        return
    # The object's middle lands as far from the target as it was from the grasp.
    expected = (
        cycle.target[0] + before.centre[0] - grasp[0],
        cycle.target[1] + before.centre[1] - grasp[1],
    )
    cycle.landed_at = after.centre
    cycle.error_mm = (
        (after.centre[0] - expected[0]) * 1000.0,
        (after.centre[1] - expected[1]) * 1000.0,
    )


def parse_target(text: str) -> tuple[float, float]:
    x, y = (float(value) for value in text.split(","))
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", help="what to pick, e.g. eraser")
    parser.add_argument("targets", nargs="+", type=parse_target, help="x,y in metres")
    parser.add_argument("--rounds", type=int, default=1, help="passes over the targets")
    args = parser.parse_args()

    run_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "cycles.jsonl"

    app = Dimos.connect(timeout=CONNECT_TIMEOUT_S)
    cycles: list[Cycle] = []
    try:
        rig = Rig(app)
        last_seen: tuple[float, float] | None = None
        for index, target in enumerate(args.targets * args.rounds, start=1):
            cycle = Cycle(index, target)
            started = time.monotonic()
            run_cycle(rig, cycle, args.prompt, last_seen)
            cycle.seconds = time.monotonic() - started
            frame = run_dir / f"{index:02d}.png"
            save_frame(frame)
            cycle.frame = frame.name
            cycles.append(cycle)
            with log_path.open("a") as log:
                log.write(json.dumps(asdict(cycle)) + "\n")
            if not cycle.succeeded:
                print(f"{index:2d} {target}  FAILED  pick: {cycle.pick}  place: {cycle.place}")
                break
            last_seen = cycle.landed_at
            dx, dy = cycle.error_mm or (math.nan, math.nan)
            print(
                f"{index:2d} {target}  landed {cycle.miss_mm:4.1f} mm off "
                f"({dx:+.1f}, {dy:+.1f})  {cycle.seconds:.0f}s"
            )
        rig.go_home()
    finally:
        app.stop()

    misses = [cycle.miss_mm for cycle in cycles if cycle.succeeded]
    planned = len(args.targets) * args.rounds
    print(f"{len(misses)}/{planned} cycles succeeded; log in {run_dir}")
    if misses:
        seconds = float(np.mean([cycle.seconds for cycle in cycles if cycle.succeeded]))
        print(
            f"miss: median {np.median(misses):.1f} mm, worst {max(misses):.1f} mm; "
            f"{seconds:.0f}s per cycle"
        )
    if len(misses) != planned:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
