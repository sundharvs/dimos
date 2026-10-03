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

"""Auto-research loop for the container pick skill on a live stack.

One trial = pick_up_container -> judge the lift -> rotate the held container by
a scheduled yaw -> set_down_container -> log. Varying the yaw between trials is
what exposes wall-classification and release bugs that a single orientation
never shows. Success is judged by the arm's gripper readback and, when a fixed
scene camera is given, by the object's colour blob rising in that camera.

    python -m dimos.manipulation.grasping.demo_container_pick_research out/ \\
        --trials 10 --scene-camera /dev/video8 --hsv 20,170,150,36,255,255

Keep a hand on the e-stop: the skill guards its own motions, but the loop is
unattended by design. See docs/capabilities/manipulation/container-pick.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

from dimos.porcelain.dimos import Dimos

DEFAULT_SCHEDULE: list[dict[str, Any]] = [
    {"rim": {"wall_select": "lever", "sides": "long"}, "rotate_deg": 45},
    {"rim": {"wall_select": "lever", "sides": "long"}, "rotate_deg": 45},
    {"rim": {"wall_select": "highest", "sides": "long"}, "rotate_deg": -90},
    {"rim": {"wall_select": "lowest", "sides": "long"}, "rotate_deg": 45},
    {"rim": {"wall_select": "lever", "sides": "short"}, "rotate_deg": 45},
    {"rim": {"wall_select": "lever", "sides": "long"}, "rotate_deg": 60},
    {"rim": {"wall_select": "lowest", "sides": "long"}, "rotate_deg": -45},
    {"rim": {"wall_select": "highest", "sides": "long"}, "rotate_deg": 45},
    {"rim": {"wall_select": "lever", "sides": "short"}, "rotate_deg": -60},
    {"rim": {"wall_select": "lever", "sides": "long"}, "rotate_deg": 45},
]


class SceneBlob:
    """Largest colour blob in a fixed V4L2 camera, for an independent lift check."""

    def __init__(self, device: str, hsv: tuple[int, ...], crop_left_half: bool) -> None:
        import cv2

        self.cv2 = cv2
        self.lo, self.hi = tuple(hsv[:3]), tuple(hsv[3:])
        self.crop_left_half = crop_left_half
        index = int(device) if device.isdigit() else device
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(6):
            self.cap.read()

    def measure(self, path: Path | None = None) -> dict[str, int] | None:
        import numpy as np

        frame = None
        for _ in range(12):  # flush buffered frames; V4L2 hands back stale ones otherwise
            ok, frame = self.cap.read()
        if frame is None:
            return None
        if self.crop_left_half:
            frame = frame[:, : frame.shape[1] // 2]
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        mask = self.cv2.inRange(
            hsv, np.array(self.lo, dtype=np.uint8), np.array(self.hi, dtype=np.uint8)
        )
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, _ = self.cv2.connectedComponentsWithStats(mask)
        blob = None
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, self.cv2.CC_STAT_AREA]))
            x, y, w, h, a = (int(v) for v in stats[i])
            if a >= 1500:
                blob = {"x": x, "y": y, "w": w, "h": h, "area": a, "top": y, "bottom": y + h}
                self.cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if path is not None:
            self.cv2.imwrite(str(path), frame, [self.cv2.IMWRITE_JPEG_QUALITY, 80])
        return blob


def run_trial(
    app: Any, scene: SceneBlob | None, out: Path, k: int, cfg: dict[str, Any], min_rise_px: int
) -> dict[str, Any]:
    t: dict[str, Any] = {"trial": k, "cfg": cfg, "t0": time.time()}
    app.RimGraspModule.set_params(**cfg["rim"])
    before = scene.measure(out / f"t{k:02d}_before.jpg") if scene else None
    t["scene_before"] = before
    result = app.ContainerPickModule.pick_up_container()
    t["pick"] = {"success": result.success, "message": result.message, **result.metadata}
    if not result.success:
        t["outcome"] = f"pick_failed: {result.error_code}"
        return t
    lifted = scene.measure(out / f"t{k:02d}_lifted.jpg") if scene else None
    t["scene_lifted"] = lifted
    rise = None
    if before and lifted:
        rise = {"bottom": before["bottom"] - lifted["bottom"], "top": before["top"] - lifted["top"]}
    t["rise"] = rise
    off_table = (
        True
        if scene is None
        else bool(rise and rise["bottom"] >= min_rise_px and rise["top"] >= 2 * min_rise_px)
    )
    t["success"] = bool(off_table)
    if cfg.get("rotate_deg"):
        rot = app.ContainerPickModule.rotate_held_container(float(cfg["rotate_deg"]))
        t["rotate"] = {"success": rot.success, "message": rot.message, **rot.metadata}
    down = app.ContainerPickModule.set_down_container()
    t["set_down"] = {"success": down.success, "message": down.message, **down.metadata}
    after = scene.measure(out / f"t{k:02d}_released.jpg") if scene else None
    t["scene_released"] = after
    t["came_along"] = bool(after and before and after["bottom"] < before["bottom"] - 60)
    t["outcome"] = "success" if t["success"] else "held_not_lifted"
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path)
    parser.add_argument("--trials", type=int, default=len(DEFAULT_SCHEDULE))
    parser.add_argument("--scene-camera", default=None, help="V4L2 device/index of a fixed camera")
    parser.add_argument(
        "--hsv", default="20,170,150,36,255,255", help="HSV low,high for the object colour"
    )
    parser.add_argument(
        "--stereo-left-half",
        action="store_true",
        help="side-by-side stereo stream: use the left eye",
    )
    parser.add_argument("--min-rise-px", type=int, default=20)
    parser.add_argument(
        "--schedule", type=Path, default=None, help="JSON list overriding the default schedule"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(args.schedule.read_text()) if args.schedule else DEFAULT_SCHEDULE
    scene = None
    if args.scene_camera:
        hsv = tuple(int(v) for v in args.hsv.split(","))
        scene = SceneBlob(args.scene_camera, hsv, args.stereo_left_half)
    app: Any = Dimos.connect(timeout=30)
    log = (args.out / "trials.jsonl").open("a")
    history: list[dict[str, Any]] = []
    try:
        for k in range(min(args.trials, len(schedule))):
            cfg = schedule[k % len(schedule)]
            try:
                t = run_trial(app, scene, args.out, k, cfg, args.min_rise_px)
            except Exception as exc:
                traceback.print_exc()
                t = {"trial": k, "cfg": cfg, "outcome": f"exception: {exc}"}
                status = app.ContainerPickModule.status()
                if status.get("holding"):
                    app.ContainerPickModule.set_down_container()
            history.append(t)
            log.write(json.dumps(t, default=str) + "\n")
            log.flush()
            print(time.strftime("%H:%M:%S"), f"[{k}] {t.get('outcome')}", flush=True)
    finally:
        app.stop()
    wins = sum(1 for h in history if h.get("success"))
    print(
        f"DONE: {wins}/{len(history)} successful lifts; yaws sampled: "
        f"{sorted({round(math.degrees(0) + (h.get('pick') or {}).get('footprint', {}).get('yaw_deg', 0)) for h in history})}"
    )


if __name__ == "__main__":
    main()
