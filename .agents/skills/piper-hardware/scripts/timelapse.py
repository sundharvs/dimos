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

"""Record a timelapse from the scene camera, and keep its latest frame on disk.

    python .agents/skills/piper-hardware/scripts/timelapse.py start
    python .agents/skills/piper-hardware/scripts/timelapse.py status
    python .agents/skills/piper-hardware/scripts/timelapse.py render
    python .agents/skills/piper-hardware/scripts/timelapse.py stop

``start`` leaves a detached recorder running: one timestamped JPEG a second, so
nothing is lost if it is killed, and it reopens the camera if the camera drops
out. ``render`` turns the frames so far into an H.264 video and can run while
recording. The recorder owns the camera's colour stream, so to see the scene
read ``latest.jpg`` (its path is in ``status``) rather than opening the camera.
The camera is ``PIPER_SCENE_CAMERA`` from ``rig.ignore.env``, a V4L2 node.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import av
import cv2
import psutil
import stack

from dimos.constants import STATE_DIR

ROOT = STATE_DIR / "piper-hardware" / "timelapse"
CURRENT = ROOT / "current.json"
CAPTURE_SIZE = (1280, 720)
JPEG_QUALITY = 85
# How long to wait before trying a camera that would not open, or stopped.
REOPEN_INTERVAL_S = 5.0
STOP_GRACE_S = 5.0


def _current() -> dict[str, str | int] | None:
    """The recording in progress, or None when there is no live recorder."""
    if not CURRENT.exists():
        return None
    recording: dict[str, str | int] = json.loads(CURRENT.read_text())
    return recording if psutil.pid_exists(int(recording["pid"])) else None


def record(device: str, interval: float, run_dir: Path) -> None:
    """Capture until told to stop. Runs as the detached recorder process."""
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    index = len(list(frames_dir.glob("*.jpg")))
    encoding = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while not stopping.is_set():
        camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        next_capture = time.monotonic()
        # Grabbing every frame keeps the driver's queue empty, so the one that
        # is decoded each interval is current rather than seconds old.
        while not stopping.is_set() and camera.grab():
            if time.monotonic() < next_capture:
                continue
            next_capture += interval
            decoded, frame = camera.retrieve()
            if not decoded:
                break
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(
                    frame, stamp, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, thickness
                )
            cv2.imwrite(str(frames_dir / f"{index:08d}.jpg"), frame, encoding)
            scratch = run_dir / "latest.tmp.jpg"
            cv2.imwrite(str(scratch), frame, encoding)
            os.replace(scratch, run_dir / "latest.jpg")
            index += 1
        camera.release()
        stopping.wait(REOPEN_INTERVAL_S)


def start(device: str, interval: float) -> None:
    recording = _current()
    if recording is not None:
        print(f"already recording into {recording['run_dir']}")
        return
    run_dir = ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    command = [sys.executable, __file__, "record", "--device", device]
    command += ["--interval", str(interval), "--run-dir", str(run_dir)]
    with (run_dir / "recorder.log").open("w") as log:
        recorder = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    CURRENT.write_text(json.dumps({"pid": recorder.pid, "run_dir": str(run_dir)}))
    print(f"recording {device} every {interval:g}s into {run_dir}")


def status() -> None:
    recording = _current()
    if recording is None:
        print("not recording")
        return
    run_dir = Path(str(recording["run_dir"]))
    frames = len(list((run_dir / "frames").glob("*.jpg")))
    print(f"recording, pid {recording['pid']}, {frames} frames")
    print(f"latest frame: {run_dir / 'latest.jpg'}")


def stop() -> None:
    recording = _current()
    if recording is None:
        print("not recording")
        return
    recorder = psutil.Process(int(recording["pid"]))
    recorder.terminate()
    try:
        recorder.wait(STOP_GRACE_S)
    except psutil.TimeoutExpired:
        recorder.kill()
    print(f"stopped; frames are in {recording['run_dir']}")


def render(run_dir: Path, fps: int) -> None:
    frames = sorted((run_dir / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {run_dir}")
    output = run_dir / "timelapse.mp4"
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height = CAPTURE_SIZE
        stream.pix_fmt = "yuv420p"
        for path in frames:
            image = cv2.imread(str(path))
            if image is None:
                # The frame being written when the recorder was killed.
                continue
            frame = av.VideoFrame.from_ndarray(cv2.resize(image, CAPTURE_SIZE), format="bgr24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    seconds = len(frames) / fps
    print(f"{output}: {len(frames)} frames, {seconds:.0f}s at {fps} fps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["start", "status", "render", "stop", "record"])
    parser.add_argument("--device", default=stack.rig_environment().get("PIPER_SCENE_CAMERA"))
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between frames")
    parser.add_argument("--fps", type=int, default=30, help="playback rate of the rendered video")
    parser.add_argument("--run-dir", type=Path, help="recording to render (default: the latest)")
    args = parser.parse_args()

    if args.command in ("start", "record") and args.device is None:
        raise SystemExit(f"no camera: pass --device or set PIPER_SCENE_CAMERA in {stack.RIG_ENV}")
    if args.command == "record":
        record(args.device, args.interval, args.run_dir)
    elif args.command == "start":
        start(args.device, args.interval)
    elif args.command == "status":
        status()
    elif args.command == "stop":
        stop()
    else:
        runs = sorted(path for path in ROOT.iterdir() if path.is_dir()) if ROOT.exists() else []
        if args.run_dir is None and not runs:
            raise SystemExit("nothing has been recorded")
        render(args.run_dir or runs[-1], args.fps)


if __name__ == "__main__":
    main()
