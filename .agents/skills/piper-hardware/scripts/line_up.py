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

"""Line the scene up with a reference wrist-camera image before a run.

    python .agents/skills/piper-hardware/scripts/line_up.py /home/guest/astra/piper_observation.png
    python .agents/skills/piper-hardware/scripts/line_up.py ref.png --once overlay.png
    python .agents/skills/piper-hardware/scripts/line_up.py ref.png --image frame.png --once overlay.png

Shows the live wrist view blended with the reference, the object's outline in
both (reference magenta, live green), and what to change: how far and which way
to slide the object in the image, and how much to turn it. The object is found
by colour (yellow by default, ``--hsv`` for another). The background outside
the object is compared too: when it has shifted, the camera is not where it was
for the reference, so put the arm in the reference's pose before moving the
object.

``--teleop`` starts ``keyboard-teleop-piper`` alongside when no stack is running,
so the arm can be jogged into the reference's pose while the overlay updates:
click the Keyboard Teleop window and use W/S, A/D, Q/E to translate, R/F, T/G,
Y/H to rotate and [ ] for the gripper. That stack does not use the camera. It
is left running on exit, because stopping a stack homes the arm and would lose
the pose; stop it with stack.py when done.

Keys in the line-up window: SPACE flips between the blend, the live view and the
reference; S saves the overlay next to the reference; Q or ESC quits. ``--once``
writes one overlay, prints the same numbers and exits 0 when lined up, 1 when
not, for use without a display. The camera is opened directly, so no stack may
be holding the wrist camera's colour stream; with a stack running, save a frame
with grab_frame.py and pass it as ``--image``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
import stack

from dimos.core.run_registry import get_most_recent

Image = NDArray[Any]

# The wrist D405's colour node, whatever its serial.
WRIST_CAMERA_GLOB = "usb-Intel_R__RealSense_TM__Depth_Camera_405_*-video-index4"
# OpenCV HSV (H 0-179): the yellow shelf bin under the lab's lights.
DEFAULT_HSV = (18, 90, 90, 38, 255, 255)
# Lined up when the object is within these of the reference.
CENTER_TOLERANCE_PX = 6.0
ANGLE_TOLERANCE_DEG = 2.0
OVERLAP_MIN = 0.90
# The camera counts as moved when the background has shifted or turned this much.
CAMERA_SHIFT_TOLERANCE_PX = 6.0
CAMERA_TURN_TOLERANCE_DEG = 1.5
MIN_OBJECT_AREA_FRACTION = 0.01
# Reversed must fit this much better than as-is to call the object flipped.
FLIP_MARGIN = 0.03
MIN_BACKGROUND_MATCHES = 12
FRAMES_TO_DROP = 5
TELEOP_BLUEPRINT = "keyboard-teleop-piper"
REFERENCE_COLOUR = (255, 0, 255)
LIVE_COLOUR = (0, 255, 0)


@dataclass
class Blob:
    """The object's silhouette in one image."""

    mask: Image
    center: NDArray[np.float64]
    # Direction of the long axis, degrees, anticlockwise positive as seen in the image.
    angle_deg: float
    contour: NDArray[np.int32]


@dataclass
class Comparison:
    shift_px: NDArray[np.float64] | None  # live minus reference, image x right, y down
    turn_deg: float | None  # live minus reference, anticlockwise positive
    flipped: bool
    overlap: float
    camera_shift_px: float | None
    camera_turn_deg: float | None

    @property
    def camera_moved(self) -> bool:
        return self.camera_shift_px is not None and (
            self.camera_shift_px > CAMERA_SHIFT_TOLERANCE_PX
            or abs(self.camera_turn_deg or 0.0) > CAMERA_TURN_TOLERANCE_DEG
        )

    @property
    def lined_up(self) -> bool:
        return (
            self.shift_px is not None
            and self.turn_deg is not None
            and not self.flipped
            and not self.camera_moved
            and float(np.linalg.norm(self.shift_px)) <= CENTER_TOLERANCE_PX
            and abs(self.turn_deg) <= ANGLE_TOLERANCE_DEG
            and self.overlap >= OVERLAP_MIN
        )


def find_object(image: Image, hsv_range: tuple[int, ...]) -> Blob | None:
    """The largest patch of the object's colour, or None when there is none to speak of."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_range[:3]), np.array(hsv_range[3:]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_OBJECT_AREA_FRACTION * mask.size:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    ys, xs = np.nonzero(filled)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0:
        axis = -axis
    return Blob(
        mask=filled,
        center=center,
        # Image y points down, so negate it for an anticlockwise-positive angle.
        angle_deg=math.degrees(math.atan2(-axis[1], axis[0])),
        contour=contour,
    )


def background_motion(
    reference: Image, live: Image, reference_mask: Image | None, live_mask: Image | None
) -> tuple[float, float] | None:
    """How far (px) and how much (deg) the scene outside the object moved, or None if unknown."""

    def outside(mask: Image | None, shape: tuple[int, ...]) -> Image:
        if mask is None:
            return np.full(shape[:2], 255, np.uint8)
        return cv2.bitwise_not(cv2.dilate(mask, np.ones((25, 25), np.uint8)))

    orb = cv2.ORB_create(nfeatures=1500)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    points_ref, features_ref = orb.detectAndCompute(
        gray_ref, outside(reference_mask, gray_ref.shape)
    )
    points_live, features_live = orb.detectAndCompute(
        gray_live, outside(live_mask, gray_live.shape)
    )
    if features_ref is None or features_live is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(features_ref, features_live)
    if len(matches) < MIN_BACKGROUND_MATCHES:
        return None
    source = np.array([points_ref[m.queryIdx].pt for m in matches], np.float32)
    target = np.array([points_live[m.trainIdx].pt for m in matches], np.float32)
    transform, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
    if transform is None or inliers is None or int(inliers.sum()) < MIN_BACKGROUND_MATCHES:
        return None
    # Where the image centre went, so a turn about the centre is not read as a shift.
    middle = np.array([reference.shape[1] / 2.0, reference.shape[0] / 2.0, 1.0])
    shift = float(np.linalg.norm(transform @ middle - middle[:2]))
    turn = -math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    return shift, turn


def _is_flipped(blob_ref: Blob, blob_live: Blob, turn_deg: float) -> bool:
    """Whether the live silhouette fits the reference better turned end for end.

    Only an object that is lopsided along its length (a scoop front) can tell.
    """
    height, width = blob_ref.mask.shape
    center = (float(blob_live.center[0]), float(blob_live.center[1]))
    shift = blob_ref.center - blob_live.center
    overlaps = []
    for extra in (0.0, 180.0):
        # Undo the live object's offset and turn, then try it as is and reversed.
        transform = cv2.getRotationMatrix2D(center, -turn_deg + extra, 1.0)
        transform[:, 2] += shift
        moved = cv2.warpAffine(blob_live.mask, transform, (width, height), flags=cv2.INTER_NEAREST)
        both = np.count_nonzero(moved & blob_ref.mask)
        overlaps.append(both / max(np.count_nonzero(moved | blob_ref.mask), 1))
    return overlaps[1] > overlaps[0] + FLIP_MARGIN


def compare(
    reference: Image,
    live: Image,
    hsv_range: tuple[int, ...],
    live_hsv_range: tuple[int, ...] | None = None,
) -> tuple[Comparison, Blob | None, Blob | None]:
    blob_ref = find_object(reference, hsv_range)
    blob_live = find_object(live, live_hsv_range or hsv_range)
    camera = background_motion(
        reference,
        live,
        blob_ref.mask if blob_ref else None,
        blob_live.mask if blob_live else None,
    )
    shift = turn = None
    flipped = False
    overlap = 0.0
    if blob_ref is not None and blob_live is not None:
        shift = blob_live.center - blob_ref.center
        # A long axis has no direction: the difference is taken modulo a half turn.
        turn = (blob_live.angle_deg - blob_ref.angle_deg + 90.0) % 180.0 - 90.0
        flipped = _is_flipped(blob_ref, blob_live, turn)
        both = np.count_nonzero(blob_ref.mask & blob_live.mask)
        either = np.count_nonzero(blob_ref.mask | blob_live.mask)
        overlap = both / max(either, 1)
    comparison = Comparison(
        shift_px=shift,
        turn_deg=turn,
        flipped=flipped,
        overlap=overlap,
        camera_shift_px=camera[0] if camera else None,
        camera_turn_deg=camera[1] if camera else None,
    )
    return comparison, blob_ref, blob_live


def advice(comparison: Comparison) -> list[str]:
    """What to change, most important first, in words for someone standing at the table."""
    lines = []
    if comparison.camera_shift_px is None:
        lines.append("camera: too little background to check its pose")
    elif comparison.camera_moved:
        lines.append(
            f"CAMERA MOVED {comparison.camera_shift_px:.0f} px, {comparison.camera_turn_deg:+.1f} deg:"
            " put the arm in the reference pose first"
        )
    else:
        lines.append("camera: where it was for the reference")
    if comparison.shift_px is None or comparison.turn_deg is None:
        lines.append("object: not found in the reference or in the live view")
        return lines
    if comparison.flipped:
        lines.append("object: turn it end for end (half a turn)")
    dx, dy = comparison.shift_px
    moves = []
    if abs(dx) > CENTER_TOLERANCE_PX / 2:
        moves.append(f"{abs(dx):.0f} px {'left' if dx > 0 else 'right'}")
    if abs(dy) > CENTER_TOLERANCE_PX / 2:
        moves.append(f"{abs(dy):.0f} px {'up' if dy > 0 else 'down'}")
    lines.append(
        "object: slide it " + " and ".join(moves) + " in the image" if moves else "object: centred"
    )
    if abs(comparison.turn_deg) > ANGLE_TOLERANCE_DEG:
        way = "clockwise" if comparison.turn_deg > 0 else "anticlockwise"
        lines.append(f"object: turn it {abs(comparison.turn_deg):.1f} deg {way} in the image")
    else:
        lines.append("object: angle matches")
    lines.append(f"overlap {comparison.overlap:.2f} (lined up at {OVERLAP_MIN:.2f})")
    lines.append("LINED UP" if comparison.lined_up else "not lined up yet")
    return lines


def draw(
    reference: Image,
    live: Image,
    comparison: Comparison,
    blob_ref: Blob | None,
    blob_live: Blob | None,
    view: str = "blend",
) -> Image:
    if view == "live":
        canvas = live.copy()
    elif view == "reference":
        canvas = reference.copy()
    else:
        canvas = cv2.addWeighted(reference, 0.5, live, 0.5, 0.0)
    if blob_ref is not None:
        cv2.drawContours(canvas, [blob_ref.contour], -1, REFERENCE_COLOUR, 2)
        cv2.drawMarker(
            canvas,
            tuple(int(v) for v in blob_ref.center),
            REFERENCE_COLOUR,
            cv2.MARKER_CROSS,
            16,
            2,
        )
    if blob_live is not None:
        cv2.drawContours(canvas, [blob_live.contour], -1, LIVE_COLOUR, 2)
        cv2.drawMarker(
            canvas, tuple(int(v) for v in blob_live.center), LIVE_COLOUR, cv2.MARKER_CROSS, 16, 2
        )
    for row, line in enumerate([f"[{view}]  magenta: reference  green: live", *advice(comparison)]):
        origin = (8, 20 + 20 * row)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return canvas


def wrist_camera_device() -> str:
    found = sorted(Path("/dev/v4l/by-id").glob(WRIST_CAMERA_GLOB))
    if not found:
        raise SystemExit("no wrist D405 under /dev/v4l/by-id; pass --device or --image")
    return str(found[0])


def open_camera(device: str, size: tuple[int, int]) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not camera.isOpened():
        raise SystemExit(f"cannot open {device}: is a stack or another tool holding the camera?")
    for _ in range(FRAMES_TO_DROP):
        camera.grab()
    return camera


def read_frame(camera: cv2.VideoCapture, size: tuple[int, int]) -> Image:
    ok, frame = camera.read()
    if not ok:
        raise SystemExit("wrist camera read failed")
    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size)
    return np.asarray(frame, dtype=np.uint8)


def start_teleop(can_port: str) -> None:
    """Bring up the keyboard teleop stack unless a stack already has the arm."""
    running = get_most_recent()
    if running is not None:
        print(f"{running.blueprint} is already running; jog the arm with it, not starting teleop")
        return
    if not stack.start(TELEOP_BLUEPRINT, can_port):
        raise SystemExit(f"{TELEOP_BLUEPRINT} did not start")
    print(
        "teleop: click the Keyboard Teleop window; W/S A/D Q/E move, R/F T/G Y/H turn, [ ] gripper"
    )
    print("teleop: left running on exit; `stack.py stop` homes the arm")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("reference", type=Path, help="the image to line the scene up with")
    parser.add_argument("--device", help="V4L2 colour node of the wrist camera (default: the D405)")
    parser.add_argument("--image", type=Path, help="compare this saved frame instead of the camera")
    parser.add_argument("--once", type=Path, metavar="OVERLAY", help="write one overlay and exit")
    parser.add_argument(
        "--hsv",
        default=",".join(str(v) for v in DEFAULT_HSV),
        help="object colour as lo_h,lo_s,lo_v,hi_h,hi_s,hi_v in OpenCV HSV",
    )
    parser.add_argument(
        "--live-hsv",
        help="the object's colour in the live view, when the lighting differs from the "
        "reference's (default: --hsv)",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help=f"start {TELEOP_BLUEPRINT} alongside to jog the arm from the keyboard",
    )
    parser.add_argument("--can-port", default="can0")
    args = parser.parse_args()

    if args.teleop:
        start_teleop(args.can_port)

    reference = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
    if reference is None:
        raise SystemExit(f"cannot read {args.reference}")
    size = (reference.shape[1], reference.shape[0])
    hsv_range = tuple(int(v) for v in args.hsv.split(","))
    if len(hsv_range) != 6:
        raise SystemExit("--hsv needs six comma-separated values")
    live_hsv_range = tuple(int(v) for v in args.live_hsv.split(",")) if args.live_hsv else None
    if live_hsv_range is not None and len(live_hsv_range) != 6:
        raise SystemExit("--live-hsv needs six comma-separated values")

    if args.image is not None:
        still = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if still is None:
            raise SystemExit(f"cannot read {args.image}")
        if (still.shape[1], still.shape[0]) != size:
            still = cv2.resize(still, size)
        camera = None
    else:
        still = None
        camera = open_camera(args.device or wrist_camera_device(), size)

    def frame() -> Image:
        return still if camera is None else read_frame(camera, size)

    try:
        if args.once is not None:
            live = frame()
            comparison, blob_ref, blob_live = compare(reference, live, hsv_range, live_hsv_range)
            print("\n".join(advice(comparison)))
            cv2.imwrite(str(args.once), draw(reference, live, comparison, blob_ref, blob_live))
            print(f"overlay: {args.once}")
            raise SystemExit(0 if comparison.lined_up else 1)

        views = ["blend", "live", "reference"]
        view = 0
        window = f"line up with {args.reference.name}"
        while True:
            live = frame()
            comparison, blob_ref, blob_live = compare(reference, live, hsv_range, live_hsv_range)
            canvas = draw(reference, live, comparison, blob_ref, blob_live, views[view])
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                view = (view + 1) % len(views)
            if key == ord("s"):
                saved = args.reference.with_name(args.reference.stem + "_line_up.png")
                cv2.imwrite(str(saved), canvas)
                print(f"saved {saved}")
        raise SystemExit(0 if comparison.lined_up else 1)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
