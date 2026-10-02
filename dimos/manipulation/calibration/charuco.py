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

"""The ChArUco calibration board: print one, then find its pose in a frame.

WHY CHARUCO AND NOT A CHESSBOARD OR A SINGLE TAG
    A chessboard has to be seen whole; half of one detects as nothing. A single
    tag fits a pose from four corners and flips between two mirror solutions
    when seen face-on. ChArUco carries an ArUco marker in every white square, so
    each chessboard corner identifies itself, a partial view still contributes,
    and the pose comes from dozens of corners refined to sub-pixel accuracy.

MEASURE THE PRINT
    `render()` writes an image at an exact scale, but printers rescale and
    "fit to page" silently shrinks by a few percent. The calibration inherits
    that as a scale error in the camera's translation. Put a rule across a known
    number of squares and pass what you actually measure as the square size.

    python -m dimos.manipulation.calibration.charuco --out charuco.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# Leaves a white quiet zone round each marker.
MARKER_RATIO = 22.0 / 29.0
# A PnP fit on four near-collinear corners is numerically miserable and still
# returns a pose, so ask for a few more than the theoretical minimum.
MIN_CORNERS = 6


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board, lengths in metres.

    The defaults are a 5x7 board of 34 mm squares, which prints on A4 with a
    usable margin and gives 24 interior corners.
    """

    squares_x: int = 5
    squares_y: int = 7
    square_m: float = 0.034
    marker_m: float = 0.034 * MARKER_RATIO
    dictionary: str = "DICT_5X5_1000"

    def __post_init__(self) -> None:
        if not 0.5 <= self.marker_m / self.square_m <= 0.9:
            raise ValueError(
                f"marker/square = {self.marker_m / self.square_m:.2f}; keep it near 0.75. "
                "Too large and the marker touches the corner it identifies; too small "
                "and it stops decoding."
            )
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"Unknown ArUco dictionary {self.dictionary!r}")

    @classmethod
    def from_mm(
        cls,
        squares_x: int,
        squares_y: int,
        square_mm: float,
        marker_mm: float | None = None,
        dictionary: str = "DICT_5X5_1000",
    ) -> BoardSpec:
        marker = marker_mm if marker_mm is not None else MARKER_RATIO * square_mm
        return cls(squares_x, squares_y, square_mm / 1000.0, marker / 1000.0, dictionary)

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    @property
    def n_corners(self) -> int:
        """Interior chessboard corners -- the points actually localised."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def cv_board(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_m, self.marker_m, dictionary
        )

    def describe(self) -> str:
        width, height = self.size_m
        return (
            f"ChArUco {self.squares_x}x{self.squares_y}, square {self.square_m * 1000:.1f} mm, "
            f"marker {self.marker_m * 1000:.1f} mm, {self.dictionary} -> "
            f"{width * 1000:.0f} x {height * 1000:.0f} mm, {self.n_corners} corners"
        )

    def render(self, dpi: int = 600, margin_mm: float = 10.0) -> NDArray[np.uint8]:
        """The board at true scale for printing at `dpi`, 100%, scaling off."""
        px_per_m = dpi / 0.0254
        width, height = self.size_m
        size = (round(width * px_per_m), round(height * px_per_m))
        image = self.cv_board().generateImage(size)
        margin = round(margin_mm / 1000.0 * px_per_m)
        bordered = cv2.copyMakeBorder(
            image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
        )
        return np.asarray(bordered, dtype=np.uint8)


@dataclass(frozen=True)
class BoardPose:
    """Where the board is in the camera frame that saw it."""

    board_in_camera: NDArray[np.float64]
    n_corners: int
    reproj_px: float  # RMS over the matched corners, the honest quality number
    corners: NDArray[Any]  # matched ChArUco corners in pixels, for drawing
    ids: NDArray[Any]


class CharucoDetector:
    """Board pose from one frame, reusing the OpenCV detector across frames."""

    def __init__(self, spec: BoardSpec, min_corners: int = MIN_CORNERS) -> None:
        self.spec = spec
        self.min_corners = min_corners
        self._board = spec.cv_board()
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def detect(
        self,
        gray: NDArray[Any],
        camera_matrix: NDArray[np.float64],
        dist_coeffs: NDArray[np.float64],
    ) -> BoardPose | None:
        """`camera_matrix` must describe THIS image: a K from another resolution
        puts the board at a plausible wrong distance and nothing downstream notices."""
        detected: tuple[Any, ...] = self._detector.detectBoard(gray)
        corners, ids = detected[0], detected[1]
        if ids is None or len(ids) < self.min_corners:
            return None
        object_points, image_points = self._board.matchImagePoints(corners, ids)
        if object_points is None or len(object_points) < self.min_corners:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        cam_T_board = np.eye(4)
        cam_T_board[:3, :3] = cv2.Rodrigues(rvec)[0]
        cam_T_board[:3, 3] = tvec.reshape(3)
        return BoardPose(
            board_in_camera=cam_T_board,
            n_corners=len(ids),
            reproj_px=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            corners=np.asarray(corners),
            ids=np.asarray(ids),
        )


def draw(
    bgr: NDArray[np.uint8],
    pose: BoardPose,
    camera_matrix: NDArray[np.float64],
    dist_coeffs: NDArray[np.float64],
    axis_m: float = 0.05,
) -> None:
    """Corners and the board frame, drawn in place."""
    cv2.aruco.drawDetectedCornersCharuco(bgr, pose.corners, pose.ids, (0, 255, 0))
    rvec = cv2.Rodrigues(pose.board_in_camera[:3, :3])[0]
    cv2.drawFrameAxes(bgr, camera_matrix, dist_coeffs, rvec, pose.board_in_camera[:3, 3], axis_m, 3)


def main(argv: list[str] | None = None) -> int:
    """Write a printable board and say what to check after printing it."""
    parser = argparse.ArgumentParser(description="generate a ChArUco board to print")
    parser.add_argument("--out", default="charuco.png")
    parser.add_argument("--squares-x", type=int, default=BoardSpec.squares_x)
    parser.add_argument("--squares-y", type=int, default=BoardSpec.squares_y)
    parser.add_argument("--square-mm", type=float, default=BoardSpec.square_m * 1000)
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--dictionary", default=BoardSpec.dictionary)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)

    spec = BoardSpec.from_mm(
        args.squares_x, args.squares_y, args.square_mm, args.marker_mm, args.dictionary
    )
    cv2.imwrite(args.out, spec.render(dpi=args.dpi))
    width, _ = spec.size_m
    print(spec.describe())
    print(f"wrote {args.out} at {args.dpi} dpi")
    print("print at 100% -- no 'fit to page', no 'shrink to printable area'.")
    print(f"then measure {args.squares_x} squares across: it should be {width * 1000:.1f} mm,")
    print("and pass the measured square size as --square-mm when calibrating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
