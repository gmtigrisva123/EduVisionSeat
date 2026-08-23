"""Pinhole camera model — the optional, calibrated half of HIEM.

**Read this first, because it decides how much of the file you need.**

HIEM's core output is a ratio of two lengths measured in the same image region
at the same depth. Every perspective term — the focal length, the depth, the
off-axis radial factor — appears identically in the numerator and the
denominator and *cancels exactly*. That is why the normaliser in
:mod:`src.hiem.normalize` needs no camera model at all, and why HIEM works on
uncalibrated classroom video where absolute height estimation does not.

A camera model buys exactly one thing: the right to compare the body SCALES of
two students against each other. A student in the front row subtends more pixels
than an equally tall student at the back, so ``S`` alone ranks students by
distance-to-camera, not by size. Undoing that needs the depth of each seat and
the intrinsics — hence this module.

The boundary is deliberate and is enforced in code, not merely documented:
:class:`~src.hiem.scale.BodyScale` carries a ``comparable_across_students`` flag
that is false unless a camera model and a depth were supplied.

Also see the checklist at the end of ``docs/HEIGHT_ESTIMATION.md``: *"Is
scale_mode other than 'none'? If not -> relative ranking only."*
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics of a single pinhole camera, with optional radial distortion.

    ``fx``/``fy`` are in pixels, ``cx``/``cy`` are the principal point in pixels,
    and ``distortion`` is the OpenCV ``(k1, k2, p1, p2, k3)`` vector.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    #: RMS reprojection error of the calibration that produced these numbers, in
    #: pixels. ``None`` means the intrinsics were guessed from a field of view
    #: rather than calibrated, which the checklist treats as uncalibrated.
    calibration_rms_px: Optional[float] = None

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"image size must be positive, got {self.width}x{self.height}")
        if len(self.distortion) != 5:
            raise ValueError(f"distortion must be the 5-vector (k1,k2,p1,p2,k3), got {self.distortion}")

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraModel":
        """Guess the intrinsics from a horizontal field of view.

        ``f_px = (W/2) / tan(HFOV/2)``. Square pixels and a centred principal
        point are assumed, which is close enough for a webcam and wrong enough
        for a wide-angle security camera that :meth:`focal_is_plausible` exists.

        This is NOT calibration: ``calibration_rms_px`` stays ``None`` and
        anything that requires a calibrated camera will keep refusing.
        """
        if not 1.0 < hfov_deg < 179.0:
            raise ValueError(f"hfov_deg must be within (1, 179), got {hfov_deg}")
        focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @classmethod
    def from_exif(
        cls, width: int, height: int, focal_mm: float, sensor_width_mm: float
    ) -> "CameraModel":
        """``f_px = f_mm * W_px / sensor_width_mm`` (docs/HEIGHT_ESTIMATION.md §3.2)."""
        if focal_mm <= 0 or sensor_width_mm <= 0:
            raise ValueError("focal_mm and sensor_width_mm must both be positive")
        focal = focal_mm * width / sensor_width_mm
        return cls(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    # ------------------------------------------------------------------ #
    # Sanity
    # ------------------------------------------------------------------ #
    def focal_is_plausible(self) -> bool:
        """The ``0.7*W <= f_px <= W`` rule of thumb for webcams and phones.

        A focal length outside this band means the field of view was guessed
        wrongly, and docs/HEIGHT_ESTIMATION.md §3.1 puts the cost of a 20% focal
        error at roughly 33 cm of height error. Fail loudly rather than publish.
        """
        return 0.7 * self.width <= self.fx <= float(self.width)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 intrinsic matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def backproject(self, u: float, v: float, depth: float) -> Tuple[float, float, float]:
        """Back-project a pixel to a 3D point at ``depth`` along the OPTICAL AXIS.

        Goes through the full inverse of ``K``, i.e. ``depth * K^-1 [u, v, 1]^T``,
        rather than the abbreviated ``S = s*Z/f`` that ignores where in the frame
        the point sits. docs/HEIGHT_ESTIMATION.md §4.3 puts the cost of that
        shortcut at 19% for a student near the edge of a wide frame.

        ``depth`` is the z-coordinate, not the range from the camera centre; use
        :meth:`radial_factor` to convert between the two.
        """
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        return ((u - self.cx) / self.fx * depth, (v - self.cy) / self.fy * depth, float(depth))

    def radial_factor(self, u: float, v: float) -> float:
        """Range / depth for a pixel: ``sqrt(1 + x_n^2 + y_n^2)`` in normalised units.

        A person in image column ``u`` is not at distance ``Z`` from the camera
        but at ``Z * radial_factor``, so a segment there is imaged smaller than
        the naive ``s = f*S/Z`` predicts, by exactly this factor.

        For a 70 degrees horizontal field of view the factor at the horizontal
        edge is ``1 / cos(35 degrees) = 1.221``. (docs/HEIGHT_ESTIMATION.md §4.3
        quotes ~1.19 for the same effect, which is the value at roughly a 66
        degrees field of view; the formula, not the quoted figure, is what is
        implemented and tested here.)
        """
        x_n = (u - self.cx) / self.fx
        y_n = (v - self.cy) / self.fy
        return math.sqrt(1.0 + x_n * x_n + y_n * y_n)

    def metric_length(self, pixel_length: float, depth: float, at: Optional[Point] = None) -> float:
        """Convert a pixel length at a known depth into metres.

        ``S = s * Z / f``, corrected for the off-axis foreshortening of §4.3 when
        the image position is supplied. Without ``at`` the point is assumed to be
        on the optical axis, which is the same shortcut §4.3 warns about — pass
        the position.
        """
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        focal = math.sqrt(self.fx * self.fy)
        length = pixel_length * depth / focal
        return length * self.radial_factor(*at) if at is not None else length

    def pixels_per_metre(self, depth: float, at: Optional[Point] = None) -> float:
        """Inverse of :meth:`metric_length`: how many pixels one metre spans."""
        return 1.0 / self.metric_length(1.0, depth, at)

    # ------------------------------------------------------------------ #
    def undistort(self, points: Sequence[Point]) -> list:
        """Remove lens distortion from image points, using OpenCV if it is present.

        A camera with a >= 90 degrees field of view shows 3-8% barrel distortion
        at the edges, which is 5-13 cm of height error (§4.4). With a zero
        distortion vector this is the identity, so callers need not branch.
        """
        if not any(self.distortion):
            return [tuple(map(float, p)) for p in points]

        import cv2

        source = np.asarray(points, dtype=float).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(
            source, self.matrix, np.asarray(self.distortion, dtype=float), P=self.matrix
        )
        return [tuple(map(float, p[0])) for p in undistorted]

    def describe(self) -> str:
        """One-line summary for run logs."""
        state = (
            f"rms={self.calibration_rms_px:.3f}px"
            if self.calibration_rms_px is not None
            else "UNCALIBRATED"
        )
        plausible = "ok" if self.focal_is_plausible() else "IMPLAUSIBLE"
        return (
            f"{self.width}x{self.height} f=({self.fx:.1f},{self.fy:.1f}) "
            f"hfov={self.hfov_deg:.1f}deg focal={plausible} {state}"
        )


__all__ = ["CameraModel"]
