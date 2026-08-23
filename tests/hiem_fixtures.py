"""Shared skeleton builder for the HIEM tests.

The landmark positions are chosen so that every segment's stature-equivalent
lands on the same value, i.e. a body whose proportions match the population
ratios in :mod:`src.hiem.constants` exactly. That matters: a fixture with
inconsistent proportions would trip the consistency gate, and every test would
then be measuring the gate rather than whatever it meant to measure.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from src.hiem.constants import (
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MEDIAPIPE_TO_ANSUR_CORRECTION as CORRECTION,
    NOSE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SEGMENT_TO_STATURE as RATIO,
)

Point = Tuple[float, float]

#: Stature-equivalent the fixture is built to, in pixels.
REFERENCE_SCALE = 1000.0


def _segment(name: str, stature: float = REFERENCE_SCALE) -> float:
    """Pixel length of a segment on a body of the given stature-equivalent."""
    return stature * RATIO[name] / CORRECTION[name]


def skeleton(
    stature: float = REFERENCE_SCALE,
    *,
    raised_hand: bool = False,
    centre: Point = (400.0, 300.0),
) -> Dict[int, Point]:
    """A seated student whose proportions match the population ratios exactly.

    ``centre`` is the shoulder midpoint. With ``raised_hand`` the subject's left
    wrist goes above the head; everything else is unchanged, so the two variants
    differ in exactly one behaviour.
    """
    cx, cy = centre
    half_shoulder = _segment("shoulder_width") * stature / REFERENCE_SCALE / 2
    upper_arm = _segment("upper_arm") * stature / REFERENCE_SCALE
    forearm = _segment("forearm") * stature / REFERENCE_SCALE
    half_head = _segment("head_width") * stature / REFERENCE_SCALE / 2
    half_hip = _segment("hip_width") * stature / REFERENCE_SCALE / 2
    unit = stature / REFERENCE_SCALE

    points: Dict[int, Point] = {
        LEFT_SHOULDER: (cx + half_shoulder, cy),
        RIGHT_SHOULDER: (cx - half_shoulder, cy),
        LEFT_EAR: (cx + half_head, cy - 100.0 * unit),
        RIGHT_EAR: (cx - half_head, cy - 100.0 * unit),
        LEFT_EYE: (cx + 20.0 * unit, cy - 95.0 * unit),
        RIGHT_EYE: (cx - 20.0 * unit, cy - 95.0 * unit),
        NOSE: (cx, cy - 80.0 * unit),
        LEFT_ELBOW: (cx + half_shoulder, cy + upper_arm),
        RIGHT_ELBOW: (cx - half_shoulder, cy + upper_arm),
        LEFT_HIP: (cx + half_hip, cy + 320.0 * unit),
        RIGHT_HIP: (cx - half_hip, cy + 320.0 * unit),
    }

    # Forearms angled inwards towards the desk, at their full anatomical length.
    inward, downward = 0.5 * forearm, math.sqrt(0.75) * forearm
    points[RIGHT_WRIST] = (cx - half_shoulder + inward, cy + upper_arm + downward)
    if raised_hand:
        points[LEFT_WRIST] = (cx + half_shoulder, cy + upper_arm - 2 * downward)
    else:
        points[LEFT_WRIST] = (cx + half_shoulder - inward, cy + upper_arm + downward)
    return points


def transform(
    points: Dict[int, Point],
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    rotation_deg: float = 0.0,
) -> Dict[int, Point]:
    """Apply a similarity transform: uniform scale, then rotation, then shift."""
    theta = math.radians(rotation_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    return {
        index: (scale * (cos * x - sin * y) + dx, scale * (sin * x + cos * y) + dy)
        for index, (x, y) in points.items()
    }


def yawed(points: Dict[int, Point], yaw_deg: float) -> Dict[int, Point]:
    """Foreshorten horizontally about the body's vertical axis.

    The projection of a rotation about the vertical: horizontal offsets contract
    by ``cos(yaw)`` while vertical ones are untouched. This is the geometry that
    makes a per-frame shoulder width the wrong denominator.
    """
    centre = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2
    factor = math.cos(math.radians(yaw_deg))
    return {index: (centre + (x - centre) * factor, y) for index, (x, y) in points.items()}


def visibility(points: Dict[int, Point], hidden: Optional[Tuple[int, ...]] = None) -> Dict[int, float]:
    """Full visibility everywhere, except the landmarks named in ``hidden``."""
    hidden = hidden or ()
    return {index: (0.05 if index in hidden else 0.99) for index in points}
