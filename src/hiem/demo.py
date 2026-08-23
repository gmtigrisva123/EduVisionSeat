"""Worked demonstration of HIEM on this repository's own images.

::

    python -m src.hiem.demo                       # everything, default images
    python -m src.hiem.demo --images data/pose/*  # a chosen set
    python -m src.hiem.demo --save-dir data/images/output   # annotated copies

Five sections, each answering a different question:

1. **Field measurement.** Students found in the classroom photographs, each with
   their body scale and their signals in pixels and in body scales side by side.
2. **Invariance audit on real students.** Do the pixel signals track apparent
   body size, and does HIEM break that link? In a single photograph apparent size
   is dominated by row depth rather than by height, which makes these images a
   natural experiment for exactly the confound HIEM exists to remove.
3. **Controlled scale sweep.** One real skeleton, resized across a 3x range. The
   behaviour is held fixed by construction, so any change in a HIEM feature is a
   defect, and the exact figure can be read off rather than argued about.
4. **Rotation control.** Confirms the taxonomy in
   :data:`~src.hiem.normalize.FEATURE_SPECS`: similarity features survive an
   image rotation, gravity features are not supposed to and do not.
5. **Temporal validation.** A student turning to a neighbour, built from a real
   skeleton with known ground truth. This is where per-frame normalisation —
   what a reasonable implementation does by default — visibly fails, and where
   the percentile-locked scale earns its complexity.
6. **Equity audit.** The labelled half of the fairness question: train one
   engagement model on pixels and one on HIEM ratios, then compare each model's
   accuracy between shorter and taller students. Necessarily synthetic — these
   images carry no engagement labels, and inventing some would be worse than
   saying so — but the code path is the one a labelled deployment runs.

Person detection here runs MediaPipe over overlapping tiles rather than YOLO.
The pipeline proper uses :class:`~src.detect.person_detector.PersonDetector`;
tiling keeps this demonstration runnable without torch, which matters because
the point of the file is HIEM, not detection.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import REPO_ROOT
from .constants import (
    LEFT_EAR,
    LEFT_EYE,
    LEFT_SHOULDER,
    MEDIAPIPE_TO_ANSUR_CORRECTION,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_SHOULDER,
    SEGMENT_TO_STATURE,
)
from .fairness import HiemAuditReport, audit_features, equity_audit, invariance_audit
from .normalize import FEATURE_SPECS, HiemNormaliser, Invariance, PoseObservation
from .scale import BodyScale, BodyScaleEstimator, measure_segments

logger = logging.getLogger(__name__)

DEFAULT_IMAGES = ("data/images/input/*.jpg", "data/images/input/*.jpeg", "data/pose/*.jpg")
POSE_MODEL = "models/pose_landmarker_full.task"

#: A scale of exactly one pixel. Normalising by it is the identity, so the "raw
#: pixels" column comes out of the SAME code path as the HIEM column and differs
#: only in the denominator. Reimplementing the features a second time to produce
#: the baseline would leave the comparison open to the obvious objection.
RAW_PIXEL_SCALE = BodyScale(value=1.0, state="locked", n_samples=1, weight=1.0)

#: Features shown in the tables. Angles are excluded: HIEM does not touch them,
#: so a before/after column would be identical by construction and would only
#: dilute the ones that mean something.
REPORTED = ("hand_raise", "wrist_gap", "wrist_rise", "hand_to_face", "neck_drop", "neck_axis", "head_width")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


#: Landmarks that must be present and visible for a detection to be believed.
#: MediaPipe will hallucinate a whole skeleton from a fragment of a person left
#: on a tile edge, and those phantoms would otherwise enter the audit as extra
#: students.
CORE_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER)
HEAD_LANDMARKS = (LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

#: Smallest believable shoulder width, in pixels. At the ~3 px landmark noise
#: that docs/HEIGHT_ESTIMATION.md §4.1 assumes, a 12 px shoulder carries 25%
#: relative error, and everything computed from it is noise wearing a number.
MIN_SHOULDER_PX = 12.0

#: Shortest tile side worth running the pose model on. Below this the model is
#: being handed a crop with no room for a person.
MIN_TILE_PX = 280


def _tile_grid(height: int, width: int, maximum: int) -> Tuple[int, int]:
    """Choose the tile count per axis from the image shape.

    A fixed NxN grid is wrong for anything that is not roughly square: a 298x808
    portrait cut into three columns leaves 100 px-wide strips, and MediaPipe
    responds to a strip through the middle of a person by inventing a complete
    skeleton for it. Sizing the grid to the image avoids manufacturing those.
    """
    return (
        max(1, min(maximum, width // MIN_TILE_PX)),
        max(1, min(maximum, height // MIN_TILE_PX)),
    )


def _tiles(height: int, width: int, nx: int, ny: int, overlap: float):
    """Overlapping tile boxes, full frame first. The overlap keeps a student off a seam."""
    yield (0, 0, width, height)
    if nx <= 1 and ny <= 1:
        return
    tile_w = width / (nx - (nx - 1) * overlap) if nx > 1 else width
    tile_h = height / (ny - (ny - 1) * overlap) if ny > 1 else height
    for i, j in itertools.product(range(ny), range(nx)):
        y0, x0 = int(i * tile_h * (1 - overlap)), int(j * tile_w * (1 - overlap))
        yield (x0, y0, int(min(x0 + tile_w, width)), int(min(y0 + tile_h, height)))


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection over union of two landmark bounding boxes."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def detect_poses(
    image_bgr: np.ndarray,
    *,
    model_path: Path,
    tiles: int = 3,
    overlap: float = 0.35,
    num_poses: int = 5,
    min_confidence: float = 0.25,
    min_visibility: float = 0.5,
    merge_iou: float = 0.35,
) -> List[PoseObservation]:
    """Find every student in one frame, in FULL-FRAME pixel coordinates.

    A single full-frame pass of MediaPipe finds one or two students in a
    classroom photograph; over overlapping tiles it finds six to ten, because
    each student then occupies a workable fraction of the input.

    Three filters keep the extra recall from turning into phantom students:
    both shoulders and at least one head landmark must be visible, the shoulder
    span must clear :data:`MIN_SHOULDER_PX`, and overlapping detections are
    merged by bounding-box IoU with the largest kept — the largest being the one
    from the tile where that student was best resolved. Merging on the shoulder
    midpoint alone is not enough, because a partial detection puts the midpoint
    somewhere else entirely.

    The conversion to full-frame pixels is the step to get right: MediaPipe
    normalises x by the tile WIDTH and y by the tile HEIGHT separately, so any
    distance computed before this conversion is wrong by the tile's aspect ratio.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
        )
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    nx, ny = _tile_grid(height, width, tiles)

    candidates: List[Tuple[float, Tuple[float, float, float, float], Dict, Dict]] = []
    try:
        for x0, y0, x1, y1 in _tiles(height, width, nx, ny, overlap):
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            if crop.size == 0:
                continue
            crop_h, crop_w = crop.shape[:2]
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            for pose in result.pose_landmarks:
                points = {
                    index: (landmark.x * crop_w + x0, landmark.y * crop_h + y0)
                    for index, landmark in enumerate(pose)
                }
                visibility = {index: float(landmark.visibility or 0.0) for index, landmark in enumerate(pose)}

                if min(visibility.get(i, 0.0) for i in CORE_LANDMARKS) < min_visibility:
                    continue
                if max(visibility.get(i, 0.0) for i in HEAD_LANDMARKS) < min_visibility:
                    continue
                shoulder = math.dist(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
                if shoulder < MIN_SHOULDER_PX:
                    continue
                xs = [p[0] for p in points.values()]
                ys = [p[1] for p in points.values()]
                candidates.append((shoulder, (min(xs), min(ys), max(xs), max(ys)), points, visibility))
    finally:
        landmarker.close()

    observations: List[PoseObservation] = []
    kept: List[Tuple[float, float, float, float]] = []
    for _, box, points, visibility in sorted(candidates, key=lambda c: -c[0]):
        if any(_iou(box, other) > merge_iou for other in kept):
            continue
        kept.append(box)
        observations.append(PoseObservation(track_id=len(observations), points=points, visibility=visibility))
    return observations


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


@dataclass
class StudentRow:
    """One detected student, measured both ways."""

    image: str
    track_id: int
    scale: Optional[BodyScale]
    raw: Dict[str, Optional[float]]
    hiem: Dict[str, Optional[float]]


def measure(observation: PoseObservation, image: str) -> StudentRow:
    """Run the observation through HIEM, and through the identity, once each."""
    hiem = HiemNormaliser(observation.track_id).observe(observation)
    raw = HiemNormaliser(observation.track_id, fixed_scale=RAW_PIXEL_SCALE).observe(observation)
    return StudentRow(
        image=image,
        track_id=observation.track_id,
        scale=hiem.scale,
        raw={name: raw.as_dict().get(name) for name in REPORTED},
        hiem={name: hiem.as_dict().get(name) for name in REPORTED},
    )


def shoulder_only_scale(points, visibility) -> Optional[float]:
    """The per-frame denominator a straightforward implementation would use.

    Shoulder width alone, in stature-equivalent pixels, from this frame only —
    no percentile, no fusion, no lock. It is the honest baseline to beat, and it
    is what ``src.pose.pose_landmarks`` implicitly divides by when it reports
    ``wrist_gap`` and ``neck_drop`` in shoulder widths.
    """
    segments = measure_segments(points, visibility)
    width = segments.get("shoulder_width")
    if not width:
        return None
    return width * MEDIAPIPE_TO_ANSUR_CORRECTION["shoulder_width"] / SEGMENT_TO_STATURE["shoulder_width"]


def _fmt(value: Optional[float], width: int = 9, places: int = 3) -> str:
    return f"{'--':>{width}}" if value is None else f"{value:>{width}.{places}f}"


def _rule(title: str) -> str:
    return f"\n{title}\n{'=' * 78}"


# --------------------------------------------------------------------------- #
# 1 + 2. Field measurement and the audit on real students
# --------------------------------------------------------------------------- #


def run_field(
    paths: Sequence[Path], model_path: Path, tiles: int, save_dir: Optional[Path]
) -> List[StudentRow]:
    import cv2

    print(_rule("1. Field measurement — students found in the repository's images"))
    rows: List[StudentRow] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"\n  {path.name}: unreadable, skipped")
            continue

        observations = detect_poses(image, model_path=model_path, tiles=tiles)
        print(f"\n  {path.name}  {image.shape[1]}x{image.shape[0]}  ->  {len(observations)} student(s)")
        if not observations:
            continue

        header = "".join(f"{name.replace('_', ' '):>15}" for name in REPORTED)
        print(f"    {'id':>3} {'body scale':>26}  {'unit':<10}{header}")
        for observation in observations:
            row = measure(observation, path.name)
            rows.append(row)
            if row.scale is None:
                scale_text = "unmeasurable"
            elif not row.scale.is_usable:
                # Shown rather than hidden: the number exists, it just cannot be
                # trusted, and section 2 explains which rule rejected it.
                scale_text = f"{row.scale.value:8.1f}px REJECTED     "
            else:
                scale_text = f"{row.scale.value:8.1f}px {row.scale.state:<13}"
            print(
                f"    {row.track_id:>3} {scale_text:>26}  {'pixels':<10}"
                + "".join(_fmt(row.raw[n], 15, 1) for n in REPORTED)
            )
            print(
                f"    {'':>3} {'':>26}  {'HIEM':<10}"
                + "".join(_fmt(row.hiem[n], 15, 4) for n in REPORTED)
            )
        if save_dir is not None:
            _annotate(image, observations, save_dir / f"hiem_{path.stem}.jpg")

    return rows


def _annotate(image_bgr, observations: Sequence[PoseObservation], destination: Path) -> None:
    """Draw each student's box, body scale and hand-raise onto a copy of the frame.

    Students whose scale failed the consistency rule are boxed in red and
    labelled, rather than omitted. Seeing which detections were thrown out is
    how you find out whether the rule is doing its job or eating the class.
    """
    import cv2

    usable_colour, rejected_colour = (60, 220, 60), (60, 60, 235)
    canvas = image_bgr.copy()
    for observation in observations:
        features = HiemNormaliser(observation.track_id).observe(observation)
        usable = features.scale is not None and features.scale.is_usable
        colour = usable_colour if usable else rejected_colour

        xs = [p[0] for p in observation.points.values()]
        ys = [p[1] for p in observation.points.values()]
        top_left = (int(min(xs)), int(min(ys)))
        cv2.rectangle(canvas, top_left, (int(max(xs)), int(max(ys))), colour, 2)
        for index in (LEFT_SHOULDER, RIGHT_SHOULDER):
            if index in observation.points:
                point = observation.points[index]
                cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (0, 200, 255), -1)

        if features.scale is None:
            label = "no scale"
        elif not usable:
            label = f"S={features.scale.value:.0f}px REJECTED"
        else:
            label = f"S={features.scale.value:.0f}px"
            if features.hand_raise is not None:
                label += f" raise={features.hand_raise:+.3f}"
        cv2.putText(
            canvas, label, (top_left[0], max(14, top_left[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)
    print(f"    annotated -> {destination}  (green = usable, red = failed the consistency rule)")


def run_audit(rows: Sequence[StudentRow], n_boot: int) -> None:
    print(_rule("2. Invariance audit on the real students found above"))
    usable = [row for row in rows if row.scale is not None and row.scale.is_usable]
    excluded = [row for row in rows if row not in usable]

    print(
        f"\n  {len(rows)} students detected · {len(usable)} usable · {len(excluded)} excluded"
        " before the audit."
    )
    if excluded:
        print(
            "\n  Excluded because the pose contradicts itself — segments that were measured"
            "\n  disagree by more than projection can explain, so the denominator cannot be"
            "\n  trusted. Publishing a score for these would be the failure mode HIEM exists"
            "\n  to prevent, so they are dropped rather than normalised:\n"
        )
        for row in excluded:
            why = "; ".join(
                v for k, v in (row.scale.reasons if row.scale else {}).items()
                if k in ("consistency", "weight")
            ) or "no measurable body scale"
            print(f"    {row.image:<34} id={row.track_id:<3} {why}")
    if len(usable) < 3:
        print(f"\n  Only {len(usable)} usable students; the audit needs at least 3.")
        return

    scales = [row.scale.value for row in usable]
    report = audit_features(
        raw={name: [row.raw[name] for row in usable] for name in REPORTED},
        normalised={name: [row.hiem[name] for row in usable] for name in REPORTED},
        scale=scales,
        n_boot=n_boot,
        notes=(
            "Every body scale here is 'instantaneous': one photograph affords no percentile and no "
            "lock, which is HIEM's weakest tier by design. Sections 3 and 5 exercise the rest.",
            "One unit is one student. In a still photograph apparent body scale is driven mostly by "
            "row depth, so this measures invariance to apparent size — a superset of height.",
            "A residual correlation is not automatically a HIEM defect: taller students may genuinely "
            "behave differently. Section 3 is the control where behaviour is held fixed by construction.",
        ),
    )
    print()
    print(report.summary())
    print(f"\n  body scales spanned: {min(scales):.0f}px to {max(scales):.0f}px "
          f"({max(scales) / max(min(scales), 1e-9):.1f}x)")


# --------------------------------------------------------------------------- #
# 3. Controlled scale sweep
# --------------------------------------------------------------------------- #


def _transform(points, scale: float, dx: float = 0.0, dy: float = 0.0, rotation_deg: float = 0.0):
    """Apply a similarity transform: uniform scale, rotation, translation."""
    theta = math.radians(rotation_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    return {
        index: (
            scale * (cos * x - sin * y) + dx,
            scale * (sin * x + cos * y) + dy,
        )
        for index, (x, y) in points.items()
    }


def run_sweep(observation: PoseObservation, label: str, n_boot: int, seed: int) -> None:
    print(_rule("3. Controlled scale sweep — one real skeleton, resized across a 3x range"))
    print(
        f"\n  Skeleton from {label}. Each row is the SAME behaviour on a body imaged at a\n"
        "  different size — a taller student, or the same student nearer the camera; the\n"
        "  image cannot tell the two apart, and neither has to.\n"
    )

    factors = [0.6, 0.8, 1.0, 1.3, 1.6, 1.8]
    header = "".join(f"{name.replace('_', ' '):>15}" for name in REPORTED)
    print(f"  {'x':>5} {'unit':<10}{header}")
    raw_table: Dict[str, List[Optional[float]]] = {name: [] for name in REPORTED}
    hiem_table: Dict[str, List[Optional[float]]] = {name: [] for name in REPORTED}
    scales: List[float] = []

    for factor in factors:
        moved = PoseObservation(
            track_id=0,
            points=_transform(observation.points, factor, dx=17.0 * factor, dy=-9.0 * factor),
            visibility=observation.visibility,
        )
        row = measure(moved, label)
        scales.append(row.scale.value)
        for name in REPORTED:
            raw_table[name].append(row.raw[name])
            hiem_table[name].append(row.hiem[name])
        print(f"  {factor:>5.1f} {'pixels':<10}" + "".join(_fmt(row.raw[n], 15, 1) for n in REPORTED))
        print(f"  {'':>5} {'HIEM':<10}" + "".join(_fmt(row.hiem[n], 15, 6) for n in REPORTED))

    print(f"\n  {'feature':<22}{'raw spread':>14}{'HIEM spread':>16}   {'verdict':<32}")
    print(f"  {'-' * 22}{'-' * 14}{'-' * 16}   {'-' * 32}")
    for name in REPORTED:
        raw_values = [v for v in raw_table[name] if v is not None]
        hiem_values = [v for v in hiem_table[name] if v is not None]
        if len(hiem_values) < 2:
            continue
        raw_span = (max(raw_values) - min(raw_values)) / max(abs(np.mean(raw_values)), 1e-12)
        hiem_span = (max(hiem_values) - min(hiem_values)) / max(abs(np.mean(hiem_values)), 1e-12)
        verdict = "invariant to machine precision" if hiem_span < 1e-9 else f"residual {hiem_span:.2e}"
        print(f"  {name:<22}{raw_span:>13.1%}{hiem_span:>16.2e}   {verdict}")

    print("\n  With realistic landmark noise (sigma = 3 px, 40 draws per size):\n")
    _noisy_sweep(observation, factors, n_boot, seed)


def _noisy_sweep(observation: PoseObservation, factors: Sequence[float], n_boot: int, seed: int) -> None:
    """The same sweep, but with the landmark noise a real detector produces.

    Exact invariance is an algebraic property and survives nothing but algebra.
    What matters in the field is whether the dependence on body scale survives
    the noise, so this repeats the sweep with jitter and audits the result.
    """
    rng = np.random.default_rng(seed)
    raw_table: Dict[str, List[Optional[float]]] = {name: [] for name in REPORTED}
    hiem_table: Dict[str, List[Optional[float]]] = {name: [] for name in REPORTED}
    scales: List[float] = []

    for factor in factors:
        for _ in range(40):
            scaled = _transform(observation.points, factor)
            jittered = {i: (x + rng.normal(0, 3.0), y + rng.normal(0, 3.0)) for i, (x, y) in scaled.items()}
            row = measure(
                PoseObservation(track_id=0, points=jittered, visibility=observation.visibility), "noisy"
            )
            if row.scale is None or not row.scale.is_usable:
                continue
            scales.append(row.scale.value)
            for name in REPORTED:
                raw_table[name].append(row.raw[name])
                hiem_table[name].append(row.hiem[name])

    report = audit_features(raw_table, hiem_table, scales, n_boot=n_boot, seed=seed)
    print(report.summary())


# --------------------------------------------------------------------------- #
# 4. Rotation control
# --------------------------------------------------------------------------- #


def run_rotation(observation: PoseObservation, degrees: float = 25.0) -> None:
    print(_rule("4. Rotation control — checking the invariance taxonomy is honest"))
    print(
        f"\n  The image is rotated by {degrees:.0f} degrees. Similarity features are ratios of\n"
        "  lengths and must not move. Gravity features encode 'above the shoulder line',\n"
        "  which a rotation genuinely changes, so they must move — a feature claiming\n"
        "  gravity invariance while surviving a rotation would be mislabelled.\n"
    )
    upright = measure(observation, "upright")
    turned = measure(
        PoseObservation(
            track_id=0,
            points=_transform(observation.points, 1.0, rotation_deg=degrees),
            visibility=observation.visibility,
        ),
        "rotated",
    )

    print(f"  {'feature':<22}{'class':<12}{'upright':>12}{'rotated':>12}{'change':>12}   expected")
    print(f"  {'-' * 22}{'-' * 12}{'-' * 12}{'-' * 12}{'-' * 12}   {'-' * 12}")
    for name in REPORTED:
        before, after = upright.hiem[name], turned.hiem[name]
        if before is None or after is None:
            continue
        spec = FEATURE_SPECS[name]
        delta = abs(after - before) / max(abs(before), 1e-12)
        expected = "unchanged" if spec.invariance is Invariance.SIMILARITY else "changes"
        held = delta < 1e-9
        ok = "ok" if held == (spec.invariance is Invariance.SIMILARITY) else "MISLABELLED"
        print(
            f"  {name:<22}{spec.invariance.value:<12}{before:>12.6f}{after:>12.6f}"
            f"{delta:>12.2e}   {expected:<10} {ok}"
        )


# --------------------------------------------------------------------------- #
# 5. Temporal validation
# --------------------------------------------------------------------------- #


def run_temporal(observation: PoseObservation, label: str, seed: int, n_frames: int = 120) -> None:
    print(_rule("5. Temporal validation — a student turning, and why per-frame normalisation fails"))
    print(
        f"\n  {n_frames} frames built from the {label} skeleton. The student turns towards a\n"
        "  neighbour and back, so the yaw wanders over +/-40 degrees, and 3 px of landmark\n"
        "  noise is added. The behaviour never changes, so the true hand-raise is a constant\n"
        "  and every deviation below is measurement error.\n\n"
        "  The mechanism: a yaw of theta contracts the shoulder width by cos(theta) but leaves\n"
        "  vertical distances alone. Divide a vertical distance by a shoulder width measured in\n"
        "  the same frame and the answer is inflated by 1/cos(theta) — 15% at 30 degrees, and the\n"
        "  inflation arrives exactly when the student turns away, which is when an engagement\n"
        "  model is being asked its hardest question.\n"
    )
    rng = np.random.default_rng(seed)
    points = dict(observation.points)
    centre_x = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2

    truth_row = measure(observation, label)
    if truth_row.scale is None or truth_row.hiem["hand_raise"] is None:
        print("  The reference skeleton has no measurable hand-raise; nothing to validate.")
        return
    truth = truth_row.hiem["hand_raise"]

    estimator = BodyScaleEstimator(0)
    frames: List[PoseObservation] = []
    per_frame: List[float] = []
    for index in range(n_frames):
        yaw = math.radians(40.0) * math.sin(2 * math.pi * index / 55.0)
        turned = {
            i: (centre_x + (x - centre_x) * math.cos(yaw) + rng.normal(0, 3.0), y + rng.normal(0, 3.0))
            for i, (x, y) in points.items()
        }
        frame = PoseObservation(
            track_id=0, points=turned, visibility=observation.visibility,
            frame_index=index, timestamp_s=index / 30.0,
        )
        frames.append(frame)
        estimator.update(turned, observation.visibility)

        baseline_scale = shoulder_only_scale(turned, observation.visibility)
        raw = measure(frame, "t").raw["hand_raise"]
        if baseline_scale and raw is not None:
            per_frame.append(raw / baseline_scale)

    locked = estimator.estimate()
    normaliser = HiemNormaliser(0, fixed_scale=locked)
    hiem_values = [
        f.hand_raise for f in (normaliser.observe(frame) for frame in frames) if f.hand_raise is not None
    ]

    print(f"  body scale after {locked.n_samples} frames: {locked.describe()}\n")
    print(f"  {'estimator':<34}{'mean':>10}{'bias':>10}{'std':>10}{'MAE':>10}")
    print(f"  {'-' * 34}{'-' * 10}{'-' * 10}{'-' * 10}{'-' * 10}")
    for name, values in (
        ("per-frame shoulder width", np.asarray(per_frame)),
        ("HIEM (percentile-locked, fused)", np.asarray(hiem_values)),
    ):
        if values.size == 0:
            continue
        bias = values.mean() - truth
        print(
            f"  {name:<34}{values.mean():>10.4f}{bias:>+10.4f}{values.std():>10.4f}"
            f"{np.abs(values - truth).mean():>10.4f}"
        )
    print(f"  {'ground truth (frontal, noise-free)':<34}{truth:>10.4f}")

    # The comparison that matters here is between the two ESTIMATORS, not
    # between pixels and ratios. Body size is constant through this sequence, so
    # the pixel measurement does not track apparent scale at all — only the
    # DENOMINATOR does, and the question is which denominator lets that leak
    # through into the answer.
    scale_series = [shoulder_only_scale(f.points, f.visibility) for f in frames]
    audit = invariance_audit(
        "hand_raise", per_frame, hiem_values, scale_series, n_boot=500, seed=seed
    )
    print(
        f"\n  Spearman rho against the per-frame apparent scale (yaw is what moves it):"
        f"\n    per-frame shoulder width {audit.rho_raw:+.3f}   ->   HIEM {audit.rho_hiem:+.3f}"
        f"   [95% CI on the drop {audit.reduction_ci[0]:+.3f}, {audit.reduction_ci[1]:+.3f}]"
    )
    print(
        "\n  Read the bias column, not the standard deviation. Noise averages out over a\n"
        "  window; a bias that tracks head yaw does not, and it is the bias that turns\n"
        "  'this student turned away' into 'this student raised their hand higher'."
    )


# --------------------------------------------------------------------------- #
# 6. Equity audit
# --------------------------------------------------------------------------- #


def run_equity(seed: int, n_students: int = 400, n_boot: int = 2000) -> None:
    print(_rule("6. Equity audit — the same model trained on pixels, and on HIEM ratios"))
    print(
        "\n  Synthetic, and it has to be: these photographs carry no engagement labels, and\n"
        "  inventing some to make a chart would be worse than saying so. What is real is the\n"
        "  code path — this is the audit a labelled deployment runs, on the arrangement the\n"
        "  pipeline describes.\n\n"
        f"  {n_students} students. Engagement drives a behaviour ratio identically for everyone;\n"
        "  apparent body scale is independent of engagement, so a fair model cannot gain\n"
        "  anything from it. Each model sees one feature: the pixel measurement, or the same\n"
        "  measurement divided by that student's body scale.\n"
    )
    rng = np.random.default_rng(seed)
    scale = rng.uniform(150.0, 650.0, n_students)
    engagement = rng.uniform(0.0, 1.0, n_students)
    ratio = 0.05 + 0.35 * engagement + rng.normal(0.0, 0.02, n_students)
    pixels = ratio * scale

    split = n_students // 2
    train, test = slice(None, split), slice(split, None)
    results = []
    for name, feature in (("pixels (no HIEM)", pixels), ("HIEM ratios", ratio)):
        slope, intercept = np.polyfit(feature[train], engagement[train], 1)
        prediction = slope * feature[test] + intercept
        result = equity_audit(
            engagement[test], prediction, scale[test], metric_name="MAE",
            label=f"model trained on {name}", n_boot=n_boot, seed=seed,
        )
        results.append((name, result))

    print(f"  {'model input':<20}{'overall MAE':>13}{'shorter half':>15}{'taller half':>14}"
          f"{'gap':>9}{'p':>9}")
    print(f"  {'-' * 20}{'-' * 13}{'-' * 15}{'-' * 14}{'-' * 9}{'-' * 9}")
    for name, result in results:
        total = sum(g.n for g in result.groups)
        overall = sum(g.value * g.n for g in result.groups) / total
        shorter, taller = result.groups[0].value, result.groups[1].value
        verdict = "SIGNIFICANT" if result.significant else "chance"
        print(
            f"  {name:<20}{overall:>13.4f}{shorter:>15.4f}{taller:>14.4f}"
            f"{result.gap:>9.4f}{result.p_value:>9.4f}  {verdict}"
        )

    raw_gap, hiem_gap = results[0][1].gap, results[1][1].gap
    factor = raw_gap / max(hiem_gap, 1e-12)
    print(
        "\n  The pixel model is not merely less accurate overall — its error is UNEVENLY"
        f"\n  distributed, {factor:.1f}x the HIEM model's gap between the two halves of the class."
        "\n  That is the harm the audit exists to find: an overall accuracy figure hides it"
        "\n  completely, and the students it lands on are the same ones every lesson.\n"
    )
    print(HiemAuditReport(equity=(results[0][1], results[1][1])).summary())


def resolve_images(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / pattern
        matches = sorted(glob.glob(str(candidate)))
        paths.extend(Path(m) for m in matches if Path(m).is_file())
    return sorted(dict.fromkeys(paths))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.hiem.demo", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--images", nargs="+", default=list(DEFAULT_IMAGES),
                        help="image paths or glob patterns, relative to the repository root")
    parser.add_argument("--model", default=POSE_MODEL, help="MediaPipe pose landmarker bundle")
    parser.add_argument("--tiles", type=int, default=3, help="tiles per axis for detection")
    parser.add_argument("--save-dir", default=None, help="write annotated copies here")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", default=None,
                        choices=["field", "audit", "sweep", "rotation", "temporal", "equity"],
                        help="run only these sections")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / args.model
    if not model_path.is_file():
        print(f"Pose model not found: {model_path}", file=sys.stderr)
        print("It ships in models/; see src/pose/pose_landmarks.py for the download URL.", file=sys.stderr)
        return 2

    paths = resolve_images(args.images)
    if not paths:
        print(f"No images matched {args.images}", file=sys.stderr)
        return 2

    sections = set(args.only or ["field", "audit", "sweep", "rotation", "temporal", "equity"])
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_absolute():
            save_dir = REPO_ROOT / args.save_dir

    print("HIEM — Height-Invariant Engagement Metrics")
    print(f"images: {len(paths)} · model: {model_path.name} · tiles: {args.tiles}x{args.tiles}")

    rows: List[StudentRow] = []
    if {"field", "audit"} & sections:
        rows = run_field(paths, model_path, args.tiles, save_dir)
    if "audit" in sections and rows:
        run_audit(rows, args.n_boot)

    reference: Optional[PoseObservation] = None
    reference_label = ""
    if {"sweep", "rotation", "temporal"} & sections:
        import cv2

        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            found = detect_poses(image, model_path=model_path, tiles=args.tiles)
            if found:
                # The largest skeleton in the set: the controls need a subject
                # whose landmarks are well resolved, so that a residual is
                # attributable to HIEM rather than to the pose model.
                best = max(found, key=lambda o: math.dist(o.points[LEFT_SHOULDER], o.points[RIGHT_SHOULDER]))
                if reference is None or math.dist(
                    best.points[LEFT_SHOULDER], best.points[RIGHT_SHOULDER]
                ) > math.dist(reference.points[LEFT_SHOULDER], reference.points[RIGHT_SHOULDER]):
                    reference, reference_label = best, path.name

    if "equity" in sections and reference is None:
        run_equity(args.seed, n_boot=args.n_boot)

    if reference is None:
        if {"sweep", "rotation", "temporal"} & sections:
            print("\nNo skeleton found for the controlled sections.", file=sys.stderr)
            return 0 if rows else 1
        return 0

    if "sweep" in sections:
        run_sweep(reference, reference_label, args.n_boot, args.seed)
    if "rotation" in sections:
        run_rotation(reference)
    if "temporal" in sections:
        run_temporal(reference, reference_label, args.seed)
    if "equity" in sections:
        run_equity(args.seed, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
