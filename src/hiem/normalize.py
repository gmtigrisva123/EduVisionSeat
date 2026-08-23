"""HIEM — Height-Invariant Engagement Metrics.

The normalisation layer that sits between raw pose geometry and any engagement
model. It takes distance-based observations in pixels and returns dimensionless
ratios, so that **two students performing the same behaviour receive the same
numbers regardless of how tall they are or how far from the camera they sit**.

Why this is a fairness problem and not a preprocessing detail
-------------------------------------------------------------
A pixel measurement of "how high is that hand raised" confounds three things:

* the behaviour, which is what is wanted;
* the student's body size, because a taller student has longer arms;
* the distance to the camera, because the back row is imaged smaller.

Train an engagement model on raw pixels and it learns all three. It will then
score the tall student in the front row differently from the short student at
the back who is doing exactly the same thing. Since seating position correlates
with eyesight, conduct and attainment, that error is not random — it lands on
the same children every lesson.

HIEM removes the second and third confounds by construction, not by hoping the
model learns to ignore them. Every distance is divided by ``S``, the student's
own body scale in pixels from :mod:`src.hiem.scale`, which turns a length into a
ratio. Ratios of lengths measured in the same image region are invariant under
the similarity group — uniform scaling, translation and rotation — and body size
and camera distance both act on the image as uniform scalings.

The guarantee, stated precisely
-------------------------------
For any similarity transform ``g: x -> s.R.x + t`` with ``s > 0``:

    ``HIEM(g . P) == HIEM(P)`` for every feature tagged
    :attr:`Invariance.SIMILARITY`, and for every feature tagged
    :attr:`Invariance.GRAVITY` whenever ``R`` is the identity.

Gravity-referenced features — how high a hand is, how far a head has dropped —
must keep the image vertical, because "up" is part of their meaning. They are
invariant to scale and translation but not to rotating the camera. That
distinction is recorded per feature in :data:`FEATURE_SPECS` and the test suite
iterates over that registry rather than over a hand-written list, so a new
feature cannot be added without declaring which class it belongs to.

What HIEM does not do
---------------------
* It does not estimate height in centimetres. ``S`` is a pixel length; see
  :mod:`src.hiem.constants`.
* It does not touch angles. Head yaw, pitch, roll and the torso angle are
  already dimensionless and are passed through untouched — with a test that
  asserts they are bit-identical on the way out.
* It does not absorb the seat geometry. A student at the left-hand edge of the
  room legitimately turns 25-30 degrees to see the board; correcting for that
  needs the per-student yaw baseline of docs/ATTENTION_INDEX.md §5, which is a
  different mechanism for a different confound. HIEM handles body size; the
  baseline handles seat position. Both are required, and neither substitutes
  for the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..pose.pose_landmarks import torso_angle as _torso_angle
from .constants import (
    HAND_RAISE_ENTER,
    HAND_RAISE_ENTER_SAMPLES,
    HAND_RAISE_EXIT,
    HAND_RAISE_EXIT_SAMPLES,
    LEFT_EAR,
    LEFT_EYE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MAX_MOTION_GAP_S,
    MIN_LANDMARK_VISIBILITY,
    MOTION_LANDMARKS,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from .scale import BodyScale, BodyScaleEstimator

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


class Invariance(str, Enum):
    """Which transforms a feature is provably unchanged by."""

    #: Unchanged by scaling, translation AND in-plane rotation. These are pure
    #: ratios of lengths.
    SIMILARITY = "similarity"
    #: Unchanged by scaling and translation only. The image vertical carries
    #: meaning — "above the shoulder line" is not a rotation-invariant claim.
    GRAVITY = "gravity"


@dataclass(frozen=True)
class FeatureSpec:
    """What one HIEM feature is, and what it is guaranteed to be invariant to."""

    name: str
    invariance: Invariance
    unit: str
    description: str
    #: False for angles and rates, which are dimensionless before HIEM sees them.
    divided_by_scale: bool = True


#: The registry. Adding a feature without adding it here makes the invariance
#: test fail, which is the point: the guarantee is enforced, not asserted.
FEATURE_SPECS: Dict[str, FeatureSpec] = {
    spec.name: spec
    for spec in (
        FeatureSpec(
            "hand_raise",
            Invariance.GRAVITY,
            "body scales",
            "Highest wrist above the shoulder line. Positive means raised.",
        ),
        FeatureSpec(
            "wrist_gap",
            Invariance.SIMILARITY,
            "body scales",
            "Separation of the two wrists. Wide means writing, narrow means a phone (§3.B).",
        ),
        FeatureSpec(
            "wrist_rise",
            Invariance.GRAVITY,
            "body scales",
            "Wrist midpoint above the shoulder line.",
        ),
        FeatureSpec(
            "hand_to_face",
            Invariance.SIMILARITY,
            "body scales",
            "Nearest wrist to the eye midpoint. Small means a hand at the face.",
        ),
        FeatureSpec(
            "neck_drop",
            Invariance.GRAVITY,
            "body scales",
            "Eye midpoint below the shoulder line. Negative while upright.",
        ),
        FeatureSpec(
            "neck_axis",
            Invariance.SIMILARITY,
            "body scales",
            "Shoulder midpoint to eye midpoint. Shortens as the student leans in.",
        ),
        FeatureSpec(
            "head_width",
            Invariance.SIMILARITY,
            "body scales",
            "Eye-to-eye span. Combined head size and yaw indicator; contracts as cos(yaw).",
        ),
        FeatureSpec(
            "motion",
            Invariance.SIMILARITY,
            "body scales per second",
            "RMS landmark displacement. Includes the student shifting bodily in the seat.",
        ),
        FeatureSpec(
            "motion_articulated",
            Invariance.SIMILARITY,
            "body scales per second",
            "Motion after the whole-body translation is removed. Limb movement only.",
        ),
        FeatureSpec(
            "torso_angle",
            Invariance.GRAVITY,
            "degrees",
            "Shoulder line to eye midpoint against the vertical. Bypasses HIEM: already scale-free.",
            divided_by_scale=False,
        ),
    )
}


# --------------------------------------------------------------------------- #
# Inputs and outputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PoseObservation:
    """One student in one frame: landmarks in FULL-FRAME pixels.

    Full-frame pixels, not crop-normalised coordinates. MediaPipe normalises x by
    the crop width and y by the crop height separately, so any distance computed
    on normalised coordinates is wrong by the crop's aspect ratio — the same
    warning ``src.pose.pose_landmarks`` gives in its own header.
    """

    track_id: int
    points: Mapping[int, Point]
    visibility: Mapping[int, float] = field(default_factory=dict)
    frame_index: int = 0
    timestamp_s: float = 0.0
    #: Optional head angles, carried through untouched. Anything with ``yaw``,
    #: ``pitch`` and ``roll`` attributes — a ``HeadAngles`` from
    #: :mod:`src.pose.pose_landmarks`, typically.
    angles: Optional[Any] = None

    def visible(self, index: int, threshold: float = MIN_LANDMARK_VISIBILITY) -> bool:
        return index in self.points and self.visibility.get(index, 1.0) >= threshold


@dataclass(frozen=True)
class HiemFeatures:
    """One student in one frame, in dimensionless units.

    Every optional field follows the convention of the rest of the repository:
    ``None`` means NOT MEASURABLE and never zero, with the cause recorded in
    :attr:`reasons`. A hand-raise of 0.0 is a hand at shoulder height; a
    hand-raise of ``None`` is a wrist nobody could see.
    """

    track_id: int
    frame_index: int
    timestamp_s: float
    scale: Optional[BodyScale]

    hand_raise: Optional[float] = None
    wrist_gap: Optional[float] = None
    wrist_rise: Optional[float] = None
    hand_to_face: Optional[float] = None
    neck_drop: Optional[float] = None
    neck_axis: Optional[float] = None
    head_width: Optional[float] = None
    motion: Optional[float] = None
    motion_articulated: Optional[float] = None
    torso_angle: Optional[float] = None

    #: Hysteretic state, not a bare comparison against :data:`HAND_RAISE_ENTER`.
    hand_raised: bool = False
    #: Head angles exactly as they came in. HIEM does not touch them.
    angles: Optional[Any] = None
    #: True on the single frame at which the body scale locked. Downstream
    #: variance windows must be cleared here — see :class:`BodyScaleEstimator`.
    scale_lock_event: bool = False
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """A usable scale is the precondition for every normalised number here."""
        return self.scale is not None and self.scale.is_usable

    @property
    def confidence(self) -> float:
        return self.scale.confidence() if self.scale is not None else 0.0

    def as_dict(self, include_angles: bool = True) -> Dict[str, Optional[float]]:
        """Flat mapping of the numeric features, for logging and the audit."""
        values: Dict[str, Optional[float]] = {name: getattr(self, name) for name in FEATURE_SPECS}
        if include_angles and self.angles is not None:
            for axis in ("yaw", "pitch", "roll"):
                values[axis] = getattr(self.angles, axis, None)
        return values


# --------------------------------------------------------------------------- #
# Hysteresis
# --------------------------------------------------------------------------- #


class _Hysteresis:
    """Schmitt trigger with asymmetric dwell, per docs/ATTENTION_INDEX.md §7.

    Never threshold a raw index directly. A bare comparison flickers on and off
    around the boundary, and each flicker is a false hand-raise reported to a
    teacher. The gap between enter and exit suppresses that, and the longer dwell
    on entry than on exit biases the trigger towards staying off — deliberately.
    """

    def __init__(self, enter: float, exit_: float, enter_samples: int, exit_samples: int) -> None:
        if enter <= exit_:
            raise ValueError(f"enter ({enter}) must exceed exit ({exit_}) for a Schmitt trigger")
        self.enter, self.exit = enter, exit_
        self.enter_samples, self.exit_samples = enter_samples, exit_samples
        self.state = False
        self._streak = 0

    def update(self, value: Optional[float]) -> bool:
        if value is None:
            self._streak = 0
            return self.state
        if self.state:
            self._streak = self._streak + 1 if value < self.exit else 0
            if self._streak >= self.exit_samples:
                self.state, self._streak = False, 0
        else:
            self._streak = self._streak + 1 if value > self.enter else 0
            if self._streak >= self.enter_samples:
                self.state, self._streak = True, 0
        return self.state


# --------------------------------------------------------------------------- #
# The normaliser
# --------------------------------------------------------------------------- #


class HiemNormaliser:
    """HIEM for one tracked student. Feed it observations, get ratios back.

    Holds the body-scale estimator, the previous frame (for movement) and the
    hand-raise trigger. One instance per track id; :class:`HiemTracker` manages
    a roomful.
    """

    def __init__(
        self,
        track_id: int = 0,
        *,
        estimator: Optional[BodyScaleEstimator] = None,
        min_visibility: float = MIN_LANDMARK_VISIBILITY,
        fixed_scale: Optional[BodyScale] = None,
    ) -> None:
        self.track_id = track_id
        self.min_visibility = min_visibility
        self.estimator = estimator or BodyScaleEstimator(track_id, min_visibility=min_visibility)
        #: Set by the two-pass batch path so every frame divides by the same
        #: final scale rather than by whatever was known at the time.
        self.fixed_scale = fixed_scale
        self._hand = _Hysteresis(
            HAND_RAISE_ENTER, HAND_RAISE_EXIT, HAND_RAISE_ENTER_SAMPLES, HAND_RAISE_EXIT_SAMPLES
        )
        self._previous: Optional[PoseObservation] = None

    # ------------------------------------------------------------------ #
    def observe(self, observation: PoseObservation) -> HiemFeatures:
        """Normalise one frame."""
        reasons: Dict[str, str] = {}
        points = observation.points

        if self.fixed_scale is not None:
            scale: Optional[BodyScale] = self.fixed_scale
            lock_event = False
        else:
            self.estimator.update(points, observation.visibility)
            scale = self.estimator.estimate()
            lock_event = self.estimator.just_locked

        if scale is None or not scale.is_usable:
            reasons["scale"] = "no usable body scale" if scale is None else "scale below the weight floor"
            self._previous = observation
            return HiemFeatures(
                track_id=observation.track_id,
                frame_index=observation.frame_index,
                timestamp_s=observation.timestamp_s,
                scale=scale,
                angles=observation.angles,
                scale_lock_event=lock_event,
                reasons=reasons,
            )

        s = scale.value
        shoulder_mid = self._midpoint(observation, LEFT_SHOULDER, RIGHT_SHOULDER)
        eye_mid = self._midpoint(observation, LEFT_EYE, RIGHT_EYE) or self._midpoint(
            observation, LEFT_EAR, RIGHT_EAR
        )
        wrists = {
            side: points[index]
            for side, index in (("left", LEFT_WRIST), ("right", RIGHT_WRIST))
            if observation.visible(index, self.min_visibility)
        }

        if shoulder_mid is None:
            reasons["shoulder_mid"] = "shoulders not visible"
        if eye_mid is None:
            reasons["eye_mid"] = "neither eyes nor ears visible"
        if not wrists:
            reasons["wrists"] = "no wrist visible"

        # -- gravity-referenced: the image vertical is part of the meaning ---
        hand_raise = wrist_rise = neck_drop = None
        if shoulder_mid is not None and wrists:
            hand_raise = max((shoulder_mid[1] - w[1]) for w in wrists.values()) / s
        if shoulder_mid is not None and len(wrists) == 2:
            wrist_mid = self._mean(list(wrists.values()))
            wrist_rise = (shoulder_mid[1] - wrist_mid[1]) / s
        if shoulder_mid is not None and eye_mid is not None:
            neck_drop = (eye_mid[1] - shoulder_mid[1]) / s

        # -- similarity-invariant: pure ratios of lengths --------------------
        wrist_gap = None
        if len(wrists) == 2:
            wrist_gap = self._distance(wrists["left"], wrists["right"]) / s
        hand_to_face = None
        if eye_mid is not None and wrists:
            hand_to_face = min(self._distance(eye_mid, w) for w in wrists.values()) / s
        neck_axis = None
        if shoulder_mid is not None and eye_mid is not None:
            neck_axis = self._distance(eye_mid, shoulder_mid) / s
        head_width = None
        if observation.visible(LEFT_EYE, self.min_visibility) and observation.visible(
            RIGHT_EYE, self.min_visibility
        ):
            head_width = self._distance(points[LEFT_EYE], points[RIGHT_EYE]) / s

        motion, motion_articulated = self._motion(observation, s, reasons)

        features = HiemFeatures(
            track_id=observation.track_id,
            frame_index=observation.frame_index,
            timestamp_s=observation.timestamp_s,
            scale=scale,
            hand_raise=hand_raise,
            wrist_gap=wrist_gap,
            wrist_rise=wrist_rise,
            hand_to_face=hand_to_face,
            neck_drop=neck_drop,
            neck_axis=neck_axis,
            head_width=head_width,
            motion=motion,
            motion_articulated=motion_articulated,
            torso_angle=_torso_angle(dict(points)),
            hand_raised=self._hand.update(hand_raise),
            angles=observation.angles,
            scale_lock_event=lock_event,
            reasons=reasons,
        )
        self._previous = observation
        return features

    # ------------------------------------------------------------------ #
    def _motion(
        self, observation: PoseObservation, s: float, reasons: Dict[str, str]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Displacement per second, in body scales.

        Two numbers, because they answer different questions. ``motion`` counts
        the student shifting bodily in the seat; ``motion_articulated`` removes
        the common translation first and so isolates limb movement. Camera shake
        and bounding-box jitter land almost entirely in the common translation,
        which is the other reason to separate them.
        """
        previous = self._previous
        if previous is None:
            return None, None

        dt = observation.timestamp_s - previous.timestamp_s
        if dt <= 0:
            reasons["motion"] = "non-increasing timestamp"
            return None, None
        if dt > MAX_MOTION_GAP_S:
            reasons["motion"] = f"gap of {dt:.1f}s exceeds {MAX_MOTION_GAP_S:.1f}s"
            return None, None

        shared = [
            index
            for index in MOTION_LANDMARKS
            if previous.visible(index, self.min_visibility)
            and observation.visible(index, self.min_visibility)
        ]
        if len(shared) < 3:
            reasons["motion"] = f"only {len(shared)} landmarks shared with the previous frame"
            return None, None

        delta = np.array(
            [
                (
                    observation.points[i][0] - previous.points[i][0],
                    observation.points[i][1] - previous.points[i][1],
                )
                for i in shared
            ],
            dtype=float,
        )
        total = float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))) / (s * dt)
        residual = delta - delta.mean(axis=0)
        articulated = float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))) / (s * dt)
        return total, articulated

    # ------------------------------------------------------------------ #
    def _midpoint(self, observation: PoseObservation, a: int, b: int) -> Optional[Point]:
        if not (observation.visible(a, self.min_visibility) and observation.visible(b, self.min_visibility)):
            return None
        return self._mean([observation.points[a], observation.points[b]])

    @staticmethod
    def _mean(points: Sequence[Point]) -> Point:
        array = np.asarray(points, dtype=float)
        return (float(array[:, 0].mean()), float(array[:, 1].mean()))

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # ------------------------------------------------------------------ #
    def observe_person_features(self, features: Any) -> HiemFeatures:
        """Retrofit HIEM onto a :class:`~src.pose.pose_landmarks.PersonFeatures`.

        That class already reports ``wrist_gap``, ``wrist_rise`` and
        ``neck_drop`` in shoulder widths — a per-frame denominator. This method
        multiplies them back into pixels using the frame's shoulder width and
        re-divides by the locked body scale, which upgrades all three from a
        noisy, foreshortening-sensitive divisor to a stable per-student constant
        without a single line of ``src.pose.pose_landmarks`` changing.

        Only the subset that ``PersonFeatures`` carries can be recovered: it
        exposes derived values rather than landmarks, so movement and hand-raise
        need :meth:`observe` and the full landmark dictionary.
        """
        reasons: Dict[str, str] = {}
        shoulder_px = getattr(features, "shoulder_width", None)
        if not shoulder_px:
            reasons["scale"] = "PersonFeatures carries no shoulder width"
            return HiemFeatures(
                track_id=getattr(features, "track_id", self.track_id),
                frame_index=getattr(features, "frame_index", 0),
                timestamp_s=getattr(features, "timestamp_s", 0.0),
                scale=None,
                angles=getattr(features, "angles", None),
                reasons=reasons,
            )

        if self.fixed_scale is not None:
            scale: Optional[BodyScale] = self.fixed_scale
            lock_event = False
        else:
            segments = {"shoulder_width": float(shoulder_px)}
            iod = getattr(features, "iod", None)
            self.estimator.update_segments(segments)
            if iod:
                reasons["iod"] = "interocular distance logged, not fused: no stature ratio is defined for it"
            scale = self.estimator.estimate()
            lock_event = self.estimator.just_locked

        if scale is None or not scale.is_usable:
            reasons["scale"] = "no usable body scale"
            return HiemFeatures(
                track_id=getattr(features, "track_id", self.track_id),
                frame_index=getattr(features, "frame_index", 0),
                timestamp_s=getattr(features, "timestamp_s", 0.0),
                scale=scale,
                angles=getattr(features, "angles", None),
                scale_lock_event=lock_event,
                reasons=reasons,
            )

        # PersonFeatures states these in shoulder widths; restore the pixels.
        ratio = float(shoulder_px) / scale.value
        hands = getattr(features, "hands", None)

        def rescale(value: Optional[float]) -> Optional[float]:
            return None if value is None else float(value) * ratio

        neck_drop = rescale(getattr(features, "neck_drop", None))
        wrist_gap = rescale(getattr(hands, "wrist_gap", None)) if hands else None
        wrist_rise = rescale(getattr(hands, "wrist_rise", None)) if hands else None
        hand_raise = wrist_rise  # the best available proxy: only the midpoint is reported

        return HiemFeatures(
            track_id=getattr(features, "track_id", self.track_id),
            frame_index=getattr(features, "frame_index", 0),
            timestamp_s=getattr(features, "timestamp_s", 0.0),
            scale=scale,
            hand_raise=hand_raise,
            wrist_gap=wrist_gap,
            wrist_rise=wrist_rise,
            neck_drop=neck_drop,
            head_width=(float(getattr(features, "iod", 0) or 0) / scale.value) or None,
            torso_angle=getattr(features, "torso_angle", None),
            hand_raised=self._hand.update(hand_raise),
            angles=getattr(features, "angles", None),
            scale_lock_event=lock_event,
            reasons=reasons,
        )


# --------------------------------------------------------------------------- #
# Many students at once
# --------------------------------------------------------------------------- #


class HiemTracker:
    """One :class:`HiemNormaliser` per track id, created on demand.

    Track ids are recycled by every multi-object tracker sooner or later, and a
    recycled id inheriting the previous student's body scale would silently
    normalise one child by another child's proportions. ``retire`` exists to make
    that impossible; call it when the tracker drops an id.
    """

    def __init__(self, **normaliser_kwargs: Any) -> None:
        self._normalisers: Dict[int, HiemNormaliser] = {}
        self._kwargs = normaliser_kwargs

    def __len__(self) -> int:
        return len(self._normalisers)

    @property
    def track_ids(self) -> List[int]:
        return sorted(self._normalisers)

    def normaliser(self, track_id: int) -> HiemNormaliser:
        if track_id not in self._normalisers:
            self._normalisers[track_id] = HiemNormaliser(track_id, **self._kwargs)
        return self._normalisers[track_id]

    def observe(self, observation: PoseObservation) -> HiemFeatures:
        return self.normaliser(observation.track_id).observe(observation)

    def observe_frame(self, observations: Iterable[PoseObservation]) -> List[HiemFeatures]:
        return [self.observe(observation) for observation in observations]

    def scales(self) -> Dict[int, Optional[BodyScale]]:
        return {tid: n.estimator.estimate() for tid, n in self._normalisers.items()}

    def retire(self, track_id: int) -> None:
        self._normalisers.pop(track_id, None)


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #


def normalise_sequence(
    observations: Sequence[PoseObservation],
    *,
    two_pass: bool = True,
    **normaliser_kwargs: Any,
) -> List[HiemFeatures]:
    """Normalise a whole recording, optionally in two passes.

    Online, the first frames of a track can only be divided by an
    ``instantaneous`` scale, so early features are systematically noisier than
    late ones — an artefact that a model trained on the output will happily learn.
    When the whole recording is in hand there is no reason to accept that: pass
    one estimates each student's scale from every frame available, pass two
    normalises everything by that one final value.

    Use ``two_pass=False`` to reproduce exactly what a live run would have
    produced, which is what an honest latency or deployment evaluation needs.
    """
    if not two_pass:
        tracker = HiemTracker(**normaliser_kwargs)
        return [tracker.observe(observation) for observation in observations]

    estimators: Dict[int, BodyScaleEstimator] = {}
    for observation in observations:
        estimator = estimators.setdefault(
            observation.track_id,
            BodyScaleEstimator(
                observation.track_id,
                min_visibility=normaliser_kwargs.get("min_visibility", MIN_LANDMARK_VISIBILITY),
            ),
        )
        estimator.update(observation.points, observation.visibility)

    final: Dict[int, Optional[BodyScale]] = {tid: est.estimate() for tid, est in estimators.items()}
    normalisers: Dict[int, HiemNormaliser] = {}
    results: List[HiemFeatures] = []
    for observation in observations:
        tid = observation.track_id
        if tid not in normalisers:
            normalisers[tid] = HiemNormaliser(tid, fixed_scale=final.get(tid), **normaliser_kwargs)
        results.append(normalisers[tid].observe(observation))
    return results


__all__ = [
    "FEATURE_SPECS",
    "FeatureSpec",
    "HiemFeatures",
    "HiemNormaliser",
    "HiemTracker",
    "Invariance",
    "PoseObservation",
    "normalise_sequence",
]
