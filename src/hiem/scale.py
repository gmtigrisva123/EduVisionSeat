"""Body-scale estimation: the denominator that makes HIEM fair.

The whole module exists to answer one question well:

    *Given a stream of noisy 2-D landmarks for one student, what single pixel
    length best represents that student's body size?*

Getting it wrong quietly ruins everything downstream, because that length
divides every distance-based signal. Four things make it hard, and the module
addresses them in this order:

1. **Foreshortening.** Out-of-plane rotation projects a rigid segment SHORTER,
   never longer. A student who turns to a neighbour has their shoulder width
   contract by ``cos(yaw)`` — 13% at 30 degrees. Dividing by that inflates every
   ratio at exactly the moment the student moves.
   *Fix:* aggregate each segment at the upper percentile of a long window rather
   than per frame. The upper tail is the unforeshortened length, because nothing
   can push a projection above it. docs/HEIGHT_ESTIMATION.md §4.2.

2. **Noise in the denominator.** Landmark jitter of ~3 px is 2.6% of a shoulder
   width at 3 m and 10.3% at 12 m. Noise in a divisor is worse than noise in a
   dividend: it is not merely noisy but BIASED, because ``E[x/S] != x/E[S]``, so
   per-frame normalisation inflates every ratio for the back rows specifically.
   *Fix:* divide by one locked constant per student, not by a per-frame value.

3. **Which segment.** The upper arm is the best available indicator and its
   ratio to stature is sex-invariant to 0.1%; shoulder width is always visible
   but carries a 5.5% sex difference; the head is nearly useless at distance.
   *Fix:* measure all of them, put them on a common footing through the stature
   ratios in :mod:`src.hiem.constants`, and fuse with inverse-variance weights.

4. **Outliers, and they are not rare.** A desk hides the arms, and MediaPipe
   answers occlusion by extrapolating a plausible limb rather than by lowering
   its visibility score. Measured on this repository's own classroom images, an
   elbow misplaced badly enough to imply an upper arm 7.5x too long still
   carried a visibility of 0.96 — so a visibility gate, which is the obvious
   defence, does not catch these at all.
   *Fix:* two layers. A geometric consistency gate (:func:`gate_segments`)
   discards a segment that disagrees with the rest of the body by more than
   projection can explain, and a Huber M-estimator fuses whatever survives.

What this module deliberately does NOT do is convert the result to centimetres.
See :mod:`src.hiem.constants` and ``docs/HEIGHT_ESTIMATION.md`` §7.1.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .camera import CameraModel
from .constants import (
    CONSISTENCY_ANCHOR,
    FORESHORTENING_PERCENTILE,
    FUSION_WEIGHTS,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    LOCK_AFTER_SAMPLES,
    MAX_SCALE_DISPERSION,
    MEDIAPIPE_TO_ANSUR_CORRECTION,
    MIN_LANDMARK_VISIBILITY,
    MIN_SAMPLES_PER_SEGMENT,
    MIN_SEGMENT_WEIGHT,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SCALE_WINDOW_SAMPLES,
    SEGMENT_CONSISTENCY_RATIO,
    SEGMENT_TO_STATURE,
)

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

#: Which landmark pairs make up each segment. A segment with two sides is
#: measured on both and the LONGER is kept — a spatial version of the percentile
#: trick, and correct for the same reason: the shorter side is the more
#: foreshortened one.
SEGMENT_PAIRS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "shoulder_width": ((LEFT_SHOULDER, RIGHT_SHOULDER),),
    "upper_arm": ((LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)),
    "forearm": ((LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST)),
    "head_width": ((LEFT_EAR, RIGHT_EAR),),
    "hip_width": ((LEFT_HIP, RIGHT_HIP),),
}


# --------------------------------------------------------------------------- #
# Robust statistics. Pure numpy, no scipy — this has to run wherever the
# pipeline runs.
# --------------------------------------------------------------------------- #


def weighted_median(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Value at which the cumulative weight first reaches half the total."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("weighted_median needs at least one value")
    if weights is None:
        return float(np.median(array))

    w = np.asarray(weights, dtype=float)
    if w.shape != array.shape:
        raise ValueError(f"weights shape {w.shape} does not match values shape {array.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        return float(np.median(array))

    order = np.argsort(array)
    cumulative = np.cumsum(w[order])
    return float(array[order][int(np.searchsorted(cumulative, total / 2.0))])


def robust_dispersion(values: Sequence[float]) -> float:
    """Relative MAD: ``1.4826 * MAD / |median|``, i.e. a robust coefficient of variation.

    The 1.4826 makes it agree with the standard deviation for Gaussian data,
    while a single wild sample moves it by nothing.
    """
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    centre = float(np.median(array))
    if abs(centre) < 1e-9:
        return float("inf")
    return float(1.4826 * np.median(np.abs(array - centre)) / abs(centre))


def huber_location(
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    c: float = 1.345,
    max_iter: int = 30,
    tol: float = 1e-9,
) -> float:
    """Weighted Huber M-estimate of location, by iteratively reweighted least squares.

    Behaves like a weighted mean for clean data and like a weighted median once a
    sample sits more than ``c`` robust standard deviations out — which is what a
    mis-detected elbow looks like. ``c = 1.345`` is the classical constant giving
    95% efficiency under normality.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("huber_location needs at least one value")
    if array.size == 1:
        return float(array[0])

    w = np.ones_like(array) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != array.shape:
        raise ValueError(f"weights shape {w.shape} does not match values shape {array.shape}")

    location = weighted_median(array, w)
    for _ in range(max_iter):
        residual = array - location
        spread = 1.4826 * weighted_median(np.abs(residual), w)
        if spread < 1e-12:
            return float(location)
        scaled = residual / spread
        # Huber psi(u)/u: unity in the quadratic core, c/|u| in the linear tails.
        gain = np.where(np.abs(scaled) <= c, 1.0, c / np.maximum(np.abs(scaled), 1e-12))
        effective = w * gain
        total = effective.sum()
        if total <= 0:
            return float(location)
        updated = float((effective * array).sum() / total)
        if abs(updated - location) < tol * max(1.0, abs(location)):
            return updated
        location = updated
    return float(location)


def foreshortening_corrected_length(
    samples: Sequence[float],
    percentile: float = FORESHORTENING_PERCENTILE,
    min_samples: int = MIN_SAMPLES_PER_SEGMENT,
) -> Optional[float]:
    """Recover the unforeshortened projected length of a rigid segment.

    An observed length is ``L_true * cos(theta) * (1 + noise)``, and ``cos`` is
    bounded above by 1, so the observation distribution has its support capped at
    ``L_true`` up to noise. The upper percentile therefore estimates ``L_true``
    **and** removes the rotation bias in a single step, which is why this is the
    single most valuable trick in the pipeline (docs/HEIGHT_ESTIMATION.md §4.2).

    The maximum would be the obvious estimator and is the wrong one: it tracks
    the upper tail of the landmark noise instead of the geometry. 92.5 sits far
    enough up to have shed the foreshortening and far enough down to have shed
    the noise.

    Returns ``None`` — never a number — when there are too few samples for a
    percentile to mean anything.
    """
    array = np.asarray([s for s in samples if s is not None and np.isfinite(s) and s > 0], dtype=float)
    if array.size < min_samples:
        return None
    return float(np.percentile(array, percentile))


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BodyScale:
    """One student's body scale: the denominator HIEM divides by.

    ``value`` is a length in PIXELS expressed in stature-equivalent units — that
    is, the pixel length the student's full stature would subtend at this depth,
    inferred from the limb segments that are actually visible. It is a
    normalisation constant, **not a height measurement**; see the module header
    of :mod:`src.hiem.constants`.
    """

    value: float
    #: ``instantaneous`` (one frame), ``provisional`` (percentile, not yet
    #: locked) or ``locked``. Only ``locked`` gives the stable denominator that
    #: the fairness argument depends on.
    state: str
    n_samples: int
    #: Sum of the fusion weights of the segments that contributed. Compare
    #: against ``MIN_SEGMENT_WEIGHT`` — the validity floor of the scale.
    weight: float
    #: Summed fusion weight of the segments that WERE measured but failed the
    #: consistency gate. The distinction from a simply low ``weight`` is the
    #: whole point: a student whose arms are behind a desk contributes no arm
    #: segments and is measured from the shoulders alone, which is fine; a
    #: student whose arms were measured and contradict the rest of the body is a
    #: detection failure wearing the same low weight, and is not fine.
    rejected_weight: float = 0.0
    #: Per-segment stature-equivalent estimates, in pixels, before fusion.
    segments: Dict[str, float] = field(default_factory=dict)
    #: Robust relative spread of the per-frame instantaneous scale. This is how
    #: much foreshortening and noise the percentile step had to remove, so it
    #: reads high for a student who turns a lot — informative, not a defect.
    dispersion: float = 0.0
    #: Largest relative disagreement between two contributing segments. High
    #: values mean either a landmark error or a student whose proportions depart
    #: from the population ratios; both are worth surfacing.
    agreement: float = 0.0
    #: False unless a camera model AND a seat depth were supplied. Without them
    #: a bigger ``value`` may mean a taller student or merely a nearer one, so
    #: ranking students by size is not permitted.
    comparable_across_students: bool = False
    #: Metres, and only ever set when ``comparable_across_students`` is true.
    metric_stature_m: Optional[float] = None
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def locked(self) -> bool:
        return self.state == "locked"

    @property
    def is_usable(self) -> bool:
        """A finite length, enough surviving weight, and a body that agrees with itself.

        The last clause is the one that earns its keep. On this repository's own
        images a student was detected with a 16 px shoulder span and a 100 px
        upper arm — physically impossible, and yet every landmark carried a
        visibility above 0.95. The gate rejects the arms, the shoulders alone
        clear the weight floor, and without this clause a body scale of 76 px
        would be published for a student whose true scale is several times that.
        Requiring the survivors to outweigh the rejects catches it.
        """
        if not (self.value > 0 and np.isfinite(self.value)):
            return False
        if self.weight < MIN_SEGMENT_WEIGHT:
            return False
        return self.rejected_weight <= self.weight

    @property
    def is_stable(self) -> bool:
        """Locked, and not thrashing. The precondition for trusting a comparison."""
        return self.locked and self.dispersion <= MAX_SCALE_DISPERSION

    @property
    def shoulder_equivalent(self) -> float:
        """The same scale expressed in shoulder widths.

        ``src.pose.pose_landmarks`` reports ``wrist_gap``, ``wrist_rise`` and
        ``neck_drop`` in shoulder widths, and docs/ATTENTION_INDEX.md §3.B states
        its thresholds in those units. This property converts without either
        module having to know about the other.
        """
        return self.value * SEGMENT_TO_STATURE["shoulder_width"]

    def confidence(self) -> float:
        """A single number in [0, 1] summarising how much to trust this scale.

        The product of four independent things going right: enough segment
        weight, a body that agrees with itself, a settled temporal estimate, and
        low dispersion. Multiplicative rather than additive because any one of
        them failing is disqualifying, and an average would let three good terms
        hide a fatal fourth.
        """
        if not self.is_usable:
            return 0.0
        measured = self.weight + self.rejected_weight
        weight_term = min(1.0, self.weight / 0.65)
        agreement_term = self.weight / measured if measured > 0 else 1.0
        state_term = {"locked": 1.0, "provisional": 0.6, "instantaneous": 0.3}.get(self.state, 0.0)
        spread_term = max(0.0, 1.0 - self.dispersion / MAX_SCALE_DISPERSION) if self.n_samples > 1 else 1.0
        return float(weight_term * agreement_term * state_term * min(1.0, spread_term))

    def describe(self) -> str:
        metric = f" ~{self.metric_stature_m * 100:.0f}cm" if self.metric_stature_m else ""
        dropped = f"/{self.rejected_weight:.2f}rej" if self.rejected_weight else ""
        return (
            f"S={self.value:.1f}px[{self.state}] n={self.n_samples} w={self.weight:.2f}{dropped} "
            f"disp={self.dispersion:.3f} conf={self.confidence():.2f}"
            f" segs={{{', '.join(f'{k}:{v:.0f}' for k, v in sorted(self.segments.items()))}}}{metric}"
        )


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def measure_segments(
    points: Mapping[int, Point],
    visibility: Optional[Mapping[int, float]] = None,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> Dict[str, float]:
    """Pixel length of every body segment that is measurable in this frame.

    Both endpoints must be present and visible enough. A two-sided segment keeps
    the LONGER side, for the reason given at :data:`SEGMENT_PAIRS`.

    Missing segments are simply absent from the result — never zero. A zero
    shoulder width would sail through the fusion and produce a denominator that
    silently divides by nothing.
    """
    visibility = visibility or {}
    measured: Dict[str, float] = {}

    for name, pairs in SEGMENT_PAIRS.items():
        best = 0.0
        for a, b in pairs:
            if a not in points or b not in points:
                continue
            if visibility.get(a, 1.0) < min_visibility or visibility.get(b, 1.0) < min_visibility:
                continue
            (ax, ay), (bx, by) = points[a], points[b]
            length = float(np.hypot(ax - bx, ay - by))
            if length > best:
                best = length
        if best > 1e-6:
            measured[name] = best
    return measured


def stature_equivalents(segments: Mapping[str, float]) -> Dict[str, float]:
    """Map each segment's pixel length onto a common stature-equivalent scale.

    ``H_j = L_j * mediapipe_correction_j / ratio_j``. The correction restates a
    MediaPipe joint-centre segment in ANSUR landmark terms; the ratio then
    divides out the anatomy, leaving one comparable pixel length per segment.
    """
    return {
        name: length * MEDIAPIPE_TO_ANSUR_CORRECTION.get(name, 1.0) / SEGMENT_TO_STATURE[name]
        for name, length in segments.items()
        if name in SEGMENT_TO_STATURE and length > 0
    }


def gate_segments(
    equivalents: Mapping[str, float],
    anchor: Optional[str] = CONSISTENCY_ANCHOR,
    ratio: float = SEGMENT_CONSISTENCY_RATIO,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Discard segments that disagree with the body by more than projection allows.

    Two segments of one student, each divided by its own population ratio to
    stature, should land close together: individual variation in limb proportion
    is a few percent. Projection widens that — a segment ``ratio`` out of plane
    shortens by that factor and the anchor may be foreshortened too — but not
    without limit. A segment landing outside the window is not an unusual body,
    it is a landmark in the wrong place, and fusing it corrupts the denominator
    that every downstream signal is divided by.

    This is a Huber estimator's weak point and the reason it is not enough on its
    own. Huber downweights an outlier relative to the *consensus*, and with four
    or five segments carrying uneven weights the consensus is itself movable: a
    single confidently-wrong arm with the largest fusion weight becomes the
    consensus and the correct shoulder measurement is what gets downweighted.
    Anchoring the window on a segment chosen in advance removes that failure.

    Returns ``(kept, rejected)`` where ``rejected`` maps each discarded segment
    to the reason, so the caller can report it rather than silently lose it.
    """
    usable = {k: float(v) for k, v in equivalents.items() if v and np.isfinite(v) and v > 0}
    if len(usable) < 2:
        return usable, {}

    if anchor is not None and anchor in usable:
        centre = usable[anchor]
    else:
        names = sorted(usable)
        centre = weighted_median(
            [usable[n] for n in names], [FUSION_WEIGHTS.get(n, 0.0) for n in names]
        )
    if centre <= 0:
        return usable, {}

    low, high = centre / ratio, centre * ratio
    kept: Dict[str, float] = {}
    rejected: Dict[str, str] = {}
    for name, value in usable.items():
        if low <= value <= high:
            kept[name] = value
        else:
            rejected[name] = f"{value / centre:.2f}x the {anchor or 'consensus'} estimate"
    return (kept, rejected) if kept else (usable, {})


def fuse_segments(
    equivalents: Mapping[str, float],
    anchor: Optional[str] = CONSISTENCY_ANCHOR,
) -> Tuple[Optional[float], float, float, Dict[str, str]]:
    """Gate, then fuse, per-segment stature-equivalents into one scale.

    Returns ``(value, weight, agreement, rejected)``. ``weight`` is the summed
    fusion weight of the segments that survived the gate — so a body measured
    from one segment reports a low weight and the caller can refuse it — and
    ``agreement`` is the largest relative disagreement among the survivors.
    """
    kept, rejected = gate_segments(equivalents, anchor)
    if not kept:
        return None, 0.0, 0.0, rejected

    names = sorted(kept)
    values = np.array([kept[n] for n in names], dtype=float)
    weights = np.array([FUSION_WEIGHTS.get(n, 0.0) for n in names], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return None, 0.0, 0.0, rejected

    fused = huber_location(values, weights)
    spread = float(values.max() - values.min()) / fused if fused > 0 and values.size > 1 else 0.0
    return fused, total_weight, spread, rejected


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #


class BodyScaleEstimator:
    """Accumulates observations of one student and reports their body scale.

    One instance per track id. The estimator moves through three states, and the
    state is reported rather than hidden, because a caller normalising by an
    ``instantaneous`` scale is doing something much weaker than one normalising
    by a ``locked`` scale and is entitled to know:

    ``instantaneous``
        Fewer than :data:`MIN_SAMPLES_PER_SEGMENT` frames. The scale is whatever
        the latest frame says, foreshortening and all. This is what a single
        still photograph gets, and it is honest about being the weakest tier.
    ``provisional``
        Enough frames for the percentile to mean something, not yet enough to
        lock. Foreshortening is corrected; the value still moves.
    ``locked``
        Past :data:`LOCK_AFTER_SAMPLES`. The denominator is now a per-student
        constant, which is what removes the ``E[x/S] != x/E[S]`` bias.

    **The lock is a step change**, and downstream windows must be told. This
    mirrors the concern raised for the yaw baseline in
    docs/ATTENTION_INDEX.md §5: when a baseline locks, every derived signal
    jumps, and a variance-based signal reads that jump as instability —
    penalising the student for the system having finished calibrating.
    :attr:`just_locked` is true for exactly the one observation on which the
    transition happened, so a caller can clear its windows.
    """

    def __init__(
        self,
        track_id: int = 0,
        *,
        percentile: float = FORESHORTENING_PERCENTILE,
        min_samples: int = MIN_SAMPLES_PER_SEGMENT,
        lock_after: int = LOCK_AFTER_SAMPLES,
        window: int = SCALE_WINDOW_SAMPLES,
        min_visibility: float = MIN_LANDMARK_VISIBILITY,
    ) -> None:
        if lock_after < min_samples:
            raise ValueError(f"lock_after ({lock_after}) must be >= min_samples ({min_samples})")
        self.track_id = track_id
        self.percentile = percentile
        self.min_samples = min_samples
        self.lock_after = lock_after
        self.min_visibility = min_visibility

        self._history: Dict[str, Deque[float]] = {name: deque(maxlen=window) for name in SEGMENT_PAIRS}
        self._instantaneous: Deque[float] = deque(maxlen=window)
        self._latest: Dict[str, float] = {}
        self._n_observations = 0
        self._was_locked = False
        self.just_locked = False

    # ------------------------------------------------------------------ #
    @property
    def n_observations(self) -> int:
        return self._n_observations

    @property
    def n_samples(self) -> int:
        """Frames in which at least one segment was measurable."""
        return len(self._instantaneous)

    @property
    def locked(self) -> bool:
        return self.n_samples >= self.lock_after

    # ------------------------------------------------------------------ #
    def update(
        self,
        points: Mapping[int, Point],
        visibility: Optional[Mapping[int, float]] = None,
    ) -> Dict[str, float]:
        """Fold one frame of landmarks in. Returns the segments it measured."""
        self._n_observations += 1
        segments = measure_segments(points, visibility, self.min_visibility)
        return self.update_segments(segments)

    def update_segments(self, segments: Mapping[str, float]) -> Dict[str, float]:
        """Fold in already-measured segment lengths, in pixels.

        Split out from :meth:`update` so a caller holding lengths from some other
        source — a different pose model, or a replayed log — can use the
        estimator without reconstructing landmark dictionaries.
        """
        self.just_locked = False
        if not segments:
            return {}

        for name, length in segments.items():
            if name in self._history and length > 0:
                self._history[name].append(float(length))
        self._latest = dict(segments)

        instant, _, _, _ = fuse_segments(stature_equivalents(segments))
        if instant is not None:
            was_locked = self._was_locked
            self._instantaneous.append(instant)
            if self.locked and not was_locked:
                self.just_locked = True
                self._was_locked = True
                logger.debug("track %s: body scale locked after %d samples", self.track_id, self.n_samples)
        return dict(segments)

    # ------------------------------------------------------------------ #
    def estimate(self) -> Optional[BodyScale]:
        """Best current body scale, or ``None`` if nothing is measurable yet."""
        if not self._instantaneous:
            return None

        reasons: Dict[str, str] = {}
        aggregated: Dict[str, float] = {}
        for name, samples in self._history.items():
            corrected = foreshortening_corrected_length(samples, self.percentile, self.min_samples)
            if corrected is not None:
                aggregated[name] = corrected

        if aggregated:
            state = "locked" if self.locked else "provisional"
            source = aggregated
        else:
            # Too few frames for a percentile: fall back to the newest frame and
            # say so, rather than pretending the window was long enough.
            state = "instantaneous"
            source = self._latest
            reasons["aggregation"] = f"fewer than {self.min_samples} samples"

        value, weight, agreement, rejected = fuse_segments(stature_equivalents(source))
        if value is None:
            return None
        if weight < MIN_SEGMENT_WEIGHT:
            reasons["weight"] = f"segment weight {weight:.2f} below floor {MIN_SEGMENT_WEIGHT:.2f}"
        rejected_weight = float(sum(FUSION_WEIGHTS.get(name, 0.0) for name in rejected))
        for name, why in rejected.items():
            reasons[f"segment.{name}"] = f"failed the consistency gate: {why}"
        if rejected_weight > weight:
            reasons["consistency"] = (
                f"rejected segments outweigh the survivors ({rejected_weight:.3f} > {weight:.3f}): "
                "the pose contradicts itself, so the scale is not trustworthy"
            )

        dispersion = robust_dispersion(list(self._instantaneous))
        if dispersion > MAX_SCALE_DISPERSION:
            reasons["dispersion"] = f"{dispersion:.3f} above {MAX_SCALE_DISPERSION:.2f}"

        return BodyScale(
            value=value,
            state=state,
            n_samples=self.n_samples,
            weight=weight,
            rejected_weight=rejected_weight,
            segments=dict(gate_segments(stature_equivalents(source))[0]),
            dispersion=dispersion,
            agreement=agreement,
            reasons=reasons,
        )

    def reset(self) -> None:
        """Forget everything. Call this on a track-id reuse, never mid-track."""
        for history in self._history.values():
            history.clear()
        self._instantaneous.clear()
        self._latest = {}
        self._n_observations = 0
        self._was_locked = False
        self.just_locked = False


# --------------------------------------------------------------------------- #
# The calibrated extension
# --------------------------------------------------------------------------- #


def with_metric_scale(
    scale: BodyScale,
    camera: CameraModel,
    depth_m: float,
    at: Optional[Point] = None,
) -> BodyScale:
    """Attach a metric stature and unlock cross-student comparison.

    This is the ONLY route by which ``comparable_across_students`` becomes true.
    Both inputs are needed and neither can be guessed: the seat depth comes from
    a tape measure or an ArUco marker, and the intrinsics come from a
    calibration. ``docs/HEIGHT_ESTIMATION.md`` §3 tabulates why the alternatives
    — monocular depth networks at ~20% relative error, or MediaPipe's
    ``pose_world_landmarks``, which returns a canonical body rather than a
    measurement — do not qualify.

    A warning is logged rather than an exception raised when the focal length
    fails its plausibility check, because the caller may knowingly be running a
    wide-angle camera; the check is recorded in ``reasons`` either way.
    """
    reasons = dict(scale.reasons)
    if not camera.focal_is_plausible():
        reasons["focal"] = f"f={camera.fx:.0f}px outside 0.7W..W for W={camera.width}"
        logger.warning("implausible focal length for metric scale: %s", camera.describe())
    if camera.calibration_rms_px is None:
        reasons["calibration"] = "intrinsics not calibrated; treat the metric value as indicative"

    metres = camera.metric_length(scale.value, depth_m, at)
    return BodyScale(
        value=scale.value,
        state=scale.state,
        n_samples=scale.n_samples,
        weight=scale.weight,
        rejected_weight=scale.rejected_weight,
        segments=dict(scale.segments),
        dispersion=scale.dispersion,
        agreement=scale.agreement,
        comparable_across_students=True,
        metric_stature_m=metres,
        reasons=reasons,
    )


__all__ = [
    "SEGMENT_PAIRS",
    "BodyScale",
    "BodyScaleEstimator",
    "foreshortening_corrected_length",
    "fuse_segments",
    "gate_segments",
    "huber_location",
    "measure_segments",
    "robust_dispersion",
    "stature_equivalents",
    "weighted_median",
    "with_metric_scale",
]
