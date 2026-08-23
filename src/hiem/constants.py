"""Anthropometric constants and tunables for HIEM.

Every number here is either **measured** (with the source named) or a **prior**
(explicitly labelled as one). The distinction matters: priors are starting values
that must be refitted on your own footage before any claim is made about them,
exactly as ``docs/ATTENTION_INDEX.md`` §4 requires for the VOTO weights.

The one thing to understand before reading further:

    **HIEM's body scale ``S`` is a length in PIXELS, not a height in
    centimetres.** The stature ratios below are used only to put several
    different limb segments onto one common footing so they can be fused. They
    are never used to print a number in centimetres — ``docs/HEIGHT_ESTIMATION.md``
    §7.1 is unambiguous that uncalibrated video cannot support that, and HIEM
    does not try.

So ``S`` is best read as *"the pixel length that this student's body subtends,
expressed in stature-equivalent units"*. Dividing a pixel measurement by it
yields a dimensionless ratio that is invariant to how tall the student is and to
how far away they are sitting. That ratio is the whole point.
"""

from __future__ import annotations

from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# MediaPipe Pose landmark indices
# --------------------------------------------------------------------------- #
# ``src.pose.pose_landmarks`` defines the subset it needs; HIEM needs the elbows
# and hips as well, so the full set it uses is listed here rather than reaching
# into another module's private surface. "left" is the SUBJECT's left, which is
# the IMAGE right for a student facing the camera.

NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

#: Landmarks whose frame-to-frame displacement defines the movement signal.
#: Restricted to the upper body: a seated student's legs are behind a desk, and
#: an occluded landmark that flickers in and out would read as violent motion.
MOTION_LANDMARKS: Tuple[int, ...] = (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)

# --------------------------------------------------------------------------- #
# Segment length as a fraction of stature
# --------------------------------------------------------------------------- #
# Sources, per row:
#
# * ``shoulder_width`` — ANSUR II biacromial breadth / stature, 0.2367 (men) and
#   0.2244 (women); the pooled figure is stored. The 5.5% sex difference is the
#   reason this segment is NOT the primary indicator, and the reason HIEM never
#   converts ``S`` into centimetres. See docs/HEIGHT_ESTIMATION.md §2.
# * ``upper_arm`` — ANSUR II acromion-radiale / stature, k = 5.239 (men) /
#   5.233 (women), i.e. 1/5.236 pooled. **Sex-invariant to within 0.1%**, which
#   is what makes it the primary indicator.
# * ``forearm`` — Drillis & Contini (1966) segment proportions, 0.146 H. Their
#   upper-arm figure of 0.186 H sits within 3% of the ANSUR-derived 0.1910,
#   which is a useful independent cross-check on the table as a whole.
# * ``head_width`` — ANSUR II bitragion breadth (~14.5 cm) over a 170 cm stature.
# * ``hip_width`` — ANSUR II bi-iliac breadth. Usually hidden by a desk; kept so
#   that a standing subject can still use it.

SEGMENT_TO_STATURE: Dict[str, float] = {
    "shoulder_width": 0.2306,
    "upper_arm": 0.1910,
    "forearm": 0.1460,
    "head_width": 0.0853,
    "hip_width": 0.1670,
}

#: Standard error of estimate, in cm, when stature is predicted from that
#: segment alone (docs/HEIGHT_ESTIMATION.md §2). Smaller is better. These drive
#: the inverse-variance part of the fusion weight.
SEGMENT_SEE_CM: Dict[str, float] = {
    "shoulder_width": 5.55,
    "upper_arm": 4.05,
    "forearm": 4.75,
    "head_width": 6.40,
    "hip_width": 5.90,
}

#: **PRIOR.** How much a segment can be trusted in a real classroom frame, over
#: and above its anthropometric quality. This is where seated visibility and
#: sex-invariance enter:
#:
#: * shoulder width and upper arm survive a seated pose intact — 1.0;
#: * the forearm is heavily foreshortened when the hands rest on the desk, and
#:   its ratio to stature carries a 2.9% sex difference — 0.6;
#: * the head is only 15-25 px across in the back rows, so its relative
#:   landmark noise is several times that of the torso — 0.5;
#: * the hips are behind the desk for most of the lesson — 0.4.
#:
#: Refit these on your own footage before quoting them (see docs/HIEM.md §7).
SEGMENT_RELIABILITY: Dict[str, float] = {
    "shoulder_width": 1.0,
    "upper_arm": 1.0,
    "forearm": 0.6,
    "head_width": 0.5,
    "hip_width": 0.4,
}


def fusion_weights() -> Dict[str, float]:
    """Normalised weight per segment: ``reliability / SEE**2``, summing to 1.

    Inverse-variance weighting is the right combination rule for independent
    estimates of one quantity; the reliability factor then downweights segments
    that are anatomically fine but practically unusable in this setting.
    """
    raw = {
        name: SEGMENT_RELIABILITY[name] / (SEGMENT_SEE_CM[name] ** 2)
        for name in SEGMENT_TO_STATURE
    }
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


#: Precomputed for callers that want to inspect the weighting without a call.
FUSION_WEIGHTS: Dict[str, float] = fusion_weights()

#: **PRIOR.** MediaPipe's "shoulder" is the glenohumeral JOINT CENTRE, not the
#: acromion, and its "elbow" is the joint centre, not the radiale. The measured
#: upper arm is therefore systematically 2-4 cm shorter than the ANSUR
#: definition (docs/HEIGHT_ESTIMATION.md §2). Multiply a MediaPipe segment by
#: this to reach ANSUR terms. Refit it in situ before trusting it.
#:
#: HIEM is insensitive to a mis-set value here in the way that matters: a
#: constant factor applied to every student cancels out of every ratio. It only
#: shifts the absolute size of ``S``.
MEDIAPIPE_TO_ANSUR_CORRECTION: Dict[str, float] = {
    "upper_arm": 1.09,
    "forearm": 1.04,
    "shoulder_width": 1.06,
    "head_width": 1.00,
    "hip_width": 1.00,
}

#: ANSUR II arm span / stature: the span EXCEEDS stature by ~3% in men and ~2%
#: in women, and is not 1.000 as the Vitruvian rule of thumb has it. Stored as
#: the reciprocal so a span in pixels maps to a stature-equivalent by
#: multiplication. Not used by the seated pipeline — a student at a desk never
#: reaches full span — but recorded so the ratio is not inverted by mistake.
STATURE_TO_ARM_SPAN = 0.9679
ARM_SPAN_TO_STATURE: Dict[str, float] = {"male": 1.0330, "female": 1.0195}

# --------------------------------------------------------------------------- #
# Scale estimation tunables
# --------------------------------------------------------------------------- #

#: The percentile used to undo foreshortening. Out-of-plane rotation can only
#: ever SHORTEN the projection of a rigid segment, so the upper tail of a long
#: observation window is the unforeshortened length. 92.5 rather than 100
#: because the maximum picks up the upper tail of the landmark noise instead.
#: docs/HEIGHT_ESTIMATION.md §4.2 calls this the single most valuable trick in
#: the pipeline; ``test_percentile_aggregation_recovers_unforeshortened_length``
#: guards it.
FORESHORTENING_PERCENTILE = 92.5

#: Below this many samples a percentile is meaningless, so the segment reports
#: nothing rather than a confident-looking number from four frames.
MIN_SAMPLES_PER_SEGMENT = 8

#: Samples required before the body scale LOCKS. At the 1 Hz feature rate of
#: docs/ATTENTION_INDEX.md §6 this is about half a minute of observation, which
#: matches the 60-90 s baseline window that §5 locks the per-student yaw
#: baseline over.
LOCK_AFTER_SAMPLES = 30

#: A locked scale is refreshed on a rolling window rather than frozen forever,
#: so a student who leans back permanently is followed. Longer than the lock
#: threshold so the estimate keeps improving after locking.
SCALE_WINDOW_SAMPLES = 600

#: Landmark visibility below which a point is not used in a length measurement.
#: MediaPipe's visibility is a calibrated-ish occlusion score; a wrist behind a
#: desk typically sits well under 0.5.
MIN_LANDMARK_VISIBILITY = 0.5

#: Widest ratio by which two segments' stature-equivalents may disagree before
#: one of them is treated as a detection failure rather than as anatomy.
#:
#: Two segments of the same body should agree closely: individual variation in
#: limb-to-stature proportion is a few percent (the SEE column above is 4-6 cm
#: against a ~165 cm stature). The slack is for projection, not for anatomy — a
#: segment 48 degrees out of plane projects to 0.67 of its length, and the
#: anchor segment may itself be foreshortened, so the window has to be roughly
#: 1.5x either way. A disagreement beyond that is not a body; it is a landmark
#: in the wrong place.
SEGMENT_CONSISTENCY_RATIO = 1.5

#: Segment the consistency gate measures the others against, or ``None`` to use
#: the weighted median of all of them.
#:
#: Shoulder width is the anchor because in a seated classroom it is the only
#: segment whose two endpoints are both large, high-contrast, above the desk
#: line and essentially never occluded. The arms are precisely what a desk
#: hides, and MediaPipe answers occlusion by EXTRAPOLATING a plausible limb
#: rather than by lowering its visibility score — measured on this repository's
#: own images, an elbow placed so badly that the implied upper arm was 7.5x too
#: long still carried a visibility of 0.96. See docs/HIEM.md §4.
#:
#: The cost of the choice is that a strongly yawed shoulder line drags the
#: anchor low. The percentile aggregation removes exactly that over a window,
#: which is why this gate matters most for single still frames.
CONSISTENCY_ANCHOR = "shoulder_width"

#: Sum of fusion weights that must be present before a scale is reported at all.
#: Directly analogous to the validity floor of docs/ATTENTION_INDEX.md §4: a
#: confident-looking denominator built from one noisy segment is worse than
#: admitting the measurement failed.
MIN_SEGMENT_WEIGHT = 0.20

#: Relative dispersion (robust MAD / median) above which a locked scale is
#: flagged as unstable. A student who is genuinely still should sit near 0.03.
MAX_SCALE_DISPERSION = 0.18

# --------------------------------------------------------------------------- #
# Signal thresholds, expressed in BODY SCALES
# --------------------------------------------------------------------------- #
# Every threshold below is dimensionless. That is the fairness property: it
# cannot be met more easily by a tall student than by a short one, because the
# quantity it is compared against has already been divided by that student's own
# body scale.

#: **PRIOR.** How far above the shoulder line the wrist must rise, in body
#: scales, to count as a raised hand. A bent-elbow raise at head height puts the
#: wrist about 0.12 stature above the acromion; a full overhead raise about
#: 0.35. The enter threshold sits above the bent-elbow figure so that a stretch
#: or a hand run through the hair does not qualify, and the exit threshold at
#: roughly half of it.
#:
#: Schmitt hysteresis, as docs/ATTENTION_INDEX.md §7 requires of every
#: thresholded quantity: enter high, leave low, and dwell on entry. The gap and
#: the asymmetric dwell both bias the system towards NOT firing.
HAND_RAISE_ENTER = 0.15
HAND_RAISE_EXIT = 0.08
HAND_RAISE_ENTER_SAMPLES = 3
HAND_RAISE_EXIT_SAMPLES = 2

#: **PRIOR.** Wrist separation dividing writing from a phone, from the hand
#: geometry of docs/ATTENTION_INDEX.md §3.B. That section states the rule in
#: shoulder widths (> 0.6); converted to stature-equivalent units here by the
#: shoulder ratio above, 0.6 x 0.2306 = 0.138.
WRIST_GAP_WRITING = 0.138

#: Longest gap, in seconds, across which a movement estimate is still made. A
#: longer gap means the track was lost and re-acquired, and differencing across
#: it would report a teleport.
MAX_MOTION_GAP_S = 2.0

__all__ = [
    "ARM_SPAN_TO_STATURE",
    "CONSISTENCY_ANCHOR",
    "FORESHORTENING_PERCENTILE",
    "FUSION_WEIGHTS",
    "HAND_RAISE_ENTER",
    "HAND_RAISE_ENTER_SAMPLES",
    "HAND_RAISE_EXIT",
    "HAND_RAISE_EXIT_SAMPLES",
    "LEFT_EAR",
    "LEFT_ELBOW",
    "LEFT_EYE",
    "LEFT_HIP",
    "LEFT_SHOULDER",
    "LEFT_WRIST",
    "LOCK_AFTER_SAMPLES",
    "MAX_MOTION_GAP_S",
    "MAX_SCALE_DISPERSION",
    "MEDIAPIPE_TO_ANSUR_CORRECTION",
    "MIN_LANDMARK_VISIBILITY",
    "MIN_SAMPLES_PER_SEGMENT",
    "MIN_SEGMENT_WEIGHT",
    "MOTION_LANDMARKS",
    "NOSE",
    "RIGHT_EAR",
    "RIGHT_ELBOW",
    "RIGHT_EYE",
    "RIGHT_HIP",
    "RIGHT_SHOULDER",
    "RIGHT_WRIST",
    "SCALE_WINDOW_SAMPLES",
    "SEGMENT_CONSISTENCY_RATIO",
    "SEGMENT_RELIABILITY",
    "SEGMENT_SEE_CM",
    "SEGMENT_TO_STATURE",
    "STATURE_TO_ARM_SPAN",
    "WRIST_GAP_WRITING",
    "fusion_weights",
]
