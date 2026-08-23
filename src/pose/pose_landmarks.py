"""Per-frame landmark extraction and head-angle geometry for the VOTO index.

This module covers §1 and §2 of ``docs/ATTENTION_INDEX.md`` and nothing beyond
them: it turns one cropped person into measured geometry. Per-student baselines
(§5), temporal smoothing (§6), signal fusion (§3-§4) and hysteresis (§7) live in
their own modules — the numbers produced here are *raw observations*, not scores.

Things to keep in mind:

1. **Angles are defined by the face's forward vector, not by an Euler order.**
   ``_euler_from_rotation_matrix`` projects the model's forward and up axes and
   reads the angles off those. Euler extraction with an assumed order is where
   the ±180° sign bugs described in §1.1 come from; this formulation has no
   order to get wrong and is covered by unit tests on synthetic rotations.

   The repo convention, in image terms:

   * ``yaw``   > 0 — head turned towards the RIGHT-HAND SIDE OF THE IMAGE
   * ``pitch`` > 0 — chin UP; head-down (the reading/writing band) is NEGATIVE
   * ``roll``  > 0 — top of the head tilted towards the right of the image

2. **Landmarks are converted to full-frame pixels before any geometry.**
   MediaPipe normalises x by the crop width and y by the crop height
   *separately*, so a distance computed on normalised coordinates is wrong by
   the crop's aspect ratio. Every helper here takes pixels.

3. **``None`` means NOT MEASURABLE.** Never substitute 0. Each missing value is
   accompanied by a reason code in :attr:`PersonFeatures.reasons`.

4. **The ``PROXY_*_GAIN_DEG`` constants are initial values, not results.** §1.2
   rule 3 is explicit: map the proxy to degrees by FITTING it on your own
   footage. Until that fit is done, treat ``Tier.PROXY`` angles as ordinal.

5. FaceLandmarker/PoseLandmarker run in ``IMAGE`` mode, not ``VIDEO``. VIDEO mode
   requires monotonically increasing timestamps on a single stream, and we feed
   it N independent person crops per frame — the timestamps would not be
   monotonic per person unless we kept N landmarker instances alive.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import REPO_ROOT

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

# --------------------------------------------------------------------------- #
# Landmark indices
# --------------------------------------------------------------------------- #

#: FaceMesh eye contours for EAR, ordered p1..p6 as in Soukupova & Cech (2016).
EAR_RIGHT_EYE = (33, 160, 158, 133, 153, 144)
EAR_LEFT_EYE = (362, 385, 387, 263, 373, 380)

#: FaceMesh INNER lip contour for MAR: (p78, p308) horizontal, then two verticals.
MAR_HORIZONTAL = (78, 308)
MAR_VERTICALS = ((81, 178), (311, 402))

#: Outer eye corners, used for the interocular distance that selects the tier.
FACE_EYE_OUTER_RIGHT = 33
FACE_EYE_OUTER_LEFT = 263

#: The six points handed to solvePnP, in the order of :data:`MODEL_POINTS_CANONICAL`.
PNP_FACE_INDICES = (1, 152, 263, 33, 291, 61)

#: Approximate anthropometric face model in millimetres, expressed in the
#: canonical MediaPipe convention: **y-up, z-out-of-face**. It is converted to
#: the OpenCV convention (y-down, z-into-scene) inside :func:`angles_from_solvepnp`.
MODEL_POINTS_CANONICAL = (
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye, outer corner
    (225.0, 170.0, -135.0),   # right eye, outer corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),  # right mouth corner
)

#: MediaPipe Pose indices. "left" is the SUBJECT's left, i.e. image right for a
#: student facing the camera.
POSE_NOSE = 0
POSE_LEFT_EYE, POSE_RIGHT_EYE = 2, 5
POSE_LEFT_EAR, POSE_RIGHT_EAR = 7, 8
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16

# --------------------------------------------------------------------------- #
# Tunables. Everything here is a documented decision, not a magic number.
# --------------------------------------------------------------------------- #

#: Tier boundaries in FULL-FRAME pixels of interocular distance (§1).
IOD_FULL_PX = 40.0
IOD_POSE_ONLY_PX = 20.0

#: EAR is computed in the image plane, so an oblique view contracts its
#: denominator and INFLATES the value. §2.1 gates the eye signal off past this.
EYE_YAW_GATE_DEG = 35.0

#: Initial proxy -> degree mappings. FIT THESE (§1.2 rule 3, scripts/fit_proxy_mapping.py).
PROXY_YAW_SHOULDER_GAIN_DEG = 90.0
PROXY_YAW_EAR_GEO_GAIN_DEG = 55.0
PROXY_YAW_EAR_VIS_GAIN_DEG = 70.0
PROXY_PITCH_GAIN_DEG = 60.0

#: Nominal value of ``pitch_proxy`` for a head held level. Subtracting a global
#: constant is the WEAK substitute; §5 requires this to become the student's own
#: baseline as soon as one is locked.
PROXY_PITCH_REST = 0.55

#: Confidence attached to each tier, straight from the table in §1.
TIER_CONF = {"full": 1.00, "pose_only": 0.70, "proxy": 0.35}

#: Fraction of the person box, measured from the top, handed to the face model.
#: FaceLandmarker finds nothing on a full-body crop — the face is a few percent
#: of those pixels — so running it on the whole person box silently pushes every
#: student down to the proxy tier. Cropping to the head first is what makes the
#: full tier reachable at all.
HEAD_CROP_FRACTION = 0.35

#: Model bundles. Missing files are not fatal — the extractor degrades and says so.
FACE_MODEL_FILENAME = "face_landmarker.task"
POSE_MODEL_FILENAME = "pose_landmarker_full.task"
_MODEL_URLS = {
    FACE_MODEL_FILENAME: (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    POSE_MODEL_FILENAME: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
}


class Tier(str, Enum):
    """Quality tier of a head-angle estimate. See the table in §1."""

    FULL = "full"
    POSE_ONLY = "pose_only"
    PROXY = "proxy"

    @property
    def conf(self) -> float:
        return TIER_CONF[self.value]


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeadAngles:
    """Head orientation in degrees, with the tier it was measured at."""

    yaw: float
    pitch: float
    roll: float
    tier: Tier

    @property
    def conf(self) -> float:
        return self.tier.conf

    @property
    def is_head_down(self) -> bool:
        """Inside the reading/writing band of Xue (2025).

        This says NOTHING about attention on its own: §3.B makes head-down a
        NEUTRAL state precisely because a student taking notes is the most
        focused one in the room.
        """
        return -60.0 <= self.pitch <= -20.0


@dataclass(frozen=True)
class HandGeometry:
    """Wrist geometry for the writing-vs-phone discriminator of §3.B.

    Deliberately raw: this module reports geometry, it does not classify. The
    decision needs the phone detections from :mod:`src.detect` as well, and §3.B
    ranks object detection above hand geometry as evidence.
    """

    #: Distance between the wrists, in shoulder widths. Writing puts them far
    #: apart (> 0.6); a phone draws them together towards the centre.
    wrist_gap: Optional[float]
    #: Height of the wrist midpoint above the shoulder line, in shoulder widths.
    wrist_rise: Optional[float]
    left_visibility: Optional[float]
    right_visibility: Optional[float]


@dataclass(frozen=True)
class PersonFeatures:
    """Everything measured for one person in one frame.

    Any field may be ``None``, which means NOT MEASURABLE — never zero.
    """

    track_id: int
    frame_index: int
    timestamp_s: float

    angles: Optional[HeadAngles] = None
    iod: Optional[float] = None
    ear: Optional[float] = None
    mar: Optional[float] = None
    shoulder_width: Optional[float] = None
    torso_angle: Optional[float] = None
    neck_drop: Optional[float] = None
    hands: Optional[HandGeometry] = None
    #: Reason codes for whatever could not be measured, e.g. ``{"ear": "yaw_gate"}``.
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def tier(self) -> Optional[Tier]:
        return self.angles.tier if self.angles else None


# --------------------------------------------------------------------------- #
# Pure geometry. No MediaPipe, no I/O — all of this is unit tested.
# --------------------------------------------------------------------------- #


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def eye_aspect_ratio(points: Sequence[Point]) -> Optional[float]:
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|) — Soukupova & Cech, CVWW 2016.

    ``points`` must be the six landmarks in p1..p6 order. Returns ``None`` when
    the horizontal span collapses, which happens at extreme yaw.
    """
    if len(points) != 6:
        raise ValueError(f"EAR needs exactly 6 points, got {len(points)}")
    horizontal = distance(points[0], points[3])
    if horizontal <= 1e-6:
        return None
    return (distance(points[1], points[5]) + distance(points[2], points[4])) / (2.0 * horizontal)


def mouth_aspect_ratio(
    horizontal: Tuple[Point, Point], verticals: Sequence[Tuple[Point, Point]]
) -> Optional[float]:
    """MAR on the INNER lip contour (§2.4). Outer-contour points move when the
    jaw does not, which is what makes the widely copied outer-lip MAR noisy."""
    span = distance(*horizontal)
    if span <= 1e-6:
        return None
    return sum(distance(a, b) for a, b in verticals) / (2.0 * span)


def _euler_from_rotation_matrix(rotation: np.ndarray) -> Tuple[float, float, float]:
    """Read yaw/pitch/roll off the face's forward and up axes.

    ``rotation`` maps model coordinates into the OpenCV camera frame
    (x right, y DOWN, z into the scene). At rest the face looks back along -z.

    Deriving the angles from projected axes instead of an assumed Euler order is
    what keeps this free of the ±180° flips described in §1.1: there is no
    convention left to get wrong, and both estimation paths share this one
    function so they cannot drift apart.
    """
    forward = rotation @ np.array([0.0, 0.0, -1.0])  # out of the face
    up = rotation @ np.array([0.0, -1.0, 0.0])       # towards the top of the head

    yaw = math.degrees(math.atan2(float(forward[0]), float(-forward[2])))
    pitch = math.degrees(math.atan2(float(-forward[1]), math.hypot(float(forward[0]), float(forward[2]))))
    roll = math.degrees(math.atan2(float(up[0]), float(-up[1])))
    return yaw, pitch, roll


def angles_from_transformation_matrix(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Decompose the 4x4 facial transformation matrix from FaceLandmarker.

    Preferred over a hand-written solvePnP (§1.1): MediaPipe solves PnP on its
    own metric mesh and returns a clean rigid transform, whereas a 6-point PnP on
    a near-coplanar face is ill-conditioned and adds 5-10° of noise.

    The columns are renormalised first — the matrix carries the mesh scale, and
    an unnormalised basis quietly corrupts the projected axes.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transformation matrix, got {matrix.shape}")

    rotation = matrix[:3, :3].copy()
    norms = np.linalg.norm(rotation, axis=0)
    if float(np.min(norms)) <= 1e-9:
        raise ValueError("Degenerate transformation matrix: a basis vector has zero length")
    rotation /= norms
    return _euler_from_rotation_matrix(rotation)


def angles_from_solvepnp(
    image_points: Sequence[Point], image_size: Tuple[int, int]
) -> Optional[Tuple[float, float, float]]:
    """Fallback 6-point PnP, in :data:`PNP_FACE_INDICES` order.

    The detail nearly every tutorial gets wrong (§1.1): the canonical model is
    **y-up, z-out-of-face** while OpenCV wants **y-down, z-into-scene**. Negating
    Y and Z is a rotation of pi about X — still a proper rotation, det(R) = +1 —
    and without it a front-facing student comes back as R ~ Rx(180°) with
    inverted signs.

    The camera matrix is a guess (focal = image width, principal point at the
    centre). ``docs/HEIGHT_ESTIMATION.md`` spells out what uncalibrated video
    costs; this path is a fallback, not a measurement instrument.
    """
    import cv2

    if len(image_points) != len(MODEL_POINTS_CANONICAL):
        raise ValueError(f"solvePnP needs {len(MODEL_POINTS_CANONICAL)} points, got {len(image_points)}")

    # y-up, z-out  ->  y-down, z-in
    model = np.array([(x, -y, -z) for x, y, z in MODEL_POINTS_CANONICAL], dtype=np.float64)
    image = np.array(image_points, dtype=np.float64)

    width, height = image_size
    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    ok, rvec, _ = cv2.solvePnP(
        model, image, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return _euler_from_rotation_matrix(rotation)


def angles_from_pose_proxy(
    points: Dict[int, Point],
    visibility: Optional[Dict[int, float]] = None,
    iod: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """Scale-invariant proxy angles from body landmarks alone (§1.2).

    Basis: Araya & Sossa-Rivera (2021), who filmed classrooms from behind where
    "in most of the scene the students' faces are not visible". This tier carries
    a confidence of 0.35 for good reason — but it is also the tier that covers
    the back rows, where every face-based signal has already failed.

    Rule 1 of §1.2 is enforced here: every quantity is divided by SHOULDER WIDTH,
    an in-image length that scales with the subject and survives the loss of the
    face. IOD is only ever used as a secondary denominator, because it collapses
    to 0 at oblique angles.
    """
    visibility = visibility or {}
    required = (POSE_NOSE, POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER)
    if any(idx not in points for idx in required):
        return None

    nose = points[POSE_NOSE]
    shoulder_l, shoulder_r = points[POSE_LEFT_SHOULDER], points[POSE_RIGHT_SHOULDER]
    shoulder_width = distance(shoulder_l, shoulder_r)
    if shoulder_width <= 1e-6:
        return None
    shoulder_mid = midpoint(shoulder_l, shoulder_r)

    estimates: List[Tuple[float, float]] = []  # (degrees, weight)
    estimates.append(((nose[0] - shoulder_mid[0]) / shoulder_width * PROXY_YAW_SHOULDER_GAIN_DEG, 1.0))

    if POSE_LEFT_EAR in points and POSE_RIGHT_EAR in points:
        ear_l, ear_r = points[POSE_LEFT_EAR], points[POSE_RIGHT_EAR]
        ear_span = distance(ear_l, ear_r)
        ear_mid = midpoint(ear_l, ear_r)
        # The geometric ear ratio is sharp head-on and degenerate in profile,
        # where the ears project on top of one another; weight it by how open
        # the span still is, relative to the shoulders.
        openness = ear_span / shoulder_width
        if ear_span > 1e-6 and openness > 0.15:
            yaw_ear_geo = 2.0 * (nose[0] - ear_mid[0]) / ear_span
            estimates.append((yaw_ear_geo * PROXY_YAW_EAR_GEO_GAIN_DEG, min(openness / 0.4, 1.0)))

        # Visibility asymmetry takes over exactly where the geometry dies: in
        # profile one ear is occluded by the head.
        vis_l, vis_r = visibility.get(POSE_LEFT_EAR), visibility.get(POSE_RIGHT_EAR)
        if vis_l is not None and vis_r is not None and (vis_l + vis_r) > 1e-6:
            yaw_ear_vis = (vis_l - vis_r) / (vis_l + vis_r)
            # Subject-left ear more visible => head turned towards image LEFT.
            estimates.append((-yaw_ear_vis * PROXY_YAW_EAR_VIS_GAIN_DEG, 0.6))

    total_weight = sum(w for _, w in estimates)
    yaw = sum(value * w for value, w in estimates) / total_weight if total_weight > 0 else 0.0

    eye_mid = _eye_midpoint(points)
    if eye_mid is None:
        return yaw, 0.0, 0.0

    denominator = max(iod or 0.0, 0.25 * shoulder_width)
    pitch_proxy = (nose[1] - eye_mid[1]) / denominator
    pitch = (pitch_proxy - PROXY_PITCH_REST) * PROXY_PITCH_GAIN_DEG

    roll = 0.0
    if POSE_LEFT_EYE in points and POSE_RIGHT_EYE in points:
        eye_l, eye_r = points[POSE_LEFT_EYE], points[POSE_RIGHT_EYE]
        roll = math.degrees(math.atan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0]))

    return yaw, pitch, roll


def _eye_midpoint(points: Dict[int, Point]) -> Optional[Point]:
    if POSE_LEFT_EYE in points and POSE_RIGHT_EYE in points:
        return midpoint(points[POSE_LEFT_EYE], points[POSE_RIGHT_EYE])
    if POSE_LEFT_EAR in points and POSE_RIGHT_EAR in points:
        return midpoint(points[POSE_LEFT_EAR], points[POSE_RIGHT_EAR])
    return None


def torso_angle(points: Dict[int, Point]) -> Optional[float]:
    """Angle at the shoulder midpoint between the neck axis and straight down.

    180° is perfectly upright; slumping pulls it towards 130°, the lower end of
    the ramp in §3.C. Computed from the SHOULDER LINE to the EYE MIDPOINT rather
    than the full spine, because a desk hides the hips of a seated student.

    Posture degrades gracefully with distance, which is why §3.C weights it as
    heavily as it does: it still works in the back rows, where the face is gone.
    """
    if POSE_LEFT_SHOULDER not in points or POSE_RIGHT_SHOULDER not in points:
        return None
    eye_mid = _eye_midpoint(points)
    if eye_mid is None:
        return None

    shoulder_mid = midpoint(points[POSE_LEFT_SHOULDER], points[POSE_RIGHT_SHOULDER])
    neck = (eye_mid[0] - shoulder_mid[0], eye_mid[1] - shoulder_mid[1])
    length = math.hypot(*neck)
    if length <= 1e-6:
        return None
    # Angle against the downward vertical (0, +1) in image coordinates.
    cosine = min(max(neck[1] / length, -1.0), 1.0)
    return math.degrees(math.acos(cosine))


def neck_drop(points: Dict[int, Point]) -> Optional[float]:
    """(eye_mid.y - shoulder_mid.y) / shoulder_width — negative when upright."""
    if POSE_LEFT_SHOULDER not in points or POSE_RIGHT_SHOULDER not in points:
        return None
    eye_mid = _eye_midpoint(points)
    if eye_mid is None:
        return None
    shoulder_l, shoulder_r = points[POSE_LEFT_SHOULDER], points[POSE_RIGHT_SHOULDER]
    width = distance(shoulder_l, shoulder_r)
    if width <= 1e-6:
        return None
    return (eye_mid[1] - midpoint(shoulder_l, shoulder_r)[1]) / width


def hand_geometry(points: Dict[int, Point], visibility: Optional[Dict[int, float]] = None) -> HandGeometry:
    """Wrist separation and height, both in shoulder widths (§3.B evidence #2)."""
    visibility = visibility or {}
    vis_l, vis_r = visibility.get(POSE_LEFT_WRIST), visibility.get(POSE_RIGHT_WRIST)

    have_shoulders = POSE_LEFT_SHOULDER in points and POSE_RIGHT_SHOULDER in points
    have_wrists = POSE_LEFT_WRIST in points and POSE_RIGHT_WRIST in points
    if not (have_shoulders and have_wrists):
        return HandGeometry(None, None, vis_l, vis_r)

    width = distance(points[POSE_LEFT_SHOULDER], points[POSE_RIGHT_SHOULDER])
    if width <= 1e-6:
        return HandGeometry(None, None, vis_l, vis_r)

    wrist_l, wrist_r = points[POSE_LEFT_WRIST], points[POSE_RIGHT_WRIST]
    shoulder_mid = midpoint(points[POSE_LEFT_SHOULDER], points[POSE_RIGHT_SHOULDER])
    gap = distance(wrist_l, wrist_r) / width
    rise = (shoulder_mid[1] - midpoint(wrist_l, wrist_r)[1]) / width
    return HandGeometry(gap, rise, vis_l, vis_r)


def tier_for_iod(iod: Optional[float], has_face: bool) -> Tier:
    """Pick the quality tier from the interocular distance, in FULL-FRAME pixels.

    The proxy tier matters more than one would expect: in a 1080p frame of a
    30-seat classroom the back row typically sits below 20 px of IOD, so every
    face-based signal fails exactly where it is needed most.
    """
    if not has_face or iod is None:
        return Tier.PROXY
    if iod >= IOD_FULL_PX:
        return Tier.FULL
    if iod >= IOD_POSE_ONLY_PX:
        return Tier.POSE_ONLY
    return Tier.PROXY


# --------------------------------------------------------------------------- #
# MediaPipe wrapper
# --------------------------------------------------------------------------- #


class LandmarkExtractor:
    """FaceLandmarker + PoseLandmarker over one person crop at a time.

    Both bundles are optional. A missing pose bundle is fatal (nothing can be
    normalised without shoulders); a missing face bundle only forces every
    student down to :attr:`Tier.PROXY`, which is logged and recorded in
    :attr:`degradations` so a run can never quietly report proxy numbers as if
    they were measurements.
    """

    def __init__(self, models_dir: Optional[Path] = None, crop_padding: float = 0.15):
        self.models_dir = Path(models_dir) if models_dir else REPO_ROOT / "models"
        self.crop_padding = crop_padding
        self.degradations: List[str] = []
        self._pose = self._load_pose_landmarker()
        self._face = self._load_face_landmarker()

    # ------------------------------------------------------------------ #
    def _bundle(self, filename: str) -> Optional[Path]:
        path = self.models_dir / filename
        return path if path.is_file() else None

    def _load_pose_landmarker(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        bundle = self._bundle(POSE_MODEL_FILENAME)
        if bundle is None:
            raise FileNotFoundError(
                f"Pose bundle not found: {self.models_dir / POSE_MODEL_FILENAME}\n"
                f"Download it with:\n  curl -L -o '{self.models_dir / POSE_MODEL_FILENAME}' "
                f"{_MODEL_URLS[POSE_MODEL_FILENAME]}"
            )
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(bundle)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,  # one crop, one student
        )
        logger.info("Loaded pose landmarker: %s", bundle)
        return PoseLandmarker.create_from_options(options)

    def _load_face_landmarker(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        bundle = self._bundle(FACE_MODEL_FILENAME)
        if bundle is None:
            msg = (
                f"Face bundle not found ({self.models_dir / FACE_MODEL_FILENAME}) -> every student "
                f"drops to the '{Tier.PROXY.value}' tier (conf {Tier.PROXY.conf}). Head angles become "
                f"ordinal, and EAR/MAR are unavailable entirely. Download it with:\n  curl -L -o "
                f"'{self.models_dir / FACE_MODEL_FILENAME}' {_MODEL_URLS[FACE_MODEL_FILENAME]}"
            )
            logger.warning(msg)
            self.degradations.append(msg)
            return None

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(bundle)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,  # the whole point of the FULL tier
        )
        logger.info("Loaded face landmarker: %s", bundle)
        return FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ #
    def _crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        padding: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """Return (crop_rgb, x_offset, y_offset), padded and clamped to the frame."""
        height, width = frame.shape[:2]
        pad = self.crop_padding if padding is None else padding
        x1, y1, x2, y2 = bbox
        pad_x = (x2 - x1) * pad
        pad_y = (y2 - y1) * pad
        x1 = max(int(x1 - pad_x), 0)
        y1 = max(int(y1 - pad_y), 0)
        x2 = min(int(x2 + pad_x), width)
        y2 = min(int(y2 + pad_y), height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        import cv2

        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), x1, y1

    @staticmethod
    def _to_pixels(landmarks, crop_shape: Tuple[int, int], offset: Tuple[int, int]) -> Dict[int, Point]:
        """Normalised crop coordinates -> full-frame pixels.

        This conversion is not cosmetic. MediaPipe normalises x by the crop width
        and y by the crop height separately, so any distance taken before this
        step is skewed by the crop's aspect ratio.
        """
        crop_h, crop_w = crop_shape
        off_x, off_y = offset
        return {
            i: (lm.x * crop_w + off_x, lm.y * crop_h + off_y) for i, lm in enumerate(landmarks)
        }

    @staticmethod
    def _visibilities(landmarks) -> Dict[int, float]:
        return {
            i: float(lm.visibility)
            for i, lm in enumerate(landmarks)
            if getattr(lm, "visibility", None) is not None
        }

    # ------------------------------------------------------------------ #
    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> PersonFeatures:
        """Measure one person. Never raises on a bad crop — it reports a reason."""
        import mediapipe as mp

        reasons: Dict[str, str] = {}
        cropped = self._crop(frame, bbox)
        if cropped is None:
            return PersonFeatures(track_id, frame_index, timestamp_s, reasons={"all": "crop_degenerate"})
        crop_rgb, off_x, off_y = cropped
        offset = (off_x, off_y)
        crop_shape = crop_rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        pose_result = self._pose.detect(image)
        if not pose_result.pose_landmarks:
            return PersonFeatures(track_id, frame_index, timestamp_s, reasons={"all": "no_pose"})
        body = self._to_pixels(pose_result.pose_landmarks[0], crop_shape, offset)
        body_vis = self._visibilities(pose_result.pose_landmarks[0])

        face: Optional[Dict[int, Point]] = None
        matrix = None
        if self._face is None:
            reasons["face"] = "no_face_model"
        else:
            found = self._detect_face(frame, bbox, image, crop_shape, offset)
            if found is None:
                reasons["face"] = "no_face"
            else:
                face, matrix = found

        iod = None
        if face is not None:
            iod = distance(face[FACE_EYE_OUTER_RIGHT], face[FACE_EYE_OUTER_LEFT])

        angles = self._head_angles(face, matrix, body, body_vis, iod, crop_shape, reasons)
        ear = self._ear(face, angles, reasons)
        mar = self._mar(face, reasons)

        shoulder_width = None
        if POSE_LEFT_SHOULDER in body and POSE_RIGHT_SHOULDER in body:
            shoulder_width = distance(body[POSE_LEFT_SHOULDER], body[POSE_RIGHT_SHOULDER]) or None

        return PersonFeatures(
            track_id=track_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            angles=angles,
            iod=iod,
            ear=ear,
            mar=mar,
            shoulder_width=shoulder_width,
            torso_angle=torso_angle(body),
            neck_drop=neck_drop(body),
            hands=hand_geometry(body, body_vis),
            reasons=reasons,
        )

    # ------------------------------------------------------------------ #
    def _detect_face(self, frame, bbox, person_image, person_shape, person_offset):
        """Look for the face in the head region first, then in the whole person box.

        The head-first order is not an optimisation. On a full-body crop the face
        occupies a few percent of the pixels and FaceLandmarker returns nothing at
        all, so a person-box-only search would report ``no_face`` for essentially
        every seated student and hand the whole class to the proxy tier.

        The second attempt costs one extra inference, and only on crops where the
        first found nothing — worth it, because the difference between the tiers
        is a confidence of 1.00 against 0.35.
        """
        import mediapipe as mp

        x1, y1, x2, y2 = bbox
        head_bbox = (x1, y1, x2, y1 + (y2 - y1) * HEAD_CROP_FRACTION)
        attempts = []

        head = self._crop(frame, head_bbox, padding=0.0)
        if head is not None:
            head_rgb, head_x, head_y = head
            head_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=head_rgb)
            attempts.append((head_image, head_rgb.shape[:2], (head_x, head_y)))
        attempts.append((person_image, person_shape, person_offset))

        for image, shape, offset in attempts:
            result = self._face.detect(image)
            if not result.face_landmarks:
                continue
            matrices = result.facial_transformation_matrixes
            return (
                self._to_pixels(result.face_landmarks[0], shape, offset),
                np.asarray(matrices[0]) if matrices else None,
            )
        return None

    # ------------------------------------------------------------------ #
    def _head_angles(
        self,
        face: Optional[Dict[int, Point]],
        matrix: Optional[np.ndarray],
        body: Dict[int, Point],
        body_vis: Dict[int, float],
        iod: Optional[float],
        crop_shape: Tuple[int, int],
        reasons: Dict[str, str],
    ) -> Optional[HeadAngles]:
        """Walk down the three tiers of §1, stopping at the first that applies."""
        tier = tier_for_iod(iod, has_face=face is not None)

        if tier in (Tier.FULL, Tier.POSE_ONLY) and face is not None:
            if matrix is not None:
                try:
                    yaw, pitch, roll = angles_from_transformation_matrix(matrix)
                    return HeadAngles(yaw, pitch, roll, tier)
                except ValueError as exc:
                    logger.debug("Transformation matrix unusable (%s), falling back to solvePnP.", exc)
            try:
                crop_h, crop_w = crop_shape
                solved = angles_from_solvepnp(
                    [face[i] for i in PNP_FACE_INDICES], (crop_w, crop_h)
                )
            except (ImportError, ValueError, KeyError) as exc:
                logger.debug("solvePnP unavailable for this face (%s).", exc)
                solved = None
            if solved is not None:
                # PnP on a near-coplanar face is ill-conditioned (§1.1), so it
                # never earns the FULL tier however large the face happens to be.
                return HeadAngles(solved[0], solved[1], solved[2], Tier.POSE_ONLY)

        proxy = angles_from_pose_proxy(body, body_vis, iod)
        if proxy is None:
            reasons["angles"] = "no_shoulders"
            return None
        return HeadAngles(proxy[0], proxy[1], proxy[2], Tier.PROXY)

    @staticmethod
    def _ear(
        face: Optional[Dict[int, Point]], angles: Optional[HeadAngles], reasons: Dict[str, str]
    ) -> Optional[float]:
        if face is None:
            return None
        if angles is not None and abs(angles.yaw) > EYE_YAW_GATE_DEG:
            # EAR lives in the image plane: an oblique view contracts the
            # horizontal denominator and inflates the ratio (§2.1).
            reasons["ear"] = "yaw_gate"
            return None
        values = [
            eye_aspect_ratio([face[i] for i in EAR_RIGHT_EYE]),
            eye_aspect_ratio([face[i] for i in EAR_LEFT_EYE]),
        ]
        usable = [v for v in values if v is not None]
        if not usable:
            reasons["ear"] = "degenerate_eye_span"
            return None
        return sum(usable) / len(usable)

    @staticmethod
    def _mar(face: Optional[Dict[int, Point]], reasons: Dict[str, str]) -> Optional[float]:
        if face is None:
            return None
        value = mouth_aspect_ratio(
            (face[MAR_HORIZONTAL[0]], face[MAR_HORIZONTAL[1]]),
            [(face[a], face[b]) for a, b in MAR_VERTICALS],
        )
        if value is None:
            reasons["mar"] = "degenerate_mouth_span"
        return value

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        for landmarker in (self._face, self._pose):
            if landmarker is not None:
                landmarker.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


__all__ = [
    "HandGeometry",
    "HeadAngles",
    "LandmarkExtractor",
    "PersonFeatures",
    "Tier",
    "angles_from_pose_proxy",
    "angles_from_solvepnp",
    "angles_from_transformation_matrix",
    "eye_aspect_ratio",
    "hand_geometry",
    "mouth_aspect_ratio",
    "neck_drop",
    "tier_for_iod",
    "torso_angle",
]
