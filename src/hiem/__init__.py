"""HIEM — Height-Invariant Engagement Metrics.

The fairness layer of the EduVisionSeat pipeline. It sits between pose geometry
and any engagement model, and it does one job: turn pixel measurements into
ratios of that student's own body scale, so that **the same behaviour scores the
same whether the student is tall or short, in the front row or the back**.

::

    from src.hiem import HiemTracker, PoseObservation

    tracker = HiemTracker()
    features = tracker.observe(PoseObservation(track_id=7, points=landmarks_px,
                                               visibility=vis, timestamp_s=t))
    features.hand_raise      # body scales, not pixels
    features.scale.describe()

Four modules, in the order they are worth reading:

* :mod:`src.hiem.normalize` — the normaliser, the feature registry and the
  invariance guarantee it enforces.
* :mod:`src.hiem.scale` — the body-scale estimator: percentile aggregation to
  undo foreshortening, inverse-variance fusion across limb segments, and a
  Huber M-estimator so one bad elbow cannot move the result.
* :mod:`src.hiem.fairness` — the audit that measures whether any of it worked.
* :mod:`src.hiem.camera` — the optional calibrated path, needed only for
  comparing one student's body scale against another's.

Run ``python -m src.hiem.demo`` for a worked example on the repository's own
images. The design and its evidence are written up in ``docs/HIEM.md``.
"""

from __future__ import annotations

from .camera import CameraModel
from .constants import FUSION_WEIGHTS, SEGMENT_TO_STATURE
from .fairness import (
    HiemAuditReport,
    InvarianceResult,
    audit_features,
    equity_audit,
    invariance_audit,
    spearman,
)
from .normalize import (
    FEATURE_SPECS,
    FeatureSpec,
    HiemFeatures,
    HiemNormaliser,
    HiemTracker,
    Invariance,
    PoseObservation,
    normalise_sequence,
)
from .scale import (
    BodyScale,
    BodyScaleEstimator,
    foreshortening_corrected_length,
    measure_segments,
    with_metric_scale,
)

__version__ = "1.0.0"

__all__ = [
    "FEATURE_SPECS",
    "FUSION_WEIGHTS",
    "SEGMENT_TO_STATURE",
    "BodyScale",
    "BodyScaleEstimator",
    "CameraModel",
    "FeatureSpec",
    "HiemAuditReport",
    "HiemFeatures",
    "HiemNormaliser",
    "HiemTracker",
    "Invariance",
    "InvarianceResult",
    "PoseObservation",
    "__version__",
    "audit_features",
    "equity_audit",
    "foreshortening_corrected_length",
    "invariance_audit",
    "measure_segments",
    "normalise_sequence",
    "spearman",
    "with_metric_scale",
]


__version__ = "1.0.0"