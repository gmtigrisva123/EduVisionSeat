"""EduVisionSeat — detection and posture analysis for classroom seating research.

Sub-packages:

* :mod:`src.detect` — person detection and tracking (Ultralytics YOLO).
* :mod:`src.pose` — pose estimation and landmarks (MediaPipe).
* :mod:`src.notebook` — demonstration notebooks.

Only :mod:`src.config` is re-exported here; the sub-packages pull in torch and
mediapipe, so they are left to be imported explicitly.
"""

from __future__ import annotations

from .config import REPO_ROOT, Config

__version__ = "1.0.0"

__all__ = ["Config", "REPO_ROOT", "__version__"]
