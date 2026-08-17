"""Pose estimation package for EduVisionSeat.

Groups the MediaPipe-based pose work: :mod:`pose_landmarker` (the MediaPipe
Tasks Pose Landmarker pipeline) and the companion notebook
``[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb``.

``pose_landmarker`` is a script, not a library: at import time it resolves the
repository root, downloads the ``pose_landmarker.task`` bundle when missing,
runs inference over ``data/pose`` and renders figures. Importing this package
must stay cheap and side-effect free, so nothing is re-exported -- invoke the
pipeline explicitly from the repository root::

    python -m src.pose.pose_landmarker

Shared, side-effect-free helpers (landmark drawing, image display) should be
promoted into this package and listed in ``__all__``.
"""

from __future__ import annotations

__all__: list[str] = []
