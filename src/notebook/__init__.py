"""Exploratory notebooks and companion scripts for EduVisionSeat.

This package groups the notebook-style code used to explore the detection and
pose pipelines: :mod:`detect` (YOLOv8 object detection over
``data/images/input``) and ``pose.ipynb``.

``detect`` is a script, not a library: at import time it loads the YOLO
weights, reads every image from ``data/images/input`` and writes annotated
copies to ``data/images/output``. Importing this package must stay cheap and
side-effect free, so nothing is re-exported -- run the pipeline explicitly from
the repository root::

    python -m src.notebook.detect

Reusable logic belongs in :mod:`src.detect` or :mod:`src.pose`, not here.
"""

from __future__ import annotations

__all__: list[str] = []
