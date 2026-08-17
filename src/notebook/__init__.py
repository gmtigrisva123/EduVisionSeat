"""Legacy home of the exploratory detection script.

The YOLOv8 pipeline that used to live here as ``detect.py`` now lives in
:mod:`src.detect.detect`, and the YOLOv8 weights moved to ``models/yolov8n.pt``.
Exploratory notebooks now sit next to the code they exercise:
``src/detect/detect_notebook.ipynb`` and ``src/pose/pose_landmarker_notebook.ipynb``.

This package is kept rather than removed so that older commits, research notes and
notebook checkouts that refer to ``src/notebook`` still resolve. It exports nothing
and should not be imported by new code -- use :mod:`src.detect` or :mod:`src.pose`.
"""

from __future__ import annotations

__all__: list[str] = []
