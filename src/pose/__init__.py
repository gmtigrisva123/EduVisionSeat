"""Pose estimation package for EduVisionSeat.

Wraps the MediaPipe Tasks Pose Landmarker: 33 body landmarks plus a segmentation
mask per person, which feed the posture and gaze signals used to estimate
observable behavioral engagement.

:mod:`src.pose.pose_landmarker` imports MediaPipe, OpenCV and matplotlib at module
level, so the public symbols below are resolved **lazily** on first attribute
access (PEP 562). That keeps ``import src.pose`` cheap and side-effect free while
still offering a real package API::

    from src.pose import detect_pose, draw_landmarks_on_image

    mp_image, result = detect_pose()
    annotated = draw_landmarks_on_image(mp_image.numpy_view(), result)

The same pipeline is available as a command line entry point::

    python -m src.pose --no-show
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from .pose_landmarker import (
        POSE_LANDMARK_NAMES,
        create_detector,
        detect_pose,
        draw_landmarks_on_image,
        read_image,
        segmentation_mask_image,
        show_image,
    )

__all__ = [
    "POSE_LANDMARK_NAMES",
    "create_detector",
    "detect_pose",
    "draw_landmarks_on_image",
    "read_image",
    "segmentation_mask_image",
    "show_image",
]


def __getattr__(name: str) -> object:
    """Resolve the public API on demand, importing the heavy pipeline module once."""
    if name in __all__:
        from . import pose_landmarker

        return getattr(pose_landmarker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
