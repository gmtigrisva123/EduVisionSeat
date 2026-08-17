"""Object detection package for EduVisionSeat.

Hosts the detection layer of the project: locating students in classroom stills so
their behaviour can later be tied to a seat on the classroom chart.

:mod:`src.detect.detect` is a runnable pipeline, so the public symbols below are
resolved **lazily** on first attribute access (PEP 562). Importing this package
therefore costs nothing, touches neither the network nor the disk, and never
shadows the module when it is run as a script.

Typical use::

    from src.detect import run_detection

    written = run_detection(limit=5)

The same pipeline is available as a command line entry point::

    python -m src.detect --limit 5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from .detect import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_WEIGHTS,
        find_images,
        load_model,
        run_detection,
    )

__all__ = [
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_WEIGHTS",
    "find_images",
    "load_model",
    "run_detection",
]


def __getattr__(name: str) -> object:
    """Resolve the public API on demand, importing the pipeline module once."""
    if name in __all__:
        from . import detect

        return getattr(detect, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
