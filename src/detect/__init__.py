"""Object detection package for EduVisionSeat.

Hosts the detection layer of the project (locating students and chairs in
classroom images and video). The package is intentionally side-effect free:
importing it must never load a model, read the filesystem or open a window, so
that callers can import it from tests, notebooks and CLI entry points alike.

The implementation currently lives in the exploratory scripts under
``src/notebook`` and is being promoted into this package module by module.
Anything meant to be part of the public API should be imported here and listed
in ``__all__``.
"""

from __future__ import annotations

__all__: list[str] = []
