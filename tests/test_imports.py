"""The packages must import cheaply and without side effects.

Both pipelines execute real work when run as scripts. Importing them must not:
loading torch or MediaPipe at import time slows every test run down, and any
import-time inference would make the packages unusable from other code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODULES = ("torch", "ultralytics", "mediapipe", "cv2", "matplotlib")


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter rooted at the repository."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("package", ["src.detect", "src.pose", "src.notebook"])
def test_import_pulls_in_no_heavy_dependency(package: str) -> None:
    result = run_python(
        f"import sys, {package};print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"{package} imported heavy modules: {result.stdout}"


def test_detect_exposes_its_api() -> None:
    from src import detect

    assert detect.__all__ == sorted(detect.__all__), "__all__ should stay sorted"
    for name in detect.__all__:
        assert hasattr(detect, name), name


def test_pose_resolves_its_api_lazily() -> None:
    """`src.pose` promises names it only imports on demand (PEP 562)."""
    from src import pose

    assert pose.__all__ == sorted(pose.__all__), "__all__ should stay sorted"
    assert sorted(dir(pose)) == sorted(pose.__all__)
    with pytest.raises(AttributeError):
        _ = pose.definitely_not_a_real_symbol
