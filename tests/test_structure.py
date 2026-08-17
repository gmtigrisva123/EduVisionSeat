"""Structural guarantees for the repository layout.

These tests encode decisions that are easy to break silently: the two pipeline
packages must stay symmetrical (module + notebook + package marker), and no
student media may ever be tracked by Git.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PACKAGES = ("detect", "pose")


@pytest.mark.parametrize("package", PIPELINE_PACKAGES)
def test_package_has_init(package: str) -> None:
    assert (REPO_ROOT / "src" / package / "__init__.py").is_file()


@pytest.mark.parametrize("package", PIPELINE_PACKAGES)
def test_package_has_a_notebook(package: str) -> None:
    notebooks = sorted((REPO_ROOT / "src" / package).glob("*.ipynb"))
    assert notebooks, f"src/{package} must ship at least one notebook"


@pytest.mark.parametrize("package", PIPELINE_PACKAGES)
def test_package_has_a_console_entry_point(package: str) -> None:
    """`python -m src.<package>` must work, so each package ships a __main__.py."""
    assert (REPO_ROOT / "src" / package / "__main__.py").is_file()


@pytest.mark.parametrize("package", PIPELINE_PACKAGES)
def test_package_has_a_pipeline_module(package: str) -> None:
    modules = [
        path for path in (REPO_ROOT / "src" / package).glob("*.py") if path.name != "__init__.py"
    ]
    assert modules, f"src/{package} must ship at least one pipeline module"


def test_notebooks_are_valid_json() -> None:
    notebooks = sorted(REPO_ROOT.glob("src/*/*.ipynb"))
    assert notebooks, "no notebooks found under src/"
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4, notebook
        assert payload["cells"], notebook


def test_no_student_media_is_tracked() -> None:
    """`data/images` and `data/videos` hold classroom media and must stay untracked."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", "data/images", "data/videos"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == [], f"student media is tracked by Git: {tracked}"
