"""Unit tests for the detection pipeline's pure logic.

Nothing here loads a model: `find_images` and the CLI parser are the parts that
decide *what* gets processed, so they are the parts worth pinning down.
"""

from __future__ import annotations

from pathlib import Path

from src.detect import detect as detect_module
from src.detect import find_images, run_detection


def make_images(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (directory / name).write_bytes(b"not a real image")


def test_find_images_sorts_by_name(tmp_path: Path) -> None:
    make_images(tmp_path, ("b.jpg", "a.jpg", "c.jpg"))
    assert [p.name for p in find_images(tmp_path)] == ["a.jpg", "b.jpg", "c.jpg"]


def test_find_images_matches_suffixes_case_insensitively(tmp_path: Path) -> None:
    make_images(tmp_path, ("a.JPG", "b.jpeg", "c.PNG", "notes.txt", "clip.mp4"))
    assert [p.name for p in find_images(tmp_path)] == ["a.JPG", "b.jpeg", "c.PNG"]


def test_find_images_applies_the_limit(tmp_path: Path) -> None:
    make_images(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))
    assert len(find_images(tmp_path, limit=2)) == 2
    assert len(find_images(tmp_path, limit=None)) == 3


def test_find_images_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "nested.jpg").mkdir()
    assert find_images(tmp_path) == []


def test_run_detection_on_an_empty_directory_is_a_no_op(tmp_path: Path) -> None:
    """No input means no model load, so this stays fast and torch-free."""
    output_dir = tmp_path / "out"
    assert run_detection(tmp_path, output_dir, limit=5) == []
    assert not output_dir.exists(), "no output directory should be created"


def test_defaults_point_inside_the_repository() -> None:
    root = detect_module.REPO_ROOT
    assert (root / "src").is_dir()
    assert root / "data" / "images" / "input" == detect_module.DEFAULT_INPUT_DIR
    assert root / "models" / "yolov8n.pt" == detect_module.DEFAULT_WEIGHTS


def test_cli_defaults_match_the_module_defaults() -> None:
    args = detect_module.build_parser().parse_args([])
    assert args.input_dir == detect_module.DEFAULT_INPUT_DIR
    assert args.output_dir == detect_module.DEFAULT_OUTPUT_DIR
    assert args.limit == detect_module.DEFAULT_LIMIT


def test_cli_accepts_overrides() -> None:
    args = detect_module.build_parser().parse_args(
        ["--input-dir", "/tmp/in", "--output-dir", "/tmp/out", "--limit", "0"]
    )
    assert args.input_dir == Path("/tmp/in")
    assert args.output_dir == Path("/tmp/out")
    assert args.limit == 0
