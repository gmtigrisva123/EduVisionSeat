"""YOLOv8 object detection over classroom stills.

Reads images from ``data/images/input``, runs YOLOv8 detection over them and
writes annotated copies to ``data/images/output``.

Run it from anywhere::

    python -m src.detect.detect
    python -m src.detect.detect --limit 10 --input-dir /path/to/images

Importing this module is cheap and free of side effects: ``ultralytics`` (and the
``torch`` stack behind it) is imported only when a model is actually loaded.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

#: Repository root, derived from this file so the module works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "images" / "input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "images" / "output"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolov8n.pt"
DEFAULT_LIMIT = 5
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
OUTPUT_PREFIX = "detected_"


def find_images(
    directory: Path,
    *,
    limit: int | None = None,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> list[Path]:
    """Return the image files in ``directory``, sorted by name.

    Suffixes are matched case-insensitively, so ``.JPG`` and ``.jpg`` both count.
    ``limit`` keeps only the first N results; ``None`` keeps all of them.
    """
    wanted = {suffix.lower() for suffix in suffixes}
    images = sorted(
        path for path in directory.glob("*") if path.is_file() and path.suffix.lower() in wanted
    )
    return images if limit is None else images[:limit]


def load_model(weights: Path = DEFAULT_WEIGHTS):
    """Load the YOLO model from ``weights``.

    When the file is missing, the bare file name is handed to Ultralytics, which
    downloads the matching pretrained checkpoint into the working directory.
    """
    from ultralytics import YOLO  # imported here: it pulls in the torch stack

    if weights.is_file():
        LOGGER.info("Loading weights: %s", weights)
        return YOLO(str(weights))

    LOGGER.warning("Weights not found at %s; letting Ultralytics fetch %s", weights, weights.name)
    return YOLO(weights.name)


def run_detection(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    weights: Path = DEFAULT_WEIGHTS,
    limit: int | None = DEFAULT_LIMIT,
) -> list[Path]:
    """Detect objects in the images of ``input_dir`` and save annotated copies.

    Returns the paths that were written, in processing order. An empty input
    directory is not an error: nothing is written and no model is loaded.
    """
    images = find_images(input_dir, limit=limit)
    if not images:
        LOGGER.warning("No images found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(weights)
    LOGGER.info("Analyzing %d image(s) from %s", len(images), input_dir)

    saved: list[Path] = []
    for image_path in images:
        destination = output_dir / f"{OUTPUT_PREFIX}{image_path.name}"
        for result in model(str(image_path)):
            result.save(filename=str(destination))
        saved.append(destination)
        LOGGER.info("Processed and saved: %s", destination)

    return saved


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for this pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="directory of input images"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for annotated output"
    )
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help="path to the YOLO weights"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="process at most N images (0 or negative means no limit)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m src.detect.detect``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    saved = run_detection(
        args.input_dir,
        args.output_dir,
        weights=args.weights,
        limit=args.limit if args.limit > 0 else None,
    )
    LOGGER.info("Done: %d image(s) written to %s", len(saved), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
