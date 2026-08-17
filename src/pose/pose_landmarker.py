"""MediaPipe Tasks Pose Landmarker over a still image.

Detects the 33 pose landmarks and the segmentation mask for a single image, and
renders the result with matplotlib.

Run it from anywhere::

    python -m src.pose.pose_landmarker
    python -m src.pose.pose_landmarker --image path/to/image.jpg --no-show

The model bundle is downloaded into ``models/`` on first use if it is missing.
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

#: Repository root, derived from this file so the module works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = REPO_ROOT / "data" / "pose" / "images_man_standing.jpeg"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

#: Landmark names in the order MediaPipe reports them.
POSE_LANDMARK_NAMES = [landmark.name for landmark in solutions.pose.PoseLandmark]


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Return the model bundle path, downloading it first if it is missing."""
    if model_path.is_file():
        return model_path

    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download the model bundle over a non-HTTPS URL: {url}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading the pose_landmarker model bundle from %s", url)
    # The URL scheme is validated above, so B310 (audit url open) does not apply here.
    urllib.request.urlretrieve(url, model_path)  # nosec B310
    return model_path


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from disk as BGR, raising a clear error when that fails."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        raise ValueError(f"Failed to read the image: {image_path}")
    return bgr_image


def draw_landmarks_on_image(rgb_image, detection_result) -> np.ndarray:
    """Draw the detected pose landmarks on an RGB image and return the annotated copy."""
    # mp.Image.create_from_file() may return a 4-channel (RGBA) image, drawing_utils takes 3.
    annotated_image = np.copy(np.asarray(rgb_image)[:, :, :3])

    for pose_landmarks in detection_result.pose_landmarks:
        # Convert the landmark list to protobuf so it can be used with drawing_utils.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks]
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks_proto,
            connections=solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2
            ),
        )

    return annotated_image


def segmentation_mask_image(detection_result) -> np.ndarray | None:
    """Return the first segmentation mask as a 3-channel uint8 image, if present."""
    masks = getattr(detection_result, "segmentation_masks", None)
    if not masks:
        return None

    mask = np.squeeze(masks[0].numpy_view())
    visualized = (mask * 255).astype(np.uint8)
    return np.stack([visualized] * 3, axis=-1)


def show_image(rgb_image, title: str | None = None, width: float = 8) -> None:
    """Display an RGB image inline (the local replacement for Colab's cv2_imshow)."""
    rgb_image = np.asarray(rgb_image)[:, :, :3]
    height = width * rgb_image.shape[0] / rgb_image.shape[1]
    plt.figure(figsize=(width, height))
    plt.imshow(rgb_image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def create_detector(model_path: Path = DEFAULT_MODEL_PATH) -> vision.PoseLandmarker:
    """Create a Pose Landmarker that also outputs segmentation masks."""
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
        output_segmentation_masks=True,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(
    image_path: Path = DEFAULT_IMAGE_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Run the Pose Landmarker on one image and return ``(mp_image, detection_result)``."""
    read_image(image_path)  # fail early with a readable error if the file is unusable
    detector = create_detector(model_path)
    mp_image = mp.Image.create_from_file(str(image_path))
    return mp_image, detector.detect(mp_image)


def log_landmarks(detection_result, limit: int = 8) -> None:
    """Log the first ``limit`` landmarks of the first detected pose."""
    if not detection_result.pose_landmarks:
        LOGGER.warning("No pose detected")
        return

    landmarks = detection_result.pose_landmarks[0]
    for index, landmark in enumerate(landmarks[:limit]):
        LOGGER.info(
            "%2d %-18s x=%.3f y=%.3f z=%.3f vis=%.2f",
            index,
            POSE_LANDMARK_NAMES[index],
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility,
        )
    LOGGER.info("%d landmarks in total", len(landmarks))


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for this pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="image to analyze")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="path to the .task model bundle"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="skip the matplotlib figures (headless runs)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m src.pose.pose_landmarker``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    LOGGER.info("Image: %s", args.image)
    mp_image, detection_result = detect_pose(args.image, args.model)
    LOGGER.info("Detected poses: %d", len(detection_result.pose_landmarks))

    if not args.no_show:
        show_image(mp_image.numpy_view(), title=args.image.name)
        show_image(
            draw_landmarks_on_image(mp_image.numpy_view(), detection_result), title="Pose landmarks"
        )
        mask = segmentation_mask_image(detection_result)
        if mask is None:
            LOGGER.warning("No segmentation mask available")
        else:
            show_image(mask, title="Segmentation mask")

    log_landmarks(detection_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
