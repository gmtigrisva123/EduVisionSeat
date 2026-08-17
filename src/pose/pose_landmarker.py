import importlib.util
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


missing = [m for m in ("mediapipe", "cv2", "matplotlib", "numpy") if importlib.util.find_spec(m) is None]
print("Missing packages:", missing if missing else "none (all installed)")

def find_repo_root(start: Path) -> Path:
    """Return the first directory, walking up from `start`, that contains data/pose."""
    for path in (start, *start.parents):
        if (path / "data" / "pose").is_dir():
            return path
    raise FileNotFoundError(f"Could not locate the repository root (data/pose) starting from {start}")


ROOT = find_repo_root(Path.cwd().resolve())
IMAGE_PATH = ROOT / "data" / "pose" / "images_man_standing.jpeg"
MODEL_PATH = ROOT / "models" / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

assert IMAGE_PATH.is_file(), f"Input image not found: {IMAGE_PATH}"

if not MODEL_PATH.is_file():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the pose_landmarker.task model bundle ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print("ROOT       :", ROOT)
print("IMAGE_PATH :", IMAGE_PATH)
print("MODEL_PATH :", MODEL_PATH, f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw the detected pose landmarks on an RGB image and return the annotated copy."""
    # mp.Image.create_from_file() may return a 4-channel (RGBA) image, drawing_utils takes 3 channels.
    annotated_image = np.copy(np.asarray(rgb_image)[:, :, :3])

    for pose_landmarks in detection_result.pose_landmarks:
        # Convert the landmark list to protobuf so it can be used with drawing_utils.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks_proto,
            connections=solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

    return annotated_image


def show_image(rgb_image, title=None, width=8):
    """Display an RGB image inline (the local replacement for Colab's cv2_imshow)."""
    rgb_image = np.asarray(rgb_image)[:, :, :3]
    height = width * rgb_image.shape[0] / rgb_image.shape[1]
    plt.figure(figsize=(width, height))
    plt.imshow(rgb_image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


bgr_image = cv2.imread(str(IMAGE_PATH))
assert bgr_image is not None, f"Failed to read the image: {IMAGE_PATH}"

print("Image shape (H, W, C):", bgr_image.shape)
show_image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), title=IMAGE_PATH.name)


base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

image = mp.Image.create_from_file(str(IMAGE_PATH))

detection_result = detector.detect(image)
print("Detected poses:", len(detection_result.pose_landmarks))

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
show_image(annotated_image, title="Pose landmarks")

segmentation_mask = np.squeeze(detection_result.segmentation_masks[0].numpy_view())

# Convert to a 3-channel uint8 image for visualization.
visualized_mask = (segmentation_mask * 255).astype(np.uint8)
visualized_mask = np.stack([visualized_mask] * 3, axis=-1)

show_image(visualized_mask, title="Segmentation mask")

POSE_LANDMARK_NAMES = [lm.name for lm in solutions.pose.PoseLandmark]

for idx, landmark in enumerate(detection_result.pose_landmarks[0][:8]):
    print(f"{idx:2d} {POSE_LANDMARK_NAMES[idx]:<18} "
          f"x={landmark.x:.3f} y={landmark.y:.3f} z={landmark.z:.3f} vis={landmark.visibility:.2f}")

print(f"... {len(detection_result.pose_landmarks[0])} landmarks in total")
