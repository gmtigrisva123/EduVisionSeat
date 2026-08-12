import math
from pathlib import Path
import glob
import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from IPython.display import display


# Configure MediaPipe shortcuts
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
plt.rcParams['figure.figsize'] = (6,6)


# Load images from a local folder (data/images/input).
REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = REPO_ROOT / 'data' / 'images' / 'input'
if not IMAGE_DIR.exists():
    IMAGE_DIR = REPO_ROOT

OUTPUT_DIR = REPO_ROOT / 'src' / 'notebook' / 'data' / 'images' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def save_output_image(image, name, prefix='pose'):
    save_path = OUTPUT_DIR / f'{prefix}_{name}'
    ok = cv2.imwrite(str(save_path), image)
    if ok:
        print(f'Saved output to: {save_path}')
    else:
        print(f'Failed to save output to: {save_path}')
    return save_path

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    # Convert BGR to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Read images with OpenCV from IMAGE_DIR
image_paths = sorted(IMAGE_DIR.glob('*'))
images = {}
for p in image_paths:
    img = cv2.imread(str(p))
    if img is None:
        continue
    images[p.name] = img
# Preview the images.
if not images:
    print(f'No images found in {IMAGE_DIR} (check the path).')
else:
    for name, image in images.items():
        print(name)
        resize_and_show(image)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

help(mp_pose.Pose)

# Run MediaPipe Pose and draw pose landmarks.
with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    for name, image in images.items():
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print nose landmark (pixel coordinates).
        image_height, image_width, _ = image.shape
        if not results.pose_landmarks:
            continue
        nose_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
        nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
        print(f'Nose coordinates: ({nose_x:.1f}, {nose_y:.1f})')

        # Draw pose landmarks.
        print(f'Pose landmarks of {name}:')
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        save_output_image(annotated_image, name, prefix='pose_landmarks')
        resize_and_show(annotated_image)

# Run MediaPipe Pose and plot 3d pose world landmarks.
with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    for name, image in images.items():
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print the real-world 3D coordinates of nose in meters with the origin at
        # the center between hips.
        if not getattr(results, 'pose_world_landmarks', None):
            continue
        print('Nose world landmark:' )
        print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
        
        # Plot pose world landmarks (if available).
        try:
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception as e:
            print('Plot not available:', e)


# Run MediaPipe Pose with `enable_segmentation=True` to get pose segmentation.
with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, 
    model_complexity=2, enable_segmentation=True
) as pose:
    for name, image in images.items():
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw pose segmentation.
        if getattr(results, 'segmentation_mask', None) is None:
            print(f'No segmentation available for {name}')
            continue
        print(f'Pose segmentation of {name}:')
        annotated_image = image.copy()
        red_img = np.zeros_like(annotated_image, dtype=np.uint8)
        red_img[:, :] = (255,255,255)
        segm_2class = 0.2 + 0.8 * results.segmentation_mask
        segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
        annotated_image = (annotated_image * segm_2class + red_img * (1 - segm_2class)).astype(np.uint8)
        save_output_image(annotated_image, name, prefix='pose_segmentation')
        resize_and_show(annotated_image)