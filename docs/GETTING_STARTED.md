# Getting Started

## 1. Requirements

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.10 – 3.11 | CI tests 3.10 and 3.11; the current development environment uses 3.11 |
| Disk space | ~2 GB | Mostly `torch` (pulled in by `ultralytics`) and `mediapipe` |
| Network | First run only | To download the model bundle if it is not already in `models/` |

No GPU is required: both pipelines run on CPU.

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
```

Install dependencies — use `requirements-dev.txt` if you intend to change code, since
it already includes `-r requirements.txt`:

```bash
pip install -r requirements-dev.txt   # development: adds pytest, ruff, bandit, pip-audit
pip install -r requirements.txt       # running the pipelines only
```

`ultralytics` pulls in `torch`, which is a large download — expect the first install to
take a while. Versions verified in the current development environment:
`ultralytics 8.4.118`, `torch 2.2.2`.

## 3. Prepare the data

Input images live in `data/images/input/`. That directory is **not committed** (see
[data-and-ethics.md](data-and-ethics.md)), so you must supply your own images after
cloning:

```bash
mkdir -p data/images/input data/images/output
cp /path/to/images/*.jpg data/images/input/
```

The only committed sample image is `data/pose/images_man_standing.jpeg`, which is what
the pose pipeline uses — so `python -m src.pose.pose_landmarker` works immediately
after a clone.

## 4. Run the object detection pipeline (YOLOv8)

```bash
python -m src.notebook.detect
```

What it does:

- Reads the **first 5 images** (sorted by name) in `data/images/input/` with the
  extensions `.jpg`, `.jpeg`, `.png` (uppercase variants included).
- Loads the `yolov8n.pt` weights from the **current working directory**, which is why
  the command must be run from the repository root where `yolov8n.pt` lives.
- Writes annotated images into `data/images/output/` with a `detected_` prefix.
- If the input directory is empty, the script prints a warning and exits without
  loading the model.

## 5. Run the pose analysis pipeline (MediaPipe Tasks)

```bash
python -m src.pose.pose_landmarker
```

What it does:

- Processes `data/pose/images_man_standing.jpeg`; the script asserts if the image is
  missing.
- Requires `models/pose_landmarker.task`. That file is committed; if it is missing, the
  script downloads the `pose_landmarker_heavy` bundle (~31 MB) from Google Cloud
  Storage.
- Displays the original image, the landmark overlay and the segmentation mask through
  `matplotlib`, and prints the coordinates of the first 8 landmarks.

Because the script opens `matplotlib` windows, run it on a machine with a display (or
from the `src/pose/pose_landmarker_notebook.ipynb` notebook). On a
headless machine, select a non-interactive backend:

```bash
MPLBACKEND=Agg python -m src.pose.pose_landmarker
```

## 6. Working with the notebook

```bash
jupyter notebook src/pose/pose_landmarker_notebook.ipynb
```

Start Jupyter from the **repository root** so that relative paths inside the notebook
resolve correctly.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: Could not locate the repository root (data/pose)` | Command run outside the repository | `cd` to the repository root and retry |
| `ModuleNotFoundError: No module named 'ultralytics'` | Dependencies not installed in the active environment | Activate the virtualenv, then `pip install -r requirements.txt` |
| `AssertionError: Input image not found` | `data/pose/images_man_standing.jpeg` is missing | `git restore data/pose/images_man_standing.jpeg` |
| `Not found any images in the folder: .../data/images/input` | No input data yet | Copy images into `data/images/input/` |
