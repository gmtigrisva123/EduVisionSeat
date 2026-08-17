# Getting Started

## 1. Requirements

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.10 – 3.11 | CI tests 3.10 and 3.11; the current development environment uses 3.11 |
| Disk space | ~2 GB | Mostly `torch` (pulled in by `ultralytics`) and `mediapipe` |
| Network | First run only | To download a model bundle if it is not already in `models/` |

No GPU is required: both pipelines run on CPU.

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
```

The repository declares dependencies in three files, so you install only what you need:

| File | Contents | Install it when |
| --- | --- | --- |
| `requirements.txt` | Runtime: numpy, OpenCV, matplotlib, MediaPipe, Ultralytics, Jupyter | You want to run the pipelines |
| `requirements-tools.txt` | Tooling: pytest, ruff, bandit, pip-audit, nbformat | You only want to lint or test (no torch download) |
| `requirements-dev.txt` | Both of the above | You are developing |

```bash
pip install -r requirements-dev.txt
```

`ultralytics` pulls in `torch`, which is a large download — expect the first install to
take a while. Versions verified in the current development environment:
`ultralytics 8.4.118`, `torch 2.2.2`.

## 3. Prepare the data

Input images live in `data/images/input/`. That directory is **not committed** (see
[DATA_AND_ETHICS.md](DATA_AND_ETHICS.md)), so you must supply your own images after
cloning:

```bash
mkdir -p data/images/input
cp /path/to/images/*.jpg data/images/input/
```

The only committed sample image is `data/pose/images_man_standing.jpeg`, which is what
the pose pipeline uses — so the pose pipeline works immediately after a clone.

## 4. Run the detection pipeline (YOLOv8)

```bash
python -m src.detect
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | `data/images/input` | Directory of input images |
| `--output-dir` | `data/images/output` | Where annotated copies are written |
| `--weights` | `models/yolov8n.pt` | YOLO checkpoint; downloaded by Ultralytics if absent |
| `--limit` | `5` | Process at most N images; `0` means no limit |

```bash
python -m src.detect --limit 20
python -m src.detect --input-dir /data/session-3 --output-dir /tmp/annotated --limit 0
```

What it does: selects images by name order (`.jpg`, `.jpeg`, `.png`, case-insensitive),
runs detection, and writes `detected_<original name>` into the output directory. An empty
input directory is not an error — nothing is written and no model is loaded.

## 5. Run the pose pipeline (MediaPipe Tasks)

```bash
python -m src.pose
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--image` | `data/pose/images_man_standing.jpeg` | Image to analyze |
| `--model` | `models/pose_landmarker.task` | Model bundle; downloaded on first use if absent |
| `--no-show` | off | Skip the matplotlib figures — use this on headless machines |

What it does: detects the 33 pose landmarks plus a segmentation mask, displays the
original image, the landmark overlay and the mask, and logs the first 8 landmarks.

The model bundle (`pose_landmarker_heavy`, ~31 MB) is committed. If it is missing, it is
downloaded from Google Cloud Storage over HTTPS.

```bash
python -m src.pose --no-show                       # headless, no figures
python -m src.pose --image data/pose/other.jpg     # a different image
```

## 6. Working with the notebooks

Each pipeline package ships one notebook next to the code it exercises:

```bash
jupyter notebook src/detect/detect_notebook.ipynb
jupyter notebook src/pose/pose_landmarker_notebook.ipynb
```

Both notebooks locate the repository root themselves, so they work whether the kernel
starts in the notebook's folder or at the repository root.

## 7. Run the checks

The same gates CI runs (see [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
ruff check src tests
ruff format --check --exclude '*.ipynb' src tests
bandit -c pyproject.toml -r src -q
pytest
```

## Working directory

Run commands from the **repository root**: `python -m src.detect` needs `src` to be
importable, and it is found through the working directory. The data paths themselves no
longer depend on your working directory — both pipelines resolve `data/`, `models/` and
the repository root from their own file location.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `No module named src` | Command run outside the repository root | `cd` to the repository root and retry |
| `ModuleNotFoundError: No module named 'ultralytics'` | Dependencies not installed in the active environment | Activate the virtualenv, then `pip install -r requirements.txt` |
| `FileNotFoundError: Input image not found` | The pose sample image is missing | `git restore data/pose/images_man_standing.jpeg` |
| `No images found in .../data/images/input` | No input data yet | Copy images into `data/images/input/` |
| Figures block the run, or crash over SSH | No display available | Add `--no-show`, or set `MPLBACKEND=Agg` |
