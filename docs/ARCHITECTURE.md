# Architecture

## Repository layout

```
EduVisionSeat/
├── data/                       # Datasets (mostly uncommitted — see data-and-ethics.md)
│   ├── images/input/           #   classroom input images       [gitignored]
│   ├── images/output/          #   annotated results            [gitignored]
│   ├── pose/                   #   sample image for the pose pipeline [committed]
│   └── videos/                 #   input video                  [gitignored]
├── docs/                       # Documentation (this directory)
├── models/
│   └── pose_landmarker.task    # MediaPipe model bundle         [committed]
├── src/
│   ├── detect/                 # Detection package — currently empty
│   ├── notebook/               # Exploratory scripts
│   │   ├── detect.py           #   YOLOv8 pipeline (script)
│   │   └── yolov8n.pt
│   └── pose/                   # Pose package
│       ├── pose_landmarker.py  #   MediaPipe Tasks pipeline (script)
│       └── pose_landmarker_notebook.ipynb
├── yolov8n.pt                  # YOLOv8-nano weights at the repository root
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Development dependencies (includes requirements.txt)
```

## The two pipelines

### Object detection — `src/notebook/detect.py`

```
data/images/input/*.{jpg,jpeg,png}
        │
        ▼
  YOLO("yolov8n.pt")            ← loaded from the current working directory
        │
        ▼
data/images/output/detected_<original name>
```

Path resolution: `find_repo_root()` walks up from `Path.cwd()` through the parent
directories and returns the first one containing `data/pose`. Consequence: the script
is **working-directory dependent** and must be run from inside the repository.

### Pose analysis — `src/pose/pose_landmarker.py`

```
data/pose/images_man_standing.jpeg
        │
        ▼
  MediaPipe PoseLandmarker      ← models/pose_landmarker.task (downloaded if missing)
        │
        ├──▶ 33 landmarks + segmentation mask
        └──▶ rendered through matplotlib
```

This uses the **MediaPipe Tasks** API (`mediapipe.tasks.python.vision`), which is
distinct from the older `mp.solutions.pose` API. Results are only displayed on screen;
nothing is written to disk.

## Package conventions

The repository follows a **src layout**: `src/` is a container directory, not a package
(there is no `src/__init__.py`), while `src/detect`, `src/notebook` and `src/pose` are
real packages. Thanks to namespace packages (PEP 420), `python -m
src.pose.pose_landmarker` still resolves correctly when run from the repository root.

### Rule: `__init__.py` must be free of side effects

Both `detect.py` and `pose_landmarker.py` **execute their pipeline at module level**:
loading weights, asserting that files exist, downloading a model over the network,
calling `plt.show()`. If an `__init__.py` re-exported them, a single `import src.pose`
would be enough to trigger the whole inference run — or to break `pytest` collection.

For that reason all three `__init__.py` files contain only a docstring and
`__all__: list[str] = []`, and import nothing. To run a pipeline, invoke it explicitly
with `python -m`.

### Intended direction

When extracting reusable logic out of the scripts:

1. Move pure, side-effect-free functions into `src/detect/` or `src/pose/` — for
   example `draw_landmarks_on_image()` and `show_image()` in `pose_landmarker.py`.
2. Export them from the corresponding `__init__.py` and add them to `__all__`.
3. Keep the execution part inside an `if __name__ == "__main__":` block or in a
   notebook, so the module stays importable without doing any work.
4. Keep `src/notebook/` for exploratory code only; anything that has stabilised belongs
   in `src/detect/` or `src/pose/`.

## Model assets

| File | Size | Source | Status |
| --- | --- | --- | --- |
| `yolov8n.pt` (repository root) | ~6.5 MB | Ultralytics YOLOv8-nano | Committed |
| `src/notebook/yolov8n.pt` | ~6.5 MB | Duplicate copy for the notebooks | Committed |
| `models/pose_landmarker.task` | ~31 MB | MediaPipe `pose_landmarker_heavy` float16 | Committed; downloaded if missing |

There are two identical copies of `yolov8n.pt` in the repository. `detect.py` uses the
one at the repository root (it loads relative to the working directory); the copy under
`src/notebook/` serves notebooks run from that directory.
