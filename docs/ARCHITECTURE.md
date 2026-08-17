# Architecture

## Repository layout

```
EduVisionSeat/
├── .github/
│   ├── dependabot.yml          # Weekly pip and actions updates
│   └── workflows/              # CI, Security, Pipeline smoke test, Release
├── data/                       # Datasets — classroom media is gitignored, never committed
│   ├── images/input/           #   classroom input images       [gitignored]
│   ├── images/output/          #   annotated results            [gitignored]
│   ├── pose/                   #   sample image for the pose pipeline [committed]
│   └── videos/                 #   input video                  [gitignored]
├── docs/                       # Documentation (this directory)
├── models/
│   ├── pose_landmarker.task    # MediaPipe model bundle         [committed]
│   └── yolov8n.pt              # YOLOv8-nano weights            [committed]
├── src/
│   ├── detect/                 # Detection pipeline
│   │   ├── __init__.py         #   lazy public API
│   │   ├── __main__.py         #   `python -m src.detect`
│   │   ├── detect.py           #   YOLOv8 pipeline + CLI
│   │   └── detect_notebook.ipynb
│   ├── notebook/               # Legacy location, kept for older references
│   │   └── __init__.py
│   └── pose/                   # Pose pipeline
│       ├── __init__.py         #   lazy public API
│       ├── __main__.py         #   `python -m src.pose`
│       ├── pose_landmarker.py  #   MediaPipe Tasks pipeline + CLI
│       └── pose_landmarker_notebook.ipynb
├── tests/                      # Unit and structural tests
├── pyproject.toml              # ruff, pytest and bandit configuration
├── requirements.txt            # Runtime dependencies
├── requirements-tools.txt      # Lint/test/audit tooling only
└── requirements-dev.txt        # Runtime + tooling
```

The two pipeline packages are deliberately symmetrical — module, CLI entry point,
notebook, lazy API — so that a third pipeline (seat mapping, engagement scoring) has an
obvious shape to follow. `tests/test_structure.py` enforces that symmetry.

## The two pipelines

### Detection — `src/detect/detect.py`

```
data/images/input/*.{jpg,jpeg,png}
        │
        ▼
  YOLO(models/yolov8n.pt)       ← downloaded by Ultralytics if the file is absent
        │
        ▼
data/images/output/detected_<original name>
```

Public API: `find_images`, `load_model`, `run_detection`. `run_detection` returns the
paths it wrote, so callers and notebooks can inspect the result instead of scraping logs.

### Pose — `src/pose/pose_landmarker.py`

```
data/pose/images_man_standing.jpeg
        │
        ▼
  MediaPipe PoseLandmarker      ← models/pose_landmarker.task (downloaded if missing)
        │
        ├──▶ 33 landmarks + segmentation mask
        └──▶ rendered through matplotlib (skip with --no-show)
```

Public API: `detect_pose`, `draw_landmarks_on_image`, `segmentation_mask_image`,
`show_image`, `create_detector`, `read_image`, `POSE_LANDMARK_NAMES`. Uses the
**MediaPipe Tasks** API (`mediapipe.tasks.python.vision`), not the older
`mp.solutions.pose` API.

## Conventions

### Paths resolve from the source file, not the working directory

Both modules derive `REPO_ROOT` from `Path(__file__).resolve().parents[2]`, so
`data/`, `models/` and outputs land in the right place no matter where you invoke them
from. You still start `python -m src.detect` at the repository root, but only because
that is how `src` becomes importable.

### `__init__.py` exposes an API, and stays free of side effects

Both pipeline modules run real work when executed as scripts and import heavy
dependencies (torch through Ultralytics; MediaPipe, OpenCV and matplotlib). Importing a
package must not pay that cost, and must never trigger inference.

Both `__init__.py` files therefore resolve their public names **lazily** through a
module-level `__getattr__` (PEP 562):

```python
from src.detect import run_detection   # imports src.detect.detect on first access
from src.pose import detect_pose       # imports MediaPipe only at this point
```

`import src.detect` and `import src.pose` load nothing heavier than the standard
library — `tests/test_imports.py` asserts exactly that in a fresh interpreter. The same
laziness avoids a second failure mode: an `__init__.py` that eagerly imports its own
submodule makes `python -m src.detect.detect` emit a `RuntimeWarning`, because the
module is then already in `sys.modules` when runpy executes it.

### Execution lives behind `main()`

Each module keeps its work in functions, exposes an `argparse` CLI through
`build_parser()`/`main()`, and ends with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

`__main__.py` in each package forwards to that same `main()`, so `python -m src.detect`
and `python -m src.detect.detect` behave identically.

### Heavy imports are deferred inside functions

`src/detect/detect.py` imports `ultralytics` inside `load_model()` rather than at module
level. That keeps the module importable — and the test suite fast — on a machine with no
torch installed at all.

## Tests

| File | What it pins down |
| --- | --- |
| `tests/test_structure.py` | Each pipeline package has `__init__.py`, `__main__.py`, a module and a notebook; notebooks are valid nbformat 4; no student media is tracked by Git |
| `tests/test_imports.py` | Importing any package pulls in no heavy dependency; the lazy APIs resolve; `__all__` stays sorted |
| `tests/test_detect.py` | Image selection (order, suffixes, limit, directories) and CLI defaults |

The suite needs only `requirements-tools.txt`, which is why CI can run it in under a
minute without downloading the runtime stack.

## Model assets

| File | Size | Source | Status |
| --- | --- | --- | --- |
| `models/yolov8n.pt` | ~6.5 MB | Ultralytics YOLOv8-nano | Committed; used by the detection pipeline |
| `models/pose_landmarker.task` | ~31 MB | MediaPipe `pose_landmarker_heavy` float16 | Committed; downloaded if missing |
| `yolov8n.pt` (repository root) | ~6.5 MB | Older copy, kept for provenance | Committed; not used by the code any more |

Release archives exclude the model files: both pipelines fetch what they need on first
run.
