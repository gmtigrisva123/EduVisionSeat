# EduVisionSeat Documentation

Technical documentation for EduVisionSeat — an object detection and pose analysis
system for classroom images and video, built for educational research.

## Contents

| Document | Covers |
| --- | --- |
| [getting-started.md](getting-started.md) | Environment setup and running each pipeline |
| [architecture.md](architecture.md) | Source layout, data flow, package conventions |
| [data-and-ethics.md](data-and-ethics.md) | Data policy and student privacy |
| [contributing.md](contributing.md) | Branching workflow, CI checks, releases |

## Project status

The project is at an **exploratory** stage: the current code consists of scripts and
notebooks used to evaluate models, not a stable library. Specifically:

- `src/notebook/detect.py` and `src/pose/pose_landmarker.py` run their whole pipeline
  at import time (loading weights, reading images, drawing figures). They are
  **scripts**, not modules meant to be imported.
- `src/detect/` is a package scaffold that is still empty — it is the destination for
  reusable logic once it is extracted out of those scripts.

This documentation therefore describes **how to run** the pipelines rather than a
public API. See [architecture.md](architecture.md) for package boundaries and the
intended direction.

## Quick reference

```bash
# Set up the environment (details in getting-started.md)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Object detection with YOLOv8 over data/images/input
python -m src.notebook.detect

# Pose analysis with the MediaPipe Pose Landmarker
python -m src.pose.pose_landmarker
```

Run every command from the **repository root** — the scripts resolve their paths from
the current working directory.
