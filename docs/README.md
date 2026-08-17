# EduVisionSeat Documentation

Technical documentation for EduVisionSeat — an object detection and pose analysis system
for classroom images and video, built for educational research.

## Contents

| Document | Covers |
| --- | --- |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Environment setup, running each pipeline, CLI options, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Source layout, data flow, package conventions, tests, model assets |
| [DATA_AND_ETHICS.md](DATA_AND_ETHICS.md) | Data policy and student privacy |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branching workflow, CI gates, releases |

## Project status

The project is at an **exploratory** stage: the two perception pipelines work end to end,
and the classroom-level layers they feed into — seat mapping, engagement estimation,
fairness-aware seat optimization — are not built yet.

What exists today:

- `src/detect` — YOLOv8 detection over classroom stills, with a CLI, a public API and a
  notebook walkthrough.
- `src/pose` — MediaPipe Tasks Pose Landmarker (33 landmarks plus segmentation mask),
  with a CLI, a public API and a notebook.
- `tests/` and four GitHub Actions workflows covering lint, security, data hygiene,
  end-to-end pipeline runs and releases.

Both packages follow the same shape, so the next pipeline has a template to copy. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the conventions that shape is built on.

## Quick reference

```bash
# Set up the environment (details in GETTING_STARTED.md)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Pose landmarks on the committed sample image — works right after cloning
python -m src.pose

# Object detection over data/images/input
python -m src.detect --limit 5

# The checks CI runs
ruff check src tests && bandit -c pyproject.toml -r src -q && pytest
```

Run every command from the **repository root** so that `src` is importable.
