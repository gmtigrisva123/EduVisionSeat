# EduVisionSeat

**Privacy-preserving AI-supported classroom seating recommendation, driven by
teacher-validated observable behavioral engagement.**

[![CI](https://github.com/gmtigrisva123/EduVisionSeat/actions/workflows/ci.yml/badge.svg)](https://github.com/gmtigrisva123/EduVisionSeat/actions/workflows/ci.yml)
[![Security](https://github.com/gmtigrisva123/EduVisionSeat/actions/workflows/security.yml/badge.svg)](https://github.com/gmtigrisva123/EduVisionSeat/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)

Seating charts in a 40–50 student classroom are still decided by eye. Teachers weigh
eyesight, height, attention, behaviour, peer relationships and support needs — with no
objective record of how a given seat actually affects a given student. EduVisionSeat
studies that question with computer vision, and turns the answer into seating
recommendations a teacher can accept, reject or adjust.

The system is a decision-support tool, not a replacement for teacher judgement, and it
is designed to keep student media local (see [Data and ethics](#data-and-ethics)).

> The research documentation refers to this work as **EduSeatAI**; `EduVisionSeat` is
> the repository name.

## Research paper

| | |
| --- | --- |
| **Topic** | Privacy-Preserving AI-Supported Classroom Seating Recommendation Using Teacher-Validated Observable Behavioral Engagement |
| **Paper** | <!-- TODO: replace with the paper URL --> _link to be added_ |
| **Status** | In preparation |

<!-- TODO: fill in once the paper is published -->
```bibtex
@article{eduseatai,
  title   = {Privacy-Preserving AI-Supported Classroom Seating Recommendation
             Using Teacher-Validated Observable Behavioral Engagement},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  url     = {TODO}
}
```

## Research questions

1. **Association** — How is students' seating position associated with observable
   behavioral engagement in classroom videos or structured classroom observations?
2. **Prediction** — How accurately can computer-vision-based or manually coded
   behavioral features predict teacher/human-rated behavioral engagement across
   different seating positions?
3. **Decision support** — How can behavioral engagement estimates be used in a
   fairness-aware mathematical optimization model to generate explainable seating
   recommendations for teachers?

Question 3 is what keeps the project honest: recommendations must be *explainable* and
*fairness-aware*, so a seat assignment can always be justified to the teacher.

## Approach

Classroom video is the input. The intended pipeline is:

1. **Detect** students in the frame.
2. **Locate** each student on the classroom seating chart.
3. **Track** observable behavioral signals — gaze direction, body posture, interaction
   frequency, participation in class activities.
4. **Estimate** engagement indicators per student and per seat.
5. **Compare** indicators across seating positions to surface high- and low-engagement
   regions of the room.
6. **Recommend** a seating chart through fairness-aware optimization, with the reasoning
   exposed to the teacher.

Only observable behaviour is modelled, and every engagement estimate is validated
against teacher/human ratings rather than treated as ground truth on its own.

## Project status

The repository is at an **exploratory stage**: the two perception building blocks run
end to end on still images, and the classroom-level layers are not built yet.

| Component | Status |
| --- | --- |
| Person/object detection on stills (YOLOv8-nano) | Working — [`src/detect`](src/detect) (CLI, API, notebook) |
| Pose landmarks, 33 points + segmentation (MediaPipe Tasks) | Working — [`src/pose`](src/pose) (CLI, API, notebook) |
| Video ingestion and per-student tracking | Not started |
| Seat position mapping onto a classroom chart | Not started |
| Engagement indicators and teacher-rating validation | Not started |
| Fairness-aware seating optimization | Not started |
| Teacher-facing reports and engagement heat map | Not started |

Each pipeline is a package with a command line entry point, a lazily-resolved public API
and one notebook. Importing a package is free of side effects — no model load, no
inference — so the pipelines can be driven from code, from a notebook or from the shell.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the conventions.

## Quick start

Requires Python 3.10 or 3.11. No GPU needed — both pipelines run on CPU.

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

```bash
# Pose landmarks on the committed sample image — works right after cloning
python -m src.pose

# Object detection over your own images in data/images/input
python -m src.detect --limit 5
```

Run commands from the repository root so that `src` is importable. Every CLI option, and
notes for headless machines, are in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Repository layout

```
data/                  # Datasets — classroom media is gitignored, never committed
docs/                  # Technical documentation
models/                # YOLOv8 weights and the MediaPipe model bundle
src/detect/            # Detection pipeline: module, CLI, API, notebook
src/pose/              # Pose pipeline: module, CLI, API, notebook
src/notebook/          # Legacy location, kept for older references
tests/                 # Unit and structural tests
```

The two pipeline packages are deliberately symmetrical, so the next pipeline — seat
mapping, engagement scoring — has an obvious shape to follow; a test enforces it.

## Documentation

| Document | Covers |
| --- | --- |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Environment setup, running each pipeline, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Source layout, data flow, package conventions, model assets |
| [docs/DATA_AND_ETHICS.md](docs/DATA_AND_ETHICS.md) | Data policy, student privacy, consent and ethics approval |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Branching workflow, CI checks, notebook hygiene, releases |

## Data and ethics

This project processes images of students — personal data belonging to minors. Two rules
are non-negotiable:

- **No student media in version control.** `data/images/` and `data/videos/` are
  gitignored. Annotated outputs count as personal data too: faces remain identifiable.
- **Inference stays local.** YOLOv8 and MediaPipe both run on the machine. The only
  network access is a one-time model download, which uploads nothing.

Data collection requires guardian consent and institutional ethics approval; the
consent scope, retention period and deletion procedure live in the project's research
documentation, not in this repository. Read
[docs/DATA_AND_ETHICS.md](docs/DATA_AND_ETHICS.md) before adding data or publishing
results.

## Intended value

- **Teachers** — objective evidence to support seating decisions, and early warning for
  seats that appear to hurt engagement, instead of relying on subjective observation
  alone.
- **Students** — placement that better fits individual needs, and earlier support when
  they are struggling.
- **Schools** — a grounded basis for research on learning environments and classroom
  management, and a step towards smart-classroom practice.

## Contributing

Branch off `main`, then run the same gates CI runs — they need only
`requirements-tools.txt`, so no `torch` download:

```bash
ruff check src tests
ruff format --check --exclude '*.ipynb' src tests
bandit -c pyproject.toml -r src -q
pytest
```

Conventions, the workflow map and notebook output hygiene are in
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
