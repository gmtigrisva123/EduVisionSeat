# Contributing

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

The code imports modules as `src.detect`, `src.pose`, ... so commands are run
from the repository root. Alternatively install the project in editable mode:

```bash
pip install -e ".[dev]"
```

## Everyday commands

```bash
make test    # pytest
make lint    # ruff check
make format  # ruff format
make check   # lint + test, what to run before opening a PR
```

## Layout

| Path | Contents |
| --- | --- |
| `src/config.py` | Every tunable, as one frozen `Config` value |
| `src/detect/` | Person detection and tracking (Ultralytics YOLO) |
| `src/pose/` | Pose estimation and landmarks (MediaPipe) |
| `src/notebook/` | Demonstration notebooks |
| `configs/` | Tracker YAML files |
| `models/` | Downloaded checkpoints — contents never committed |
| `data/` | Input and output imagery — never committed, see below |
| `tests/` | Test suite; runs without weights or a network |

## Ground rules

1. **No participant data in git.** Read [docs/DATA_AND_ETHICS.md](DATA_AND_ETHICS.md)
   before your first commit. Clear notebook outputs before committing them.
2. **Configuration goes in `src/config.py`.** A new threshold becomes a field on
   the relevant dataclass, not a literal buried in a function.
3. **Tests must not need the network.** The suite runs without model weights;
   anything that downloads a checkpoint gets the `slow` marker and is skipped in
   the default run.
4. **Degrade loudly.** When the pipeline silently falls back to a worse setting
   (smaller weights, ReID disabled), it must log the fallback and record it, so a
   result is never quietly produced by a different configuration than intended.

## Notebooks

Notebooks are excluded from linting on purpose — they use top-level statements
and display side effects that the rules would flag. Keep the logic in `src/` and
let the notebook call it, so what a notebook demonstrates is what the command
line actually runs.
