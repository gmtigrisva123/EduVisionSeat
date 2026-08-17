# Contributing

## Branching workflow

The main branch is `main`. Do not commit to `main` directly — create a branch and open a
pull request. The naming conventions currently in use are:

```
feat/<topic>-<name>        # e.g. feat/detect-notebook-vietanh
agents/<topic>             # e.g. agents/fix-pose-ipynb-output-error
```

```bash
git switch main && git pull
git switch -c feat/<topic>-<name>
```

## Checks to run before opening a PR

Run exactly what CI will run, so you do not wait on a round trip through GitHub:

```bash
ruff check src            # lint
bandit -r src -q          # source security scan
python -m compileall src  # syntax check
pytest -q                 # tests (none exist yet)
```

Audit dependencies the way the `Security` workflow does:

```bash
pip-audit -r requirements.txt
```

## CI workflows

| Workflow | Trigger | What it runs |
| --- | --- | --- |
| `CI` (`.github/workflows/ci.yml`) | push/PR to `main`, `master`, `develop` | Ruff, Bandit, `compileall`, `pytest` on Python 3.10 and 3.11 |
| `Security` (`.github/workflows/security.yml`) | push/PR to `main`, `master`; plus Mondays at 03:00 UTC | Bandit and `pip-audit` |
| `Release` (`.github/workflows/release.yml`) | tags matching `v*.*.*` | `compileall`, then packages `src`, `README.md`, `LICENSE` and `docs` into a zip and creates a GitHub release |

A note on `pytest` in CI: the workflow treats exit code `5` (no tests collected) as
success, so the test step is currently always green. Once you add real tests, failures
will turn CI red as usual.

## Code style

- Docstrings and code comments are written in **English**, and so is everything under
  `docs/`.
- Every package must have a **side-effect-free** `__init__.py`: docstring and `__all__`
  only. Never re-export modules that run a pipeline at module level — see
  [architecture.md](architecture.md#rule-__init__py-must-be-free-of-side-effects).
- Use `pathlib.Path` for paths rather than string concatenation. Do not hardcode
  absolute paths; resolve the repository root the way the existing scripts do.
- Ruff runs with its default configuration (the repository has no `pyproject.toml` or
  `ruff.toml`). If a rule needs tightening or relaxing, add explicit configuration
  instead of scattering `# noqa` comments.

## Dependencies

- `requirements.txt` — dependencies needed to **run** the pipelines.
- `requirements-dev.txt` — development tooling; it already includes
  `-r requirements.txt`, so it is the only file you need when working on code.

Pin minimum versions with `>=`, matching the existing lines. When you add a new import
under `src/`, add the corresponding dependency in the same PR. Be aware that CI will not
catch a missing dependency: Ruff, Bandit and `compileall` never actually import your
modules, so an unlisted package fails only on a fresh install. `ultralytics` was missing
from `requirements.txt` for exactly this reason.

## Notebooks

Notebooks are committed **with their outputs** in this repository, which makes diffs
large (`src/pose/pose_landmarker_notebook.ipynb` is close to 1 MB). Before committing,
strip any output containing images of students — notebook output is personal data too,
see [data-and-ethics.md](data-and-ethics.md#outputs-are-personal-data-too).

To clear all outputs from the command line:

```bash
jupyter nbconvert --clear-output --inplace src/pose/pose_landmarker_notebook.ipynb
```

## Commits and pull requests

Use Conventional Commit prefixes as the existing history does (`feat:`, `docs:`, `fix:`).
Write in the imperative mood and describe **what changed and why**.

In the pull request, state the purpose of the change, how you verified it (which commands
you ran and what they reported), and whether data handling changed. Do not attach
screenshots showing students' faces to the PR description.

## Releasing

```bash
git switch main && git pull
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

A tag matching `v*.*.*` triggers the `Release` workflow. That workflow packages the
`docs` directory, so `docs/` must always contain at least one committed file — Git does
not track empty directories, and the `zip` step fails if `docs` does not exist after
checkout.
