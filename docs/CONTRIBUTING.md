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

These are exactly the commands the `CI` workflow runs, so running them locally means no
waiting on a round trip through GitHub:

```bash
ruff check src tests
ruff format --check --exclude '*.ipynb' src tests
bandit -c pyproject.toml -r src -q
python -m compileall -q src tests
pytest
```

They need only `requirements-tools.txt`, so a lint/test environment installs in seconds
without `torch`. `ruff format` (without `--check`) applies the formatting.

Audit dependencies the way the `Security` workflow does:

```bash
pip-audit --requirement requirements.txt --requirement requirements-tools.txt
```

## CI workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `CI` (`ci.yml`) | push/PR to `main`, `master`, `develop` | **quality**: ruff, ruff format, bandit, `compileall`, pytest on Python 3.10 and 3.11. **data-hygiene**: fails if any classroom image or video is tracked by Git |
| `Security` (`security.yml`) | push/PR to `main`, `master`; Mondays 03:00 UTC | **code-scan**: bandit report. **dependency-audit**: `pip-audit` over the declared requirements |
| `Pipeline smoke test` (`pipeline.yml`) | push to `main` touching `src/**`; Mondays 03:30 UTC; manual | Installs the full runtime stack and runs both pipelines end to end — pose on the committed sample, detection on a synthesised frame |
| `Release` (`release.yml`) | tags matching `v*.*.*`; manual | **verify** (lint, bandit, compileall, pytest) then **package**: zip + SHA-256 checksum, uploaded as an artifact and attached to the GitHub release |

Design notes worth keeping:

- Every workflow declares `permissions: contents: read`. Only the release *packaging*
  job escalates to `contents: write`, and only to create the release.
- The heavy runtime install lives in `pipeline.yml`, not in `CI`, so pull requests stay
  fast and deterministic. Run it manually from the Actions tab when you change a pipeline
  on a branch.
- `dependabot.yml` opens weekly PRs for pip requirements and action versions.

## Code style

- Docstrings and code comments are written in **English**, and so is everything under
  `docs/`.
- `ruff` and `bandit` are configured in `pyproject.toml`; the ruff rule set includes
  pydocstyle (Google convention), so public functions need docstrings. Add explicit
  configuration rather than scattering `# noqa`.
- Line length is 100. Notebooks are exempt from line length and import-position rules,
  since cells are narrative.
- Use `pathlib.Path` for paths, never string concatenation or `os.path`.
- Resolve paths from `__file__`, not from the working directory.
- Keep `__init__.py` free of side effects and expose the package API lazily — see
  [ARCHITECTURE.md](ARCHITECTURE.md#__init__py-exposes-an-api-and-stays-free-of-side-effects).
- Put execution behind `main()` with an `argparse` parser, and add a `__main__.py` when
  you create a new pipeline package.

## Adding a pipeline package

`tests/test_structure.py` requires each pipeline package under `src/` to ship an
`__init__.py`, a `__main__.py`, at least one module and at least one notebook. Follow
`src/detect` as the template, then add the new package name to `PIPELINE_PACKAGES` in
that test file.

## Dependencies

| File | Purpose |
| --- | --- |
| `requirements.txt` | Runtime dependencies for the pipelines |
| `requirements-tools.txt` | Lint, test and audit tooling — no runtime dependencies |
| `requirements-dev.txt` | `-r` both of the above; the file to install while developing |

Pin minimum versions with `>=`, matching the existing lines. When you add a new import
under `src/`, add the corresponding dependency in the same PR. Be aware that lint and
tests will not catch a missing dependency: ruff, bandit and `compileall` never import
your modules, and the test suite deliberately avoids the heavy ones. The
`Pipeline smoke test` workflow is what catches it, because it installs
`requirements.txt` from scratch and runs both pipelines.

## Notebooks

Each pipeline package ships exactly one notebook, kept next to the code it exercises.
Notebooks are committed **with their outputs** in this repository, which makes diffs
large. Before committing, strip any output containing images of students — notebook
output is personal data too, see
[DATA_AND_ETHICS.md](DATA_AND_ETHICS.md#outputs-are-personal-data-too).

```bash
jupyter nbconvert --clear-output --inplace src/detect/detect_notebook.ipynb
```

## Commits and pull requests

Use Conventional Commit prefixes as the existing history does (`feat:`, `docs:`, `fix:`,
`build:`, `ci:`). Write in the imperative mood and describe **what changed and why**.

In the pull request, state the purpose of the change, how you verified it (which commands
you ran and what they reported), and whether data handling changed. Do not attach
screenshots showing students' faces to the PR description.

## Releasing

```bash
git switch main && git pull
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

A tag matching `v*.*.*` triggers the `Release` workflow: it re-runs the checks, builds
`eduvisionseat-<tag>.zip` with a `.sha256` checksum, and publishes a GitHub release with
generated notes. Model weights are excluded from the archive — both pipelines download
what they need on first run. `docs/` must always contain at least one committed file,
since the archive includes that directory.
