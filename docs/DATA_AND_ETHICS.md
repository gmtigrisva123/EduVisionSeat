# Data and Research Ethics

## Principle

This project processes **images of students in classrooms** — personal data belonging to
minors. None of the experimental images or video may enter version control. Only source
code, documentation and model weights are committed.

## What is committed

| Path | Status | Reason |
| --- | --- | --- |
| `data/images/input/` | **Not committed** (`.gitignore`) | Classroom images showing students |
| `data/images/output/` | **Not committed** (`.gitignore`) | Derived from the originals; individuals are still identifiable |
| `data/videos/` | **Not committed** (`.gitignore`) | Experimental video |
| `data/pose/images_man_standing.jpeg` | Committed | Sample image of a single adult, not part of the experimental data |

The exclusion rules are at the end of `.gitignore` (`data/images`, `data/videos`). To
check whether a specific file is excluded:

```bash
git check-ignore -v data/images/input/classroom1.jpg
```

## Before committing

Always review `git status` and `git diff --stat` before committing to confirm that no
image or video slipped in. If you have already staged one:

```bash
git restore --staged data/images        # unstage, keep the files on disk
```

If images have already been **committed**, removing them in a later commit is not
enough — the data remains in the Git history and is still downloadable after a clone.
Contact the repository maintainer to arrange a history rewrite; do not force-push to a
shared branch on your own.

## Outputs are personal data too

The images in `data/images/output/` are annotated but **still show students' faces**.
Treat them exactly like the originals: do not commit them, do not put them in reports or
slides without masking faces, and do not upload them to third-party services.

When you need an illustration for a paper or a talk, use a sample that is not
experimental data (for example `data/pose/images_man_standing.jpeg`), or an image with
faces masked and written consent on file.

## Consent and ethics approval

Collecting student data requires the consent of parents or guardians and must follow the
ethics approval of the research institution. The details of consent scope, retention
period and the data deletion procedure are recorded in the project's research
documentation, not in this repository.

Before widening the scope of collection (more classrooms, more cameras, continuous video
recording), re-check it against the existing approval.

## A note on the models

Both YOLOv8 and the MediaPipe Pose Landmarker run **entirely locally** — no image is
sent anywhere during inference. The only network access is the one-time download of
`pose_landmarker.task` from Google Cloud Storage, which fetches a model and uploads no
data. Preserve this property: do not add a cloud inference service to the pipeline
without revisiting the ethics approval.
