# Data handling and ethics

This project processes video and images of students in classrooms. That makes the
data personal data about identifiable people, most of them minors. This document
records how the repository handles it.

> **Note for maintainers:** the sections marked **[FILL IN]** describe the actual
> research protocol and must be completed by the project leads. They are
> deliberately left blank rather than guessed at — an ethics statement that is
> invented is worse than one that is missing.

## What never enters version control

`data/` is listed in `.gitignore` in full. No recording, frame, or crop of a
participant belongs in the repository, in an issue, in a pull request, or in a
committed notebook output.

Two failure modes are easy to miss:

- **Notebook outputs.** A notebook that has been run stores the rendered images
  inside the `.ipynb` file. Clear them before committing:

  ```bash
  jupyter nbconvert --clear-output --inplace <notebook>.ipynb
  ```

- **Annotated results.** `data/images/output/` holds detections drawn over the
  original frames. Faces stay fully identifiable there; these are input data, not
  results, and are covered by the same rules.

## What may be shared

Aggregate, non-reversible measurements: seat occupancy counts, posture-class
frequencies, per-session summary statistics. Anything from which an individual
student could be re-identified may not leave the secure storage location.

## Storage and retention

- Storage location: **[FILL IN]**
- Access is limited to: **[FILL IN]**
- Retention period and deletion procedure: **[FILL IN]**

## Consent and approval

- Ethics board / IRB approval reference: **[FILL IN]**
- Consent obtained from: **[FILL IN — students, guardians, teaching staff]**
- Withdrawal procedure, including deletion of already-recorded material:
  **[FILL IN]**

## Model and measurement limitations

These matter ethically, not only technically, because the outputs describe named
students:

- Detection recall drops for the back rows and for heavily occluded students, so
  absence of a detection is **not** evidence of absence.
- Without the ReID encoder, track identities swap after occlusions (the detector
  logs this degradation explicitly — see `PersonDetector.degradations`). Any
  per-student statistic computed from a degraded run is unreliable.
- Posture classification is a proxy for attention, not a measurement of it. Do
  not present it as one.

Report these limitations alongside any result derived from the pipeline.
