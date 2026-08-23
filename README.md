# EduVisionSeat

A toolkit for detecting and analysing posture/seating in images and video, built for research and educational applications.

## Overview
- **Goal:** Provide source code, notebooks and sample models for object detection (chairs, students) and posture analysis from images/video.
- **Main directories:**
	- [data](data) — holds the input images and videos. -> kept private to protect the data of students and other participants in the project; the ethics statement is documented in the research write-up and the project docs.
	- [src/detect](src/detect/__init__.py#L1) — detection code (module).
	- [src/notebook](src/notebook/detect_notebook_advanced_version.ipynb) — demonstration notebooks (ships with the small `yolov8n.pt` model).
	- [src/pose](src/pose) — posture processing code (pose estimation).

## Requirements
- Python 3.8+ (or equivalent)
- Common libraries: `torch`, `opencv-python`, `numpy`, `yolov8`/`ultralytics` (depending on the implementation). If a `requirements.txt` is present, install with:

```bash
pip install -r requirements.txt
```

## Data layout
- Input images: place them in `data/images/input/`.
- Output results (images, video, logs): saved to `src/notebook/data/images/output/` or the corresponding directory.

## Quick start
1. Set up the environment and install the packages.
2. Add your data to [data/images/input](data/images/input).
3. Run the detection module or open a demonstration notebook:

```bash
# Open a notebook (for example with jupyter)
jupyter notebook src/pose/pose.ipynb

# Or run the detection script (depending on the repo):
python -m src.detect.run --input data/images/input --output src/notebook/data/images/output
```

Note: the script/entrypoint name may differ; look inside `src/detect` for the details.

## Sample model
- A sample model file is included: `src/notebook/yolov8n.pt` (fallback). You can swap in a larger model to improve quality.

## Feedback & development
- To extend the project: add datasets, retrain the model, or add a post-processing filter for higher accuracy.
- Open an issue or a PR if you want to contribute code or report a bug.

## License
See the `LICENSE` file for the copyright details and terms of use.

---
Possible next steps:
- Add instructions for installing a full `requirements.txt` generated from the current environment.
- Write a sample runner script `src/detect/run.py` if one does not exist yet.
