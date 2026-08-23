"""Person detection & tracking with Ultralytics YOLO.

Things to keep in mind (verified against ultralytics 8.4.121):

1. YOLO26 runs NMS-free by default (``end2end=True``), and in that mode the
   ``iou`` PARAMETER IS IGNORED entirely. In a crowded classroom the boxes
   overlap heavily and the one-to-one head easily merges or drops students who
   sit close together. That is why this repo defaults to ``end2end=False`` +
   ``iou=0.75``. A/B test both on your own footage — it is the single most
   valuable experiment to run first.

2. ``persist=True`` is MANDATORY when you iterate over frames yourself,
   otherwise the tracker is re-initialised on every frame and every track ID
   resets to 1.

3. Never reuse the same ``YOLO`` object across two different videos — the
   tracker state leaks into the second one.

4. ``result.boxes.id`` is ``None`` when there are no tracks. Always check
   ``result.boxes.is_track`` before calling ``.id.int()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config, REPO_ROOT

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A detection box with a track ID attached."""

    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel)
    conf: float
    cls: int = 0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def iou(self, other: "Detection") -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FrameDetections:
    frame_index: int
    timestamp_s: float
    people: List[Detection]
    phones: List[Detection]


class PersonDetector:
    """Wrapper around Ultralytics YOLO for the classroom problem."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = self._load_model()
        self._tracker_path = self._resolve_tracker()
        self._reid_disabled = False
        self.degradations: List[str] = []

    # ------------------------------------------------------------------ #
    def _load_model(self):
        from ultralytics import YOLO

        candidates = [self.cfg.detect.weights] + list(self.cfg.detect.fallback_weights)
        # Weights bundled in the repo, used as the last resort when offline
        local_fallback = REPO_ROOT / "src" / "notebook" / "yolov8n.pt"
        if local_fallback.exists():
            candidates.append(str(local_fallback))

        errors = []
        for name in candidates:
            try:
                model = YOLO(name)
                if name != self.cfg.detect.weights:
                    logger.warning(
                        "Could not load '%s', falling back to '%s'. "
                        "A smaller model will miss students in the back rows.",
                        self.cfg.detect.weights, name,
                    )
                logger.info("Loaded detection model: %s", name)
                return model
            except Exception as exc:  # pragma: no cover - network dependent
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
        raise RuntimeError(
            "Could not load any YOLO weights. Tried:\n  "
            + "\n  ".join(errors)
            + "\nCheck your network connection or pre-download the .pt file into models/."
        )

    def _resolve_tracker(self) -> str:
        p = Path(self.cfg.track.tracker_yaml)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            return str(p)
        logger.warning(
            "%s not found, using the ultralytics default botsort.yaml. "
            "That default config enables GMC (redundant with a fixed camera) and "
            "disables ReID (which we need, because students sit still and the "
            "motion cue is therefore useless).",
            p,
        )
        return "botsort.yaml"

    # ------------------------------------------------------------------ #
    def _predict_kwargs(self) -> dict:
        d = self.cfg.detect
        classes = list(d.classes)
        if d.detect_phone and d.phone_class_id not in classes:
            classes.append(d.phone_class_id)

        kw = dict(
            classes=classes,
            conf=min(d.conf, d.phone_conf) if d.detect_phone else d.conf,
            imgsz=d.imgsz,
            max_det=d.max_det,
            augment=d.augment,
            verbose=False,
        )
        if d.device is not None:
            kw["device"] = d.device
        if d.quantize is not None:
            kw["quantize"] = d.quantize
        if d.end2end is not None:
            kw["end2end"] = d.end2end
        # `iou` only means anything when NOT running end2end
        if d.end2end is False:
            kw["iou"] = d.iou
        return kw

    def _disable_reid(self) -> bool:
        """Generate a ReID-free tracker config when the encoder cannot be downloaded.

        This happens more often than you would expect: a blocked network, the
        GitHub API rate limit, or simply running offline. Without a fallback the
        whole pipeline dies on the very first frame.

        Warning: losing ReID costs A LOT in this setting. Students sit still, so
        the motion cue is nearly useless and appearance is the main association
        signal. Without ReID, ID switches after an occlusion increase sharply.
        """
        if self._reid_disabled:
            return False
        try:
            import yaml
        except ImportError:
            return False

        try:
            with open(self._tracker_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            data = {"tracker_type": "botsort"}

        data["with_reid"] = False
        data.pop("model", None)
        # No appearance signal left -> lean back on IoU
        if "iou_weight" in data:
            data["iou_weight"] = 0.9
            data["reid_weight"] = 0.1

        out = REPO_ROOT / "configs" / "_tracker_noreid_autogen.yaml"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        except Exception:
            return False

        self._tracker_path = str(out)
        self._reid_disabled = True
        msg = (
            "Could not download the ReID encoder -> ReID DISABLED, continuing. "
            "Students sit still, so the motion cue is very weak; without ReID the "
            "rate of track ID switches after an occlusion goes up sharply. "
            "Pre-download the .onnx file and point 'model:' in the tracker yaml at it."
        )
        logger.warning(msg)
        self.degradations.append(msg)
        return True

    def track_frame(
        self, frame: np.ndarray, frame_index: int, timestamp_s: float
    ) -> FrameDetections:
        """Run detect + track on ONE frame (for use inside a manual loop)."""
        kw = self._predict_kwargs()

        def _run(kwargs: dict):
            return self.model.track(
                frame,
                persist=self.cfg.track.persist,
                tracker=self._tracker_path,
                **kwargs,
            )

        try:
            results = _run(kw)
        except TypeError as exc:
            # Older ultralytics versions do not know `end2end`/`quantize`
            for key in ("end2end", "quantize"):
                kw.pop(key, None)
            logger.debug("track() rejected an argument (%s), retrying without it.", exc)
            results = _run(kw)
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}".lower()
            if ("reid" in text or "onnx" in text or "no_suchfile" in text) and self._disable_reid():
                results = _run(kw)
            else:
                raise

        return self._parse(results[0], frame_index, timestamp_s)

    # ------------------------------------------------------------------ #
    def _parse(self, result, frame_index: int, timestamp_s: float) -> FrameDetections:
        people: List[Detection] = []
        phones: List[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return FrameDetections(frame_index, timestamp_s, people, phones)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # `is_track` is the right guard: `.id` is None while there are no tracks yet
        has_ids = bool(getattr(boxes, "is_track", False)) and boxes.id is not None
        ids = boxes.id.cpu().numpy().astype(int) if has_ids else None

        d = self.cfg.detect
        for i in range(len(xyxy)):
            box = tuple(float(v) for v in xyxy[i])
            cls = int(clss[i])
            conf = float(confs[i])
            if cls == d.phone_class_id and d.detect_phone:
                if conf >= d.phone_conf:
                    phones.append(Detection(-1, box, conf, cls))
            elif cls in d.classes:
                if conf < d.conf:
                    continue
                tid = int(ids[i]) if ids is not None else -1
                people.append(Detection(tid, box, conf, cls))

        return FrameDetections(frame_index, timestamp_s, people, phones)

    # ------------------------------------------------------------------ #
    @staticmethod
    def select_top(dets: Sequence[Detection], limit: int) -> List[Detection]:
        """Cap how many people are processed per frame, favouring large boxes (close to the camera).

        The MediaPipe cost is linear in the number of people, so a very crowded
        classroom needs a ceiling. This truncation IS LOGGED at the pipeline
        layer — never drop people silently.
        """
        if len(dets) <= limit:
            return list(dets)
        return sorted(dets, key=lambda d: d.area, reverse=True)[:limit]


def iter_video_frames(
    video_path: str,
    stride: int = 1,
    start_second: float = 0.0,
    end_second: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, frame_bgr) from a video file."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_second > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_second * 1000.0)

    emitted = 0
    idx = int(start_second * fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = idx / fps
            if end_second is not None and ts > end_second:
                break
            if (idx % max(stride, 1)) == 0:
                yield idx, ts, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def video_metadata(video_path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration_s"] = meta["frame_count"] / meta["fps"] if meta["fps"] else 0.0
    cap.release()
    return meta
