"""Central configuration for the EduVisionSeat pipeline.

Every tunable the detection and tracking layers read lives here, so a run can be
described by one object instead of a scatter of keyword arguments. The defaults
encode the decisions documented in :mod:`src.detect.person_detector`:

* ``end2end=False`` + ``iou=0.75`` — the NMS-free head merges students who sit
  close together, so this repo keeps classic NMS and a high IoU threshold.
* ``persist=True`` — mandatory when frames are iterated manually, otherwise the
  tracker is re-initialised on every frame.

The dataclasses are frozen: a ``Config`` is a value, not a mutable global. Build
a variant with :meth:`Config.merge` rather than assigning to a field.

Typical use::

    from src.config import Config
    from src.detect import PersonDetector

    cfg = Config()                                    # defaults
    cfg = Config.from_yaml("configs/experiment.yaml")  # from a file
    cfg = cfg.merge(detect={"conf": 0.4})              # one-off override

    detector = PersonDetector(cfg)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

#: Repository root, derived from this file so imports work from any directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: COCO class ids used by the pretrained YOLO weights.
PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67

#: Tracker config shipped with the repo; see ``configs/botsort_reid.yaml``.
DEFAULT_TRACKER_YAML = "configs/botsort_reid.yaml"


def _as_tuple(value: Any, name: str) -> Tuple[Any, ...]:
    """Normalise a scalar or any sequence into a tuple, so configs stay hashable."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of values, got a string: {value!r}")
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class DetectConfig:
    """Weights, thresholds and inference arguments for the YOLO detector."""

    #: Primary checkpoint. Ultralytics downloads it on first use.
    weights: str = "yolo11m.pt"
    #: Tried in order when ``weights`` cannot be loaded (offline, rate limit, ...).
    fallback_weights: Tuple[str, ...] = ("yolo11s.pt", "yolov8n.pt")

    #: Class ids kept as "people". Anything else is discarded.
    classes: Tuple[int, ...] = (PERSON_CLASS_ID,)
    #: Minimum confidence for a person detection.
    conf: float = 0.25
    #: NMS IoU threshold. Ignored by the model when ``end2end`` is true.
    iou: float = 0.75
    #: Inference resolution. Back rows are small, so this is deliberately large.
    imgsz: int = 1280
    #: Upper bound on boxes per frame, before :meth:`PersonDetector.select_top`.
    max_det: int = 300
    #: Test-time augmentation. Roughly halves throughput; off by default.
    augment: bool = False

    #: ``None`` lets ultralytics choose; otherwise "cpu", "mps", "0", "0,1", ...
    device: Optional[str] = None
    #: ``None`` leaves the argument out entirely (older ultralytics rejects it).
    quantize: Optional[bool] = None
    #: ``False`` keeps classic NMS so ``iou`` still applies. See the module docs.
    end2end: Optional[bool] = False

    #: Also report phones, for the on-task/off-task signal.
    detect_phone: bool = False
    phone_class_id: int = CELL_PHONE_CLASS_ID
    #: Phones are small and easily missed, so they get their own lower threshold.
    phone_conf: float = 0.35

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallback_weights", _as_tuple(self.fallback_weights, "fallback_weights"))
        object.__setattr__(self, "classes", _as_tuple(self.classes, "classes"))

        for name in ("conf", "iou", "phone_conf"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"detect.{name} must be within [0, 1], got {value}")
        if self.imgsz <= 0:
            raise ValueError(f"detect.imgsz must be positive, got {self.imgsz}")
        if self.max_det <= 0:
            raise ValueError(f"detect.max_det must be positive, got {self.max_det}")
        if not self.classes:
            raise ValueError("detect.classes must list at least one class id")


@dataclass(frozen=True)
class TrackConfig:
    """Multi-object tracking settings."""

    #: Path to a BoT-SORT/ByteTrack yaml, relative to the repository root.
    #: A missing file is not fatal: the detector falls back to the ultralytics
    #: default and logs what that costs.
    tracker_yaml: str = DEFAULT_TRACKER_YAML
    #: Must stay true while frames are fed in one at a time.
    persist: bool = True
    #: Ceiling on people processed per frame; the pose stage is linear in this.
    max_people_per_frame: int = 40

    def __post_init__(self) -> None:
        if self.max_people_per_frame <= 0:
            raise ValueError(
                f"track.max_people_per_frame must be positive, got {self.max_people_per_frame}"
            )


@dataclass(frozen=True)
class PathsConfig:
    """Input/output locations, all resolved against :data:`REPO_ROOT`."""

    #: Never committed — see ``docs/DATA_AND_ETHICS.md``.
    input_dir: str = "data/images/input"
    output_dir: str = "data/images/output"
    #: Downloaded model bundles and checkpoints.
    models_dir: str = "models"

    def resolve(self, attribute: str) -> Path:
        """Return one of the configured directories as an absolute path."""
        try:
            value = getattr(self, attribute)
        except AttributeError as exc:
            known = ", ".join(f.name for f in fields(self))
            raise AttributeError(f"Unknown path '{attribute}'. Known paths: {known}") from exc

        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def input(self) -> Path:
        return self.resolve("input_dir")

    @property
    def output(self) -> Path:
        return self.resolve("output_dir")

    @property
    def models(self) -> Path:
        return self.resolve("models_dir")


_SECTIONS = {"detect": DetectConfig, "track": TrackConfig, "paths": PathsConfig}


@dataclass(frozen=True)
class Config:
    """Top-level configuration handed to :class:`~src.detect.PersonDetector`."""

    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Build a config from nested mappings, rejecting unknown keys loudly.

        A silently ignored typo in a config file is the kind of bug that costs an
        afternoon of confused benchmarking, so every key is validated here.
        """
        unknown_sections = set(data) - set(_SECTIONS)
        if unknown_sections:
            raise ValueError(
                f"Unknown config section(s): {', '.join(sorted(unknown_sections))}. "
                f"Expected: {', '.join(sorted(_SECTIONS))}"
            )

        sections: Dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            values = data.get(name) or {}
            if not isinstance(values, Mapping):
                raise TypeError(f"Config section '{name}' must be a mapping, got {type(values).__name__}")

            allowed = {f.name for f in fields(section_cls)}
            unknown_keys = set(values) - allowed
            if unknown_keys:
                raise ValueError(
                    f"Unknown key(s) in section '{name}': {', '.join(sorted(unknown_keys))}. "
                    f"Expected: {', '.join(sorted(allowed))}"
                )
            sections[name] = section_cls(**values)

        return cls(**sections)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load a config from a YAML file. An empty file yields the defaults."""
        import yaml

        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    # ------------------------------------------------------------------ #
    def merge(self, **overrides: Mapping[str, Any]) -> "Config":
        """Return a copy with per-section overrides applied.

        ``cfg.merge(detect={"conf": 0.4})`` changes one field and leaves the rest
        of the section untouched, which is what an experiment sweep needs.
        """
        unknown = set(overrides) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"Unknown config section(s): {', '.join(sorted(unknown))}")

        sections = {
            name: replace(getattr(self, name), **dict(values))
            for name, values in overrides.items()
        }
        return replace(self, **sections)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict, suitable for logging or dumping to YAML."""
        return asdict(self)

    def describe(self) -> str:
        """Return a compact one-line summary for run logs."""
        d = self.detect
        return (
            f"weights={d.weights} imgsz={d.imgsz} conf={d.conf} iou={d.iou} "
            f"end2end={d.end2end} tracker={self.track.tracker_yaml} persist={self.track.persist}"
        )


#: Importable default, for callers that do not need their own instance.
DEFAULT_CONFIG = Config()

__all__ = [
    "CELL_PHONE_CLASS_ID",
    "Config",
    "DEFAULT_CONFIG",
    "DEFAULT_TRACKER_YAML",
    "DetectConfig",
    "PERSON_CLASS_ID",
    "PathsConfig",
    "REPO_ROOT",
    "TrackConfig",
]
