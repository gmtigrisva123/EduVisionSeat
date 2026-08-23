"""Tests for the central configuration object."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    CELL_PHONE_CLASS_ID,
    DEFAULT_TRACKER_YAML,
    PERSON_CLASS_ID,
    REPO_ROOT,
    Config,
    DetectConfig,
    TrackConfig,
)


class TestDefaults:
    def test_repo_root_is_the_repository(self):
        assert (REPO_ROOT / "src" / "config.py").is_file()
        assert (REPO_ROOT / "pyproject.toml").is_file()

    def test_detection_defaults_match_the_documented_decisions(self):
        """The detector docstring promises classic NMS with a high IoU threshold."""
        cfg = Config()
        assert cfg.detect.end2end is False, "end2end must stay off, or `iou` is silently ignored"
        assert cfg.detect.iou == 0.75
        assert cfg.track.persist is True, "manual frame loops need persist=True"

    def test_person_is_the_only_default_class(self):
        assert Config().detect.classes == (PERSON_CLASS_ID,)
        assert Config().detect.phone_class_id == CELL_PHONE_CLASS_ID

    def test_the_default_tracker_file_exists_and_is_loadable(self):
        yaml = pytest.importorskip("yaml")
        path = REPO_ROOT / DEFAULT_TRACKER_YAML
        assert path.is_file(), f"{DEFAULT_TRACKER_YAML} is the default, so it must ship with the repo"

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["tracker_type"] == "botsort"
        assert data["with_reid"] is True, "students sit still; appearance is the main cue"
        assert data["gmc_method"] == "none", "the classroom camera is fixed"


class TestValidation:
    @pytest.mark.parametrize("field_name", ["conf", "iou", "phone_conf"])
    @pytest.mark.parametrize("bad_value", [-0.1, 1.5])
    def test_probabilities_must_be_within_zero_and_one(self, field_name, bad_value):
        with pytest.raises(ValueError, match=field_name):
            DetectConfig(**{field_name: bad_value})

    @pytest.mark.parametrize("kwargs", [{"imgsz": 0}, {"imgsz": -640}, {"max_det": 0}])
    def test_sizes_must_be_positive(self, kwargs):
        with pytest.raises(ValueError):
            DetectConfig(**kwargs)

    def test_at_least_one_class_is_required(self):
        with pytest.raises(ValueError, match="at least one class"):
            DetectConfig(classes=())

    def test_max_people_per_frame_must_be_positive(self):
        with pytest.raises(ValueError, match="max_people_per_frame"):
            TrackConfig(max_people_per_frame=0)

    def test_a_string_is_not_accepted_as_a_sequence(self):
        """`classes="0"` is a typo, not a one-element list; fail rather than iterate chars."""
        with pytest.raises(TypeError):
            DetectConfig(classes="0")

    def test_scalars_are_promoted_to_tuples(self):
        assert DetectConfig(classes=0).classes == (0,)
        assert DetectConfig(classes=[0, 56]).classes == (0, 56)


class TestFromDict:
    def test_nested_sections_are_applied(self):
        cfg = Config.from_dict({"detect": {"conf": 0.4}, "track": {"persist": False}})
        assert cfg.detect.conf == 0.4
        assert cfg.track.persist is False
        assert cfg.detect.iou == 0.75, "untouched fields keep their default"

    def test_an_empty_mapping_yields_the_defaults(self):
        assert Config.from_dict({}) == Config()

    def test_an_unknown_section_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown config section"):
            Config.from_dict({"detekt": {"conf": 0.4}})

    def test_an_unknown_key_is_rejected(self):
        """A silently ignored typo would cost an afternoon of confused benchmarking."""
        with pytest.raises(ValueError, match="confidence"):
            Config.from_dict({"detect": {"confidence": 0.4}})

    def test_a_non_mapping_section_is_rejected(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            Config.from_dict({"detect": [0.4]})


class TestFromYaml:
    def test_a_yaml_file_round_trips(self, tmp_path: Path):
        pytest.importorskip("yaml")
        config_file = tmp_path / "experiment.yaml"
        config_file.write_text("detect:\n  conf: 0.5\n  imgsz: 960\n", encoding="utf-8")

        cfg = Config.from_yaml(config_file)
        assert (cfg.detect.conf, cfg.detect.imgsz) == (0.5, 960)

    def test_an_empty_file_yields_the_defaults(self, tmp_path: Path):
        pytest.importorskip("yaml")
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")
        assert Config.from_yaml(config_file) == Config()

    def test_a_missing_file_raises_a_readable_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_yaml(tmp_path / "nope.yaml")


class TestMergeAndDump:
    def test_merge_overrides_one_field_and_keeps_the_rest(self):
        cfg = Config().merge(detect={"conf": 0.6})
        assert cfg.detect.conf == 0.6
        assert cfg.detect.imgsz == Config().detect.imgsz

    def test_merge_does_not_mutate_the_original(self):
        original = Config()
        original.merge(detect={"conf": 0.6})
        assert original.detect.conf == 0.25

    def test_merge_rejects_an_unknown_section(self):
        with pytest.raises(ValueError, match="Unknown config section"):
            Config().merge(pose={"conf": 0.6})

    def test_to_dict_round_trips_through_from_dict(self):
        cfg = Config().merge(detect={"conf": 0.42}, track={"persist": False})
        assert Config.from_dict(cfg.to_dict()) == cfg

    def test_describe_mentions_the_settings_that_change_results(self):
        described = Config().describe()
        for expected in ("weights=", "imgsz=", "conf=", "iou=", "end2end=", "persist="):
            assert expected in described


class TestPaths:
    def test_paths_resolve_against_the_repository_root(self):
        paths = Config().paths
        assert paths.input == REPO_ROOT / "data" / "images" / "input"
        assert paths.models == REPO_ROOT / "models"

    def test_an_absolute_path_is_left_alone(self, tmp_path: Path):
        cfg = Config().merge(paths={"input_dir": str(tmp_path)})
        assert cfg.paths.input == tmp_path

    def test_an_unknown_path_name_lists_the_known_ones(self):
        with pytest.raises(AttributeError, match="input_dir"):
            Config().paths.resolve("inptu_dir")


def test_config_is_immutable():
    """A run is described by a value; sweeps build new configs instead of mutating."""
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().detect.conf = 0.9
