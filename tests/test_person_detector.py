"""Tests for the detection layer.

These exercise the pure logic only — geometry, argument construction, capping —
so the suite runs without model weights or a network connection. Anything that
needs a real checkpoint belongs behind the ``slow`` marker.
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.detect import Detection, PersonDetector, video_metadata
from src.detect.person_detector import FrameDetections


def make_detection(x1=0.0, y1=0.0, x2=10.0, y2=20.0, track_id=1, conf=0.9) -> Detection:
    return Detection(track_id=track_id, bbox=(x1, y1, x2, y2), conf=conf)


def detector_without_model(cfg: Config) -> PersonDetector:
    """Build a PersonDetector without running __init__, which would load weights."""
    detector = PersonDetector.__new__(PersonDetector)
    detector.cfg = cfg
    return detector


class TestDetectionGeometry:
    def test_width_height_and_area(self):
        det = make_detection(0, 0, 10, 20)
        assert (det.width, det.height, det.area) == (10.0, 20.0, 200.0)

    def test_center(self):
        assert make_detection(0, 0, 10, 20).center == (5.0, 10.0)

    def test_a_degenerate_box_has_no_negative_area(self):
        """Clamped rather than negative, so `select_top` cannot be fooled by a bad box."""
        assert make_detection(10, 20, 0, 0).area == 0.0

    def test_identical_boxes_fully_overlap(self):
        assert make_detection().iou(make_detection()) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        assert make_detection(0, 0, 10, 10).iou(make_detection(100, 100, 110, 110)) == 0.0

    def test_partial_overlap(self):
        # Two 10x10 boxes offset by 5 in x: intersection 5x10=50, union 100+100-50=150.
        a = make_detection(0, 0, 10, 10)
        b = make_detection(5, 0, 15, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_iou_is_symmetric(self):
        a, b = make_detection(0, 0, 10, 10), make_detection(3, 4, 12, 15)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_empty_boxes_do_not_divide_by_zero(self):
        empty = make_detection(5, 5, 5, 5)
        assert empty.iou(empty) == 0.0


class TestSelectTop:
    def test_everything_is_kept_when_under_the_limit(self):
        dets = [make_detection(), make_detection()]
        assert PersonDetector.select_top(dets, limit=10) == dets

    def test_the_largest_boxes_win(self):
        """Large box == close to the camera, which is what we can actually analyse."""
        small = make_detection(0, 0, 2, 2, track_id=1)
        medium = make_detection(0, 0, 5, 5, track_id=2)
        large = make_detection(0, 0, 50, 50, track_id=3)

        kept = PersonDetector.select_top([small, large, medium], limit=2)
        assert [d.track_id for d in kept] == [3, 2]

    def test_the_input_sequence_is_not_mutated(self):
        dets = [make_detection(0, 0, 2, 2, track_id=1), make_detection(0, 0, 50, 50, track_id=2)]
        PersonDetector.select_top(dets, limit=1)
        assert [d.track_id for d in dets] == [1, 2]


class TestPredictKwargs:
    def test_iou_is_sent_only_when_nms_is_in_play(self):
        """`iou` is silently ignored by the end2end head, so only send it otherwise."""
        with_nms = detector_without_model(Config().merge(detect={"end2end": False}))
        assert with_nms._predict_kwargs()["iou"] == 0.75

        end2end = detector_without_model(Config().merge(detect={"end2end": True}))
        assert "iou" not in end2end._predict_kwargs()

    def test_optional_arguments_are_omitted_when_unset(self):
        """Older ultralytics versions reject arguments they do not know."""
        kwargs = detector_without_model(Config())._predict_kwargs()
        assert "device" not in kwargs
        assert "quantize" not in kwargs

    def test_device_is_forwarded_when_set(self):
        detector = detector_without_model(Config().merge(detect={"device": "cpu"}))
        assert detector._predict_kwargs()["device"] == "cpu"

    def test_the_phone_class_is_added_only_when_requested(self):
        off = detector_without_model(Config())._predict_kwargs()
        assert 67 not in off["classes"]

        on = detector_without_model(Config().merge(detect={"detect_phone": True}))._predict_kwargs()
        assert 67 in on["classes"]

    def test_the_lower_of_the_two_thresholds_is_used_when_detecting_phones(self):
        """One inference pass serves both classes, so it must run at the lower threshold."""
        cfg = Config().merge(detect={"detect_phone": True, "conf": 0.5, "phone_conf": 0.2})
        assert detector_without_model(cfg)._predict_kwargs()["conf"] == 0.2

    def test_verbose_is_off_so_frame_loops_do_not_flood_the_log(self):
        assert detector_without_model(Config())._predict_kwargs()["verbose"] is False


class TestParse:
    def test_a_frame_with_no_boxes_yields_empty_lists(self):
        detector = detector_without_model(Config())

        class EmptyResult:
            boxes = None

        parsed = detector._parse(EmptyResult(), frame_index=7, timestamp_s=0.25)
        assert isinstance(parsed, FrameDetections)
        assert parsed.frame_index == 7
        assert parsed.timestamp_s == 0.25
        assert parsed.people == [] and parsed.phones == []


class TestVideoHelpers:
    def test_a_missing_video_raises_a_readable_error(self, tmp_path):
        pytest.importorskip("cv2")
        with pytest.raises(FileNotFoundError, match="Cannot open video"):
            video_metadata(str(tmp_path / "missing.mp4"))
