"""Tests for the pinhole camera model — the optional, calibrated half of HIEM.

HIEM's core needs no camera at all: a ratio of two lengths at the same depth
cancels every perspective term. The camera model exists for the one question
ratios cannot answer, which is whether one student's body is genuinely larger
than another's or merely nearer. These tests pin the geometry against the
figures tabulated in ``docs/HEIGHT_ESTIMATION.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.hiem.camera import CameraModel


@pytest.fixture
def camera():
    """1080p at a 70 degree horizontal field of view — the doc's worked example."""
    return CameraModel.from_hfov(1920, 1080, 70.0)


class TestConstruction:
    def test_the_focal_length_matches_the_documented_figure(self, camera):
        """docs/HEIGHT_ESTIMATION.md §4.1 works its pixel table at f = 1371 px."""
        assert camera.fx == pytest.approx(1371.0, abs=0.5)

    def test_the_field_of_view_round_trips(self, camera):
        assert camera.hfov_deg == pytest.approx(70.0, abs=1e-9)

    def test_exif_intrinsics_use_the_sensor_width(self):
        model = CameraModel.from_exif(4000, 3000, focal_mm=26.0, sensor_width_mm=9.6)
        assert model.fx == pytest.approx(26.0 * 4000 / 9.6)

    def test_a_non_positive_focal_length_is_rejected(self):
        with pytest.raises(ValueError):
            CameraModel(fx=0.0, fy=1.0, cx=1.0, cy=1.0, width=100, height=100)

    def test_a_malformed_distortion_vector_is_rejected(self):
        with pytest.raises(ValueError):
            CameraModel(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=10, height=10, distortion=(0.0, 0.0))

    def test_an_impossible_field_of_view_is_rejected(self):
        with pytest.raises(ValueError):
            CameraModel.from_hfov(1920, 1080, 190.0)


class TestPlausibility:
    def test_a_normal_webcam_focal_length_passes(self, camera):
        """The 0.7W <= f <= W rule of thumb of §3.2."""
        assert camera.focal_is_plausible()

    def test_a_wide_angle_lens_falls_outside_the_band(self):
        assert not CameraModel.from_hfov(1920, 1080, 120.0).focal_is_plausible()

    def test_a_telephoto_guess_also_falls_outside(self):
        assert not CameraModel.from_hfov(1920, 1080, 25.0).focal_is_plausible()

    def test_uncalibrated_intrinsics_say_so_in_the_description(self, camera):
        assert "UNCALIBRATED" in camera.describe()

    def test_a_calibrated_camera_reports_its_rms(self):
        model = CameraModel(fx=1371.0, fy=1371.0, cx=960.0, cy=540.0,
                            width=1920, height=1080, calibration_rms_px=0.31)
        assert "0.310px" in model.describe()


class TestBackprojection:
    def test_a_point_on_the_optical_axis_projects_to_the_axis(self, camera):
        assert camera.backproject(960.0, 540.0, 5.0) == pytest.approx((0.0, 0.0, 5.0))

    def test_backprojection_goes_through_the_full_inverse_of_k(self, camera):
        """Not the abbreviated S = s*Z/f, which ignores where in the frame the
        point sits — §4.3 puts the cost of that shortcut at 19%."""
        x, y, z = camera.backproject(1500.0, 300.0, 6.0)
        expected = np.linalg.inv(camera.matrix) @ np.array([1500.0, 300.0, 1.0]) * 6.0
        assert (x, y, z) == pytest.approx(tuple(expected))

    def test_a_non_positive_depth_is_rejected(self, camera):
        with pytest.raises(ValueError):
            camera.backproject(100.0, 100.0, 0.0)


class TestOffAxisGeometry:
    def test_the_radial_factor_is_one_on_the_optical_axis(self, camera):
        assert camera.radial_factor(960.0, 540.0) == pytest.approx(1.0)

    def test_the_frame_edge_carries_the_documented_penalty(self, camera):
        """At 70 degrees the horizontal edge sits 35 degrees off axis, so the
        range exceeds the depth by 1/cos(35) = 1.221."""
        assert camera.radial_factor(1920.0, 540.0) == pytest.approx(
            1.0 / math.cos(math.radians(35.0)), rel=1e-9
        )

    def test_ignoring_the_radial_factor_understates_a_segment_at_the_edge(self, camera):
        naive = camera.metric_length(100.0, 5.0)
        corrected = camera.metric_length(100.0, 5.0, at=(1920.0, 540.0))
        assert corrected > naive
        assert corrected / naive == pytest.approx(1.2208, rel=1e-3)

    def test_the_correction_is_symmetric_about_the_principal_point(self, camera):
        assert camera.radial_factor(0.0, 540.0) == pytest.approx(camera.radial_factor(1920.0, 540.0))


class TestMetricConversion:
    def test_metric_length_and_pixels_per_metre_are_inverses(self, camera):
        metres = camera.metric_length(250.0, 4.0, at=(700.0, 400.0))
        assert metres * camera.pixels_per_metre(4.0, at=(700.0, 400.0)) == pytest.approx(250.0)

    def test_a_segment_twice_as_far_away_measures_twice_as_long_for_the_same_pixels(self, camera):
        assert camera.metric_length(100.0, 8.0) == pytest.approx(2.0 * camera.metric_length(100.0, 4.0))

    def test_a_two_percent_scale_error_costs_the_documented_height_error(self, camera):
        """§3.1: a 2% scale error is 3.3 cm on a 165 cm student — level with the
        anthropometric floor, which is why 2% is the accuracy target."""
        true_height = camera.metric_length(1000.0, 5.0)
        biased = camera.metric_length(1000.0, 5.0 * 1.02)
        assert (biased - true_height) / true_height == pytest.approx(0.02, rel=1e-9)


class TestUndistortion:
    def test_undistorting_without_coefficients_is_the_identity(self, camera):
        points = [(10.0, 20.0), (1900.0, 1000.0)]
        assert camera.undistort(points) == pytest.approx(points)

    def test_barrel_distortion_pulls_the_edges_in(self):
        pytest.importorskip("cv2")
        model = CameraModel(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720,
                            distortion=(-0.28, 0.09, 0.0, 0.0, 0.0))
        (corrected,) = model.undistort([(1270.0, 360.0)])
        assert corrected[0] > 1270.0
        assert corrected[1] == pytest.approx(360.0, abs=1e-6)
