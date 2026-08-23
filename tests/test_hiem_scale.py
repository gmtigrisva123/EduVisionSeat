"""Tests for the body-scale estimator — the denominator every HIEM signal divides by.

Getting this wrong is the expensive failure. A denominator that drifts with head
yaw, or that one mis-detected elbow can move, does not merely add noise: it adds
noise that CORRELATES with the student's distance from the camera and with how
much they turn, which is precisely the bias HIEM exists to remove.

Pure numpy and pure geometry, so the suite runs without MediaPipe.
"""

from __future__ import annotations

import numpy as np
import pytest
from hiem_fixtures import REFERENCE_SCALE, skeleton, transform, visibility, yawed

from src.hiem.camera import CameraModel
from src.hiem.constants import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from src.hiem.scale import (
    BodyScale,
    BodyScaleEstimator,
    foreshortening_corrected_length,
    fuse_segments,
    gate_segments,
    huber_location,
    measure_segments,
    robust_dispersion,
    stature_equivalents,
    weighted_median,
    with_metric_scale,
)


def foreshortened_samples(true_length=100.0, n=300, max_yaw_deg=40.0, noise=0.02, seed=7):
    """Observations of a rigid segment seen at varying, unknown angles."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, np.radians(max_yaw_deg), n)
    return true_length * np.cos(angles) * (1.0 + rng.normal(0.0, noise, n))


class TestPercentileAggregation:
    """docs/HEIGHT_ESTIMATION.md §4.2 — the most valuable trick in the pipeline."""

    def test_percentile_aggregation_recovers_unforeshortened_length(self):
        """Rotation can only SHORTEN a projection, so the upper tail is the truth."""
        recovered = foreshortening_corrected_length(foreshortened_samples())
        assert recovered == pytest.approx(100.0, rel=0.03)

    def test_the_mean_is_biased_low_by_foreshortening(self):
        """The obvious estimator is wrong, and wrong in one direction every time.

        A bias that always shortens the denominator always inflates the ratio, so
        it never averages away over a lesson.
        """
        samples = foreshortened_samples()
        assert float(np.mean(samples)) < 0.95 * 100.0

    def test_the_maximum_chases_the_noise_instead(self):
        """Why 92.5 and not 100: the maximum estimates the noise ceiling."""
        samples = foreshortened_samples()
        percentile = foreshortening_corrected_length(samples)
        assert float(np.max(samples)) > 100.0
        assert abs(np.max(samples) - 100.0) > abs(percentile - 100.0)

    def test_a_wider_swing_is_still_recovered(self):
        """A student who turns right round to a neighbour, not merely glances."""
        recovered = foreshortening_corrected_length(foreshortened_samples(max_yaw_deg=70.0))
        assert recovered == pytest.approx(100.0, rel=0.05)

    def test_too_few_samples_report_none_rather_than_a_number(self):
        """A percentile of four frames is a number with no meaning attached."""
        assert foreshortening_corrected_length([90.0, 95.0, 99.0, 100.0]) is None

    def test_non_positive_samples_are_discarded(self):
        assert foreshortening_corrected_length([0.0, -3.0] + [100.0] * 8) == pytest.approx(100.0, rel=0.01)


class TestRobustStatistics:
    def test_huber_matches_the_weighted_mean_on_clean_data(self):
        values, weights = [10.0, 10.2, 9.8, 10.1], [1.0, 1.0, 1.0, 1.0]
        assert huber_location(values, weights) == pytest.approx(np.mean(values), abs=0.05)

    def test_huber_resists_a_single_wild_segment(self):
        """One hallucinated elbow must not move the denominator."""
        clean = [100.0, 101.0, 99.0, 100.5]
        assert huber_location(clean + [700.0]) == pytest.approx(100.0, abs=2.0)
        assert float(np.mean(clean + [700.0])) > 200.0

    def test_weighted_median_respects_the_weights(self):
        assert weighted_median([1.0, 2.0, 100.0], [0.1, 0.1, 0.8]) == 100.0

    def test_a_single_value_survives_unchanged(self):
        assert huber_location([42.0]) == 42.0

    def test_robust_dispersion_ignores_an_outlier(self):
        steady = [100.0] * 20
        assert robust_dispersion(steady) == pytest.approx(0.0)
        assert robust_dispersion(steady + [900.0]) == pytest.approx(0.0)

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError):
            huber_location([1.0, 2.0], [1.0])


class TestSegmentMeasurement:
    def test_the_fixture_is_anatomically_consistent(self):
        """Guards the guard: every segment must agree, or later tests measure the gate."""
        equivalents = stature_equivalents(measure_segments(skeleton()))
        assert len(equivalents) == 5
        for value in equivalents.values():
            assert value == pytest.approx(REFERENCE_SCALE, rel=1e-9)

    def test_the_longer_of_two_sides_is_kept(self):
        """Foreshortening only shortens, so the shorter arm is the more rotated one."""
        points = dict(skeleton())
        points[RIGHT_ELBOW] = (points[RIGHT_SHOULDER][0], points[RIGHT_SHOULDER][1] + 10.0)
        long_side = np.hypot(*np.subtract(points[LEFT_ELBOW], points[LEFT_SHOULDER]))
        assert measure_segments(points)["upper_arm"] == pytest.approx(long_side)

    def test_an_invisible_landmark_is_not_measured(self):
        points = skeleton()
        hidden = visibility(points, hidden=(LEFT_ELBOW, RIGHT_ELBOW))
        assert "upper_arm" not in measure_segments(points, hidden)

    def test_a_missing_segment_is_absent_rather_than_zero(self):
        """A zero shoulder width would sail through the fusion and divide by nothing."""
        points = {k: v for k, v in skeleton().items() if k not in (LEFT_SHOULDER, RIGHT_SHOULDER)}
        assert "shoulder_width" not in measure_segments(points)

    def test_collapsed_landmarks_do_not_produce_a_zero_length_segment(self):
        points = dict(skeleton())
        points[LEFT_SHOULDER] = points[RIGHT_SHOULDER]
        assert "shoulder_width" not in measure_segments(points)


class TestConsistencyGate:
    """MediaPipe answers occlusion by extrapolating, not by lowering visibility."""

    def test_an_anatomically_impossible_limb_is_rejected(self):
        equivalents = stature_equivalents(measure_segments(skeleton()))
        equivalents["upper_arm"] *= 7.5  # the worst case seen on data/images
        kept, rejected = gate_segments(equivalents)
        assert "upper_arm" in rejected and "upper_arm" not in kept

    def test_a_merely_foreshortened_segment_is_kept(self):
        """A 30 degree rotation is geometry, not a detection failure."""
        equivalents = stature_equivalents(measure_segments(skeleton()))
        equivalents["forearm"] *= np.cos(np.radians(30.0))
        assert "forearm" in gate_segments(equivalents)[0]

    def test_the_gate_says_why_it_rejected_something(self):
        equivalents = stature_equivalents(measure_segments(skeleton()))
        equivalents["hip_width"] *= 4.0
        assert "4.00x" in gate_segments(equivalents)[1]["hip_width"]

    def test_a_lone_segment_is_never_gated_away(self):
        """With nothing to compare against there is no evidence of a failure."""
        kept, rejected = gate_segments({"shoulder_width": 250.0})
        assert kept == {"shoulder_width": 250.0} and not rejected

    def test_the_anchor_stops_a_heavy_outlier_becoming_the_consensus(self):
        """The upper arm carries the largest fusion weight, so a Huber estimator
        anchored on the consensus would treat the correct shoulders as the outlier."""
        equivalents = {"shoulder_width": 1000.0, "upper_arm": 6000.0, "forearm": 5500.0}
        kept, _ = gate_segments(equivalents, anchor="shoulder_width")
        assert set(kept) == {"shoulder_width"}
        value, _, _, _ = fuse_segments(equivalents)
        assert value == pytest.approx(1000.0)


class TestBodyScaleValidity:
    def test_arms_behind_a_desk_still_give_a_usable_scale(self):
        """Segments that were never measured are not evidence of anything wrong."""
        scale = BodyScale(value=300.0, state="locked", n_samples=60, weight=0.226)
        assert scale.is_usable

    def test_a_pose_that_contradicts_itself_is_not_usable(self):
        """Measured segments that disagree ARE evidence, and they disqualify.

        The case this encodes was observed on data/images/input/classroom4.jpg: a
        16 px shoulder span with a 100 px upper arm, every landmark above 0.95
        visibility. The shoulders alone clear the weight floor, so without this
        rule a body scale several times too small would be published.
        """
        scale = BodyScale(value=76.0, state="instantaneous", n_samples=1, weight=0.31, rejected_weight=0.69)
        assert not scale.is_usable and scale.confidence() == 0.0

    def test_the_weight_floor_rejects_a_scale_built_from_the_head_alone(self):
        assert not BodyScale(value=300.0, state="locked", n_samples=60, weight=0.085).is_usable

    def test_confidence_ranks_the_three_states(self):
        def at(state):
            return BodyScale(value=300.0, state=state, n_samples=60, weight=1.0).confidence()

        assert at("locked") > at("provisional") > at("instantaneous") > 0.0

    def test_the_shoulder_equivalent_converts_to_the_units_the_pose_layer_uses(self):
        """src.pose.pose_landmarks reports its ratios in shoulder widths."""
        scale = BodyScale(value=1000.0, state="locked", n_samples=60, weight=1.0)
        assert scale.shoulder_equivalent == pytest.approx(230.6)

    def test_a_stable_scale_must_be_both_locked_and_settled(self):
        assert not BodyScale(value=300.0, state="provisional", n_samples=20, weight=1.0).is_stable
        assert BodyScale(value=300.0, state="locked", n_samples=60, weight=1.0, dispersion=0.02).is_stable
        assert not BodyScale(value=300.0, state="locked", n_samples=60, weight=1.0, dispersion=0.9).is_stable


class TestEstimatorStates:
    def test_a_single_frame_is_only_instantaneous(self):
        """One photograph affords no percentile, and the tier says so."""
        estimator = BodyScaleEstimator(1)
        estimator.update(skeleton())
        assert estimator.estimate().state == "instantaneous"

    def test_enough_frames_reach_provisional_then_locked(self):
        estimator = BodyScaleEstimator(1, min_samples=8, lock_after=30)
        states = []
        for _ in range(30):
            estimator.update(skeleton())
            states.append(estimator.estimate().state)
        assert states[0] == "instantaneous"
        assert states[7] == "provisional"
        assert states[-1] == "locked"

    def test_the_lock_is_announced_exactly_once(self):
        """Every derived signal steps when the denominator changes, and a
        variance-based signal reads that step as instability — penalising the
        student for the system having finished calibrating. Downstream windows
        have to be cleared, so the transition has to be observable."""
        estimator = BodyScaleEstimator(1, min_samples=4, lock_after=10)
        announcements = []
        for index in range(20):
            estimator.update(skeleton())
            if estimator.just_locked:
                announcements.append(index)
        assert announcements == [9]

    def test_a_locked_scale_is_steadier_than_the_frame_it_came_from(self):
        """The point of locking. A per-frame denominator inherits every wobble of
        the pose model; the ratio then inherits it too, and because E[x/S] is not
        x/E[S] the wobble is a bias, not merely noise."""
        rng = np.random.default_rng(3)
        estimator = BodyScaleEstimator(1, min_samples=8, lock_after=30)
        per_frame, locked = [], []
        for index in range(90):
            noisy = {i: (x + rng.normal(0, 3), y + rng.normal(0, 3)) for i, (x, y) in skeleton().items()}
            estimator.update(noisy)
            single = BodyScaleEstimator(2)
            single.update(noisy)
            per_frame.append(single.estimate().value)
            if index >= 30:
                locked.append(estimator.estimate().value)
        assert np.std(locked) < 0.2 * np.std(per_frame)

    def test_yaw_shrinks_the_per_frame_scale_but_not_the_locked_one(self):
        """The failure mode in one test: a student turning away contracts every
        horizontal segment, so a per-frame denominator shrinks exactly when the
        student moves. The percentile recovers the frontal value."""
        estimator = BodyScaleEstimator(1, min_samples=8, lock_after=30)
        frontal = BodyScaleEstimator(2)
        frontal.update(skeleton())
        truth = frontal.estimate().value

        for index in range(60):
            estimator.update(yawed(skeleton(), 40.0 * abs(np.sin(index / 9.0))))
        turned = BodyScaleEstimator(3)
        turned.update(yawed(skeleton(), 40.0))

        assert turned.estimate().value < 0.9 * truth
        assert estimator.estimate().value == pytest.approx(truth, rel=0.05)

    def test_reset_forgets_everything(self):
        estimator = BodyScaleEstimator(1, min_samples=4, lock_after=8)
        for _ in range(10):
            estimator.update(skeleton())
        assert estimator.locked
        estimator.reset()
        assert estimator.n_samples == 0 and estimator.estimate() is None

    def test_an_empty_frame_changes_nothing(self):
        estimator = BodyScaleEstimator(1)
        estimator.update({})
        assert estimator.estimate() is None

    def test_locking_before_the_minimum_sample_count_is_rejected(self):
        with pytest.raises(ValueError):
            BodyScaleEstimator(1, min_samples=30, lock_after=10)

    def test_the_scale_is_unchanged_by_where_the_student_sits_in_the_frame(self):
        near = BodyScaleEstimator(1)
        near.update(skeleton())
        far = BodyScaleEstimator(2)
        far.update(transform(skeleton(), scale=1.0, dx=600.0, dy=-120.0))
        assert far.estimate().value == pytest.approx(near.estimate().value, rel=1e-9)

    def test_the_scale_tracks_apparent_size_proportionally(self):
        """S is a pixel length; doubling the imaged body must double it exactly,
        which is what makes the ratios that divide by it invariant."""
        small = BodyScaleEstimator(1)
        small.update(skeleton())
        large = BodyScaleEstimator(2)
        large.update(transform(skeleton(), scale=2.0))
        assert large.estimate().value == pytest.approx(2.0 * small.estimate().value, rel=1e-9)


class TestMetricScale:
    """The calibrated extension — the only route to comparing students by size."""

    def test_an_uncalibrated_scale_may_not_be_compared_across_students(self):
        estimator = BodyScaleEstimator(1)
        estimator.update(skeleton())
        assert not estimator.estimate().comparable_across_students

    def test_a_camera_and_a_depth_unlock_the_comparison(self):
        estimator = BodyScaleEstimator(1)
        estimator.update(skeleton())
        camera = CameraModel.from_hfov(1920, 1080, 70.0)
        metric = with_metric_scale(estimator.estimate(), camera, depth_m=4.0, at=(960.0, 540.0))
        assert metric.comparable_across_students
        assert metric.metric_stature_m == pytest.approx(1000.0 * 4.0 / camera.fx, rel=1e-9)

    def test_guessed_intrinsics_are_recorded_as_uncalibrated(self):
        estimator = BodyScaleEstimator(1)
        estimator.update(skeleton())
        metric = with_metric_scale(estimator.estimate(), CameraModel.from_hfov(1920, 1080, 70.0), 4.0)
        assert "calibration" in metric.reasons

    def test_two_students_at_different_depths_rank_by_metric_size_not_by_pixels(self):
        """The reason the flag exists: in pixels the near student always wins.

        A shorter child at 2 m and a 15% taller child at 6 m. The taller child
        subtends 1.15/3 of the pixels, so ranking on ``value`` puts them last —
        which is why ``comparable_across_students`` stays false until a depth and
        a camera are supplied, and why the pipeline's default output is a ranking
        WITHIN a row rather than across the room.
        """
        camera = CameraModel.from_hfov(1920, 1080, 70.0)
        near, far = BodyScaleEstimator(1), BodyScaleEstimator(2)
        near.update(transform(skeleton(), scale=1.0))
        far.update(transform(skeleton(), scale=1.15 / 3.0))
        assert far.estimate().value < near.estimate().value  # backwards, in pixels

        near_m = with_metric_scale(near.estimate(), camera, depth_m=2.0)
        far_m = with_metric_scale(far.estimate(), camera, depth_m=6.0)
        assert far_m.metric_stature_m > near_m.metric_stature_m  # correct, in metres
        assert far_m.metric_stature_m / near_m.metric_stature_m == pytest.approx(1.15, rel=1e-9)


class TestWristGeometryIsNotPartOfTheScale:
    def test_a_raised_hand_does_not_change_the_body_scale(self):
        """A behaviour must never move the denominator it will be divided by,
        or the normalisation quietly cancels the very signal it is measuring."""
        resting, raised = BodyScaleEstimator(1), BodyScaleEstimator(2)
        resting.update(skeleton(raised_hand=False), visibility(skeleton()))
        raised.update(skeleton(raised_hand=True), visibility(skeleton()))
        assert raised.estimate().value == pytest.approx(resting.estimate().value, rel=0.02)

    def test_the_wrists_are_not_treated_as_a_scale_segment(self):
        points = dict(skeleton())
        points[LEFT_WRIST] = (points[LEFT_WRIST][0], points[LEFT_WRIST][1] - 500.0)
        points[RIGHT_WRIST] = (points[RIGHT_WRIST][0], points[RIGHT_WRIST][1] - 500.0)
        moved = stature_equivalents(measure_segments(points))
        original = stature_equivalents(measure_segments(skeleton()))
        assert moved["shoulder_width"] == pytest.approx(original["shoulder_width"])
        assert moved["upper_arm"] == pytest.approx(original["upper_arm"])
