"""Tests for the HIEM normaliser — the fairness guarantee, enforced.

The claim HIEM makes is narrow enough to test exactly:

    two students performing the same behaviour receive the same numbers,
    whatever their height and wherever they sit.

Body size and camera distance both act on the image as a uniform scaling, so
the claim is the statement that every feature is invariant under the similarity
group. That is an algebraic property, and algebra can be checked to machine
precision rather than argued about — which is what most of this file does.

The tests iterate over :data:`~src.hiem.normalize.FEATURE_SPECS` rather than over
a hand-written list of names, so a feature added without declaring its
invariance class fails the suite instead of quietly escaping it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hiem_fixtures import skeleton, transform, visibility

from src.hiem.constants import (
    HAND_RAISE_ENTER,
    HAND_RAISE_EXIT,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from src.hiem.normalize import (
    FEATURE_SPECS,
    HiemNormaliser,
    HiemTracker,
    Invariance,
    PoseObservation,
    normalise_sequence,
)
from src.hiem.scale import BodyScale

#: Features actually produced from a single still frame. Movement needs two.
STATIC_FEATURES = tuple(name for name in FEATURE_SPECS if not name.startswith("motion"))


def observe(points, track_id=0, **kwargs):
    return HiemNormaliser(track_id).observe(
        PoseObservation(track_id=track_id, points=points, visibility=visibility(points), **kwargs)
    )


def measured(features):
    """The features that came back with a value, as a plain dict."""
    return {k: v for k, v in features.as_dict(include_angles=False).items() if v is not None}


class TestSimilarityInvariance:
    """x -> s.R.x + t leaves the answer alone. This is the whole guarantee."""

    def test_two_students_of_different_height_doing_the_same_thing_score_identically(self):
        """The fairness claim, stated as a test.

        The taller student is 40% larger and sits elsewhere in the frame. Nothing
        about the behaviour differs, so nothing about the scores may.
        """
        short = observe(skeleton())
        tall = observe(transform(skeleton(), scale=1.4, dx=310.0, dy=-95.0))
        assert set(measured(short)) == set(measured(tall))
        for name, value in measured(short).items():
            assert measured(tall)[name] == pytest.approx(value, abs=1e-9), name

    def test_the_same_student_in_the_back_row_scores_the_same_as_in_the_front(self):
        """Depth acts on the image as a uniform scaling, so it is the same test
        with a different story attached — and it is the one that decides whether
        seating position leaks into an engagement score."""
        front = observe(skeleton())
        back = observe(transform(skeleton(), scale=0.28, dx=740.0, dy=110.0))
        for name, value in measured(front).items():
            assert measured(back)[name] == pytest.approx(value, abs=1e-9), name

    @pytest.mark.parametrize("factor", [0.25, 0.5, 1.0, 2.0, 4.0])
    def test_every_feature_survives_a_uniform_rescaling(self, factor):
        reference = measured(observe(skeleton()))
        rescaled = measured(observe(transform(skeleton(), scale=factor)))
        assert set(rescaled) == set(reference)
        for name, value in reference.items():
            assert rescaled[name] == pytest.approx(value, abs=1e-9), f"{name} at {factor}x"

    def test_translation_changes_nothing_at_all(self):
        reference = measured(observe(skeleton()))
        moved = measured(observe(transform(skeleton(), dx=-260.0, dy=480.0)))
        for name, value in reference.items():
            assert moved[name] == pytest.approx(value, abs=1e-12), name

    def test_similarity_features_survive_an_image_rotation(self):
        reference = measured(observe(skeleton()))
        rotated = measured(observe(transform(skeleton(), rotation_deg=25.0)))
        for name, value in reference.items():
            if FEATURE_SPECS[name].invariance is Invariance.SIMILARITY:
                assert rotated[name] == pytest.approx(value, abs=1e-9), name

    def test_gravity_features_do_not_claim_to_survive_a_rotation(self):
        """'Above the shoulder line' is not a rotation-invariant statement, and a
        feature that survived a rotation while declaring itself gravity-referenced
        would be mislabelled — which would make the registry decorative."""
        reference = measured(observe(skeleton(raised_hand=True)))
        rotated = measured(observe(transform(skeleton(raised_hand=True), rotation_deg=35.0)))
        moved = [
            name
            for name, value in reference.items()
            if FEATURE_SPECS[name].invariance is Invariance.GRAVITY
            and abs(rotated[name] - value) > 1e-6
        ]
        assert moved, "no gravity feature responded to a 35 degree rotation"

    def test_the_scale_itself_is_proportional_rather_than_invariant(self):
        """S must track apparent size — it is the thing being divided out."""
        small = observe(skeleton()).scale.value
        large = observe(transform(skeleton(), scale=3.0)).scale.value
        assert large == pytest.approx(3.0 * small, rel=1e-9)


class TestFeatureRegistry:
    def test_every_reported_feature_declares_an_invariance_class(self):
        features = observe(skeleton(raised_hand=True))
        for name in features.as_dict(include_angles=False):
            assert name in FEATURE_SPECS, f"{name} is reported but undeclared"

    def test_every_declared_feature_names_a_unit(self):
        for name, spec in FEATURE_SPECS.items():
            assert spec.unit, name
            assert spec.description.endswith("."), f"{name}: description should be a sentence"

    def test_only_the_angle_bypasses_the_scale(self):
        bypassing = {name for name, spec in FEATURE_SPECS.items() if not spec.divided_by_scale}
        assert bypassing == {"torso_angle"}


class TestAnglesBypassHiem:
    """HIEM normalises distances. Angles are already dimensionless."""

    def test_head_angles_pass_through_untouched(self):
        class Angles:
            yaw, pitch, roll = -12.5, -33.0, 4.25

        angles = Angles()
        features = HiemNormaliser(0).observe(
            PoseObservation(track_id=0, points=skeleton(), angles=angles)
        )
        assert features.angles is angles
        assert (features.angles.yaw, features.angles.pitch) == (-12.5, -33.0)

    def test_the_torso_angle_is_unchanged_by_body_size(self):
        assert observe(transform(skeleton(), scale=2.6)).torso_angle == pytest.approx(
            observe(skeleton()).torso_angle, abs=1e-9
        )

    def test_the_torso_angle_stays_in_degrees(self):
        """Dividing an angle by a length would be a units error that no assertion
        on invariance alone would catch, because an angle is invariant either way."""
        assert 0.0 <= observe(skeleton()).torso_angle <= 180.0


class TestMissingIsNotZero:
    def test_a_hidden_wrist_reports_none_rather_than_zero(self):
        """A wrist_gap of zero reads as a phone held in both hands (§3.B)."""
        points = skeleton()
        hidden = visibility(points, hidden=(LEFT_WRIST, RIGHT_WRIST))
        features = HiemNormaliser(0).observe(
            PoseObservation(track_id=0, points=points, visibility=hidden)
        )
        assert features.wrist_gap is None and features.hand_raise is None
        assert features.neck_drop is not None

    def test_the_reason_names_what_was_missing(self):
        points = skeleton()
        hidden = visibility(points, hidden=(LEFT_WRIST, RIGHT_WRIST))
        features = HiemNormaliser(0).observe(
            PoseObservation(track_id=0, points=points, visibility=hidden)
        )
        assert "wrists" in features.reasons

    def test_a_body_with_no_measurable_scale_yields_no_features(self):
        features = HiemNormaliser(0).observe(
            PoseObservation(track_id=0, points={LEFT_SHOULDER: (10.0, 10.0)})
        )
        assert not features.is_valid and features.confidence == 0.0
        assert measured(features) == {}
        assert "scale" in features.reasons

    def test_an_unusable_scale_is_reported_rather_than_used(self):
        points = dict(skeleton())
        points[LEFT_SHOULDER] = (points[LEFT_SHOULDER][0] * 0.1, points[LEFT_SHOULDER][1])
        features = HiemNormaliser(0).observe(
            PoseObservation(track_id=0, points=points, visibility=visibility(points))
        )
        if not features.is_valid:
            assert "scale" in features.reasons


class TestHandRaiseHysteresis:
    """docs/ATTENTION_INDEX.md §7: never threshold a raw index directly."""

    def _hold(self, normaliser, value, n):
        states = []
        for index in range(n):
            points = self._at_height(value)
            states.append(
                normaliser.observe(
                    PoseObservation(track_id=0, points=points, visibility=visibility(points),
                                    frame_index=index, timestamp_s=index)
                ).hand_raised
            )
        return states

    @staticmethod
    def _at_height(raise_ratio):
        """A skeleton whose left wrist sits ``raise_ratio`` body scales up."""
        points = dict(skeleton())
        shoulder_y = points[LEFT_SHOULDER][1]
        points[LEFT_WRIST] = (points[LEFT_WRIST][0], shoulder_y - raise_ratio * 1000.0)
        return points

    def test_a_clear_raise_is_reported(self):
        assert self._hold(HiemNormaliser(0), 0.30, 6)[-1] is True

    def test_a_resting_hand_is_not(self):
        assert self._hold(HiemNormaliser(0), -0.20, 6)[-1] is False

    def test_entering_needs_a_dwell_so_a_stretch_does_not_count(self):
        states = self._hold(HiemNormaliser(0), 0.30, 6)
        assert states[0] is False and states[1] is False
        assert states[2] is True

    def test_a_hand_hovering_between_the_thresholds_does_not_flicker(self):
        """The gap is what suppresses a hand resting on the boundary from being
        reported and withdrawn several times a minute."""
        between = (HAND_RAISE_ENTER + HAND_RAISE_EXIT) / 2
        assert not any(self._hold(HiemNormaliser(0), between, 10))

    def test_a_raised_hand_is_not_dropped_by_a_dip_between_the_thresholds(self):
        normaliser = HiemNormaliser(0)
        self._hold(normaliser, 0.30, 5)
        assert all(self._hold(normaliser, (HAND_RAISE_ENTER + HAND_RAISE_EXIT) / 2, 6))

    def test_a_tall_and_a_short_student_trigger_on_the_same_gesture(self):
        """The fairness point of stating the threshold in body scales: an
        absolute pixel threshold is met by a tall student's smaller gesture."""
        short_points = self._at_height(0.30)
        tall_points = transform(short_points, scale=1.5)
        short_states = [
            HiemNormaliser(0).observe(PoseObservation(0, short_points, visibility(short_points)))
            for _ in range(1)
        ]
        tall_states = [
            HiemNormaliser(1).observe(PoseObservation(1, tall_points, visibility(tall_points)))
            for _ in range(1)
        ]
        assert short_states[0].hand_raise == pytest.approx(tall_states[0].hand_raise, abs=1e-9)


class TestMotion:
    def _sequence(self, points_a, points_b, dt=1 / 30.0, track_id=0):
        normaliser = HiemNormaliser(track_id)
        normaliser.observe(PoseObservation(track_id, points_a, visibility(points_a), 0, 0.0))
        return normaliser.observe(PoseObservation(track_id, points_b, visibility(points_b), 1, dt))

    def test_the_first_frame_has_no_movement_to_report(self):
        assert observe(skeleton()).motion is None

    def test_movement_is_measured_in_body_scales_per_second(self):
        moved = {i: (x + 10.0, y) for i, (x, y) in skeleton().items()}
        features = self._sequence(skeleton(), moved)
        assert features.motion == pytest.approx(10.0 / (features.scale.value / 30.0), rel=1e-9)

    def test_the_same_gesture_reads_the_same_at_any_body_size(self):
        """A pixel displacement scales with the body, so unnormalised movement
        reports a taller student as more restless for identical behaviour."""
        def gesture(factor):
            base = transform(skeleton(), scale=factor)
            moved = dict(base)
            moved[LEFT_WRIST] = (base[LEFT_WRIST][0] + 40.0 * factor, base[LEFT_WRIST][1])
            return self._sequence(base, moved)

        assert gesture(1.7).motion == pytest.approx(gesture(1.0).motion, rel=1e-9)

    def test_articulated_movement_ignores_a_pure_translation(self):
        """A student shifting bodily in the seat is not fidgeting, and camera
        shake is not either — both land entirely in the common translation."""
        shifted = {i: (x + 25.0, y - 6.0) for i, (x, y) in skeleton().items()}
        features = self._sequence(skeleton(), shifted)
        assert features.motion > 0
        assert features.motion_articulated == pytest.approx(0.0, abs=1e-9)

    def test_a_limb_moving_alone_registers_as_articulated(self):
        moved = dict(skeleton())
        moved[LEFT_WRIST] = (moved[LEFT_WRIST][0] + 55.0, moved[LEFT_WRIST][1] - 30.0)
        assert self._sequence(skeleton(), moved).motion_articulated > 0

    def test_a_track_gap_reports_none_rather_than_a_teleport(self):
        features = self._sequence(skeleton(), transform(skeleton(), dx=400.0), dt=9.0)
        assert features.motion is None and "motion" in features.reasons

    def test_a_repeated_timestamp_does_not_divide_by_zero(self):
        assert self._sequence(skeleton(), skeleton(), dt=0.0).motion is None


class TestTracker:
    def test_each_student_gets_their_own_scale(self):
        tracker = HiemTracker()
        tracker.observe(PoseObservation(1, skeleton()))
        tracker.observe(PoseObservation(2, transform(skeleton(), scale=2.0)))
        scales = tracker.scales()
        assert scales[2].value == pytest.approx(2.0 * scales[1].value, rel=1e-9)

    def test_retiring_a_track_forgets_its_body(self):
        """Every tracker recycles ids. A recycled id inheriting the previous
        student's proportions would normalise one child by another child's body."""
        tracker = HiemTracker()
        tracker.observe(PoseObservation(1, skeleton()))
        tracker.retire(1)
        assert tracker.track_ids == []

    def test_a_frame_of_several_students_is_normalised_together(self):
        tracker = HiemTracker()
        observations = [
            PoseObservation(index, transform(skeleton(), scale=0.5 + 0.4 * index))
            for index in range(4)
        ]
        results = tracker.observe_frame(observations)
        assert len(results) == 4
        raises = [f.hand_raise for f in results]
        assert all(value == pytest.approx(raises[0], abs=1e-9) for value in raises)


class TestBatch:
    def _stream(self, n=60, seed=5):
        rng = np.random.default_rng(seed)
        frames = []
        for index in range(n):
            yaw = math.radians(35.0) * math.sin(index / 8.0)
            points = skeleton()
            centre = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2
            turned = {
                i: (centre + (x - centre) * math.cos(yaw) + rng.normal(0, 2), y + rng.normal(0, 2))
                for i, (x, y) in points.items()
            }
            frames.append(PoseObservation(0, turned, visibility(turned), index, index / 30.0))
        return frames

    def test_two_passes_divide_every_frame_by_the_same_scale(self):
        """Online, the first frames can only use an instantaneous scale, so early
        features are systematically noisier than late ones — an artefact a model
        trained on the output will happily learn."""
        results = normalise_sequence(self._stream())
        values = {round(f.scale.value, 9) for f in results}
        assert len(values) == 1

    def test_one_pass_reproduces_what_a_live_run_would_have_produced(self):
        results = normalise_sequence(self._stream(), two_pass=False)
        assert results[0].scale.state == "instantaneous"
        assert results[-1].scale.state == "locked"

    def test_the_two_pass_result_is_steadier_than_the_live_one(self):
        stream = self._stream()
        live = [f.hand_raise for f in normalise_sequence(stream, two_pass=False) if f.hand_raise]
        batch = [f.hand_raise for f in normalise_sequence(stream) if f.hand_raise]
        assert np.std(batch) < np.std(live)


class TestPersonFeaturesRetrofit:
    """HIEM upgrades src.pose.pose_landmarks without editing a line of it."""

    def test_shoulder_width_ratios_are_restated_on_the_body_scale(self):
        from src.pose.pose_landmarks import HandGeometry, PersonFeatures

        person = PersonFeatures(
            track_id=4,
            frame_index=0,
            timestamp_s=0.0,
            shoulder_width=230.6,          # exactly one body scale of 1000 px
            neck_drop=-0.5,                # in shoulder widths
            hands=HandGeometry(wrist_gap=1.2, wrist_rise=0.4, left_visibility=0.9, right_visibility=0.9),
        )
        features = HiemNormaliser(4).observe_person_features(person)
        ratio = 230.6 / features.scale.value
        assert features.neck_drop == pytest.approx(-0.5 * ratio, rel=1e-9)
        assert features.wrist_gap == pytest.approx(1.2 * ratio, rel=1e-9)

    def test_a_person_without_shoulders_is_reported_not_guessed(self):
        from src.pose.pose_landmarks import PersonFeatures

        features = HiemNormaliser(4).observe_person_features(
            PersonFeatures(track_id=4, frame_index=0, timestamp_s=0.0)
        )
        assert features.scale is None and "scale" in features.reasons


class TestScaleLockEvent:
    def test_the_lock_is_surfaced_on_the_features(self):
        """Downstream variance windows must be cleared when the denominator
        steps, or the student is marked unstable for the system having finished
        calibrating."""
        normaliser = HiemNormaliser(0)
        events = []
        for index in range(45):
            points = skeleton()
            features = normaliser.observe(
                PoseObservation(0, points, visibility(points), index, index / 30.0)
            )
            if features.scale_lock_event:
                events.append(index)
        assert len(events) == 1

    def test_a_fixed_scale_never_announces_a_lock(self):
        fixed = BodyScale(value=1000.0, state="locked", n_samples=99, weight=1.0)
        normaliser = HiemNormaliser(0, fixed_scale=fixed)
        assert not normaliser.observe(PoseObservation(0, skeleton())).scale_lock_event
