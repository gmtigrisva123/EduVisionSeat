"""Tests for the landmark geometry layer.

Pure geometry only, so the suite runs without the MediaPipe bundles. The point
of most of these is the SIGN CONVENTION: docs/ATTENTION_INDEX.md §1.1 documents
how easily head-angle code ends up mirrored or off by 180°, and a mirrored yaw
is invisible in aggregate statistics while being completely wrong per student.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.pose.pose_landmarks import (
    EAR_LEFT_EYE,
    MODEL_POINTS_CANONICAL,
    POSE_LEFT_EAR,
    POSE_LEFT_EYE,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_NOSE,
    POSE_RIGHT_EAR,
    POSE_RIGHT_EYE,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
    Tier,
    angles_from_pose_proxy,
    angles_from_solvepnp,
    angles_from_transformation_matrix,
    eye_aspect_ratio,
    hand_geometry,
    mouth_aspect_ratio,
    neck_drop,
    tier_for_iod,
    torso_angle,
)


def rotation_matrix(forward, up=(0.0, -1.0, 0.0), scale=1.0) -> np.ndarray:
    """Build a 4x4 transform from the face axes, in OpenCV camera coordinates.

    ``forward`` points out of the face, ``up`` towards the top of the head, both
    with y DOWN. The columns of the rotation are the model axes, and the model is
    y-down/z-into-scene, so model z = -forward and model y = -up.
    """
    z = -np.asarray(forward, dtype=float)
    z /= np.linalg.norm(z)
    y = -np.asarray(up, dtype=float)
    y = y - np.dot(y, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x, y, z]) * scale
    return matrix


def body(nose=(100.0, 60.0), shoulders=((140.0, 120.0), (60.0, 120.0)), eyes=None, ears=None, wrists=None):
    """Assemble a pose landmark dict. Subject-left landmarks sit on the IMAGE right."""
    points = {
        POSE_NOSE: nose,
        POSE_LEFT_SHOULDER: shoulders[0],
        POSE_RIGHT_SHOULDER: shoulders[1],
    }
    if eyes is not None:
        points[POSE_LEFT_EYE], points[POSE_RIGHT_EYE] = eyes
    if ears is not None:
        points[POSE_LEFT_EAR], points[POSE_RIGHT_EAR] = ears
    if wrists is not None:
        points[POSE_LEFT_WRIST], points[POSE_RIGHT_WRIST] = wrists
    return points


class TestHeadAngleConventions:
    """yaw > 0 = image right · pitch > 0 = chin up · roll > 0 = crown to image right."""

    def test_facing_the_camera_is_all_zeros(self):
        yaw, pitch, roll = angles_from_transformation_matrix(rotation_matrix((0.0, 0.0, -1.0)))
        assert (round(yaw, 6), round(pitch, 6), round(roll, 6)) == (0.0, 0.0, 0.0)

    def test_turning_towards_the_image_right_gives_a_positive_yaw(self):
        yaw, _, _ = angles_from_transformation_matrix(rotation_matrix((0.5, 0.0, -math.sqrt(0.75))))
        assert yaw == pytest.approx(30.0, abs=1e-6)

    def test_turning_towards_the_image_left_gives_a_negative_yaw(self):
        yaw, _, _ = angles_from_transformation_matrix(rotation_matrix((-0.5, 0.0, -math.sqrt(0.75))))
        assert yaw == pytest.approx(-30.0, abs=1e-6)

    def test_chin_up_is_a_positive_pitch(self):
        # y is DOWN, so a face looking upwards has a negative y component.
        _, pitch, _ = angles_from_transformation_matrix(rotation_matrix((0.0, -0.5, -math.sqrt(0.75))))
        assert pitch == pytest.approx(30.0, abs=1e-6)

    def test_head_down_is_a_negative_pitch_inside_the_reading_band(self):
        matrix = rotation_matrix((0.0, math.sin(math.radians(40)), -math.cos(math.radians(40))))
        _, pitch, _ = angles_from_transformation_matrix(matrix)
        assert pitch == pytest.approx(-40.0, abs=1e-6)
        assert -60.0 <= pitch <= -20.0  # the Xue (2025) reading/writing band

    def test_crown_tilted_towards_the_image_right_is_a_positive_roll(self):
        up = (math.sin(math.radians(20)), -math.cos(math.radians(20)), 0.0)
        _, _, roll = angles_from_transformation_matrix(rotation_matrix((0.0, 0.0, -1.0), up=up))
        assert roll == pytest.approx(20.0, abs=1e-6)

    def test_the_mesh_scale_is_divided_out(self):
        """The matrix carries the metric mesh scale; an unnormalised basis would
        skew the projected axes rather than fail loudly."""
        forward = (0.5, 0.0, -math.sqrt(0.75))
        assert angles_from_transformation_matrix(rotation_matrix(forward, scale=7.3)) == pytest.approx(
            angles_from_transformation_matrix(rotation_matrix(forward))
        )

    def test_a_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError):
            angles_from_transformation_matrix(np.eye(3))

    def test_a_degenerate_basis_is_rejected(self):
        with pytest.raises(ValueError):
            angles_from_transformation_matrix(np.zeros((4, 4)))


class TestPoseProxy:
    def test_a_centred_nose_reads_as_no_yaw(self):
        assert angles_from_pose_proxy(body())[0] == pytest.approx(0.0)

    def test_the_nose_moving_right_of_the_shoulders_is_a_positive_yaw(self):
        assert angles_from_pose_proxy(body(nose=(120.0, 60.0)))[0] > 0

    def test_the_sign_agrees_with_the_matrix_path(self):
        """Both tiers feed the same index, so a mirrored proxy would silently
        contradict the full tier for students in the back rows."""
        assert angles_from_pose_proxy(body(nose=(80.0, 60.0)))[0] < 0

    def test_the_more_visible_ear_puts_the_head_on_the_other_side(self):
        """Turning your head to your own right exposes your LEFT ear, and moves
        your face towards the LEFT of the image."""
        points = body(ears=((110.0, 65.0), (90.0, 65.0)))
        visibility = {POSE_LEFT_EAR: 0.9, POSE_RIGHT_EAR: 0.1}
        assert angles_from_pose_proxy(points, visibility)[0] < 0

    def test_the_proxy_is_scale_invariant(self):
        """§1.2 rule 1: normalise by an in-image length that scales with the
        subject, never by the image dimensions. A student in the back row must
        read the same as the same posture in the front row."""
        near = body(nose=(120.0, 60.0), ears=((130.0, 65.0), (70.0, 65.0)))
        far = {i: (x * 0.35, y * 0.35) for i, (x, y) in near.items()}
        assert angles_from_pose_proxy(far)[0] == pytest.approx(angles_from_pose_proxy(near)[0])

    def test_without_shoulders_there_is_no_estimate(self):
        assert angles_from_pose_proxy({POSE_NOSE: (100.0, 60.0)}) is None

    def test_collapsed_shoulders_do_not_divide_by_zero(self):
        assert angles_from_pose_proxy(body(shoulders=((100.0, 120.0), (100.0, 120.0)))) is None


class TestPostureSignals:
    def test_an_upright_student_reads_180_degrees(self):
        points = body(eyes=((105.0, 50.0), (95.0, 50.0)))
        assert torso_angle(points) == pytest.approx(180.0)

    def test_slumping_forward_lowers_the_angle_towards_the_ramp(self):
        upright = body(eyes=((105.0, 50.0), (95.0, 50.0)))
        slumped = body(eyes=((145.0, 90.0), (135.0, 90.0)))
        assert torso_angle(slumped) < torso_angle(upright)

    def test_the_torso_angle_falls_back_to_the_ears_when_the_eyes_are_gone(self):
        assert torso_angle(body(ears=((110.0, 50.0), (90.0, 50.0)))) == pytest.approx(180.0)

    def test_neck_drop_is_negative_while_the_head_is_above_the_shoulders(self):
        assert neck_drop(body(eyes=((105.0, 50.0), (95.0, 50.0)))) < 0

    def test_neck_drop_is_measured_in_shoulder_widths(self):
        points = body(eyes=((105.0, 80.0), (95.0, 80.0)))  # 40 px above an 80 px shoulder line
        assert neck_drop(points) == pytest.approx(-0.5)


class TestHandGeometry:
    def test_writing_keeps_the_wrists_further_than_six_tenths_apart(self):
        points = body(wrists=((160.0, 200.0), (40.0, 200.0)))  # 120 px over an 80 px span
        assert hand_geometry(points).wrist_gap == pytest.approx(1.5)

    def test_a_phone_draws_the_wrists_together(self):
        points = body(wrists=((110.0, 160.0), (90.0, 160.0)))
        assert hand_geometry(points).wrist_gap == pytest.approx(0.25)

    def test_raised_hands_give_a_positive_rise(self):
        assert hand_geometry(body(wrists=((110.0, 80.0), (90.0, 80.0)))).wrist_rise > 0

    def test_missing_wrists_report_none_rather_than_zero(self):
        """`None` = NOT MEASURABLE. A zero gap would read as a phone in the hands."""
        geometry = hand_geometry(body())
        assert geometry.wrist_gap is None and geometry.wrist_rise is None


class TestFaceRatios:
    def test_a_wide_open_eye_scores_above_a_closed_one(self):
        open_eye = [(0.0, 0.0), (2.0, -3.0), (4.0, -3.0), (6.0, 0.0), (4.0, 3.0), (2.0, 3.0)]
        closed_eye = [(0.0, 0.0), (2.0, -0.2), (4.0, -0.2), (6.0, 0.0), (4.0, 0.2), (2.0, 0.2)]
        assert eye_aspect_ratio(open_eye) == pytest.approx(1.0)
        assert eye_aspect_ratio(closed_eye) < 0.1

    def test_a_collapsed_eye_span_is_not_measurable(self):
        """The horizontal denominator vanishes in profile — that is a missing
        measurement, not an eye that happens to score zero."""
        assert eye_aspect_ratio([(1.0, 0.0)] * 6) is None

    def test_the_ear_landmark_set_is_the_documented_six(self):
        assert len(EAR_LEFT_EYE) == 6

    def test_an_open_mouth_scores_above_a_closed_one(self):
        horizontal = ((0.0, 0.0), (10.0, 0.0))
        closed = [((3.0, -0.5), (3.0, 0.5)), ((7.0, -0.5), (7.0, 0.5))]
        open_mouth = [((3.0, -5.0), (3.0, 5.0)), ((7.0, -5.0), (7.0, 5.0))]
        assert mouth_aspect_ratio(horizontal, closed) == pytest.approx(0.1)
        assert mouth_aspect_ratio(horizontal, open_mouth) == pytest.approx(1.0)


class TestTiers:
    def test_a_large_face_earns_the_full_tier(self):
        assert tier_for_iod(55.0, has_face=True) is Tier.FULL

    def test_the_middle_band_keeps_the_angles_but_loses_the_eye_signals(self):
        assert tier_for_iod(25.0, has_face=True) is Tier.POSE_ONLY

    def test_the_back_row_falls_through_to_the_proxy(self):
        """Under a 1080p 30-seat frame the back row sits below 20 px of IOD, so
        every face-based signal fails exactly where it is needed most."""
        assert tier_for_iod(12.0, has_face=True) is Tier.PROXY

    def test_no_face_means_the_proxy_regardless_of_iod(self):
        assert tier_for_iod(None, has_face=False) is Tier.PROXY

    def test_each_tier_carries_the_documented_confidence(self):
        assert (Tier.FULL.conf, Tier.POSE_ONLY.conf, Tier.PROXY.conf) == (1.00, 0.70, 0.35)


class TestSolvePnPRoundTrip:
    """Project the canonical model under a known rotation, then recover it.

    This is the test that actually pins down §1.1: the canonical mesh is y-up /
    z-out-of-face and OpenCV is y-down / z-into-scene, and skipping that negation
    makes a front-facing student come back as R ~ Rx(180°). A round trip catches
    that, where an eyeball check of one video frame does not.
    """

    IMAGE_SIZE = (640, 480)

    def project(self, forward, up=(0.0, -1.0, 0.0)):
        cv2 = pytest.importorskip("cv2")
        width, height = self.IMAGE_SIZE
        camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=float)
        model = np.array([(x, -y, -z) for x, y, z in MODEL_POINTS_CANONICAL], dtype=float)
        rotation = rotation_matrix(forward, up=up)[:3, :3]
        rvec, _ = cv2.Rodrigues(rotation)
        points, _ = cv2.projectPoints(
            model, rvec, np.array([[0.0], [0.0], [1500.0]]), camera, np.zeros((4, 1))
        )
        return [tuple(p[0]) for p in points]

    def test_a_front_facing_student_is_not_flipped_by_180_degrees(self):
        yaw, pitch, roll = angles_from_solvepnp(self.project((0.0, 0.0, -1.0)), self.IMAGE_SIZE)
        assert (yaw, pitch, roll) == pytest.approx((0.0, 0.0, 0.0), abs=0.5)

    def test_yaw_survives_the_round_trip_with_the_same_sign_as_the_matrix_path(self):
        forward = (math.sin(math.radians(25)), 0.0, -math.cos(math.radians(25)))
        yaw, _, _ = angles_from_solvepnp(self.project(forward), self.IMAGE_SIZE)
        assert yaw == pytest.approx(25.0, abs=0.5)
        assert yaw == pytest.approx(angles_from_transformation_matrix(rotation_matrix(forward))[0], abs=0.5)

    def test_a_head_bent_over_a_notebook_comes_back_negative(self):
        forward = (0.0, math.sin(math.radians(35)), -math.cos(math.radians(35)))
        _, pitch, _ = angles_from_solvepnp(self.project(forward), self.IMAGE_SIZE)
        assert pitch == pytest.approx(-35.0, abs=0.5)

    def test_the_wrong_number_of_points_is_rejected(self):
        with pytest.raises(ValueError):
            angles_from_solvepnp([(0.0, 0.0)] * 4, self.IMAGE_SIZE)
