"""Tests for the fairness audit.

The audit is the part of HIEM that could most easily be theatre — a number that
always looks good is worse than no number. So these tests check both directions:
that it detects the dependence when it is there, and that it fails to detect it
when it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hiem.fairness import (
    MIN_UNITS_FOR_INFERENCE,
    HiemAuditReport,
    accuracy,
    audit_features,
    bootstrap_ci,
    equity_audit,
    invariance_audit,
    mean_absolute_error,
    pearson,
    rankdata,
    spearman,
    stratify_by_scale,
)


def cohort(n=120, seed=1):
    """A class of students: body scale, and a behaviour independent of it."""
    rng = np.random.default_rng(seed)
    scale = rng.uniform(150.0, 650.0, n)
    behaviour = rng.uniform(0.15, 0.35, n)
    return scale, behaviour


class TestCorrelation:
    def test_ties_are_averaged(self):
        assert list(rankdata([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]

    def test_spearman_sees_through_a_monotone_curve_that_pearson_misses(self):
        x = np.linspace(0.1, 4.0, 60)
        y = np.exp(3.0 * x)
        assert spearman(x, y) == pytest.approx(1.0, abs=1e-9)
        assert pearson(x, y) < 0.9

    def test_a_constant_input_is_zero_correlation_not_a_nan(self):
        assert spearman([1.0] * 10, np.arange(10.0)) == 0.0
        assert pearson([3.0] * 10, np.arange(10.0)) == 0.0

    def test_a_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            spearman([1.0, 2.0], [1.0])


class TestInvarianceAudit:
    def test_a_raw_pixel_signal_is_found_to_track_body_scale(self):
        scale, behaviour = cohort()
        result = invariance_audit("hand_raise", behaviour * scale, behaviour, scale)
        assert result.rho_raw > 0.5

    def test_the_normalised_signal_is_found_not_to(self):
        scale, behaviour = cohort()
        result = invariance_audit("hand_raise", behaviour * scale, behaviour, scale)
        assert abs(result.rho_hiem) < 0.2
        assert result.passes

    def test_the_improvement_carries_an_interval_that_excludes_zero(self):
        scale, behaviour = cohort()
        result = invariance_audit("hand_raise", behaviour * scale, behaviour, scale, n_boot=800)
        low, high = result.reduction_ci
        assert low > 0 and high > low

    def test_a_normalisation_that_does_nothing_does_not_pass(self):
        """The audit must be able to fail, or it is decoration."""
        scale, behaviour = cohort()
        raw = behaviour * scale
        assert not invariance_audit("broken", raw, raw, scale, n_boot=400).passes

    def test_a_partly_effective_normalisation_reports_the_residual(self):
        scale, behaviour = cohort()
        half_fixed = behaviour * np.sqrt(scale)
        result = invariance_audit("partial", behaviour * scale, half_fixed, scale, n_boot=400)
        assert result.reduction > 0 and abs(result.rho_hiem) > 0.2
        assert not result.passes

    def test_missing_values_are_dropped_rather_than_imputed(self):
        scale, behaviour = cohort(n=60)
        raw = list(behaviour * scale)
        normalised = list(behaviour)
        raw[3] = raw[9] = None
        result = invariance_audit("gappy", raw, normalised, scale)
        assert result.n_units == 58

    def test_a_small_cohort_is_warned_about_rather_than_silently_trusted(self):
        scale, behaviour = cohort(n=12)
        result = invariance_audit("small", behaviour * scale, behaviour, scale, n_boot=200)
        assert any("indicative" in w for w in result.warnings)
        assert result.n_units < MIN_UNITS_FOR_INFERENCE

    def test_a_cohort_too_small_for_any_statistic_returns_empty_rather_than_raising(self):
        result = invariance_audit("tiny", [1.0, 2.0], [1.0, 2.0], [1.0, 2.0])
        assert result.rho_raw == 0.0 and np.isnan(result.reduction_ci[0])

    def test_the_reduction_is_reported_as_a_percentage_too(self):
        scale, behaviour = cohort()
        result = invariance_audit("hand_raise", behaviour * scale, behaviour, scale, n_boot=200)
        assert 50.0 < result.reduction_pct <= 100.0


class TestBootstrap:
    def test_the_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 1.0, 200)
        low, high = bootstrap_ci(lambda index: float(data[index].mean()), data.size, n_boot=500)
        assert low < 5.0 < high

    def test_a_single_unit_gives_no_interval(self):
        low, high = bootstrap_ci(lambda index: 1.0, 1)
        assert np.isnan(low) and np.isnan(high)


class TestStratify:
    def test_groups_run_from_shortest_to_tallest(self):
        assignment, names = stratify_by_scale([100.0, 500.0, 200.0, 400.0])
        assert list(assignment) == [0, 1, 0, 1]
        assert "shorter" in names[0] and "taller" in names[1]

    def test_three_groups_are_named_by_thirds(self):
        _, names = stratify_by_scale(np.arange(30.0), n_groups=3)
        assert names == ["shorter third", "middle third", "taller third"]

    def test_custom_labels_must_match_the_group_count(self):
        with pytest.raises(ValueError):
            stratify_by_scale(np.arange(10.0), n_groups=2, labels=["only one"])

    def test_too_few_units_for_the_requested_groups_is_rejected(self):
        with pytest.raises(ValueError):
            stratify_by_scale([1.0], n_groups=2)


class TestEquityAudit:
    def test_a_model_that_fails_the_shorter_students_shows_a_gap(self):
        """The audit the pipeline asks for: accuracy for taller against shorter."""
        rng = np.random.default_rng(2)
        n = 200
        scale = rng.uniform(150.0, 650.0, n)
        truth = rng.uniform(0.0, 1.0, n)
        # Error inversely proportional to apparent size — the back rows suffer.
        prediction = truth + rng.normal(0.0, 1.0, n) * (600.0 / scale) * 0.08
        result = equity_audit(truth, prediction, scale, n_boot=600)
        assert result.gap > 0 and result.significant and result.p_value < 0.05
        assert result.groups[0].value > result.groups[1].value

    def test_the_gap_statistic_alone_can_never_exonerate_a_model(self):
        """max - min is non-negative by construction, so its bootstrap interval
        excludes zero even for a provably fair model. This is why the verdict
        comes from a permutation test; the interval is a precision statement."""
        rng = np.random.default_rng(3)
        scale = rng.uniform(150.0, 650.0, 200)
        truth = rng.uniform(0.0, 1.0, 200)
        result = equity_audit(truth, truth + rng.normal(0.0, 0.05, 200), scale, n_boot=600)
        assert result.gap_ci[0] > 0.0
        assert not result.significant

    def test_a_fair_model_shows_no_gap_that_survives_its_null(self):
        rng = np.random.default_rng(3)
        n = 200
        scale = rng.uniform(150.0, 650.0, n)
        truth = rng.uniform(0.0, 1.0, n)
        prediction = truth + rng.normal(0.0, 0.05, n)
        result = equity_audit(truth, prediction, scale, n_boot=600)
        assert not result.significant and result.p_value > 0.05

    def test_the_metric_is_pluggable(self):
        rng = np.random.default_rng(4)
        scale = rng.uniform(150.0, 650.0, 120)
        truth = rng.integers(0, 2, 120).astype(float)
        result = equity_audit(
            truth, truth.copy(), scale, metric=accuracy, metric_name="accuracy",
            higher_is_better=True, n_boot=200,
        )
        assert all(group.value == 1.0 for group in result.groups)
        assert any("score, not an error" in w for w in result.warnings)

    def test_a_small_cohort_is_warned_about(self):
        rng = np.random.default_rng(5)
        scale, truth = rng.uniform(150.0, 650.0, 20), rng.uniform(0.0, 1.0, 20)
        result = equity_audit(truth, truth + 0.1, scale, n_boot=200)
        assert any("indicative" in w for w in result.warnings)

    def test_mean_absolute_error_is_what_it_says(self):
        assert mean_absolute_error(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.5)


class TestReport:
    def test_the_summary_renders_both_audits_and_the_warnings(self):
        scale, behaviour = cohort(n=40)
        invariance = audit_features(
            {"hand_raise": behaviour * scale}, {"hand_raise": behaviour}, scale, n_boot=200
        )
        equity = equity_audit(behaviour, behaviour + 0.01, scale, n_boot=200)
        report = HiemAuditReport(
            invariance=invariance.invariance, equity=(equity,), notes=("a note",)
        )
        text = report.summary()
        assert "Invariance" in text and "Equity" in text and "a note" in text
        assert "hand_raise" in text and "MAE" in text

    def test_an_all_pass_report_says_so(self):
        scale, behaviour = cohort()
        report = audit_features(
            {"hand_raise": behaviour * scale}, {"hand_raise": behaviour}, scale, n_boot=400
        )
        assert report.all_pass

    def test_an_empty_report_does_not_claim_success(self):
        assert not HiemAuditReport().all_pass

    def test_auditing_a_feature_absent_from_one_table_is_rejected(self):
        with pytest.raises(ValueError):
            audit_features({"a": [1.0, 2.0]}, {"b": [1.0, 2.0]}, [1.0, 2.0])
