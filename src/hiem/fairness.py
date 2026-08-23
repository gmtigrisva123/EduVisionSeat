"""Fairness audit for HIEM — does the normalisation actually remove the confound?

A normalisation scheme that is *argued* to be fair and never *measured* is worth
very little. This module measures it, two ways, and both are reported because
they answer different questions.

**1. The invariance audit — no labels needed.**
If HIEM works, a normalised feature carries no information about how big the
student appears. So correlate each feature against the body scale, before and
after normalisation, and watch the correlation collapse. Spearman rather than
Pearson: the relationship between a pixel measurement and body scale is
monotone but not linear, and one student sitting unusually close to the camera
would otherwise dominate the statistic.

This audit needs no engagement labels at all, which makes it runnable on the
first day of a deployment, on data nobody has annotated, and repeatedly
thereafter as a regression test.

**2. The equity audit — labels needed.**
Split students into groups by body scale and compare the model's accuracy
between them. This is the audit the pipeline description asks for: *"comparing
prediction accuracy between taller and shorter students to verify whether HIEM
meaningfully narrows that gap."* It is the stronger claim and the more expensive
one, because it needs ground truth.

**One honesty note that belongs in any write-up of these numbers.** A residual
correlation between a HIEM feature and body scale is not automatically a HIEM
failure. Taller students may genuinely behave differently — older children in a
mixed-age room, for one. The audit measures statistical dependence, and
dependence has more than one possible cause. What *is* unambiguous is the
synthetic control: scale one student's skeleton up and down while holding the
behaviour fixed, and the correlation must be exactly zero. The test suite runs
that control; this module runs the field version, whose residual has to be
interpreted rather than merely read off.

Confidence intervals come from a bootstrap that resamples **students**, not
frames. Frames within a student are strongly autocorrelated, so resampling
frames would treat 10 000 correlated observations as 10 000 independent ones and
report an interval several times too narrow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Below this many independent units, a correlation and its interval are
#: decorative. The audit still computes them and attaches a warning.
MIN_UNITS_FOR_INFERENCE = 30

#: Residual |rho| at or below which a normalised feature is treated as having
#: shed its dependence on body scale. A convention, not a law of nature: it is
#: the conventional "negligible" end of the small-effect band, and it is stated
#: here so that a reader can disagree with it explicitly.
RESIDUAL_RHO_TOLERANCE = 0.20


# --------------------------------------------------------------------------- #
# Statistics, numpy only
# --------------------------------------------------------------------------- #


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Ranks, with ties averaged. The tie handling is what makes it Spearman."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(1, array.size + 1, dtype=float)

    ordered = array[order]
    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation, returning 0.0 for a constant input rather than NaN."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size < 2:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator < 1e-12 else float(a @ b / denominator)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson on the ranks."""
    return pearson(rankdata(x), rankdata(y))


def _paired_complete(
    x: Sequence[Optional[float]], y: Sequence[Optional[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is missing. Missing means missing, not zero."""
    a = np.asarray([np.nan if v is None else v for v in x], dtype=float)
    b = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_units: int,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for a statistic of a resampled index set.

    ``statistic`` receives an array of indices into the units and returns a
    scalar. Passing indices rather than data lets a caller compute two statistics
    on the SAME resample, which is what a paired comparison requires.
    """
    if n_units < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = statistic(rng.integers(0, n_units, n_units))
    finite = draws[np.isfinite(draws)]
    if finite.size < n_boot // 10:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(finite, 100 * alpha / 2)),
        float(np.percentile(finite, 100 * (1 - alpha / 2))),
    )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InvarianceResult:
    """One feature's dependence on body scale, before and after HIEM."""

    feature: str
    rho_raw: float
    rho_hiem: float
    n_units: int
    #: ``|rho_raw| - |rho_hiem|``. Positive means HIEM removed dependence.
    reduction: float
    #: Bootstrap interval for :attr:`reduction`, over resampled students.
    reduction_ci: Tuple[float, float]
    warnings: Tuple[str, ...] = ()

    @property
    def reduction_pct(self) -> float:
        return 100.0 * self.reduction / abs(self.rho_raw) if abs(self.rho_raw) > 1e-9 else 0.0

    @property
    def passes(self) -> bool:
        """Residual dependence is negligible AND the improvement excludes zero.

        Both halves matter. A feature that was never scale-dependent passes the
        first test trivially and tells you nothing about HIEM; one that improved
        a great deal but is still strongly scale-dependent has not been fixed.
        """
        low = self.reduction_ci[0]
        improved = np.isfinite(low) and low > 0.0
        return abs(self.rho_hiem) <= RESIDUAL_RHO_TOLERANCE and improved


@dataclass(frozen=True)
class GroupMetric:
    name: str
    n: int
    value: float


#: Significance level for the permutation test behind :attr:`EquityResult.significant`.
EQUITY_ALPHA = 0.05


@dataclass(frozen=True)
class EquityResult:
    """Model accuracy compared between body-scale groups."""

    metric: str
    groups: Tuple[GroupMetric, ...]
    #: Worst group minus best group, in the metric's own units. For an error
    #: metric this is how much worse the disadvantaged group has it.
    gap: float
    #: Bootstrap interval for :attr:`gap`. A statement about PRECISION only —
    #: see :attr:`significant` for why it is not a significance test.
    gap_ci: Tuple[float, float]
    #: Permutation p-value: the share of random regroupings whose gap is at
    #: least as large as the observed one.
    p_value: float
    n_units: int
    #: What was audited, for reports that compare several models side by side.
    label: str = ""
    warnings: Tuple[str, ...] = ()

    @property
    def significant(self) -> bool:
        """Whether a gap this large would be surprising if the groups did not matter.

        **The obvious test is wrong here, and quietly so.** ``gap`` is
        ``max - min`` across groups, which is non-negative by construction, so
        its bootstrap distribution never straddles zero and a percentile interval
        excludes zero *always* — including for a model that is provably fair. An
        audit built that way reports discrimination every time it is run and is
        therefore worthless in both directions: it cannot exonerate, and its
        accusations carry no information.

        The gap has to be compared against its own null distribution instead.
        Shuffling the group labels while keeping the group sizes generates
        exactly that, and the p-value is the share of shuffles reaching the
        observed gap. It is valid for any number of groups and assumes nothing
        about the metric's distribution.
        """
        return self.p_value < EQUITY_ALPHA


@dataclass(frozen=True)
class HiemAuditReport:
    """Everything the audit found, with a renderer for the run log."""

    invariance: Tuple[InvarianceResult, ...] = ()
    equity: Tuple[EquityResult, ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def all_pass(self) -> bool:
        return bool(self.invariance) and all(result.passes for result in self.invariance)

    def summary(self) -> str:
        lines = ["HIEM fairness audit", "=" * 78]

        if self.invariance:
            lines += [
                "",
                "Invariance — Spearman rho against body scale (no labels required)",
                "",
                f"  {'feature':<22}{'raw':>8}{'HIEM':>8}{'drop':>9}{'95% CI on drop':>22}{'':>4}",
                f"  {'-' * 22}{'-' * 8}{'-' * 8}{'-' * 9}{'-' * 22}{'-' * 4}",
            ]
            for result in sorted(self.invariance, key=lambda r: -abs(r.rho_raw)):
                low, high = result.reduction_ci
                interval = f"[{low:+.3f}, {high:+.3f}]" if np.isfinite(low) else "[insufficient data]"
                mark = "PASS" if result.passes else "----"
                lines.append(
                    f"  {result.feature:<22}{result.rho_raw:>+8.3f}{result.rho_hiem:>+8.3f}"
                    f"{result.reduction:>+9.3f}{interval:>22}  {mark}"
                )
            worst = max(self.invariance, key=lambda r: abs(r.rho_hiem))
            counts = sorted({r.n_units for r in self.invariance})
            units = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
            lines += [
                "",
                f"  units (students) = {units}, varying by feature because a signal is only"
                " counted where it was measurable",
                f"  residual tolerance |rho| <= {RESIDUAL_RHO_TOLERANCE:.2f}"
                f" · largest residual = {worst.feature} at {worst.rho_hiem:+.3f}",
            ]

        for index, result in enumerate(self.equity, start=1):
            heading = f"{result.metric} by body-scale group"
            if result.label:
                heading += f" — {result.label}"
            counter = "" if len(self.equity) == 1 else f" [{index}/{len(self.equity)}]"
            lines += ["", f"Equity{counter} — {heading}", ""]
            for group in result.groups:
                lines.append(f"  {group.name:<22}n={group.n:<6}{result.metric} = {group.value:.4f}")
            low, high = result.gap_ci
            interval = f"[{low:+.4f}, {high:+.4f}]" if np.isfinite(low) else "[insufficient data]"
            verdict = "SIGNIFICANT" if result.significant else "consistent with chance"
            lines.append(
                f"  {'gap (worst - best)':<22}{result.gap:+.4f}  bootstrap 95% CI {interval}"
            )
            lines.append(
                f"  {'permutation test':<22}p = {result.p_value:.4f}  {verdict}"
                f"  ({result.n_units} units, alpha = {EQUITY_ALPHA})"
            )

        warnings = sorted(
            {w for r in self.invariance for w in r.warnings}
            | {w for r in self.equity for w in r.warnings}
        )
        if warnings:
            lines += ["", "Warnings:"] + [f"  ! {w}" for w in warnings]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The audits
# --------------------------------------------------------------------------- #


def invariance_audit(
    feature: str,
    raw: Sequence[Optional[float]],
    normalised: Sequence[Optional[float]],
    scale: Sequence[Optional[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> InvarianceResult:
    """Correlate one feature against body scale, before and after normalisation.

    ``raw`` is the measurement in pixels, ``normalised`` the same measurement in
    body scales, ``scale`` the body scale itself. All three are indexed by unit —
    one entry per student, not per frame.
    """
    if not (len(raw) == len(normalised) == len(scale)):
        raise ValueError(f"lengths differ: raw={len(raw)} normalised={len(normalised)} scale={len(scale)}")

    raw_x, raw_s = _paired_complete(raw, scale)
    hiem_x, hiem_s = _paired_complete(normalised, scale)
    n_units = min(raw_x.size, hiem_x.size)

    warnings: List[str] = []
    if n_units < MIN_UNITS_FOR_INFERENCE:
        warnings.append(
            f"only {n_units} units: correlations and intervals are indicative, not inferential"
        )
    if n_units < 3:
        empty = (float("nan"), float("nan"))
        return InvarianceResult(feature, 0.0, 0.0, n_units, 0.0, empty, tuple(warnings))

    rho_raw = spearman(raw_x, raw_s)
    rho_hiem = spearman(hiem_x, hiem_s)

    # Paired: both correlations are recomputed on the SAME resampled students,
    # so the interval is for the difference rather than for two separate
    # statistics that happen to be printed next to each other.
    paired_raw, paired_scale = _paired_complete(
        [None if (r is None or h is None) else r for r, h in zip(raw, normalised)], scale
    )
    paired_hiem, _ = _paired_complete(
        [None if (r is None or h is None) else h for r, h in zip(raw, normalised)], scale
    )

    def reduction_of(index: np.ndarray) -> float:
        sampled_scale = paired_scale[index]
        if np.ptp(sampled_scale) < 1e-12:
            return float("nan")
        return abs(spearman(paired_raw[index], sampled_scale)) - abs(
            spearman(paired_hiem[index], sampled_scale)
        )

    ci = bootstrap_ci(reduction_of, paired_raw.size, n_boot=n_boot, seed=seed)
    return InvarianceResult(
        feature=feature,
        rho_raw=rho_raw,
        rho_hiem=rho_hiem,
        n_units=n_units,
        reduction=abs(rho_raw) - abs(rho_hiem),
        reduction_ci=ci,
        warnings=tuple(warnings),
    )


def stratify_by_scale(
    scale: Sequence[float], n_groups: int = 2, labels: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Split students into equal-count groups by body scale, smallest first.

    Two groups is the default because it is the comparison the pipeline asks for
    — shorter against taller — and because splitting a class of thirty into
    tertiles leaves ten students per cell, which no interval will survive.
    """
    array = np.asarray(scale, dtype=float)
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    if array.size < n_groups:
        raise ValueError(f"cannot form {n_groups} groups from {array.size} units")

    edges = np.quantile(array, np.linspace(0, 1, n_groups + 1)[1:-1])
    assignment = np.searchsorted(edges, array, side="right")

    if labels is not None:
        if len(labels) != n_groups:
            raise ValueError(f"expected {n_groups} labels, got {len(labels)}")
        names = list(labels)
    elif n_groups == 2:
        names = ["shorter (lower half)", "taller (upper half)"]
    elif n_groups == 3:
        names = ["shorter third", "middle third", "taller third"]
    else:
        names = [f"scale group {i + 1}/{n_groups}" for i in range(n_groups)]
    return assignment, names


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def equity_audit(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    scale: Sequence[float],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = mean_absolute_error,
    metric_name: str = "MAE",
    label: str = "",
    higher_is_better: bool = False,
    n_groups: int = 2,
    n_boot: int = 2000,
    n_permutations: int = 2000,
    seed: int = 0,
) -> EquityResult:
    """Compare a model's accuracy between body-scale groups.

    The gap is always reported as *worst minus best*, so a positive number always
    means "one group is served less well" whichever direction the metric runs.
    Significance comes from a permutation test, not from the bootstrap interval;
    :attr:`EquityResult.significant` explains why that distinction matters.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    sizes = np.asarray(scale, dtype=float)
    if not (truth.size == prediction.size == sizes.size):
        raise ValueError(f"lengths differ: {truth.size}, {prediction.size}, {sizes.size}")

    keep = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(sizes)
    truth, prediction, sizes = truth[keep], prediction[keep], sizes[keep]

    assignment, names = stratify_by_scale(sizes, n_groups)
    warnings: List[str] = []
    if truth.size < MIN_UNITS_FOR_INFERENCE:
        warnings.append(f"only {truth.size} units: the equity gap is indicative, not inferential")

    groups: List[GroupMetric] = []
    for index, name in enumerate(names):
        mask = assignment == index
        if mask.sum() == 0:
            warnings.append(f"group '{name}' is empty")
            continue
        if mask.sum() < 5:
            warnings.append(f"group '{name}' has only {int(mask.sum())} units")
        groups.append(GroupMetric(name, int(mask.sum()), metric(truth[mask], prediction[mask])))

    values = np.array([g.value for g in groups], dtype=float)
    gap = float(values.max() - values.min())

    def gap_of(index: np.ndarray) -> float:
        sampled = assignment[index]
        per_group = [
            metric(truth[index][sampled == i], prediction[index][sampled == i])
            for i in range(len(names))
            if (sampled == i).sum() >= 2
        ]
        if len(per_group) < 2:
            return float("nan")
        return float(max(per_group) - min(per_group))

    ci = bootstrap_ci(gap_of, truth.size, n_boot=n_boot, seed=seed)

    # The null: body-scale group carries no information about the error. Shuffling
    # the labels while holding the group sizes fixed samples from exactly that.
    rng = np.random.default_rng(seed)
    at_least_as_large = 0
    valid = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(assignment)
        per_group = [
            metric(truth[shuffled == i], prediction[shuffled == i])
            for i in range(len(names))
            if (shuffled == i).sum() >= 2
        ]
        if len(per_group) < 2:
            continue
        valid += 1
        if max(per_group) - min(per_group) >= gap:
            at_least_as_large += 1
    p_value = (1 + at_least_as_large) / (1 + valid) if valid else float("nan")

    if higher_is_better:
        warnings.append(f"{metric_name} is a score, not an error: the worst group is the LOWEST one")

    return EquityResult(
        metric=metric_name,
        groups=tuple(groups),
        gap=gap,
        gap_ci=ci,
        p_value=p_value,
        n_units=int(truth.size),
        label=label,
        warnings=tuple(warnings),
    )


def audit_features(
    raw: Mapping[str, Sequence[Optional[float]]],
    normalised: Mapping[str, Sequence[Optional[float]]],
    scale: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    notes: Sequence[str] = (),
) -> HiemAuditReport:
    """Run the invariance audit over every feature present in both tables."""
    shared = [name for name in raw if name in normalised]
    if not shared:
        raise ValueError("no feature name appears in both the raw and the normalised table")

    results = [
        invariance_audit(name, raw[name], normalised[name], scale, n_boot=n_boot, seed=seed)
        for name in shared
    ]
    return HiemAuditReport(invariance=tuple(results), notes=tuple(notes))


__all__ = [
    "MIN_UNITS_FOR_INFERENCE",
    "RESIDUAL_RHO_TOLERANCE",
    "EQUITY_ALPHA",
    "EquityResult",
    "GroupMetric",
    "HiemAuditReport",
    "InvarianceResult",
    "accuracy",
    "audit_features",
    "bootstrap_ci",
    "equity_audit",
    "invariance_audit",
    "mean_absolute_error",
    "pearson",
    "rankdata",
    "spearman",
    "stratify_by_scale",
]
