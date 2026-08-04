"""Scenario comparison: I vs II, II vs III differences with uncertainty.

Implements the *independent-replication* procedure from research-methodology.md
("Comparing scenarios: use independent replications"), not a naive difference of two
single-run confidence intervals. That procedure is unsound here on two counts: common
random numbers do not stay paired under delta tracking (Scenario I and II use
different majorants, so their variate streams desynchronize at the first null
collision - see transport.md), and even where some correlation survives,
``Var(A-B) != Var(A) + Var(B)``, so differencing two independent CIs mis-states the
uncertainty on the one number the whole study is about.

The robust alternative needs no assumption about that correlation: run each scenario
``R`` independent replicates (each replicate its own seed set; ``R >= 10``
recommended), form the *per-replicate* difference ``d_r = a_r - b_r``, and report the
mean difference with a t-interval computed from the sample variance of ``d_r`` across
replicates. This is a paired estimator regardless of whether the two scenarios turned
out to be correlated - the pairing is by replicate index, decided at the experiment-
design stage, not asserted after the fact from a shared seed.

This module is a pure function of already-computed metric values (a downstream
analysis, per data.md / metrics.md) - it does not run simulations, and it does not
import :mod:`uowc.experiments`, so the dependency direction stays
experiments -> analysis, never the reverse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats

from uowc.core.results import MetricValue

__all__ = ["ScenarioDifference", "paired_scenario_difference", "compare_scenarios"]


@dataclass(frozen=True, slots=True)
class ScenarioDifference:
    """Paired-replicate estimate of ``scenario_a - scenario_b`` for one metric.

    ``n_replicates`` below research-methodology.md's recommended minimum of 10 is not
    rejected (a t-interval is defined for any ``R >= 2``), but it is under-powered;
    callers building a reproducibility report should flag it, not just this module.
    """

    metric_name: str
    scenario_a: str
    scenario_b: str
    mean_difference: float
    std_difference: float
    n_replicates: int
    confidence: float
    ci95_low: float
    ci95_high: float

    @property
    def excludes_zero(self) -> bool:
        """True iff the confidence interval on the difference excludes zero.

        research-methodology.md: "A difference whose interval spans zero is not a
        result." This does not on its own establish practical significance - only
        that chance alone is an unlikely explanation for the sign of the difference.
        """
        return self.ci95_low > 0.0 or self.ci95_high < 0.0

    @property
    def meets_recommended_replicate_count(self) -> bool:
        return self.n_replicates >= 10


def paired_scenario_difference(
    *,
    metric_name: str,
    scenario_a: str,
    scenario_b: str,
    values_a: Sequence[float],
    values_b: Sequence[float],
    confidence: float = 0.95,
) -> ScenarioDifference:
    """Paired t-interval for the mean of ``values_a[r] - values_b[r]``.

    ``values_a`` and ``values_b`` are point estimates of the *same* metric from ``R``
    independent replicates of scenario A and B respectively, paired by replicate index
    ``r`` (each replicate shares its seed set across the two scenarios; different
    replicates use independent seeds). Requires ``R >= 2`` to estimate a variance.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must have equal length (paired by replicate)")
    if a.ndim != 1:
        raise ValueError("values_a/values_b must be 1-D (one scalar per replicate)")
    n_replicates = int(a.size)
    if n_replicates < 2:
        raise ValueError("at least 2 replicates are required to estimate a variance")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")

    differences = a - b
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    standard_error = std_difference / np.sqrt(n_replicates)

    alpha = 1.0 - confidence
    t_critical = float(scipy.stats.t.ppf(1.0 - alpha / 2.0, df=n_replicates - 1))
    half_width = t_critical * standard_error

    return ScenarioDifference(
        metric_name=metric_name,
        scenario_a=scenario_a,
        scenario_b=scenario_b,
        mean_difference=mean_difference,
        std_difference=std_difference,
        n_replicates=n_replicates,
        confidence=confidence,
        ci95_low=mean_difference - half_width,
        ci95_high=mean_difference + half_width,
    )


def _scalar_values(
    metrics_by_replicate: Sequence[Mapping[str, MetricValue]], name: str
) -> list[float] | None:
    """Extract ``name``'s point estimate from each replicate, or ``None`` if any
    replicate lacks it or holds an array-valued (curve) metric."""
    out: list[float] = []
    for metrics in metrics_by_replicate:
        if name not in metrics:
            return None
        value = metrics[name].value
        if not isinstance(value, (int, float)):
            return None
        out.append(float(value))
    return out


def compare_scenarios(
    *,
    scenario_a: str,
    scenario_b: str,
    results_a: Sequence[Mapping[str, MetricValue]],
    results_b: Sequence[Mapping[str, MetricValue]],
    confidence: float = 0.95,
) -> dict[str, ScenarioDifference]:
    """Paired comparison across every scalar metric common to both replicate sets.

    ``results_a``/``results_b`` are each a sequence of ``R`` replicates' metric
    mappings (e.g. ``ScenarioResult.metrics`` from ``R`` independent runs, one per
    replicate) - paired element-by-element, so ``results_a[r]`` and ``results_b[r]``
    must come from the same replicate (same seed set). Array-valued metrics (e.g.
    ``frequency_response``) are skipped: a paired t-interval is for scalars.
    """
    if len(results_a) != len(results_b):
        raise ValueError("results_a and results_b must have the same number of replicates (paired)")
    if not results_a:
        return {}

    common_names = sorted(set(results_a[0]) & set(results_b[0]))
    differences: dict[str, ScenarioDifference] = {}
    for name in common_names:
        values_a = _scalar_values(results_a, name)
        values_b = _scalar_values(results_b, name)
        if values_a is None or values_b is None:
            continue
        differences[name] = paired_scenario_difference(
            metric_name=name,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            values_a=values_a,
            values_b=values_b,
            confidence=confidence,
        )
    return differences
