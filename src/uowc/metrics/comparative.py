"""Comparative/ensemble metrics over multiple results: scenario deltas (I vs II, II vs
III), convergence curves. Implements :class:`~uowc.core.ports.ComparativeMetric`.

These are thin adapters onto :mod:`uowc.analysis` - the statistics (paired-replicate
differences, log-log slope convergence checks) live there, not here, so this module
does not re-derive them. What this module adds is the ``ComparativeMetric`` port
shape: given an ordered :class:`~uowc.core.results.RawResult` sequence, run a
single-run :class:`~uowc.core.ports.Metric` over each, hand the resulting point
estimates to the appropriate :mod:`uowc.analysis` function, and pack the headline
number back into one :class:`~uowc.core.results.MetricValue`.

Both adapters necessarily discard some of the richer :mod:`uowc.analysis` report (a
``MetricValue`` has one scalar-or-curve value, one std, one CI): a caller who wants
the full picture - ``ScenarioDifference.meets_recommended_replicate_count``,
``ConvergenceReport.all_consecutive_estimates_overlap``, and so on - should call
:mod:`uowc.analysis` directly rather than through this port. This module exists for
contexts (a generic metric pipeline, a results table) that need the standard
``Metric``/``ComparativeMetric`` shape and only the headline number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uowc.analysis.comparison import paired_scenario_difference
from uowc.analysis.convergence import assess_convergence
from uowc.core.ports import Metric
from uowc.core.results import MetricValue, RawResult

__all__ = ["ScenarioDeltaMetric", "ConvergenceCurveMetric"]


def _scalar(metric_value: MetricValue) -> float:
    if not isinstance(metric_value.value, (int, float)):
        raise TypeError(
            f"{metric_value.name!r}: comparative metrics require a scalar base metric, "
            f"got {type(metric_value.value).__name__}"
        )
    return float(metric_value.value)


@dataclass(frozen=True, slots=True)
class ScenarioDeltaMetric:
    """``scenario_a - scenario_b`` for ``base_metric``, as a paired-replicate
    difference (:func:`~uowc.analysis.comparison.paired_scenario_difference`).

    ``compare()`` groups ``results`` by ``RawResult.metadata.scenario``, keeping only
    the two configured scenario labels (other scenarios present are ignored) and
    requiring equal group sizes. Grouping preserves the input order within each
    group, so **the i-th scenario_a result and the i-th scenario_b result must be the
    same replicate** (same seed set) - this is the caller's responsibility, exactly as
    for :func:`~uowc.analysis.comparison.compare_scenarios`; the adapter cannot verify
    pairing from ``RunMetadata`` alone.
    """

    base_metric: Metric
    scenario_a: str
    scenario_b: str
    confidence: float = 0.95

    @property
    def name(self) -> str:
        return f"{self.base_metric.name}_delta_{self.scenario_a}_vs_{self.scenario_b}"

    def compare(self, results: Sequence[RawResult]) -> MetricValue:
        group_a = [r for r in results if r.metadata.scenario == self.scenario_a]
        group_b = [r for r in results if r.metadata.scenario == self.scenario_b]
        if len(group_a) != len(group_b):
            raise ValueError(
                f"{self.name!r}: unequal replicate counts for {self.scenario_a!r} "
                f"({len(group_a)}) and {self.scenario_b!r} ({len(group_b)})"
            )

        computed_a = [self.base_metric.compute(r) for r in group_a]
        computed_b = [self.base_metric.compute(r) for r in group_b]
        values_a = [_scalar(mv) for mv in computed_a]
        values_b = [_scalar(mv) for mv in computed_b]

        difference = paired_scenario_difference(
            metric_name=self.base_metric.name,
            scenario_a=self.scenario_a,
            scenario_b=self.scenario_b,
            values_a=values_a,
            values_b=values_b,
            confidence=self.confidence,
        )
        unit = computed_a[0].unit if computed_a else ""
        standard_error = difference.std_difference / (difference.n_replicates**0.5)
        return MetricValue(
            name=self.name,
            value=difference.mean_difference,
            unit=unit,
            n_samples=difference.n_replicates,
            std=standard_error,
            ci95_low=difference.ci95_low,
            ci95_high=difference.ci95_high,
        )


@dataclass(frozen=True, slots=True)
class ConvergenceCurveMetric:
    """Whether ``base_metric`` is converging as photon count increases
    (:func:`~uowc.analysis.convergence.assess_convergence`).

    Reports the fitted ``log(SE)`` vs ``log(N)`` slope as the headline value (the CLT
    rate is -0.5; see analysis/convergence.py), with its regression standard error as
    ``std``. Every result in ``results`` must carry a distinct
    ``RawResult.metadata.sampling.n_photons``.
    """

    base_metric: Metric
    slope_tolerance: float = 0.15

    @property
    def name(self) -> str:
        return f"{self.base_metric.name}_convergence_slope"

    def compare(self, results: Sequence[RawResult]) -> MetricValue:
        n_photons = [r.metadata.sampling.n_photons for r in results]
        values = [self.base_metric.compute(r) for r in results]

        report = assess_convergence(
            metric_name=self.base_metric.name,
            n_photons=n_photons,
            values=values,
            slope_tolerance=self.slope_tolerance,
        )
        return MetricValue(
            name=self.name,
            value=report.fitted_slope,
            unit="dimensionless",
            n_samples=len(results),
            std=report.fitted_slope_stderr,
        )
