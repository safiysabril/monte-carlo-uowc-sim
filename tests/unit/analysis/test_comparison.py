"""Paired-replicate scenario-difference estimator (research-methodology.md)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from uowc.analysis.comparison import (
    ScenarioDifference,
    compare_scenarios,
    paired_scenario_difference,
)
from uowc.core.results import MetricValue


def test_matches_scipy_paired_ttest_closed_form() -> None:
    """Cross-check against scipy's own paired t-test rather than reimplemented math."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.30, scale=0.02, size=20)
    b = rng.normal(loc=0.28, scale=0.02, size=20)

    result = paired_scenario_difference(
        metric_name="received_power_fraction",
        scenario_a="I",
        scenario_b="II",
        values_a=a,
        values_b=b,
    )

    diffs = a - b
    expected_mean = np.mean(diffs)
    expected_se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    t_crit = scipy.stats.t.ppf(0.975, df=len(diffs) - 1)

    assert result.mean_difference == pytest.approx(expected_mean)
    assert result.ci95_low == pytest.approx(expected_mean - t_crit * expected_se)
    assert result.ci95_high == pytest.approx(expected_mean + t_crit * expected_se)

    # scipy.stats.ttest_rel independently confirms the same mean difference and,
    # via its p-value/CI machinery, the same standard error.
    ttest = scipy.stats.ttest_rel(a, b)
    assert result.mean_difference == pytest.approx(float(np.mean(a) - np.mean(b)))
    scipy_ci = ttest.confidence_interval(confidence_level=0.95)
    # ttest_rel's CI is on the mean of (a - b) directly.
    assert result.ci95_low == pytest.approx(scipy_ci.low, rel=1e-6)
    assert result.ci95_high == pytest.approx(scipy_ci.high, rel=1e-6)


def test_identical_replicates_give_a_zero_width_interval_at_the_true_difference() -> None:
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.5, 0.5, 0.5, 0.5]
    result = paired_scenario_difference(
        metric_name="m", scenario_a="I", scenario_b="II", values_a=a, values_b=b
    )
    assert result.mean_difference == pytest.approx(0.5)
    assert result.std_difference == pytest.approx(0.0)
    assert result.ci95_low == pytest.approx(0.5)
    assert result.ci95_high == pytest.approx(0.5)
    assert result.excludes_zero


def test_excludes_zero_is_false_when_interval_spans_zero() -> None:
    a = [1.0, 2.0, 3.0, -4.0, 5.0]
    b = [1.1, 2.2, 2.7, -3.5, 4.6]  # noisy, small, sign-varying differences
    result = paired_scenario_difference(
        metric_name="m", scenario_a="I", scenario_b="II", values_a=a, values_b=b
    )
    assert result.ci95_low < 0.0 < result.ci95_high
    assert not result.excludes_zero


def test_recommended_replicate_count_flag() -> None:
    few = paired_scenario_difference(
        metric_name="m", scenario_a="I", scenario_b="II", values_a=[1.0, 2.0], values_b=[0.5, 1.5]
    )
    assert not few.meets_recommended_replicate_count

    many_a = list(range(10))
    many_b = [x - 1 for x in many_a]
    many = paired_scenario_difference(
        metric_name="m", scenario_a="I", scenario_b="II", values_a=many_a, values_b=many_b
    )
    assert many.meets_recommended_replicate_count


def test_rejects_mismatched_replicate_counts() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_scenario_difference(
            metric_name="m", scenario_a="I", scenario_b="II", values_a=[1.0, 2.0], values_b=[1.0]
        )


def test_rejects_fewer_than_two_replicates() -> None:
    with pytest.raises(ValueError, match="at least 2 replicates"):
        paired_scenario_difference(
            metric_name="m", scenario_a="I", scenario_b="II", values_a=[1.0], values_b=[0.5]
        )


def test_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        paired_scenario_difference(
            metric_name="m",
            scenario_a="I",
            scenario_b="II",
            values_a=[1.0, 2.0],
            values_b=[1.0, 2.0],
            confidence=1.5,
        )


def _metrics(power: float, delay: float) -> dict[str, MetricValue]:
    return {
        "received_power_fraction": MetricValue(
            name="received_power_fraction", value=power, unit="fraction", n_samples=1000
        ),
        "mean_arrival_time": MetricValue(
            name="mean_arrival_time", value=delay, unit="s", n_samples=1000
        ),
    }


def test_compare_scenarios_returns_a_difference_per_common_scalar_metric() -> None:
    results_a = [_metrics(0.30, 1.0e-7), _metrics(0.31, 1.05e-7), _metrics(0.29, 0.98e-7)]
    results_b = [_metrics(0.25, 1.1e-7), _metrics(0.26, 1.12e-7), _metrics(0.24, 1.08e-7)]

    diffs = compare_scenarios(
        scenario_a="I", scenario_b="II", results_a=results_a, results_b=results_b
    )

    assert set(diffs) == {"received_power_fraction", "mean_arrival_time"}
    assert isinstance(diffs["received_power_fraction"], ScenarioDifference)
    assert diffs["received_power_fraction"].mean_difference == pytest.approx(0.05, abs=1e-6)


def test_compare_scenarios_skips_array_valued_metrics() -> None:
    def with_curve(power: float) -> dict[str, MetricValue]:
        out = _metrics(power, 1e-7)
        out["frequency_response"] = MetricValue(
            name="frequency_response", value=np.array([1.0 + 0j, 0.5 + 0j]), unit="", n_samples=10
        )
        return out

    results_a = [with_curve(0.3), with_curve(0.31)]
    results_b = [with_curve(0.25), with_curve(0.26)]
    diffs = compare_scenarios(
        scenario_a="I", scenario_b="II", results_a=results_a, results_b=results_b
    )
    assert "frequency_response" not in diffs
    assert "received_power_fraction" in diffs


def test_compare_scenarios_skips_metrics_not_common_to_both() -> None:
    results_a = [_metrics(0.3, 1e-7), _metrics(0.31, 1.1e-7)]
    only_power = "received_power_fraction"
    results_b_only_power = [
        {only_power: _metrics(0.25, 1e-7)[only_power]},
        {only_power: _metrics(0.26, 1.1e-7)[only_power]},
    ]
    diffs = compare_scenarios(
        scenario_a="I", scenario_b="II", results_a=results_a, results_b=results_b_only_power
    )
    assert set(diffs) == {"received_power_fraction"}


def test_compare_scenarios_rejects_mismatched_replica_counts() -> None:
    with pytest.raises(ValueError, match="same number of replicates"):
        compare_scenarios(
            scenario_a="I",
            scenario_b="II",
            results_a=[_metrics(0.3, 1e-7)],
            results_b=[_metrics(0.25, 1e-7), _metrics(0.26, 1e-7)],
        )


def test_compare_scenarios_empty_input_returns_empty_mapping() -> None:
    assert compare_scenarios(scenario_a="I", scenario_b="II", results_a=[], results_b=[]) == {}
