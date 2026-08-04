"""SE proportional to N^-1/2 convergence checks (research-methodology.md)."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.analysis.convergence import ConvergenceReport, assess_convergence
from uowc.core.results import MetricValue


def _mv(value: float, std: float, z: float = 1.96) -> MetricValue:
    return MetricValue(
        name="m",
        value=value,
        unit="fraction",
        n_samples=100,
        std=std,
        ci95_low=value - z * std,
        ci95_high=value + z * std,
    )


def test_true_monte_carlo_scaling_is_recognized_as_converged() -> None:
    """SE = k / sqrt(N) exactly: the textbook case the -0.5 slope check exists for."""
    n_photons = [10_000, 40_000, 160_000, 640_000]
    k = 0.01
    true_value = 0.30
    values = [_mv(true_value, k / np.sqrt(n)) for n in n_photons]

    report = assess_convergence(
        metric_name="received_power_fraction", n_photons=n_photons, values=values
    )

    assert report.fitted_slope == pytest.approx(-0.5, abs=1e-6)
    assert report.slope_within_tolerance
    assert report.all_consecutive_estimates_overlap
    assert report.converged


def test_stalled_standard_error_is_not_converged() -> None:
    """SE that does not shrink with N (slope ~ 0) is the bug/heavy-tail case."""
    n_photons = [10_000, 40_000, 160_000, 640_000]
    values = [_mv(0.30, 0.01) for _ in n_photons]  # constant SE regardless of N

    report = assess_convergence(metric_name="rms_delay_spread", n_photons=n_photons, values=values)

    assert report.fitted_slope == pytest.approx(0.0, abs=1e-6)
    assert not report.slope_within_tolerance
    assert not report.converged


def test_slope_steeper_than_minus_half_also_fails_tolerance() -> None:
    """Falling faster than N^-1/2 is flagged too (research-methodology.md: "a falling
    trend indicates a bug"), not just a shallower-than-expected slope."""
    n_photons = [10_000, 40_000, 160_000, 640_000]
    values = [_mv(0.30, 0.01 / n) for n in n_photons]  # SE ~ 1/N, far steeper than 1/sqrt(N)

    report = assess_convergence(metric_name="m", n_photons=n_photons, values=values)

    assert report.fitted_slope < -0.9
    assert not report.slope_within_tolerance


def test_non_overlapping_consecutive_estimates_fail_convergence_even_with_good_slope() -> None:
    n_photons = [10_000, 40_000, 160_000, 640_000]
    k = 0.001  # tight enough that a real jump in the estimate won't be swallowed by the CI
    # A genuine trend/jump in the point estimate, not just noise.
    drifting_values = [0.10, 0.10, 0.50, 0.50]
    values = [_mv(v, k / np.sqrt(n)) for v, n in zip(drifting_values, n_photons, strict=True)]

    report = assess_convergence(metric_name="m", n_photons=n_photons, values=values)

    assert not report.all_consecutive_estimates_overlap
    assert not report.converged


def test_points_are_sorted_by_photon_count_regardless_of_input_order() -> None:
    n_photons = [160_000, 10_000, 640_000, 40_000]
    k = 0.01
    values = [_mv(0.3, k / np.sqrt(n)) for n in n_photons]

    report = assess_convergence(metric_name="m", n_photons=n_photons, values=values)

    assert [p.n_photons for p in report.points] == [10_000, 40_000, 160_000, 640_000]


def test_normalized_standard_error_is_flat_for_true_monte_carlo_scaling() -> None:
    n_photons = [10_000, 40_000, 160_000, 640_000]
    k = 0.01
    values = [_mv(0.3, k / np.sqrt(n)) for n in n_photons]
    report = assess_convergence(metric_name="m", n_photons=n_photons, values=values)

    normalized = [p.normalized_standard_error for p in report.points]
    assert normalized == pytest.approx([k] * len(n_photons), rel=1e-9)


def test_rejects_fewer_than_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        assess_convergence(
            metric_name="m", n_photons=[100, 200], values=[_mv(0.3, 0.1), _mv(0.3, 0.05)]
        )


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        assess_convergence(
            metric_name="m", n_photons=[100, 200, 300], values=[_mv(0.3, 0.1), _mv(0.3, 0.05)]
        )


def test_rejects_duplicate_photon_counts() -> None:
    with pytest.raises(ValueError, match="distinct"):
        assess_convergence(
            metric_name="m",
            n_photons=[100, 100, 300],
            values=[_mv(0.3, 0.1), _mv(0.3, 0.09), _mv(0.3, 0.05)],
        )


def test_rejects_nonpositive_photon_counts() -> None:
    with pytest.raises(ValueError, match="positive"):
        assess_convergence(
            metric_name="m",
            n_photons=[0, 100, 300],
            values=[_mv(0.3, 0.1), _mv(0.3, 0.09), _mv(0.3, 0.05)],
        )


def test_rejects_missing_standard_error() -> None:
    bare = MetricValue(name="m", value=0.3, unit="fraction", n_samples=10)
    with pytest.raises(ValueError, match="std"):
        assess_convergence(metric_name="m", n_photons=[100, 200, 300], values=[bare, bare, bare])


def test_rejects_zero_standard_error() -> None:
    zero_se = _mv(0.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        assess_convergence(
            metric_name="m",
            n_photons=[100, 200, 300],
            values=[zero_se, _mv(0.3, 0.05), _mv(0.3, 0.03)],
        )


def test_rejects_missing_confidence_interval() -> None:
    no_ci = MetricValue(name="m", value=0.3, unit="fraction", n_samples=10, std=0.05)
    with pytest.raises(ValueError, match="ci95"):
        assess_convergence(metric_name="m", n_photons=[100, 200, 300], values=[no_ci, no_ci, no_ci])


def test_rejects_array_valued_metric() -> None:
    curve = MetricValue(name="m", value=np.array([1.0, 2.0]), unit="", n_samples=10, std=0.1)
    with pytest.raises(TypeError, match="scalar"):
        assess_convergence(metric_name="m", n_photons=[100, 200, 300], values=[curve, curve, curve])


def test_report_is_a_convergence_report_instance() -> None:
    n_photons = [100, 200, 300]
    values = [_mv(0.3, 0.1 / np.sqrt(n)) for n in n_photons]
    assert isinstance(
        assess_convergence(metric_name="m", n_photons=n_photons, values=values), ConvergenceReport
    )
