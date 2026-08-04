"""Mean/variance/CI/sample-count reporting."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.metrics.statistics import (
    effective_sample_size,
    weighted_mean,
    weighted_rms_spread,
    weighted_standard_error,
    wilson_interval,
)


def test_weighted_mean_matches_numpy_average() -> None:
    values = np.array([1.0, 2.0, 4.0])
    weights = np.array([1.0, 3.0, 2.0])
    assert weighted_mean(values, weights) == pytest.approx(np.average(values, weights=weights))


def test_weighted_mean_is_nan_without_weight() -> None:
    assert np.isnan(weighted_mean(np.array([1.0]), np.array([0.0])))


def test_weighted_rms_spread_is_offset_invariant() -> None:
    """RMS spread about the mean must not depend on the time origin (metrics.md)."""
    times = np.array([1.0e-7, 1.2e-7, 1.5e-7])
    weights = np.array([1.0, 2.0, 1.0])
    assert weighted_rms_spread(times, weights) == pytest.approx(
        weighted_rms_spread(times + 5.0e-7, weights), rel=1e-9
    )


def test_weighted_rms_spread_is_zero_for_identical_arrivals() -> None:
    assert weighted_rms_spread(np.full(4, 3.0), np.ones(4)) == pytest.approx(0.0)


# --- Wilson interval -------------------------------------------------------------


def test_wilson_matches_closed_form() -> None:
    p, n, z = 0.3, 1000, 1.96
    se, low, high = wilson_interval(p, n)
    z2_n = z * z / n
    centre = (p + 0.5 * z2_n) / (1.0 + z2_n)
    half = (z / (1.0 + z2_n)) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    assert se == pytest.approx(np.sqrt(p * (1.0 - p) / n))
    assert low == pytest.approx(centre - half)
    assert high == pytest.approx(centre + half)


def test_wilson_is_non_degenerate_at_zero_detections() -> None:
    """The Wald failure this replaces: p=0 must not yield the interval [0, 0]."""
    n = 1_000_000
    se, low, high = wilson_interval(0.0, n)
    assert se == 0.0
    assert low == 0.0
    assert high > 0.0
    # Wilson's upper limit at p = 0 is z^2 / (n + z^2), comparable to the exact
    # one-sided "rule of three" bound 3/n.
    assert high == pytest.approx(1.96**2 / (n + 1.96**2))
    assert 3.0 / n < high < 6.0 / n


def test_wilson_stays_inside_the_unit_interval_in_the_rare_event_regime() -> None:
    for p in (1e-9, 1e-6, 1e-3):
        _, low, high = wilson_interval(p, 10_000)
        assert 0.0 <= low <= p <= high <= 1.0


def test_wilson_brackets_the_estimate_and_narrows_with_n() -> None:
    def width(n: int) -> float:
        _, low, high = wilson_interval(0.2, n)
        return high - low

    widths = [width(n) for n in (100, 10_000, 1_000_000)]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_is_safe_for_zero_trials() -> None:
    assert wilson_interval(0.0, 0) == (0.0, 0.0, 0.0)


# --- weighted estimators ---------------------------------------------------------


def test_effective_sample_size_equals_count_for_equal_weights() -> None:
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)


def test_effective_sample_size_collapses_under_a_dominant_weight() -> None:
    weights = np.concatenate([[1000.0], np.ones(99)])
    assert effective_sample_size(weights) < 5.0


def test_effective_sample_size_is_zero_without_weight() -> None:
    assert effective_sample_size(np.zeros(10)) == 0.0


def test_weighted_standard_error_reduces_to_the_binomial_case() -> None:
    """Unit-weight scores over N trials reproduce sqrt(p(1-p)/N) up to Bessel's factor."""
    n, k = 1000, 300
    se = weighted_standard_error(np.ones(k), n)
    p = k / n
    assert se == pytest.approx(np.sqrt(p * (1.0 - p) / (n - 1)), rel=1e-9)


def test_weighted_standard_error_scales_as_inverse_sqrt_n() -> None:
    rng = np.random.default_rng(0)
    scores = rng.exponential(1.0, size=4000)
    se_small = weighted_standard_error(scores[:1000], 1000)
    se_large = weighted_standard_error(scores, 4000)
    assert se_large == pytest.approx(se_small / 2.0, rel=0.25)


def test_weighted_standard_error_is_nan_below_two_trials() -> None:
    assert np.isnan(weighted_standard_error(np.array([1.0]), 1))
