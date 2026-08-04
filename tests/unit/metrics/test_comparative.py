"""Scenario-delta and convergence-curve metrics."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import (
    DetectedPhotons,
    RawResult,
    RunMetadata,
    SamplingConfig,
    SeedTree,
    Tallies,
    TransportOutput,
)
from uowc.core.ports import ComparativeMetric
from uowc.metrics import ReceivedPowerFraction
from uowc.metrics.comparative import ConvergenceCurveMetric, ScenarioDeltaMetric


def _raw(*, scenario: str, n_photons: int, detected: int, weight: float = 1.0) -> RawResult:
    times = [1e-7] * detected
    n = len(times)
    photons = DetectedPhotons(
        arrival_time_s=np.array(times),
        path_length_m=np.zeros(n),
        weight=np.full(n, weight),
        n_scatters=np.zeros(n, dtype=np.int64),
        incidence_rad=np.zeros(n),
    )
    tallies = Tallies(
        launched=n_photons,
        detected=n,
        detected_weight=weight * n,
        absorbed_weight=float(n_photons - n) * weight,
        escaped_weight=0.0,
    )
    meta = RunMetadata(
        scenario=scenario,
        medium_type="t",
        optical_model="t",
        effects=(),
        wavelength_nm=520.0,
        sampling=SamplingConfig(n_photons=n_photons, estimator="analog"),
        seed_tree=SeedTree(root_seed=0),
        rng_impl="t",
        timestamp_utc="t",
        code_version="t",
    )
    return RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=meta)


# --- ScenarioDeltaMetric -------------------------------------------------------------


def test_conforms_to_comparative_metric() -> None:
    metric = ScenarioDeltaMetric(
        base_metric=ReceivedPowerFraction(), scenario_a="I", scenario_b="II"
    )
    assert isinstance(metric, ComparativeMetric)
    assert metric.name == "received_power_fraction_delta_I_vs_II"


def test_delta_matches_hand_computed_paired_difference() -> None:
    # Scenario I: detected fractions 300/1000, 310/1000, 292/1000 (per-replicate
    # differences deliberately unequal, so the paired variance is nonzero).
    # Scenario II: detected fractions 250/1000, 260/1000, 240/1000
    results = [
        _raw(scenario="I", n_photons=1000, detected=300),
        _raw(scenario="II", n_photons=1000, detected=250),
        _raw(scenario="I", n_photons=1000, detected=310),
        _raw(scenario="II", n_photons=1000, detected=260),
        _raw(scenario="I", n_photons=1000, detected=292),
        _raw(scenario="II", n_photons=1000, detected=240),
    ]
    metric = ScenarioDeltaMetric(
        base_metric=ReceivedPowerFraction(), scenario_a="I", scenario_b="II"
    )
    result = metric.compare(results)

    expected_mean = np.mean([0.30 - 0.25, 0.31 - 0.26, 0.292 - 0.24])
    assert result.value == pytest.approx(expected_mean)
    assert result.n_samples == 3
    assert result.unit == "fraction"
    assert result.ci95_low < result.value < result.ci95_high


def test_delta_ignores_scenarios_outside_the_configured_pair() -> None:
    results = [
        _raw(scenario="I", n_photons=1000, detected=300),
        _raw(scenario="II", n_photons=1000, detected=250),
        _raw(scenario="III", n_photons=1000, detected=999),  # must be ignored
        _raw(scenario="I", n_photons=1000, detected=310),
        _raw(scenario="II", n_photons=1000, detected=260),
    ]
    metric = ScenarioDeltaMetric(
        base_metric=ReceivedPowerFraction(), scenario_a="I", scenario_b="II"
    )
    result = metric.compare(results)
    assert result.n_samples == 2


def test_delta_rejects_unequal_replicate_counts() -> None:
    results = [
        _raw(scenario="I", n_photons=1000, detected=300),
        _raw(scenario="I", n_photons=1000, detected=310),
        _raw(scenario="II", n_photons=1000, detected=250),
    ]
    metric = ScenarioDeltaMetric(
        base_metric=ReceivedPowerFraction(), scenario_a="I", scenario_b="II"
    )
    with pytest.raises(ValueError, match="unequal replicate counts"):
        metric.compare(results)


# --- ConvergenceCurveMetric -----------------------------------------------------------


def test_convergence_conforms_to_comparative_metric() -> None:
    metric = ConvergenceCurveMetric(base_metric=ReceivedPowerFraction())
    assert isinstance(metric, ComparativeMetric)
    assert metric.name == "received_power_fraction_convergence_slope"


def test_convergence_slope_near_minus_half_for_a_stable_true_probability() -> None:
    """Detected fraction stays near the same true p as N grows: genuine 1/sqrt(N)
    Monte Carlo scaling of the Wilson-interval standard error."""
    rng = np.random.default_rng(0)
    true_p = 0.30
    n_photons_list = [2_000, 8_000, 32_000, 128_000]
    results = [
        _raw(scenario="II", n_photons=n, detected=int(rng.binomial(n, true_p)))
        for n in n_photons_list
    ]
    metric = ConvergenceCurveMetric(base_metric=ReceivedPowerFraction())
    result = metric.compare(results)

    assert result.value == pytest.approx(-0.5, abs=0.2)
    assert result.n_samples == len(results)
    assert result.unit == "dimensionless"


def test_convergence_requires_distinct_photon_counts() -> None:
    results = [
        _raw(scenario="II", n_photons=1000, detected=300),
        _raw(scenario="II", n_photons=1000, detected=305),
        _raw(scenario="II", n_photons=1000, detected=295),
    ]
    metric = ConvergenceCurveMetric(base_metric=ReceivedPowerFraction())
    with pytest.raises(ValueError, match="distinct"):
        metric.compare(results)
