"""Tests for power metrics and the metric pipeline."""
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
from uowc.core.ports import Metric
from uowc.metrics import MetricPipeline, ReceivedPowerFraction


def _raw(photons: DetectedPhotons, tallies: Tallies) -> RawResult:
    meta = RunMetadata(
        scenario="T",
        medium_type="t",
        optical_model="t",
        effects=(),
        wavelength_nm=500.0,
        sampling=SamplingConfig(n_photons=0, estimator="analog"),
        seed_tree=SeedTree(root_seed=0),
        rng_impl="t",
        timestamp_utc="t",
        code_version="t",
    )
    return RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=meta)


def _photons(times) -> DetectedPhotons:
    times = np.asarray(times, dtype=float)
    n = times.size
    return DetectedPhotons(
        arrival_time_s=times,
        path_length_m=np.zeros(n),
        weight=np.ones(n),
        n_scatters=np.zeros(n, dtype=np.int64),
        incidence_rad=np.zeros(n),
    )


def test_received_power_conforms_to_metric() -> None:
    assert isinstance(ReceivedPowerFraction(), Metric)
    assert ReceivedPowerFraction().name == "received_power_fraction"


def test_received_power_fraction_value() -> None:
    tallies = Tallies(launched=1000, detected=300, detected_weight=300.0, absorbed_weight=500.0, escaped_weight=200.0)
    metric = ReceivedPowerFraction().compute(_raw(_photons(np.zeros(300)), tallies))
    assert metric.value == pytest.approx(0.3)
    assert metric.n_samples == 1000
    assert metric.std == pytest.approx(np.sqrt(0.3 * 0.7 / 1000))
    assert 0.0 <= metric.ci95_low < metric.value < metric.ci95_high <= 1.0


def test_received_power_zero_launched_is_safe() -> None:
    tallies = Tallies(launched=0, detected=0, detected_weight=0.0, absorbed_weight=0.0, escaped_weight=0.0)
    metric = ReceivedPowerFraction().compute(_raw(_photons([]), tallies))
    assert metric.value == 0.0 and metric.n_samples == 0


def test_pipeline_applies_all_metrics() -> None:
    pipeline = MetricPipeline.default()
    tallies = Tallies(launched=10, detected=2, detected_weight=2.0, absorbed_weight=5.0, escaped_weight=3.0)
    out = pipeline.run(_raw(_photons([1.0, 2.0]), tallies))
    assert set(out) == {"received_power_fraction", "mean_arrival_time", "rms_delay_spread"}
