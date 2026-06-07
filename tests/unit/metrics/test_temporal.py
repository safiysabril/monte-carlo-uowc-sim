"""Tests for temporal metrics."""
from __future__ import annotations

import math

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
from uowc.metrics import MeanArrivalTime, RmsDelaySpread


def _raw(times) -> RawResult:
    times = np.asarray(times, dtype=float)
    n = times.size
    photons = DetectedPhotons(
        arrival_time_s=times,
        path_length_m=np.zeros(n),
        weight=np.ones(n),
        n_scatters=np.zeros(n, dtype=np.int64),
        incidence_rad=np.zeros(n),
    )
    tallies = Tallies(launched=n, detected=n, detected_weight=float(n), absorbed_weight=0.0, escaped_weight=0.0)
    meta = RunMetadata(
        scenario="T", medium_type="t", optical_model="t", effects=(), wavelength_nm=500.0,
        sampling=SamplingConfig(n_photons=0, estimator="analog"), seed_tree=SeedTree(root_seed=0),
        rng_impl="t", timestamp_utc="t", code_version="t",
    )
    return RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=meta)


def test_metrics_conform() -> None:
    assert isinstance(MeanArrivalTime(), Metric)
    assert isinstance(RmsDelaySpread(), Metric)


def test_mean_arrival_time() -> None:
    metric = MeanArrivalTime().compute(_raw([1.0, 2.0, 3.0]))
    assert metric.value == pytest.approx(2.0)
    assert metric.unit == "s"
    assert metric.n_samples == 3


def test_rms_delay_spread() -> None:
    metric = RmsDelaySpread().compute(_raw([1.0, 2.0, 3.0]))
    assert metric.value == pytest.approx(math.sqrt(2.0 / 3.0))


def test_empty_is_nan_with_zero_samples() -> None:
    mean = MeanArrivalTime().compute(_raw([]))
    spread = RmsDelaySpread().compute(_raw([]))
    assert math.isnan(mean.value) and mean.n_samples == 0
    assert math.isnan(spread.value)
