"""Temporal metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core.results import MetricValue, RawResult
from uowc.metrics.statistics import weighted_mean, weighted_rms_spread

__all__ = ["MeanArrivalTime", "RmsDelaySpread"]


@dataclass(frozen=True, slots=True)
class MeanArrivalTime:
    """Weight-averaged photon arrival time at the receiver [s]."""

    @property
    def name(self) -> str:
        return "mean_arrival_time"

    def compute(self, result: RawResult) -> MetricValue:
        photons = result.output.photons
        n = int(photons.weight.size)
        mean = weighted_mean(photons.arrival_time_s, photons.weight)
        std = weighted_rms_spread(photons.arrival_time_s, photons.weight) / np.sqrt(n) if n > 0 else None
        return MetricValue(name=self.name, value=mean, unit="s", n_samples=n, std=std)


@dataclass(frozen=True, slots=True)
class RmsDelaySpread:
    """RMS spread of arrival times about the mean excess delay [s]."""

    @property
    def name(self) -> str:
        return "rms_delay_spread"

    def compute(self, result: RawResult) -> MetricValue:
        photons = result.output.photons
        n = int(photons.weight.size)
        spread = weighted_rms_spread(photons.arrival_time_s, photons.weight)
        return MetricValue(name=self.name, value=spread, unit="s", n_samples=n)
