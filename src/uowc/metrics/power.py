"""Power metrics."""
from __future__ import annotations

from dataclasses import dataclass

from uowc.core.results import MetricValue, RawResult
from uowc.metrics.statistics import wilson_interval

__all__ = ["ReceivedPowerFraction"]


@dataclass(frozen=True, slots=True)
class ReceivedPowerFraction:
    """Detected weight as a fraction of launched weight.

    For an analog run this is one quantity under three names: received power fraction,
    photon capture probability, and detection efficiency (metrics.md) - not three
    independent results.

    Uncertainty is the Wilson score interval for a proportion, which stays valid in the
    rare-capture regime this channel operates in. With zero detections it reports the
    upper bound the non-detection licenses rather than a point estimate of zero.
    """

    @property
    def name(self) -> str:
        return "received_power_fraction"

    def compute(self, result: RawResult) -> MetricValue:
        tallies = result.output.tallies
        n = int(tallies.launched)
        fraction = tallies.detected_weight / n if n > 0 else 0.0
        se, low, high = wilson_interval(fraction, n)
        return MetricValue(
            name=self.name,
            value=fraction,
            unit="fraction",
            n_samples=n,
            std=se,
            ci95_low=low,
            ci95_high=high,
        )
