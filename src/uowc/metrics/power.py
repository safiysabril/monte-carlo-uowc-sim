"""Power metrics."""
from __future__ import annotations

from dataclasses import dataclass

from uowc.core.results import MetricValue, RawResult
from uowc.metrics.statistics import wald_interval

__all__ = ["ReceivedPowerFraction"]


@dataclass(frozen=True, slots=True)
class ReceivedPowerFraction:
    """Detected weight as a fraction of launched weight.

    For an analog run this is the capture probability / normalized received power.
    Uncertainty is the Wald interval for a proportion.
    """

    @property
    def name(self) -> str:
        return "received_power_fraction"

    def compute(self, result: RawResult) -> MetricValue:
        tallies = result.output.tallies
        n = int(tallies.launched)
        fraction = tallies.detected_weight / n if n > 0 else 0.0
        se, low, high = wald_interval(fraction, n)
        return MetricValue(
            name=self.name,
            value=fraction,
            unit="fraction",
            n_samples=n,
            std=se,
            ci95_low=low,
            ci95_high=high,
        )
