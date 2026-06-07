"""A shared, ordered collection of metrics applied to a transport result."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from uowc.core.ports import Metric
from uowc.core.results import MetricValue, RawResult
from uowc.metrics.power import ReceivedPowerFraction
from uowc.metrics.temporal import MeanArrivalTime, RmsDelaySpread

__all__ = ["MetricPipeline"]


@dataclass(frozen=True, slots=True)
class MetricPipeline:
    """Applies a fixed set of metrics to a :class:`RawResult`.

    The same pipeline instance is reused across scenarios so every scenario is scored
    by an identical metric set (a fair, shared metrics pipeline).
    """

    metrics: tuple[Metric, ...]

    def run(self, result: RawResult) -> Mapping[str, MetricValue]:
        return {metric.name: metric.compute(result) for metric in self.metrics}

    @classmethod
    def default(cls) -> "MetricPipeline":
        return cls(metrics=(ReceivedPowerFraction(), MeanArrivalTime(), RmsDelaySpread()))
