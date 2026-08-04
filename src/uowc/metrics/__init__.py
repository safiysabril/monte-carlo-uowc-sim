"""Metrics (adapters): RawResult -> MetricValue. No transport, no plotting."""
from __future__ import annotations

from uowc.metrics.comparative import ConvergenceCurveMetric, ScenarioDeltaMetric
from uowc.metrics.frequency import Bandwidth3dB, FrequencyResponse
from uowc.metrics.pipeline import MetricPipeline
from uowc.metrics.power import ReceivedPowerFraction
from uowc.metrics.temporal import MeanArrivalTime, RmsDelaySpread

__all__ = [
    "ReceivedPowerFraction",
    "MeanArrivalTime",
    "RmsDelaySpread",
    "FrequencyResponse",
    "Bandwidth3dB",
    "MetricPipeline",
    "ScenarioDeltaMetric",
    "ConvergenceCurveMetric",
]
