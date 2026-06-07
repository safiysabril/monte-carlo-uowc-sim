"""The persistable record of one simulation run."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from uowc.core.results import MetricValue, RawResult

__all__ = ["SimulationResult"]

_EMPTY: Mapping[str, MetricValue] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """A complete simulation outcome: raw transport output, provenance, and metrics.

    ``raw`` carries the detected photons, tallies and full :class:`RunMetadata`
    (scenario, model, parameters, seed, environmental effects). ``metrics`` are derived
    and reproducible from ``raw``; they are persisted for convenience.
    """

    raw: RawResult
    metrics: Mapping[str, MetricValue] = _EMPTY
