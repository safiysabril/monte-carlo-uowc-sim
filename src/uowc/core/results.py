"""Result schemas: the transport -> metrics -> storage seam."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from uowc.core.config import SamplingConfig
from uowc.core.units import ComplexArray, FloatArray, IntArray, as_readonly, freeze_value

__all__ = [
    "SCHEMA_VERSION",
    "DetectedPhotons",
    "Tallies",
    "TallyResult",
    "TransportOutput",
    "SeedTree",
    "RunMetadata",
    "RawResult",
    "Axis",
    "MetricValue",
]

#: Version of the result/metadata schema; bump on any breaking change (see data.md).
SCHEMA_VERSION = 1

_EMPTY_F: Mapping[str, float] = MappingProxyType({})
_EMPTY_S: Mapping[str, str] = MappingProxyType({})
_EMPTY_P: Mapping[str, float | int | str | bool] = MappingProxyType({})
_EMPTY_T: Mapping[str, "TallyResult"] = MappingProxyType({})
_EMPTY_I: Mapping[str, int] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class DetectedPhotons:
    """Columnar record of detected photons; all arrays share length N.

    Units:
        arrival_time_s: s    (optical path time along the trajectory)
        path_length_m:  m
        weight:         dimensionless
        n_scatters:     count
        incidence_rad:  rad  (angle of arrival relative to the receiver normal)
    """

    arrival_time_s: FloatArray
    path_length_m: FloatArray
    weight: FloatArray
    n_scatters: IntArray
    incidence_rad: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrival_time_s", as_readonly(self.arrival_time_s))
        object.__setattr__(self, "path_length_m", as_readonly(self.path_length_m))
        object.__setattr__(self, "weight", as_readonly(self.weight))
        object.__setattr__(self, "n_scatters", as_readonly(self.n_scatters, dtype=np.int64))
        object.__setattr__(self, "incidence_rad", as_readonly(self.incidence_rad))


@dataclass(frozen=True, slots=True)
class Tallies:
    """Run-level scalar conservation tallies (weights dimensionless).

    ``extra`` is an escape hatch for new diagnostics (e.g. roulette losses) without
    editing this core type.
    """

    launched: int
    detected: int
    detected_weight: float
    absorbed_weight: float
    escaped_weight: float
    extra: Mapping[str, float] = _EMPTY_F


@dataclass(frozen=True, slots=True)
class TallyResult:
    """A binned (spatial/temporal) tally - the data path for outputs like power vs
    depth and the channel impulse response, accumulated online during transport.

    ``values`` has shape ``tuple(len(e) - 1 for e in edges)``.
    """

    name: str
    values: FloatArray
    edges: tuple[FloatArray, ...]
    unit: str
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", as_readonly(self.values))
        object.__setattr__(self, "edges", tuple(as_readonly(e) for e in self.edges))


@dataclass(frozen=True, slots=True)
class TransportOutput:
    """What a transport engine returns: detections, conservation tallies, and any
    requested binned tallies. Provenance is attached later by orchestration."""

    photons: DetectedPhotons
    tallies: Tallies
    binned: Mapping[str, TallyResult] = _EMPTY_T


@dataclass(frozen=True, slots=True)
class SeedTree:
    """Recorded RNG seed hierarchy for full reproducibility.

    ``streams`` maps a named sub-stream (e.g. "transport", "turbulence_field") to its
    seed, so every stochastic component of a run - including effect random fields -
    can be reproduced, not only the photon transport.
    """

    root_seed: int
    streams: Mapping[str, int] = _EMPTY_I


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to reproduce a run (see data.md, research-methodology.md).

    Provenance is typed: the sampling config and seed tree are first-class, and
    ``parameters`` is restricted to serializable scalar types rather than ``object``.
    """

    scenario: str
    medium_type: str
    optical_model: str
    effects: tuple[str, ...]
    wavelength_nm: float
    sampling: SamplingConfig
    seed_tree: SeedTree
    rng_impl: str
    timestamp_utc: str
    code_version: str
    library_versions: Mapping[str, str] = _EMPTY_S
    parameters: Mapping[str, float | int | str | bool] = _EMPTY_P


@dataclass(frozen=True, slots=True)
class RawResult:
    """A transport output plus its provenance, versioned for safe schema evolution."""

    output: TransportOutput
    metadata: RunMetadata
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class Axis:
    """A labelled independent axis for a curve-valued metric (unit-aware)."""

    name: str
    values: FloatArray
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", as_readonly(self.values))


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A computed metric with uncertainty and units (see metrics.md).

    Supports scalar, real-array, and complex-array values (e.g. the frequency
    response), one or more labelled axes, and per-element uncertainty.

    Uncertainty: standard deviation and a 95% confidence interval (``ci95_low`` /
    ``ci95_high`` may be scalars or per-element arrays), plus the sample count.
    """

    name: str
    value: float | FloatArray | ComplexArray
    unit: str
    n_samples: int
    std: float | FloatArray | None = None
    ci95_low: float | FloatArray | None = None
    ci95_high: float | FloatArray | None = None
    axes: tuple[Axis, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_value(self.value))
        object.__setattr__(self, "std", freeze_value(self.std))
        object.__setattr__(self, "ci95_low", freeze_value(self.ci95_low))
        object.__setattr__(self, "ci95_high", freeze_value(self.ci95_high))
