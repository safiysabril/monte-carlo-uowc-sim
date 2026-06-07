"""Result schemas: the transport -> metrics -> storage seam."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from uowc.core.units import FloatArray, IntArray

__all__ = [
    "DetectedPhotons",
    "Tallies",
    "RunMetadata",
    "RawResult",
    "MetricValue",
]


@dataclass(frozen=True, slots=True)
class DetectedPhotons:
    """Columnar record of detected photons; all arrays share length N.

    Parquet-ready. Storing raw detections (not only summaries) keeps every metric
    reproducible from stored output (see data.md).

    Units:
        arrival_time_s: s    (optical path time along the photon trajectory)
        path_length_m:  m
        weight:         dimensionless (statistical weight at detection)
        n_scatters:     count
        incidence_rad:  rad  (angle of arrival relative to the receiver normal)
    """

    arrival_time_s: FloatArray
    path_length_m: FloatArray
    weight: FloatArray
    n_scatters: IntArray
    incidence_rad: FloatArray


@dataclass(frozen=True, slots=True)
class Tallies:
    """Run-level scalar tallies (weights dimensionless) for conservation checks."""

    launched: int
    detected: int
    detected_weight: float
    absorbed_weight: float
    escaped_weight: float


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to reproduce a run (see data.md, research-methodology.md).

    Captures the identity of every choice that can affect results, including code
    and library versions, so a result is reproducible from metadata alone.
    """

    scenario: str
    medium_type: str
    optical_model: str
    effects: tuple[str, ...]
    wavelength_nm: float
    seed: int
    rng_impl: str
    parameters: Mapping[str, object]
    timestamp_utc: str
    code_version: str
    library_versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RawResult:
    """Complete raw output of a single transport run."""

    photons: DetectedPhotons
    tallies: Tallies
    metadata: RunMetadata


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A computed metric with uncertainty and units (see metrics.md).

    Scalar metrics set ``value`` to a float and leave ``axis`` ``None``. Curve
    metrics (CIR, frequency response, power-vs-depth) set ``value`` to an array and
    populate ``axis`` / ``axis_name`` / ``axis_unit`` so downstream figures stay
    unit-aware without re-deriving the independent variable.

    Uncertainty (per metrics.md): standard deviation, 95% confidence interval, and
    the sample count behind the estimate.
    """

    name: str
    value: float | FloatArray
    unit: str
    n_samples: int
    std: float | FloatArray | None = None
    ci95: tuple[float, float] | None = None
    axis: FloatArray | None = None
    axis_name: str | None = None
    axis_unit: str | None = None
