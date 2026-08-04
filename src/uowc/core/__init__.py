"""Core domain types and ports (Protocols).

Depends only on numpy. Re-exports the public value objects and the Protocol
interfaces so callers can ``from uowc.core import Medium, RawResult`` etc.
"""
from __future__ import annotations

from uowc.core.config import SamplingConfig
from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import BoundaryOutcome, Receiver, Region, Source
from uowc.core.ports import (
    Acceleration,
    Boundary,
    ComparativeMetric,
    Domain,
    Medium,
    Metric,
    OpticalEffect,
    OpticalField,
    OpticalPropertyModel,
    ParameterEffect,
    PhaseFunction,
    RefractiveEffect,
    Rng,
    TransportEngine,
)
from uowc.core.results import (
    SCHEMA_VERSION,
    Axis,
    DetectedPhotons,
    MetricValue,
    RawResult,
    RunMetadata,
    SeedTree,
    Tallies,
    TallyResult,
    TransportOutput,
)
from uowc.core.state import IOP, LocalOpticalState, Scattering, ScatteringComponent
from uowc.core.units import (
    BoolArray,
    ComplexArray,
    FloatArray,
    FloatOrArray,
    IntArray,
    Vector3,
)

__all__ = [
    # units / aliases
    "FloatArray",
    "IntArray",
    "ComplexArray",
    "BoolArray",
    "Vector3",
    "FloatOrArray",
    # config
    "SamplingConfig",
    # value objects
    "EnvironmentalState",
    "ScatteringComponent",
    "Scattering",
    "IOP",
    "LocalOpticalState",
    "Source",
    "Receiver",
    "Region",
    "BoundaryOutcome",
    "DetectedPhotons",
    "Tallies",
    "TallyResult",
    "TransportOutput",
    "SeedTree",
    "RunMetadata",
    "RawResult",
    "Axis",
    "MetricValue",
    "SCHEMA_VERSION",
    # ports (interfaces)
    "OpticalPropertyModel",
    "PhaseFunction",
    "ParameterEffect",
    "OpticalEffect",
    "RefractiveEffect",
    "OpticalField",
    "Domain",
    "Acceleration",
    "Medium",
    "Boundary",
    "Rng",
    "TransportEngine",
    "Metric",
    "ComparativeMetric",
]
