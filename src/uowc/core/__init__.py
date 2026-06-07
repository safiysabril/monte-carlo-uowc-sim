"""Core domain types and ports (Protocols).

Depends only on numpy. Re-exports the public value objects and the Protocol
interfaces so callers can ``from uowc.core import Medium, RawResult`` etc.
"""
from __future__ import annotations

from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import Receiver, Source
from uowc.core.ports import (
    EnvironmentalEffect,
    Medium,
    Metric,
    OpticalPropertyModel,
    PhaseFunction,
    Rng,
    TransportEngine,
)
from uowc.core.results import (
    DetectedPhotons,
    MetricValue,
    RawResult,
    RunMetadata,
    Tallies,
)
from uowc.core.state import IOP, LocalOpticalState
from uowc.core.units import FloatArray, FloatOrArray, IntArray, Vector3

__all__ = [
    # units / aliases
    "FloatArray",
    "IntArray",
    "Vector3",
    "FloatOrArray",
    # value objects
    "EnvironmentalState",
    "IOP",
    "LocalOpticalState",
    "Source",
    "Receiver",
    "DetectedPhotons",
    "Tallies",
    "RunMetadata",
    "RawResult",
    "MetricValue",
    # ports (interfaces)
    "OpticalPropertyModel",
    "PhaseFunction",
    "EnvironmentalEffect",
    "Medium",
    "Rng",
    "TransportEngine",
    "Metric",
]
