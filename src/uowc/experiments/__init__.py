"""Orchestration engine / composition root.

The only package that imports concretes from every layer and wires them together.
Physics models, profiles and effects are injected via :class:`ExperimentConfig`.
"""
from __future__ import annotations

from uowc.experiments.config import ExperimentConfig
from uowc.experiments.runner import ScenarioResult, ScenarioRunner
from uowc.experiments.scenarios import Scenario, build_medium, effect_names, medium_type

__all__ = [
    "ExperimentConfig",
    "Scenario",
    "build_medium",
    "medium_type",
    "effect_names",
    "ScenarioResult",
    "ScenarioRunner",
]
