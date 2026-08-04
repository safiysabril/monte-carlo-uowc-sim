"""Orchestration engine / composition root.

The only package that imports concretes from every layer and wires them together.
Physics models, profiles and effects are injected via :class:`ExperimentConfig`.
"""
from __future__ import annotations

from uowc.experiments.campaign import (
    Campaign,
    CampaignResult,
    ParameterMatrix,
    ParameterPoint,
    ProvenanceNode,
    replicate_seeds,
)
from uowc.experiments.config import ExperimentConfig
from uowc.experiments.registry import (
    available_effects,
    available_metrics,
    available_optical_models,
    available_profiles,
    build_effect,
    build_effects,
    build_metric,
    build_metric_pipeline,
    build_metrics,
    build_optical_model,
    build_profile,
)
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
    "build_optical_model",
    "available_optical_models",
    "build_profile",
    "available_profiles",
    "build_metric",
    "build_metrics",
    "build_metric_pipeline",
    "available_metrics",
    "build_effect",
    "build_effects",
    "available_effects",
    "Campaign",
    "CampaignResult",
    "ParameterMatrix",
    "ParameterPoint",
    "ProvenanceNode",
    "replicate_seeds",
]
