"""Reusable analysis routines over stored Parquet datasets."""

from __future__ import annotations

from uowc.analysis.comparison import (
    ScenarioDifference,
    compare_scenarios,
    paired_scenario_difference,
)
from uowc.analysis.convergence import ConvergencePoint, ConvergenceReport, assess_convergence
from uowc.analysis.sensitivity import SensitivityResult, compute_sensitivity, rank_by_influence
from uowc.analysis.uncertainty import (
    UncertaintyBudget,
    build_uncertainty_budget,
    spread_across_alternatives,
)

__all__ = [
    "ScenarioDifference",
    "paired_scenario_difference",
    "compare_scenarios",
    "ConvergencePoint",
    "ConvergenceReport",
    "assess_convergence",
    "UncertaintyBudget",
    "spread_across_alternatives",
    "build_uncertainty_budget",
    "SensitivityResult",
    "compute_sensitivity",
    "rank_by_influence",
]
