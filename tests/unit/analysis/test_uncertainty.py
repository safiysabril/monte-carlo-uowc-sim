"""Uncertainty-source decomposition: MC vs model vs environmental vs measurement."""
from __future__ import annotations

import pytest

from uowc.analysis.uncertainty import (
    UncertaintyBudget,
    build_uncertainty_budget,
    spread_across_alternatives,
)
from uowc.core.results import MetricValue


def _mc(std: float, value: float = 0.3) -> MetricValue:
    return MetricValue(name="m", value=value, unit="fraction", n_samples=1000, std=std)


# --- spread_across_alternatives ----------------------------------------------------


def test_spread_is_the_range_not_the_std() -> None:
    """The whole point (module docstring): a handful of named alternatives is not a
    sample from a distribution, so range - not standard deviation - is reported."""
    values = [0.30, 0.32, 0.28]
    assert spread_across_alternatives(values) == pytest.approx(0.04)


def test_spread_of_identical_alternatives_is_zero() -> None:
    assert spread_across_alternatives([0.3, 0.3, 0.3]) == pytest.approx(0.0)


def test_spread_requires_at_least_two_alternatives() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        spread_across_alternatives([0.3])


# --- build_uncertainty_budget --------------------------------------------------------


def test_budget_with_only_monte_carlo_assessed() -> None:
    budget = build_uncertainty_budget(metric_name="received_power_fraction", monte_carlo=_mc(0.01))
    assert budget.monte_carlo_std == pytest.approx(0.01)
    assert budget.model_spread is None
    assert budget.environmental_spread is None
    assert budget.measurement_std is None
    assert budget.dominant_source == "monte_carlo"
    assert budget.monte_carlo_dominates_model_uncertainty is None  # unassessed, not "safe"


def test_budget_rejects_monte_carlo_metric_without_std() -> None:
    bare = MetricValue(name="m", value=0.3, unit="fraction", n_samples=10)
    with pytest.raises(ValueError, match="scalar std"):
        build_uncertainty_budget(metric_name="m", monte_carlo=bare)


def test_full_budget_all_four_sources() -> None:
    budget = build_uncertainty_budget(
        metric_name="received_power_fraction",
        monte_carlo=_mc(0.005),
        model_alternatives=[0.30, 0.22, 0.34],  # e.g. HG vs Fournier-Forand vs Petzold
        environmental_alternatives=[0.30, 0.28],  # e.g. two chlorophyll profile realizations
        measurement_std=0.0006,
    )
    assert isinstance(budget, UncertaintyBudget)
    assert budget.monte_carlo_std == pytest.approx(0.005)
    assert budget.model_spread == pytest.approx(0.12)
    assert budget.environmental_spread == pytest.approx(0.02)
    assert budget.measurement_std == pytest.approx(0.0006)


def test_precision_without_accuracy_trap_is_detected() -> None:
    """A tight MC interval around a result whose model choice swings it wildly."""
    budget = build_uncertainty_budget(
        metric_name="m", monte_carlo=_mc(0.001), model_alternatives=[0.30, 0.15]
    )
    assert budget.monte_carlo_dominates_model_uncertainty is True
    assert budget.dominant_source == "model"


def test_monte_carlo_can_be_the_dominant_source_too() -> None:
    budget = build_uncertainty_budget(
        metric_name="m", monte_carlo=_mc(0.05), model_alternatives=[0.30, 0.31]
    )
    assert budget.monte_carlo_dominates_model_uncertainty is False
    assert budget.dominant_source == "monte_carlo"


def test_dominant_source_is_none_when_nothing_assessed() -> None:
    budget = build_uncertainty_budget(metric_name="m")
    assert budget.dominant_source is None
    assert budget.monte_carlo_std is None


def test_rejects_negative_measurement_std() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_uncertainty_budget(metric_name="m", measurement_std=-0.1)


def test_measurement_uncertainty_can_dominate() -> None:
    budget = build_uncertainty_budget(
        metric_name="pure_water_absorption", monte_carlo=_mc(0.0001), measurement_std=0.0006
    )
    assert budget.dominant_source == "measurement"
