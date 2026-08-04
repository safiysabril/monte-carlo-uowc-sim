"""OFAT elasticity sensitivity and derived-coefficient guard."""

from __future__ import annotations

import pytest

from uowc.analysis.sensitivity import SensitivityResult, compute_sensitivity, rank_by_influence
from uowc.core.results import MetricValue


def test_elasticity_matches_hand_computed_value() -> None:
    """chlorophyll 0.3 -> 0.33 (+10%), received power 0.20 -> 0.22 (+10%): unit elasticity."""
    result = compute_sensitivity(
        metric_name="received_power_fraction",
        parameter_name="chlorophyll",
        baseline_value=0.3,
        baseline_metric=0.20,
        perturbed_value=0.33,
        perturbed_metric=0.22,
    )
    assert result.relative_input_change == pytest.approx(0.10)
    assert result.relative_output_change == pytest.approx(0.10)
    assert result.elasticity == pytest.approx(1.0)


def test_elasticity_is_dimensionless_and_scale_free() -> None:
    """A parameter in radians and one in metres should be comparable by elasticity."""
    fov = compute_sensitivity(
        metric_name="received_power_fraction",
        parameter_name="fov_rad",
        baseline_value=0.5,
        baseline_metric=0.20,
        perturbed_value=0.6,  # +20%
        perturbed_metric=0.24,  # +20%
    )
    aperture = compute_sensitivity(
        metric_name="received_power_fraction",
        parameter_name="aperture_radius_m",
        baseline_value=0.05,
        baseline_metric=0.20,
        perturbed_value=0.06,  # +20%
        perturbed_metric=0.24,  # +20%
    )
    assert fov.elasticity == pytest.approx(aperture.elasticity)


def test_negative_elasticity_for_an_inverse_relationship() -> None:
    result = compute_sensitivity(
        metric_name="received_power_fraction",
        parameter_name="range_m",
        baseline_value=10.0,
        baseline_metric=0.20,
        perturbed_value=20.0,  # +100%
        perturbed_metric=0.05,  # -75%
    )
    assert result.elasticity < 0.0


def test_accepts_metric_value_inputs() -> None:
    baseline = MetricValue(name="m", value=0.20, unit="fraction", n_samples=100)
    perturbed = MetricValue(name="m", value=0.22, unit="fraction", n_samples=100)
    result = compute_sensitivity(
        metric_name="m",
        parameter_name="chlorophyll",
        baseline_value=0.3,
        baseline_metric=baseline,
        perturbed_value=0.33,
        perturbed_metric=perturbed,
    )
    assert result.baseline_metric == pytest.approx(0.20)
    assert result.perturbed_metric == pytest.approx(0.22)


def test_rejects_array_valued_metric_value() -> None:
    import numpy as np

    curve = MetricValue(name="m", value=np.array([1.0, 2.0]), unit="", n_samples=10)
    with pytest.raises(TypeError, match="scalar"):
        compute_sensitivity(
            metric_name="m",
            parameter_name="chlorophyll",
            baseline_value=0.3,
            baseline_metric=curve,
            perturbed_value=0.33,
            perturbed_metric=curve,
        )


@pytest.mark.parametrize(
    "name", ["a", "b", "c", "absorption", "scattering", "attenuation", "omega_0"]
)
def test_rejects_derived_coefficients_by_default(name: str) -> None:
    with pytest.raises(ValueError, match="model-derived IOP"):
        compute_sensitivity(
            metric_name="m",
            parameter_name=name,
            baseline_value=0.1,
            baseline_metric=0.2,
            perturbed_value=0.11,
            perturbed_metric=0.22,
        )


def test_derived_coefficient_allowed_as_explicit_what_if_probe() -> None:
    result = compute_sensitivity(
        metric_name="m",
        parameter_name="a",
        baseline_value=0.1,
        baseline_metric=0.2,
        perturbed_value=0.11,
        perturbed_metric=0.22,
        allow_derived_coefficient=True,
    )
    assert result.is_model_free_probe
    assert result.elasticity == pytest.approx(1.0)


def test_ordinary_input_is_not_flagged_as_a_model_free_probe() -> None:
    result = compute_sensitivity(
        metric_name="m",
        parameter_name="chlorophyll",
        baseline_value=0.3,
        baseline_metric=0.2,
        perturbed_value=0.33,
        perturbed_metric=0.22,
    )
    assert not result.is_model_free_probe


def test_rejects_zero_baseline_value() -> None:
    with pytest.raises(ZeroDivisionError, match="baseline_value"):
        compute_sensitivity(
            metric_name="m",
            parameter_name="index_above",
            baseline_value=0.0,
            baseline_metric=0.2,
            perturbed_value=0.001,
            perturbed_metric=0.21,
        )


def test_rejects_zero_baseline_metric() -> None:
    with pytest.raises(ZeroDivisionError, match="baseline_metric"):
        compute_sensitivity(
            metric_name="m",
            parameter_name="chlorophyll",
            baseline_value=0.3,
            baseline_metric=0.0,
            perturbed_value=0.33,
            perturbed_metric=0.01,
        )


def test_rejects_equal_baseline_and_perturbed_values() -> None:
    with pytest.raises(ValueError, match="must differ"):
        compute_sensitivity(
            metric_name="m",
            parameter_name="chlorophyll",
            baseline_value=0.3,
            baseline_metric=0.2,
            perturbed_value=0.3,
            perturbed_metric=0.2,
        )


def test_rank_by_influence_orders_by_absolute_elasticity_descending() -> None:
    weak = compute_sensitivity(
        metric_name="m",
        parameter_name="salinity",
        baseline_value=35.0,
        baseline_metric=0.20,
        perturbed_value=38.5,  # +10%
        perturbed_metric=0.201,  # +0.5%
    )
    strong_negative = compute_sensitivity(
        metric_name="m",
        parameter_name="range_m",
        baseline_value=10.0,
        baseline_metric=0.20,
        perturbed_value=11.0,  # +10%
        perturbed_metric=0.10,  # -50%
    )
    moderate = compute_sensitivity(
        metric_name="m",
        parameter_name="chlorophyll",
        baseline_value=0.3,
        baseline_metric=0.20,
        perturbed_value=0.33,  # +10%
        perturbed_metric=0.22,  # +10%
    )

    ranked = rank_by_influence([weak, strong_negative, moderate])

    assert ranked[0] is strong_negative
    assert ranked[1] is moderate
    assert ranked[2] is weak
    assert isinstance(ranked[0], SensitivityResult)
