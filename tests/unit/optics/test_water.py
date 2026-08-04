"""Pure-water baselines vs reference tables."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from uowc.optics.water import PureWaterAbsorption, PureWaterScattering

# --- PureWaterAbsorption (Pope & Fry 1997) ----------------------------------------


def test_absorption_minimum_matches_pope_and_fry_headline_value() -> None:
    """Pope & Fry (1997) report the pure-water absorption minimum as
    0.0044 +/- 0.0006 m^-1 at 418 nm - the table's own internal consistency check."""
    absorption = PureWaterAbsorption()
    wavelengths = np.linspace(410.0, 425.0, 200)
    values = absorption.at(wavelengths)
    minimum = float(np.min(values))
    argmin_wl = float(wavelengths[np.argmin(values)])
    assert minimum == pytest.approx(0.0044, abs=0.0006)
    assert argmin_wl == pytest.approx(418.0, abs=2.0)


def test_absorption_is_monotone_increasing_away_from_the_minimum() -> None:
    absorption = PureWaterAbsorption()
    blue = absorption.at(400.0)
    minimum = absorption.at(418.0)
    red = absorption.at(600.0)
    assert blue > minimum
    assert red > minimum
    assert red > blue  # red water absorbs far more strongly than blue


def test_absorption_scalar_in_scalar_out() -> None:
    value = PureWaterAbsorption().at(500.0)
    assert isinstance(value, float)


def test_absorption_vectorized() -> None:
    values = PureWaterAbsorption().at(np.array([400.0, 500.0, 600.0]))
    assert values.shape == (3,)
    assert np.all(values > 0.0)


def test_absorption_valid_range_matches_the_tabulated_extent() -> None:
    assert PureWaterAbsorption().valid_range_nm == (400.0, 600.0)


@pytest.mark.parametrize("wavelength_nm", [399.9, 600.1, 300.0, 700.0])
def test_absorption_refuses_to_extrapolate(wavelength_nm: float) -> None:
    with pytest.raises(ValueError):
        PureWaterAbsorption().at(wavelength_nm)


def test_absorption_matches_illustrative_haltrin_placeholder_order_of_magnitude() -> None:
    """Sanity cross-check: HaltrinModel.illustrative() uses 0.045 m^-1 at 520 nm as a
    hand-picked placeholder (haltrin.py). The real Pope & Fry value should be the same
    order of magnitude, confirming the placeholder wasn't wildly unphysical."""
    real = PureWaterAbsorption().at(520.0)
    assert real == pytest.approx(0.045, rel=0.2)


# --- PureWaterScattering (Morel 1974 / Zhang, Hu & He 2009) -----------------------


def test_scattering_equals_reference_at_the_reference_wavelength() -> None:
    scattering = PureWaterScattering(
        reference_scattering_m_inv=0.0029, reference_wavelength_nm=500.0
    )
    assert scattering.at(500.0) == pytest.approx(0.0029)


def test_scattering_follows_inverse_fourth_power_law() -> None:
    scattering = PureWaterScattering(
        reference_scattering_m_inv=0.0029, reference_wavelength_nm=500.0
    )
    doubled_wavelength = scattering.at(1000.0)
    expected_ratio = (500.0 / 1000.0) ** 4.32
    assert doubled_wavelength / 0.0029 == pytest.approx(expected_ratio)


def test_scattering_decreases_with_wavelength() -> None:
    scattering = PureWaterScattering(
        reference_scattering_m_inv=0.0029, reference_wavelength_nm=500.0
    )
    assert scattering.at(400.0) > scattering.at(500.0) > scattering.at(600.0)


def test_scattering_custom_exponent_is_respected() -> None:
    # Above the reference wavelength the ratio (ref/wl) < 1, so a *larger* exponent
    # pushes the value further below the reference (falls off faster with lambda).
    shallower = PureWaterScattering(
        reference_scattering_m_inv=0.003, reference_wavelength_nm=500.0, exponent=4.286
    )
    steeper = PureWaterScattering(
        reference_scattering_m_inv=0.003, reference_wavelength_nm=500.0, exponent=4.306
    )
    assert steeper.at(600.0) < shallower.at(600.0)


def test_scattering_scalar_in_scalar_out() -> None:
    value = PureWaterScattering(
        reference_scattering_m_inv=0.0029, reference_wavelength_nm=500.0
    ).at(450.0)
    assert isinstance(value, float)


def test_scattering_requires_a_reference_magnitude() -> None:
    """No hardcoded default magnitude is shipped (see module docstring) - the caller
    must supply one, so the constructor cannot be called with zero arguments."""
    params = inspect.signature(PureWaterScattering).parameters
    assert params["reference_scattering_m_inv"].default is inspect.Parameter.empty
    assert params["reference_wavelength_nm"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_scattering_m_inv": -0.001, "reference_wavelength_nm": 500.0},
        {"reference_scattering_m_inv": 0.003, "reference_wavelength_nm": 0.0},
        {"reference_scattering_m_inv": 0.003, "reference_wavelength_nm": 500.0, "exponent": 0.0},
    ],
)
def test_scattering_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PureWaterScattering(**kwargs)


def test_scattering_rejects_nonpositive_wavelength_query() -> None:
    scattering = PureWaterScattering(
        reference_scattering_m_inv=0.0029, reference_wavelength_nm=500.0
    )
    with pytest.raises(ValueError):
        scattering.at(0.0)
