"""Tests for the Haltrin chlorophyll-based optical-property model."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core.environment import EnvironmentalState
from uowc.core.ports import OpticalPropertyModel
from uowc.optics.haltrin import HaltrinModel


def make() -> HaltrinModel:
    # Explicit coefficients so expected values are exact (independent of defaults).
    return HaltrinModel(
        wavelength_nm=520.0,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.002,
    )


def test_conforms_to_optical_property_model() -> None:
    assert isinstance(make(), OpticalPropertyModel)
    assert make().name == "haltrin"


def test_absorption_formula() -> None:
    iop = make().evaluate(EnvironmentalState(chlorophyll=1.0), 520.0)
    # a = a_w + a_c* * C^0.602 = 0.1 + 0.05 * 1 = 0.15
    assert iop.absorption == pytest.approx(0.15)
    assert iop.wavelength_nm == 520.0


def test_scattering_is_two_population_mixture() -> None:
    iop = make().evaluate(EnvironmentalState(chlorophyll=1.0), 520.0)
    comps = iop.scattering.components
    assert len(comps) == 2
    # water (near-symmetric) then particles (forward-peaked)
    assert comps[0].phase.asymmetry == 0.0
    assert comps[1].phase.asymmetry == pytest.approx(0.924)
    assert comps[0].coefficient == pytest.approx(0.002)
    # b_p = 0.30 * (550/520) * 1^0.62
    assert comps[1].coefficient == pytest.approx(0.30 * (550.0 / 520.0))


def test_attenuation_and_albedo() -> None:
    iop = make().evaluate(EnvironmentalState(chlorophyll=1.0), 520.0)
    b_p = 0.30 * (550.0 / 520.0)
    expected_b = 0.002 + b_p
    expected_c = 0.15 + expected_b
    assert iop.scattering_coefficient == pytest.approx(expected_b)
    assert iop.attenuation == pytest.approx(expected_c)
    assert iop.single_scattering_albedo == pytest.approx(expected_b / expected_c)


def test_pure_water_limit_at_zero_chlorophyll() -> None:
    iop = make().evaluate(EnvironmentalState(chlorophyll=0.0), 520.0)
    assert iop.absorption == pytest.approx(0.1)            # a_w only
    assert iop.scattering_coefficient == pytest.approx(0.002)  # b_w only (b_p = 0)


def test_vectorized_over_chlorophyll() -> None:
    chl = np.array([0.0, 0.5, 1.0, 4.0])
    iop = make().evaluate(EnvironmentalState(chlorophyll=chl), 520.0)
    a = np.asarray(iop.absorption)
    c = np.asarray(iop.attenuation)
    assert a.shape == chl.shape
    # absorption and attenuation increase monotonically with chlorophyll
    assert np.all(np.diff(a) > 0)
    assert np.all(np.diff(c) > 0)


def test_negative_chlorophyll_is_clamped() -> None:
    iop = make().evaluate(EnvironmentalState(chlorophyll=-5.0), 520.0)
    assert iop.absorption == pytest.approx(0.1)  # treated as C = 0


def test_wavelength_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        make().evaluate(EnvironmentalState(chlorophyll=1.0), 600.0)


@pytest.mark.parametrize(
    "override",
    [
        {"wavelength_nm": 0.0},
        {"pure_water_absorption_m_inv": -0.1},
        {"chlorophyll_specific_absorption_m2_mg": -0.1},
        {"pure_water_scattering_m_inv": -0.1},
        {"particle_scattering_coeff": -0.1},
    ],
)
def test_invalid_parameters_raise(override: dict[str, float]) -> None:
    params = {
        "wavelength_nm": 520.0,
        "pure_water_absorption_m_inv": 0.1,
        "chlorophyll_specific_absorption_m2_mg": 0.05,
        "pure_water_scattering_m_inv": 0.002,
    }
    params.update(override)
    with pytest.raises(ValueError):
        HaltrinModel(**params)


def test_illustrative_constructor() -> None:
    model = HaltrinModel.illustrative(520.0)
    assert isinstance(model, OpticalPropertyModel)
    assert model.wavelength_nm == 520.0
    iop = model.evaluate(EnvironmentalState(chlorophyll=0.3), 520.0)
    assert float(iop.attenuation) > 0.0
