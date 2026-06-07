"""Tests for the Scenario I homogeneous medium."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core import EnvironmentalState, LocalOpticalState, Region
from uowc.core.ports import Acceleration, Domain, Medium, OpticalField
from uowc.media import DepthAverage, HomogeneousMedium, KamedaModel, SurfaceValue
from uowc.optics import HaltrinModel

WAVELENGTH_NM = 520.0


def model() -> HaltrinModel:
    return HaltrinModel(
        wavelength_nm=WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.002,
    )


def bounds() -> Region:
    return Region(lower=[-50.0, -50.0, -200.0], upper=[50.0, 50.0, 0.0])


def expected_attenuation(chlorophyll: float) -> float:
    return float(model().evaluate(EnvironmentalState(chlorophyll=chlorophyll), WAVELENGTH_NM).attenuation)


def build(chlorophyll: float = 0.5) -> HomogeneousMedium:
    return HomogeneousMedium.uniform(
        model=model(), chlorophyll=chlorophyll, wavelength_nm=WAVELENGTH_NM, bounds=bounds()
    )


def test_conforms_to_medium_capabilities() -> None:
    medium = build()
    assert isinstance(medium, Medium)
    assert isinstance(medium.field, OpticalField)
    assert isinstance(medium.domain, Domain)
    assert isinstance(medium.acceleration, Acceleration)


def test_extinction_is_spatially_constant() -> None:
    medium = build(0.5)
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -50.0], [10.0, 20.0, -150.0]])
    c = medium.field.extinction(pts)
    assert c.shape == (3,)
    assert np.allclose(c, c[0])                       # no depth dependence
    assert np.allclose(c, expected_attenuation(0.5))


def test_majorant_equals_constant_extinction_exactly() -> None:
    medium = build(0.5)
    expected = expected_attenuation(0.5)
    assert medium.acceleration.majorant(bounds()) == pytest.approx(expected)
    # exact and independent of the region queried
    small = Region(lower=[-1.0, -1.0, -1.0], upper=[1.0, 1.0, 0.0])
    assert medium.acceleration.majorant(small) == pytest.approx(expected)


def test_refractive_index_uniform_and_zero_gradient() -> None:
    medium = build()
    pts = np.zeros((4, 3))
    pts[:, 2] = np.linspace(-200.0, 0.0, 4)
    assert np.allclose(medium.field.refractive_index(pts), 1.34)
    assert np.allclose(medium.field.refractive_index_gradient(pts), 0.0)


def test_local_state_is_constant() -> None:
    medium = build(0.5)
    s1 = medium.field.local_state(np.array([0.0, 0.0, 0.0]))
    s2 = medium.field.local_state(np.array([0.0, 0.0, -150.0]))
    assert isinstance(s1, LocalOpticalState)
    assert float(s1.iop.attenuation) == pytest.approx(float(s2.iop.attenuation))
    assert s1.iop.wavelength_nm == WAVELENGTH_NM
    assert len(s1.iop.scattering.components) == 2


def test_domain_contains_and_bounds() -> None:
    medium = build()
    assert bool(medium.domain.contains(np.array([[0.0, 0.0, -100.0]]))[0]) is True
    assert not np.any(medium.domain.contains(np.array([[0.0, 0.0, -300.0]])))


def test_uniform_rejects_array_chlorophyll() -> None:
    with pytest.raises(ValueError):
        HomogeneousMedium.uniform(
            model=model(), chlorophyll=np.array([0.1, 0.2]), wavelength_nm=WAVELENGTH_NM, bounds=bounds()
        )


def test_from_iop_rejects_non_scalar_iop() -> None:
    array_iop = model().evaluate(EnvironmentalState(chlorophyll=np.array([0.1, 0.2])), WAVELENGTH_NM)
    with pytest.raises(ValueError):
        HomogeneousMedium.from_iop(iop=array_iop, bounds=bounds())


def test_from_iop_uses_given_iop() -> None:
    iop = model().evaluate(EnvironmentalState(chlorophyll=0.3), WAVELENGTH_NM)
    medium = HomogeneousMedium.from_iop(iop=iop, bounds=bounds())
    assert medium.acceleration.majorant(bounds()) == pytest.approx(float(iop.attenuation))


def test_from_profile_surface_rule_uses_surface_chlorophyll() -> None:
    profile = KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )
    medium = HomogeneousMedium.from_profile(
        profile=profile, model=model(), wavelength_nm=WAVELENGTH_NM, bounds=bounds(), rule=SurfaceValue()
    )
    surface_chl = float(profile.chlorophyll(np.array([0.0]))[0])
    got = medium.field.extinction(np.array([[0.0, 0.0, -100.0]]))[0]
    assert got == pytest.approx(expected_attenuation(surface_chl))


def test_from_profile_depth_average_differs_from_surface() -> None:
    profile = KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )
    surf = HomogeneousMedium.from_profile(
        profile=profile, model=model(), wavelength_nm=WAVELENGTH_NM, bounds=bounds(), rule=SurfaceValue()
    )
    avg = HomogeneousMedium.from_profile(
        profile=profile, model=model(), wavelength_nm=WAVELENGTH_NM, bounds=bounds(), rule=DepthAverage()
    )
    probe = np.array([[0.0, 0.0, -10.0]])
    # depth-average chlorophyll is higher (DCM) -> higher constant extinction
    assert avg.field.extinction(probe)[0] > surf.field.extinction(probe)[0]
