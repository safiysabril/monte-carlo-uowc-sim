"""IOP derived c/albedo; LocalOpticalState immutability.

No separate backscatter-ratio field exists on these types (state.py's own docstring:
directionality lives entirely in each scattering population's phase function, "a
single source of truth" - no separate backscatter field to bound here); this module
tests what the type actually exposes.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from uowc.core.state import IOP, LocalOpticalState, Scattering, ScatteringComponent


class _Isotropic:
    asymmetry = 0.0

    def sample_cos_theta(self, u):
        return 2.0 * np.asarray(u, dtype=float) - 1.0

    def value(self, cos_theta):
        return np.full(np.asarray(cos_theta, dtype=float).shape, 1.0 / (4.0 * np.pi))


# --- ScatteringComponent / Scattering --------------------------------------------------


def test_scattering_component_is_frozen() -> None:
    component = ScatteringComponent(coefficient=0.3, phase=_Isotropic())
    with pytest.raises(FrozenInstanceError):
        component.coefficient = 0.5  # type: ignore[misc]


def test_scattering_total_coefficient_sums_components() -> None:
    mixture = Scattering(
        (
            ScatteringComponent(0.3, _Isotropic()),
            ScatteringComponent(0.1, _Isotropic()),
        )
    )
    assert mixture.coefficient == pytest.approx(0.4)


def test_scattering_with_no_components_has_zero_coefficient() -> None:
    assert Scattering(()).coefficient == 0.0


def test_scattering_coefficient_is_vectorized() -> None:
    b = np.array([0.1, 0.2, 0.3])
    mixture = Scattering((ScatteringComponent(b, _Isotropic()),))
    np.testing.assert_allclose(mixture.coefficient, b)


# --- IOP -----------------------------------------------------------------------------


def test_attenuation_is_absorption_plus_scattering() -> None:
    iop = IOP(
        absorption=0.2,
        scattering=Scattering((ScatteringComponent(0.5, _Isotropic()),)),
        wavelength_nm=500.0,
    )
    assert iop.attenuation == pytest.approx(0.7)
    assert iop.scattering_coefficient == pytest.approx(0.5)


def test_single_scattering_albedo_is_b_over_c() -> None:
    iop = IOP(
        absorption=0.25,
        scattering=Scattering((ScatteringComponent(0.75, _Isotropic()),)),
        wavelength_nm=500.0,
    )
    assert iop.single_scattering_albedo == pytest.approx(0.75)


def test_single_scattering_albedo_is_zero_when_extinction_is_zero() -> None:
    iop = IOP(absorption=0.0, scattering=Scattering(()), wavelength_nm=500.0)
    assert iop.attenuation == 0.0
    assert iop.single_scattering_albedo == 0.0


def test_single_scattering_albedo_is_vectorized() -> None:
    a = np.array([0.1, 0.0, 0.5])
    b = np.array([0.3, 0.0, 0.5])
    iop = IOP(
        absorption=a,
        scattering=Scattering((ScatteringComponent(b, _Isotropic()),)),
        wavelength_nm=500.0,
    )
    albedo = iop.single_scattering_albedo
    np.testing.assert_allclose(albedo, [0.75, 0.0, 0.5])


def test_iop_is_tagged_with_its_wavelength() -> None:
    iop = IOP(absorption=0.1, scattering=Scattering(()), wavelength_nm=650.0)
    assert iop.wavelength_nm == 650.0


def test_iop_is_frozen() -> None:
    iop = IOP(absorption=0.1, scattering=Scattering(()), wavelength_nm=500.0)
    with pytest.raises(FrozenInstanceError):
        iop.absorption = 0.2  # type: ignore[misc]


# --- LocalOpticalState -----------------------------------------------------------------


def test_local_optical_state_is_frozen() -> None:
    iop = IOP(absorption=0.1, scattering=Scattering(()), wavelength_nm=500.0)
    state = LocalOpticalState(iop=iop, refractive_index=1.34)
    with pytest.raises(FrozenInstanceError):
        state.refractive_index = 1.4  # type: ignore[misc]


def test_local_optical_state_gradient_defaults_to_none() -> None:
    iop = IOP(absorption=0.1, scattering=Scattering(()), wavelength_nm=500.0)
    state = LocalOpticalState(iop=iop, refractive_index=1.34)
    assert state.refractive_index_gradient is None


def test_local_optical_state_gradient_is_frozen_when_provided() -> None:
    iop = IOP(absorption=0.1, scattering=Scattering(()), wavelength_nm=500.0)
    gradient = np.array([0.0, 0.0, 1e-4])
    state = LocalOpticalState(iop=iop, refractive_index=1.34, refractive_index_gradient=gradient)
    assert state.refractive_index_gradient is not None
    with pytest.raises(ValueError, match="read-only"):
        state.refractive_index_gradient[0] = 1.0
