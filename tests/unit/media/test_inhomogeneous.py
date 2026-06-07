"""Tests for the depth-dependent medium (Scenario II) and effect composition (III)."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core import EnvironmentalState, LocalOpticalState, Region
from uowc.core.ports import Acceleration, Domain, Medium, OpticalField
from uowc.effects import TurbulenceEffect
from uowc.media import InhomogeneousMedium, KamedaModel
from uowc.optics import HaltrinModel

WAVELENGTH_NM = 520.0


def build() -> tuple[InhomogeneousMedium, KamedaModel, HaltrinModel, Region]:
    profile = KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )
    model = HaltrinModel(
        wavelength_nm=WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.002,
    )
    bounds = Region(lower=[-50.0, -50.0, -200.0], upper=[50.0, 50.0, 0.0])
    medium = InhomogeneousMedium.scenario_ii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds
    )
    return medium, profile, model, bounds


def _turbulence() -> TurbulenceEffect:
    return TurbulenceEffect.isotropic(
        rms_fluctuation=1e-3, correlation_length_m=5.0, n_modes=128, seed=7
    )


# --- Scenario II (no effects) ------------------------------------------------


def test_conforms_to_medium_capabilities() -> None:
    medium, *_ = build()
    assert isinstance(medium, Medium)
    assert isinstance(medium.field, OpticalField)
    assert isinstance(medium.domain, Domain)
    assert isinstance(medium.acceleration, Acceleration)


def test_extinction_matches_profile_then_model() -> None:
    medium, profile, model, _ = build()
    pts = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -80.0], [0.0, 0.0, -150.0]])
    chl = profile.chlorophyll(-pts[:, 2])
    expected = np.asarray(model.evaluate(EnvironmentalState(chlorophyll=chl), WAVELENGTH_NM).attenuation)
    assert np.allclose(medium.field.extinction(pts), expected)


def test_extinction_is_depth_dependent_and_peaks_at_dcm() -> None:
    medium, *_ = build()
    surface = medium.field.extinction(np.array([[0.0, 0.0, 0.0]]))[0]
    peak = medium.field.extinction(np.array([[0.0, 0.0, -80.0]]))[0]   # DCM at 80 m
    deep = medium.field.extinction(np.array([[0.0, 0.0, -200.0]]))[0]
    assert peak > surface and peak > deep


def test_refractive_index_uniform_and_zero_gradient() -> None:
    medium, *_ = build()
    pts = np.zeros((5, 3))
    pts[:, 2] = np.linspace(-200.0, 0.0, 5)
    n = medium.field.refractive_index(pts)
    assert n.shape == (5,)
    assert np.allclose(n, 1.34)
    assert np.allclose(medium.field.refractive_index_gradient(pts), 0.0)


def test_local_state_at_a_point() -> None:
    medium, profile, model, _ = build()
    state = medium.field.local_state(np.array([0.0, 0.0, -80.0]))
    assert isinstance(state, LocalOpticalState)
    assert state.refractive_index == pytest.approx(1.34)
    assert state.iop.wavelength_nm == WAVELENGTH_NM
    assert len(state.iop.scattering.components) == 2
    chl = float(profile.chlorophyll(np.array([80.0]))[0])
    expected = float(model.evaluate(EnvironmentalState(chlorophyll=chl), WAVELENGTH_NM).attenuation)
    assert float(state.iop.attenuation) == pytest.approx(expected)


def test_domain_contains_and_bounds() -> None:
    medium, _, _, bounds = build()
    inside = np.array([[0.0, 0.0, -100.0]])
    outside = np.array([[0.0, 0.0, -300.0], [200.0, 0.0, -10.0]])
    assert bool(medium.domain.contains(inside)[0]) is True
    assert not np.any(medium.domain.contains(outside))
    assert medium.domain.bounds() is bounds


def test_majorant_matches_peak_extinction() -> None:
    medium, _, _, bounds = build()
    c_peak = medium.field.extinction(np.array([[0.0, 0.0, -80.0]]))[0]
    majorant = medium.acceleration.majorant(bounds)
    assert majorant == pytest.approx(c_peak, rel=1e-3)
    zs = np.linspace(-200.0, 0.0, 37)
    pts = np.column_stack([np.zeros(37), np.zeros(37), zs])
    assert np.all(medium.field.extinction(pts) <= majorant + 1e-12)


def test_majorant_safety_factor_bounds_fine_grid() -> None:
    profile = KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )
    model = HaltrinModel.illustrative(WAVELENGTH_NM)
    bounds = Region(lower=[-50.0, -50.0, -200.0], upper=[50.0, 50.0, 0.0])
    medium = InhomogeneousMedium.scenario_ii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds, majorant_safety=1.05
    )
    zs = np.linspace(-200.0, 0.0, 9999)
    pts = np.column_stack([np.zeros(9999), np.zeros(9999), zs])
    assert medium.acceleration.majorant(bounds) >= medium.field.extinction(pts).max()


# --- Scenario III (composed effects) -----------------------------------------


def test_scenario_iii_turbulence_perturbs_refractive_index() -> None:
    _, profile, model, bounds = build()
    medium = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds, effects=[_turbulence()]
    )
    pts = np.zeros((50, 3))
    pts[:, 2] = np.linspace(-150.0, -10.0, 50)
    n = medium.field.refractive_index(pts)
    assert not np.allclose(n, 1.34)                       # turbulence perturbs n(x)
    assert n.mean() == pytest.approx(1.34, abs=1e-3)      # zero-mean perturbation about base
    assert np.any(medium.field.refractive_index_gradient(pts) != 0.0)


def test_scenario_iii_refractive_effect_leaves_extinction_unchanged() -> None:
    base, profile, model, bounds = build()
    medium = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds, effects=[_turbulence()]
    )
    pts = np.zeros((20, 3))
    pts[:, 2] = np.linspace(-150.0, 0.0, 20)
    assert np.allclose(base.field.extinction(pts), medium.field.extinction(pts))
    assert base.acceleration.majorant(bounds) == pytest.approx(medium.acceleration.majorant(bounds))


def test_scenario_iii_local_state_carries_refractive_gradient() -> None:
    _, profile, model, bounds = build()
    medium = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds, effects=[_turbulence()]
    )
    state = medium.field.local_state(np.array([0.0, 0.0, -80.0]))
    assert state.refractive_index_gradient is not None
    assert state.refractive_index != 1.34


def test_scenario_iii_reproducible_by_seed() -> None:
    _, profile, model, bounds = build()

    def make_medium() -> InhomogeneousMedium:
        return InhomogeneousMedium.scenario_iii(
            profile=profile,
            model=model,
            wavelength_nm=WAVELENGTH_NM,
            bounds=bounds,
            effects=[TurbulenceEffect.isotropic(rms_fluctuation=1e-3, correlation_length_m=5.0, n_modes=128, seed=99)],
        )

    pts = np.zeros((10, 3))
    pts[:, 2] = np.linspace(-100.0, 0.0, 10)
    assert np.allclose(make_medium().field.refractive_index(pts), make_medium().field.refractive_index(pts))


def test_optical_effect_adds_extinction() -> None:
    base, profile, model, bounds = build()

    class _ProbeOpticalEffect:
        name = "probe"

        def apply(self, state, position, time_s=0.0):
            return state

        def extinction_contribution(self, positions, time_s=0.0):
            p = np.asarray(positions, dtype=float)
            return np.full(p.shape[:-1], 0.5)

        def extinction_bound(self, region):
            return 0.5

    medium = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds, effects=[_ProbeOpticalEffect()]
    )
    pts = np.array([[0.0, 0.0, -50.0]])
    assert medium.field.extinction(pts)[0] == pytest.approx(base.field.extinction(pts)[0] + 0.5)


def test_unknown_effect_raises() -> None:
    _, profile, model, bounds = build()

    class _NotAnEffect:
        pass

    with pytest.raises(TypeError):
        InhomogeneousMedium.scenario_iii(
            profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=bounds, effects=[_NotAnEffect()]
        )
