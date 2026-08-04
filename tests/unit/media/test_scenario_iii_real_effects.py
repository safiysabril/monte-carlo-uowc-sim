"""Scenario III composed with the real (non-mock) effects: sediment, bubbles,
thermocline - end-to-end through InhomogeneousMedium, not just the dispatch
mechanism (that is covered generically in test_inhomogeneous.py with a probe effect).
"""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Region
from uowc.effects import BubbleLayerEffect, SedimentEffect, ThermoclineEffect
from uowc.media import InhomogeneousMedium, KamedaModel
from uowc.optics import HaltrinModel
from uowc.optics.phase import HenyeyGreenstein

WAVELENGTH_NM = 520.0
_BOUNDS = Region(lower=[-50.0, -50.0, -200.0], upper=[50.0, 50.0, 0.0])


def _profile_and_model() -> tuple[KamedaModel, HaltrinModel]:
    profile = KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )
    model = HaltrinModel(
        wavelength_nm=WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.002,
    )
    return profile, model


def test_sediment_effect_composes_but_has_no_numerical_effect_on_haltrin() -> None:
    """Documented Case-1 caveat (sediment.py), verified through the full medium."""
    profile, model = _profile_and_model()
    baseline = InhomogeneousMedium.scenario_ii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS
    )
    with_sediment = InhomogeneousMedium.scenario_iii(
        profile=profile,
        model=model,
        wavelength_nm=WAVELENGTH_NM,
        bounds=_BOUNDS,
        effects=[SedimentEffect(concentration_g_m3=500.0)],
    )
    pts = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -80.0]])
    np.testing.assert_allclose(with_sediment.field.extinction(pts), baseline.field.extinction(pts))


def test_bubble_layer_raises_extinction_only_inside_its_region() -> None:
    profile, model = _profile_and_model()
    surface_layer = Region(lower=np.array([-50.0, -50.0, -1.0]), upper=np.array([50.0, 50.0, 0.0]))
    bubbles = BubbleLayerEffect(
        region=surface_layer, void_fraction=1e-4, mean_radius_m=1e-4, phase=HenyeyGreenstein(0.9)
    )
    baseline = InhomogeneousMedium.scenario_ii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS
    )
    with_bubbles = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS, effects=[bubbles]
    )
    shallow = np.array([[0.0, 0.0, -0.5]])
    deep = np.array([[0.0, 0.0, -50.0]])

    expected_bump = 1.5 * 1e-4 / 1e-4
    assert with_bubbles.field.extinction(shallow)[0] == pytest.approx(
        baseline.field.extinction(shallow)[0] + expected_bump
    )
    assert with_bubbles.field.extinction(deep)[0] == pytest.approx(
        baseline.field.extinction(deep)[0]
    )


def test_bubble_layer_majorant_accounts_for_the_added_extinction() -> None:
    """mediums.md: an effect that raises extinction must raise the medium's majorant."""
    profile, model = _profile_and_model()
    surface_layer = Region(lower=np.array([-50.0, -50.0, -1.0]), upper=np.array([50.0, 50.0, 0.0]))
    bubbles = BubbleLayerEffect(
        region=surface_layer, void_fraction=1e-3, mean_radius_m=1e-4, phase=HenyeyGreenstein(0.9)
    )
    baseline = InhomogeneousMedium.scenario_ii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS
    )
    with_bubbles = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS, effects=[bubbles]
    )
    assert with_bubbles.acceleration.majorant(_BOUNDS) > baseline.acceleration.majorant(_BOUNDS)


def test_thermocline_perturbs_index_and_leaves_extinction_unchanged() -> None:
    profile, model = _profile_and_model()
    thermocline = ThermoclineEffect(
        depth_m=30.0, transition_width_m=2.0, index_above=0.0, index_below=0.0015
    )
    baseline = InhomogeneousMedium.scenario_ii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS
    )
    with_thermocline = InhomogeneousMedium.scenario_iii(
        profile=profile,
        model=model,
        wavelength_nm=WAVELENGTH_NM,
        bounds=_BOUNDS,
        effects=[thermocline],
    )
    shallow = np.array([[0.0, 0.0, -1.0]])
    deep = np.array([[0.0, 0.0, -100.0]])

    assert with_thermocline.field.refractive_index(shallow)[0] == pytest.approx(
        baseline.field.refractive_index(shallow)[0], abs=1e-6
    )
    assert with_thermocline.field.refractive_index(deep)[0] == pytest.approx(
        baseline.field.refractive_index(deep)[0] + 0.0015, abs=1e-6
    )
    np.testing.assert_allclose(
        with_thermocline.field.extinction(shallow), baseline.field.extinction(shallow)
    )


def test_all_three_real_effects_compose_together() -> None:
    """A representative Scenario III: one effect from each of the three kinds,
    verifying _classify_effects routes all of them correctly at once."""
    profile, model = _profile_and_model()
    surface_layer = Region(lower=np.array([-50.0, -50.0, -1.0]), upper=np.array([50.0, 50.0, 0.0]))
    effects = [
        SedimentEffect(concentration_g_m3=100.0),
        BubbleLayerEffect(
            region=surface_layer,
            void_fraction=1e-4,
            mean_radius_m=1e-4,
            phase=HenyeyGreenstein(0.9),
        ),
        ThermoclineEffect(depth_m=30.0, transition_width_m=2.0, index_above=0.0, index_below=0.001),
    ]
    medium = InhomogeneousMedium.scenario_iii(
        profile=profile, model=model, wavelength_nm=WAVELENGTH_NM, bounds=_BOUNDS, effects=effects
    )
    shallow = np.array([[0.0, 0.0, -0.5]])
    state = medium.field.local_state(shallow[0])

    # Bubble scattering population present, thermocline index contribution present,
    # sediment silently absorbed into (unused) EnvironmentalState.nap - all at once.
    assert len(state.iop.scattering.components) == 3  # water + biogenic + bubbles
    assert state.refractive_index == pytest.approx(1.34, abs=1e-6)  # shallow: pre-thermocline
