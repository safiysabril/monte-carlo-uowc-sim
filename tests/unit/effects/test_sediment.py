"""Sediment NAP contribution to EnvironmentalState."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import Region
from uowc.core.ports import ParameterEffect
from uowc.effects.sediment import SedimentEffect


def test_conforms_to_parameter_effect() -> None:
    assert isinstance(SedimentEffect(concentration_g_m3=1.0), ParameterEffect)
    assert SedimentEffect(concentration_g_m3=1.0).name == "sediment"


def test_adds_to_absent_nap() -> None:
    state = EnvironmentalState(chlorophyll=0.5)
    positions = np.zeros((4, 3))
    out = SedimentEffect(concentration_g_m3=2.0).apply(state, positions)
    np.testing.assert_allclose(out.nap, np.full(4, 2.0))


def test_adds_to_existing_nap_rather_than_replacing_it() -> None:
    state = EnvironmentalState(chlorophyll=0.5, nap=np.array([1.0, 3.0]))
    positions = np.zeros((2, 3))
    out = SedimentEffect(concentration_g_m3=0.5).apply(state, positions)
    np.testing.assert_allclose(out.nap, [1.5, 3.5])


def test_leaves_chlorophyll_and_other_fields_untouched() -> None:
    state = EnvironmentalState(chlorophyll=0.7, cdom_440=0.1, temperature=18.0, salinity=35.0)
    out = SedimentEffect(concentration_g_m3=1.0).apply(state, np.zeros((1, 3)))
    assert out.chlorophyll == pytest.approx(0.7)
    assert out.cdom_440 == pytest.approx(0.1)
    assert out.temperature == pytest.approx(18.0)
    assert out.salinity == pytest.approx(35.0)


def test_region_confines_the_contribution() -> None:
    region = Region(lower=np.array([-1.0, -1.0, -5.0]), upper=np.array([1.0, 1.0, -2.0]))
    effect = SedimentEffect(concentration_g_m3=3.0, region=region)
    state = EnvironmentalState(chlorophyll=0.5)
    positions = np.array(
        [
            [0.0, 0.0, -3.0],  # inside the region
            [0.0, 0.0, -10.0],  # outside (too deep / below the box)
        ]
    )
    out = effect.apply(state, positions)
    np.testing.assert_allclose(out.nap, [3.0, 0.0])


def test_without_a_region_the_contribution_is_everywhere() -> None:
    effect = SedimentEffect(concentration_g_m3=1.5)
    state = EnvironmentalState(chlorophyll=0.5)
    positions = np.array([[0.0, 0.0, -1.0], [100.0, 100.0, -500.0]])
    out = effect.apply(state, positions)
    np.testing.assert_allclose(out.nap, [1.5, 1.5])


def test_rejects_negative_concentration() -> None:
    with pytest.raises(ValueError):
        SedimentEffect(concentration_g_m3=-1.0)


def test_haltrin_is_unaffected_by_sediment_because_it_is_case1_only() -> None:
    """Model-scope caveat in the module docstring, verified: HaltrinModel.evaluate()
    never reads state.nap, so adding sediment must not change its IOP output."""
    from uowc.optics.haltrin import HaltrinModel

    model = HaltrinModel.illustrative(wavelength_nm=520.0)
    state = EnvironmentalState(chlorophyll=0.5)
    positions = np.zeros((1, 3))

    baseline = model.evaluate(state, 520.0)
    # No region given: the contribution is uniform everywhere, so apply() legitimately
    # returns a scalar nap (EnvironmentalState fields may be scalar-or-array).
    sedimented = SedimentEffect(concentration_g_m3=1000.0).apply(state, positions)
    with_sediment = model.evaluate(EnvironmentalState(chlorophyll=0.5, nap=sedimented.nap), 520.0)

    assert with_sediment.attenuation == pytest.approx(baseline.attenuation)
    assert with_sediment.absorption == pytest.approx(baseline.absorption)
