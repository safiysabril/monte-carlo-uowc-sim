"""Bubble-layer scattering contribution and phase mixture."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core.geometry import Region
from uowc.core.ports import OpticalEffect
from uowc.core.state import IOP, LocalOpticalState, Scattering, ScatteringComponent
from uowc.effects.bubbles import BubbleLayerEffect
from uowc.optics.phase import HenyeyGreenstein

_NEAR_SURFACE = Region(lower=np.array([-10.0, -10.0, -1.0]), upper=np.array([10.0, 10.0, 0.0]))


def _layer(**overrides: object) -> BubbleLayerEffect:
    kwargs = dict(
        region=_NEAR_SURFACE,
        void_fraction=1e-4,
        mean_radius_m=1e-4,
        phase=HenyeyGreenstein(0.8),
    )
    kwargs.update(overrides)
    return BubbleLayerEffect(**kwargs)  # type: ignore[arg-type]


def test_conforms_to_optical_effect() -> None:
    assert isinstance(_layer(), OpticalEffect)
    assert _layer().name == "bubbles"


def test_extinction_matches_the_geometric_optics_formula() -> None:
    layer = _layer(void_fraction=2e-4, mean_radius_m=5e-5)
    expected = 1.5 * 2e-4 / 5e-5
    got = layer.extinction_contribution(np.array([[0.0, 0.0, -0.5]]))
    assert got[0] == pytest.approx(expected)


def test_extinction_is_zero_outside_the_layer() -> None:
    layer = _layer()
    positions = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, -5.0]])
    got = layer.extinction_contribution(positions)
    assert got[0] > 0.0
    assert got[1] == 0.0


def test_extinction_bound_matches_the_constant_coefficient_when_regions_overlap() -> None:
    layer = _layer(void_fraction=1e-4, mean_radius_m=1e-4)
    query_region = Region(lower=np.array([-1.0, -1.0, -0.5]), upper=np.array([1.0, 1.0, 0.0]))
    assert layer.extinction_bound(query_region) == pytest.approx(1.5 * 1e-4 / 1e-4)


def test_extinction_bound_is_zero_when_regions_do_not_overlap() -> None:
    layer = _layer()
    far_region = Region(
        lower=np.array([100.0, 100.0, -50.0]), upper=np.array([200.0, 200.0, -40.0])
    )
    assert layer.extinction_bound(far_region) == 0.0


def _plain_state() -> LocalOpticalState:
    water_component = ScatteringComponent(coefficient=0.01, phase=HenyeyGreenstein(0.0))
    iop = IOP(
        absorption=0.05, scattering=Scattering(components=(water_component,)), wavelength_nm=520.0
    )
    return LocalOpticalState(iop=iop, refractive_index=1.34)


def test_apply_adds_a_new_scattering_population_inside_the_layer() -> None:
    layer = _layer(void_fraction=1e-4, mean_radius_m=1e-4)
    state = _plain_state()
    out = layer.apply(state, np.array([0.0, 0.0, -0.5]))
    assert len(out.iop.scattering.components) == 2
    added = out.iop.scattering.components[-1]
    assert added.coefficient == pytest.approx(1.5 * 1e-4 / 1e-4)
    assert added.phase.asymmetry == pytest.approx(0.8)


def test_apply_leaves_absorption_and_existing_populations_untouched() -> None:
    layer = _layer()
    state = _plain_state()
    out = layer.apply(state, np.array([0.0, 0.0, -0.5]))
    assert out.iop.absorption == pytest.approx(state.iop.absorption)
    assert out.iop.scattering.components[0] is state.iop.scattering.components[0]


def test_apply_is_a_no_op_outside_the_layer() -> None:
    layer = _layer()
    state = _plain_state()
    out = layer.apply(state, np.array([0.0, 0.0, -5.0]))
    assert out is state


def test_apply_raises_total_scattering_by_exactly_the_bubble_contribution() -> None:
    layer = _layer(void_fraction=1e-4, mean_radius_m=1e-4)
    state = _plain_state()
    out = layer.apply(state, np.array([0.0, 0.0, -0.5]))
    expected = state.iop.scattering_coefficient + 1.5 * 1e-4 / 1e-4
    assert out.iop.scattering_coefficient == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"void_fraction": -0.1},
        {"void_fraction": 1.0},
        {"mean_radius_m": 0.0},
        {"mean_radius_m": -1e-4},
    ],
)
def test_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _layer(**kwargs)
