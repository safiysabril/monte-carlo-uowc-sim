"""Refractive-index transition, its analytic gradient, and boundary limits."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core.ports import RefractiveEffect
from uowc.effects.thermocline import ThermoclineEffect

_H = 1e-6  # finite-difference step for numerical-gradient cross-checks


def test_conforms_to_refractive_effect() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=1.0, index_above=0.0, index_below=0.002
    )
    assert isinstance(effect, RefractiveEffect)
    assert effect.name == "thermocline"


def test_index_approaches_above_value_far_above_the_layer() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=0.5, index_above=0.0, index_below=0.002
    )
    shallow = np.array([[0.0, 0.0, -0.1]])  # z = -0.1 -> depth 0.1 m, well above depth_m=10
    assert effect.index(shallow)[0] == pytest.approx(0.0, abs=1e-9)


def test_index_approaches_below_value_far_below_the_layer() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=0.5, index_above=0.0, index_below=0.002
    )
    deep = np.array([[0.0, 0.0, -50.0]])  # depth 50 m, well below depth_m=10
    assert effect.index(deep)[0] == pytest.approx(0.002, abs=1e-9)


def test_index_at_the_layer_centre_is_the_midpoint() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=0.5, index_above=0.0, index_below=0.002
    )
    centre = np.array([[0.0, 0.0, -10.0]])  # depth = 10 m = depth_m
    assert effect.index(centre)[0] == pytest.approx(0.001)


def test_index_is_monotone_with_depth_for_a_positive_step() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=1.0, index_above=0.0, index_below=0.002
    )
    depths_z = -np.linspace(0.0, 30.0, 50)
    positions = np.stack([np.zeros(50), np.zeros(50), depths_z], axis=-1)
    values = effect.index(positions)
    assert np.all(np.diff(values) >= 0.0)


def test_index_is_flat_in_the_horizontal() -> None:
    """Depth-only dependence: moving in x/y at fixed z must not change the index."""
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=1.0, index_above=0.0, index_below=0.002
    )
    positions = np.array([[0.0, 0.0, -10.0], [5.0, -3.0, -10.0], [-8.0, 8.0, -10.0]])
    values = effect.index(positions)
    np.testing.assert_allclose(values, values[0])


def test_gradient_matches_finite_difference() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=0.7, index_above=0.0, index_below=0.0015
    )
    z0 = -9.6
    p_plus = np.array([1.0, 2.0, z0 + _H])
    p_minus = np.array([1.0, 2.0, z0 - _H])
    numerical_dz = (effect.index(p_plus) - effect.index(p_minus)) / (2 * _H)

    analytic = effect.gradient(np.array([1.0, 2.0, z0]))
    assert analytic[2] == pytest.approx(numerical_dz, rel=1e-4)
    assert analytic[0] == pytest.approx(0.0)
    assert analytic[1] == pytest.approx(0.0)


def test_gradient_vanishes_far_from_the_layer() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=0.5, index_above=0.0, index_below=0.002
    )
    far_above = effect.gradient(
        np.array([0.0, 0.0, 10.0])
    )  # depth = -10 m (in the air, but fine for a field query)
    far_below = effect.gradient(np.array([0.0, 0.0, -100.0]))
    np.testing.assert_allclose(far_above, 0.0, atol=1e-9)
    np.testing.assert_allclose(far_below, 0.0, atol=1e-9)


def test_gradient_sign_for_index_increasing_with_depth() -> None:
    """Index rises with depth (index_below > index_above) => n decreases with z
    (since z points up / depth = -z), so d(index)/dz must be negative at the layer."""
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=0.5, index_above=0.0, index_below=0.002
    )
    grad = effect.gradient(np.array([0.0, 0.0, -10.0]))
    assert grad[2] < 0.0


def test_zero_step_gives_a_uniform_zero_contribution() -> None:
    effect = ThermoclineEffect(
        depth_m=10.0, transition_width_m=1.0, index_above=0.001, index_below=0.001
    )
    positions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -50.0]])
    values = effect.index(positions)
    np.testing.assert_allclose(values, 0.001)
    np.testing.assert_allclose(effect.gradient(positions[0]), 0.0, atol=1e-9)


def test_rejects_nonpositive_transition_width() -> None:
    with pytest.raises(ValueError):
        ThermoclineEffect(depth_m=10.0, transition_width_m=0.0, index_above=0.0, index_below=0.001)
