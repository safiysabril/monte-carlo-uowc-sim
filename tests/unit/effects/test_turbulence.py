"""Tests for the refractive-index turbulence effect (a realized random field)."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core.ports import RefractiveEffect
from uowc.effects.turbulence import TurbulenceEffect


def make(*, seed: int = 7, n_modes: int = 256, rms: float = 1e-4, length: float = 5.0) -> TurbulenceEffect:
    return TurbulenceEffect.isotropic(
        rms_fluctuation=rms, correlation_length_m=length, n_modes=n_modes, seed=seed
    )


def test_conforms_to_refractive_effect() -> None:
    assert isinstance(make(), RefractiveEffect)
    assert make().name == "turbulence"


def test_shapes() -> None:
    t = make()
    assert t.index(np.zeros((10, 3))).shape == (10,)
    assert np.ndim(t.index(np.array([1.0, 2.0, -3.0]))) == 0
    assert t.gradient(np.zeros((10, 3))).shape == (10, 3)
    assert t.gradient(np.array([1.0, 2.0, -3.0])).shape == (3,)


def test_is_a_realized_field_deterministic_per_position() -> None:
    t = make()
    x = np.array([[1.0, 2.0, -3.0], [0.0, 0.0, -10.0]])
    assert np.array_equal(t.index(x), t.index(x))
    assert np.array_equal(t.gradient(x), t.gradient(x))


def test_same_seed_gives_identical_field() -> None:
    a, b = make(seed=42), make(seed=42)
    x = np.array([[1.0, 1.0, -1.0], [2.0, -2.0, -5.0]])
    assert np.allclose(a.index(x), b.index(x))
    assert np.allclose(a.gradient(x), b.gradient(x))


def test_different_seed_gives_different_field() -> None:
    a, b = make(seed=1), make(seed=2)
    x = np.array([[1.0, 1.0, -1.0]])
    assert not np.allclose(a.index(x), b.index(x))


def test_zero_mean_and_correct_rms() -> None:
    rms = 2e-4
    t = make(seed=11, n_modes=512, rms=rms, length=5.0)
    points = np.random.default_rng(0).uniform(-50.0, 50.0, size=(20000, 3))
    values = t.index(points)
    assert abs(float(values.mean())) < rms * 0.1
    assert float(values.std()) == pytest.approx(rms, rel=0.1)


def test_gradient_matches_finite_difference() -> None:
    t = make(seed=5, n_modes=128, rms=1e-3, length=5.0)
    x = np.array([[1.0, -2.0, -7.0], [3.0, 0.5, -12.0]])
    analytic = t.gradient(x)
    h = 1e-4
    fd = np.zeros_like(x)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        fd[:, axis] = (t.index(x + step) - t.index(x - step)) / (2.0 * h)
    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-12)


def test_is_spatially_smooth_below_correlation_length() -> None:
    t = make(seed=3, n_modes=256, rms=1e-3, length=10.0)
    x = np.array([[0.0, 0.0, -20.0]])
    here = t.index(x)[0]
    nearby = t.index(x + np.array([0.01, 0.0, 0.0]))[0]  # 1 cm << L = 10 m
    assert abs(nearby - here) < 1e-4


@pytest.mark.parametrize(
    "override",
    [{"rms_fluctuation": -1.0}, {"correlation_length_m": 0.0}, {"n_modes": 0}],
)
def test_invalid_parameters_raise(override: dict[str, float]) -> None:
    params = {"rms_fluctuation": 1e-4, "correlation_length_m": 5.0, "n_modes": 64, "seed": 1}
    params.update(override)
    with pytest.raises(ValueError):
        TurbulenceEffect.isotropic(**params)
