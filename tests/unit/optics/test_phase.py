"""Tests for the Henyey-Greenstein phase function."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core.ports import PhaseFunction
from uowc.optics.phase import HenyeyGreenstein

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def test_conforms_to_phase_function_protocol() -> None:
    assert isinstance(HenyeyGreenstein(0.5), PhaseFunction)


def test_asymmetry_property_returns_g() -> None:
    assert HenyeyGreenstein(0.7).asymmetry == 0.7


@pytest.mark.parametrize("g", [-0.6, 0.0, 0.3, 0.924])
def test_value_is_normalized_over_solid_angle(g: float) -> None:
    # integral over 4pi sr of p(cos t) dOmega = 2*pi * integral_{-1}^{1} p(mu) dmu = 1
    pf = HenyeyGreenstein(g)
    mu = np.linspace(-1.0, 1.0, 200001)
    integral = 2.0 * np.pi * _trapz(pf.value(mu), mu)
    assert integral == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("g", [-0.6, 0.0, 0.3, 0.924])
def test_value_recovers_asymmetry(g: float) -> None:
    # 2*pi * integral mu * p(mu) dmu = <cos t> = g
    pf = HenyeyGreenstein(g)
    mu = np.linspace(-1.0, 1.0, 200001)
    mean_cos = 2.0 * np.pi * _trapz(mu * pf.value(mu), mu)
    assert mean_cos == pytest.approx(g, abs=2e-3)


@pytest.mark.parametrize("g", [0.0, 0.3, 0.6, 0.9])
def test_sampling_recovers_asymmetry(g: float) -> None:
    pf = HenyeyGreenstein(g)
    u = np.random.default_rng(12345).uniform(size=400_000)
    cos_theta = pf.sample_cos_theta(u)
    assert cos_theta.min() >= -1.0 - 1e-9
    assert cos_theta.max() <= 1.0 + 1e-9
    assert float(cos_theta.mean()) == pytest.approx(g, abs=0.01)


def test_isotropic_limit() -> None:
    pf = HenyeyGreenstein(0.0)
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0 - 1e-12])
    assert np.allclose(pf.sample_cos_theta(u), 2.0 * u - 1.0)
    assert np.allclose(pf.value(np.array([-1.0, 0.0, 1.0])), 1.0 / (4.0 * np.pi))


def test_is_forward_peaked_for_positive_g() -> None:
    pf = HenyeyGreenstein(0.8)
    assert pf.value(np.array([1.0]))[0] > pf.value(np.array([-1.0]))[0]


@pytest.mark.parametrize("g", [-1.0, 1.0, 1.5, -2.0])
def test_invalid_asymmetry_raises(g: float) -> None:
    with pytest.raises(ValueError):
        HenyeyGreenstein(g)


def test_sampling_is_deterministic() -> None:
    pf = HenyeyGreenstein(0.5)
    u = np.linspace(0.0, 0.999, 11)
    assert np.array_equal(pf.sample_cos_theta(u), pf.sample_cos_theta(u))
