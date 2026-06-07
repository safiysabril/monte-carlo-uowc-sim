"""Tests for the Kameda shifted-Gaussian vertical chlorophyll profile."""
from __future__ import annotations

import math

import numpy as np
import pytest

from uowc.media.profiles import ChlorophyllProfile, KamedaModel

# np.trapz was renamed np.trapezoid in numpy 2.0 (and removed under the old name).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def make() -> KamedaModel:
    return KamedaModel(
        background_mg_m3=0.05,
        peak_integral_mg_m2=20.0,
        peak_depth_m=80.0,
        peak_width_m=20.0,
    )


def test_conforms_to_chlorophyll_profile() -> None:
    assert isinstance(make(), ChlorophyllProfile)


def test_returns_ndarray_preserving_shape() -> None:
    model = make()
    z = np.linspace(0.0, 100.0, 50).reshape(10, 5)
    out = model.chlorophyll(z)
    assert isinstance(out, np.ndarray)
    assert out.shape == z.shape


def test_peak_amplitude_formula() -> None:
    model = make()
    expected = model.peak_integral_mg_m2 / (model.peak_width_m * math.sqrt(2.0 * math.pi))
    assert model.peak_amplitude_mg_m3 == pytest.approx(expected)


def test_maximum_is_at_peak_depth() -> None:
    model = make()
    z = np.linspace(0.0, 200.0, 4001)
    c = model.chlorophyll(z)
    assert z[int(np.argmax(c))] == pytest.approx(model.peak_depth_m, abs=0.1)
    assert c.max() == pytest.approx(
        model.background_mg_m3 + model.peak_amplitude_mg_m3, rel=1e-6
    )


def test_decays_to_background_far_from_peak() -> None:
    model = make()
    deep = model.chlorophyll(np.array([1000.0]))[0]
    assert deep == pytest.approx(model.background_mg_m3, abs=1e-9)


def test_gaussian_is_symmetric_about_peak() -> None:
    model = make()
    offset = 15.0
    above = model.chlorophyll(np.array([model.peak_depth_m - offset]))[0]
    below = model.chlorophyll(np.array([model.peak_depth_m + offset]))[0]
    assert above == pytest.approx(below, rel=1e-12)


def test_monotonic_decrease_below_peak() -> None:
    model = make()
    z = np.linspace(model.peak_depth_m, model.peak_depth_m + 120.0, 200)
    c = model.chlorophyll(z)
    assert np.all(np.diff(c) <= 1e-12)


def test_concentration_is_non_negative() -> None:
    model = make()
    z = np.linspace(0.0, 500.0, 1000)
    assert np.all(model.chlorophyll(z) >= 0.0)


def test_peak_excess_integrates_to_peak_integral() -> None:
    # Integrating (C - background) over depth must recover peak_integral_mg_m2.
    model = make()
    z = np.linspace(model.peak_depth_m - 160.0, model.peak_depth_m + 160.0, 40001)
    excess = model.chlorophyll(z) - model.background_mg_m3
    assert _trapz(excess, z) == pytest.approx(model.peak_integral_mg_m2, rel=1e-3)


def test_is_deterministic() -> None:
    model = make()
    z = np.array([10.0, 50.0, 90.0])
    assert np.array_equal(model.chlorophyll(z), model.chlorophyll(z))


def test_is_immutable() -> None:
    model = make()
    with pytest.raises(Exception):
        model.background_mg_m3 = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    [
        {"peak_width_m": 0.0},
        {"peak_width_m": -5.0},
        {"background_mg_m3": -0.1},
        {"peak_integral_mg_m2": -1.0},
        {"peak_depth_m": -10.0},
    ],
)
def test_invalid_parameters_raise(override: dict[str, float]) -> None:
    params = {
        "background_mg_m3": 0.05,
        "peak_integral_mg_m2": 20.0,
        "peak_depth_m": 80.0,
        "peak_width_m": 20.0,
    }
    params.update(override)
    with pytest.raises(ValueError):
        KamedaModel(**params)


def test_from_surface_chlorophyll_builds_valid_profile() -> None:
    model = KamedaModel.from_surface_chlorophyll(0.3)
    assert isinstance(model, KamedaModel)
    assert isinstance(model, ChlorophyllProfile)


def test_from_surface_background_increases_with_surface_chlorophyll() -> None:
    low = KamedaModel.from_surface_chlorophyll(0.05)
    high = KamedaModel.from_surface_chlorophyll(1.5)
    assert high.background_mg_m3 > low.background_mg_m3


def test_from_surface_dcm_shoals_as_surface_chlorophyll_rises() -> None:
    low = KamedaModel.from_surface_chlorophyll(0.05)
    high = KamedaModel.from_surface_chlorophyll(1.5)
    assert high.peak_depth_m < low.peak_depth_m


def test_from_surface_negative_raises() -> None:
    with pytest.raises(ValueError):
        KamedaModel.from_surface_chlorophyll(-1.0)
