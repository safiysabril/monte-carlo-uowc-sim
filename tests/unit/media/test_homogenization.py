"""Tests for homogenization rules that build the Scenario I baseline."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.media.homogenization import DepthAverage, HomogenizationRule, SurfaceValue
from uowc.media.profiles import KamedaModel

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def profile() -> KamedaModel:
    return KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )


def test_rules_conform_to_protocol() -> None:
    assert isinstance(SurfaceValue(), HomogenizationRule)
    assert isinstance(DepthAverage(), HomogenizationRule)


def test_rule_names() -> None:
    assert SurfaceValue().name == "surface"
    assert DepthAverage().name == "depth_average"


def test_surface_value_returns_surface_concentration() -> None:
    p = profile()
    assert SurfaceValue().reduce(p, 0.0, 200.0) == pytest.approx(
        float(p.chlorophyll(np.array([0.0]))[0])
    )


def test_depth_average_matches_integral() -> None:
    p = profile()
    depths = np.linspace(0.0, 200.0, 4096)
    expected = _trapz(p.chlorophyll(depths), depths) / 200.0
    assert DepthAverage(samples=4096).reduce(p, 0.0, 200.0) == pytest.approx(expected, rel=1e-9)


def test_depth_average_exceeds_surface_for_a_deep_maximum() -> None:
    # A deep chlorophyll maximum raises the column average above the shallow surface value.
    p = profile()
    assert DepthAverage().reduce(p, 0.0, 200.0) > SurfaceValue().reduce(p, 0.0, 200.0)


def test_depth_average_invalid_range_raises() -> None:
    with pytest.raises(ValueError):
        DepthAverage().reduce(profile(), 100.0, 100.0)
