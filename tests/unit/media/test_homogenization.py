"""Tests for homogenization rules that build the Scenario I baseline."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import Region
from uowc.media.homogenization import (
    DepthAverage,
    HomogenizationRule,
    OpticalDepthPreserving,
    SurfaceValue,
    depth_span_m,
)
from uowc.media.profiles import KamedaModel
from uowc.optics.haltrin import HaltrinModel

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

_WAVELENGTH_NM = 520.0


def profile() -> KamedaModel:
    return KamedaModel(
        background_mg_m3=0.05, peak_integral_mg_m2=20.0, peak_depth_m=80.0, peak_width_m=20.0
    )


def model() -> HaltrinModel:
    return HaltrinModel(
        wavelength_nm=_WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.002,
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


# --- OpticalDepthPreserving (IOP-space; a distinct interface) -------------------------


def test_optical_depth_preserving_does_not_satisfy_the_chlorophyll_space_protocol() -> None:
    # research-methodology.md: "a distinct interface... not a variant of" the
    # chlorophyll-space rule - it must not accidentally structurally match
    # HomogenizationRule's @runtime_checkable Protocol via a same-named `reduce`.
    assert not isinstance(OpticalDepthPreserving(), HomogenizationRule)


def test_optical_depth_preserving_name() -> None:
    assert OpticalDepthPreserving().name == "optical_depth"


def test_optical_depth_preserving_matches_the_integral_definition() -> None:
    p, m = profile(), model()
    depth_min_m, depth_max_m = 0.0, 200.0
    depths = np.linspace(depth_min_m, depth_max_m, 8192)
    column = m.evaluate(EnvironmentalState(chlorophyll=p.chlorophyll(depths)), _WAVELENGTH_NM)
    expected_a = _trapz(column.absorption, depths) / (depth_max_m - depth_min_m)
    expected_b = _trapz(column.scattering_coefficient, depths) / (depth_max_m - depth_min_m)

    iop = OpticalDepthPreserving(samples=8192).reduce_iop(
        profile=p,
        model=m,
        wavelength_nm=_WAVELENGTH_NM,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    assert iop.absorption == pytest.approx(expected_a, rel=1e-9)
    assert iop.scattering_coefficient == pytest.approx(expected_b, rel=1e-9)


def test_optical_depth_preserving_keeps_both_scattering_populations() -> None:
    # Haltrin's water + particle terms must survive as two distinct populations with
    # their own phase functions, not collapse into one combined coefficient.
    p, m = profile(), model()
    iop = OpticalDepthPreserving().reduce_iop(
        profile=p, model=m, wavelength_nm=_WAVELENGTH_NM, depth_min_m=0.0, depth_max_m=200.0
    )
    assert len(iop.scattering.components) == 2
    assert iop.scattering.components[0].phase is m.water_phase
    assert iop.scattering.components[1].phase is m.particle_phase


def test_optical_depth_preserving_reproduces_a_constant_profile_exactly() -> None:
    # mediums.md's profile-collapse property, restated for the IOP-space rule: a
    # constant profile must homogenize to exactly the value the model gives that
    # constant, since there is nothing to average over.
    flat = KamedaModel(
        background_mg_m3=0.3, peak_integral_mg_m2=0.0, peak_depth_m=40.0, peak_width_m=10.0
    )
    m = model()
    iop = OpticalDepthPreserving().reduce_iop(
        profile=flat, model=m, wavelength_nm=_WAVELENGTH_NM, depth_min_m=0.0, depth_max_m=100.0
    )
    expected = m.evaluate(EnvironmentalState(chlorophyll=0.3), _WAVELENGTH_NM)
    assert iop.absorption == pytest.approx(float(expected.absorption), rel=1e-6)
    assert iop.scattering_coefficient == pytest.approx(
        float(expected.scattering_coefficient), rel=1e-6
    )


def test_optical_depth_preserving_invalid_range_raises() -> None:
    with pytest.raises(ValueError):
        OpticalDepthPreserving().reduce_iop(
            profile=profile(),
            model=model(),
            wavelength_nm=_WAVELENGTH_NM,
            depth_min_m=100.0,
            depth_max_m=100.0,
        )


def test_jensen_bias_direction_iop_space_absorption_is_lower_or_equal() -> None:
    # research-methodology.md's headline claim: averaging chlorophyll first
    # (chlorophyll-space) systematically overestimates absorption relative to
    # averaging IOPs first (IOP-space), because a(C) is concave in C. Equal only for
    # a constant profile; this profile has real depth variation, so strictly lower.
    p, m = profile(), model()
    chl_avg = DepthAverage().reduce(p, 0.0, 200.0)
    a_from_chlorophyll_space = float(
        m.evaluate(EnvironmentalState(chlorophyll=chl_avg), _WAVELENGTH_NM).absorption
    )
    iop_space = OpticalDepthPreserving().reduce_iop(
        profile=p, model=m, wavelength_nm=_WAVELENGTH_NM, depth_min_m=0.0, depth_max_m=200.0
    )
    assert iop_space.absorption < a_from_chlorophyll_space


# --- depth_span_m ----------------------------------------------------------------------


def test_depth_span_m_converts_z_bounds_to_depth() -> None:
    region = Region(lower=[-10.0, -10.0, -50.0], upper=[10.0, 10.0, 0.0])
    assert depth_span_m(region) == pytest.approx((0.0, 50.0))


def test_depth_span_m_handles_a_domain_not_starting_at_the_surface() -> None:
    region = Region(lower=[-10.0, -10.0, -50.0], upper=[10.0, 10.0, -5.0])
    assert depth_span_m(region) == pytest.approx((5.0, 50.0))
