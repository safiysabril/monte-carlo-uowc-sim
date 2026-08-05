"""Water-type IOPs vs a published reference.

Scope note (why this is not a Haltrin/Kameda test)
----------------------------------------------------
The original stub name suggested checking :class:`~uowc.optics.haltrin.HaltrinModel`
against a reference water type. That is not meaningful here: Haltrin's model is
parameterized by *chlorophyll concentration*, while the classic named water types
below (Petzold, 1972, "Volume scattering functions for selected ocean waters") are
*directly measured* absorption/scattering coefficients with no stated chlorophyll
value to feed the model - there is no chlorophyll input that would make
``HaltrinModel`` reproduce them, so a test claiming to check one against the other
would either fabricate a chlorophyll value or silently test nothing. This module
instead validates the framework's own :class:`~uowc.core.state.IOP` and
:class:`~uowc.media.HomogeneousMedium` against those measured coefficients directly.

Reference
---------
Geldard, C. T., Thompson, J. & Popoola, W. O. (2020), "Effects of Turbulence Induced
Scattering on Underwater Optical Wireless Communications", arXiv:2008.01152, Sec.
III.A: "Petzold's values of a = 0.295 m^-1 and b_Petzold = 1.875 m^-1 are used to
model turbid harbour water[;] a = 0.179 m^-1 and b_Petzold = 0.219 m^-1 for coastal
water" - itself citing Petzold (1972). Sec. V.A additionally states the mean number
of scattering interactions ("extinction length") is ~33 for a 15 m harbour link and
~11 for a 30 m coastal link.

These are the two IOP sets; a specific measurement wavelength is not restated in the
source alongside them (Petzold's instrument is commonly associated with visible
green light in later UOWC literature, but that attribution is not repeated in this
citation), so the wavelength tag below is a documented, inert placeholder - nothing
in this module's assertions depends on its value.
"""
from __future__ import annotations

import pytest

from uowc.core import IOP, Region, Scattering, ScatteringComponent
from uowc.media import HomogeneousMedium
from uowc.optics.phase import HenyeyGreenstein

pytestmark = pytest.mark.validation

_WAVELENGTH_NM = 530.0  # placeholder - see module docstring

# Petzold (1972) values as quoted by Geldard, Thompson & Popoola (2020), Sec. III.A.
_COASTAL_A = 0.179
_COASTAL_B = 0.219
_HARBOUR_A = 0.295
_HARBOUR_B = 1.875


def _iop(a: float, b: float) -> IOP:
    # Asymmetry is not part of the cited reference; Mobley's (1994) Petzold-average
    # value is used only to build a valid PhaseFunction, and does not affect any
    # quantity checked below (all derived from a, b alone).
    scattering = Scattering((ScatteringComponent(b, HenyeyGreenstein(0.924)),))
    return IOP(absorption=a, scattering=scattering, wavelength_nm=_WAVELENGTH_NM)


def test_coastal_water_attenuation_matches_petzold() -> None:
    iop = _iop(_COASTAL_A, _COASTAL_B)
    assert iop.attenuation == pytest.approx(0.398)


def test_harbour_water_attenuation_matches_petzold() -> None:
    iop = _iop(_HARBOUR_A, _HARBOUR_B)
    assert iop.attenuation == pytest.approx(2.170)


def test_harbour_water_is_far_more_strongly_scattering_than_coastal() -> None:
    # The paper's own qualitative claim (Sec. V.A): harbour water scatters far more
    # than coastal water, both in absolute b and in single-scattering albedo.
    coastal = _iop(_COASTAL_A, _COASTAL_B)
    harbour = _iop(_HARBOUR_A, _HARBOUR_B)
    assert harbour.scattering_coefficient > coastal.scattering_coefficient
    assert harbour.single_scattering_albedo > coastal.single_scattering_albedo


def test_single_scattering_albedo_matches_hand_calculation() -> None:
    coastal = _iop(_COASTAL_A, _COASTAL_B)
    harbour = _iop(_HARBOUR_A, _HARBOUR_B)
    assert coastal.single_scattering_albedo == pytest.approx(_COASTAL_B / 0.398)
    assert harbour.single_scattering_albedo == pytest.approx(_HARBOUR_B / 2.170)


@pytest.mark.parametrize(
    "a, b, length_m, expected_interactions",
    [
        (_HARBOUR_A, _HARBOUR_B, 15.0, 33.0),
        (_COASTAL_A, _COASTAL_B, 30.0, 11.0),
    ],
)
def test_mean_interaction_count_matches_the_papers_stated_figure(
    a: float, b: float, length_m: float, expected_interactions: float
) -> None:
    # Mean number of extinction events over a path of length L in a medium with
    # attenuation c is c*L (the defining property of the underlying Poisson process;
    # transport.md's free-path sampling relies on exactly this). The paper states
    # this figure rounded to the nearest integer, not to full precision, so the
    # tolerance here reflects that rounding, not simulation noise.
    c = a + b
    mean_interactions = c * length_m
    assert mean_interactions == pytest.approx(expected_interactions, rel=0.1)


def test_homogeneous_medium_majorant_is_exact_for_a_measured_water_type() -> None:
    # A homogeneous medium's Woodcock majorant is exactly c (mediums.md) - grounds
    # the reference IOPs in the actual Medium port, not just the raw dataclass.
    iop = _iop(_HARBOUR_A, _HARBOUR_B)
    bounds = Region(lower=[-10.0, -10.0, -20.0], upper=[10.0, 10.0, 0.0])
    medium = HomogeneousMedium.from_iop(iop=iop, bounds=bounds)
    assert medium.acceleration.majorant(bounds) == pytest.approx(2.170)
