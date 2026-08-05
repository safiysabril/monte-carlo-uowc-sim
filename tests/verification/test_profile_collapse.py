"""Inhomogeneous medium with constant profile == homogeneous.

mediums.md's *Inhomogeneous Medium* requirement: "A constant profile must reproduce
the homogeneous medium exactly (to Monte Carlo error). This is the profile-collapse
verification test and the cheapest guard against a depth-indexing error."

With ``peak_integral_mg_m2=0.0`` the shifted-Gaussian deep-chlorophyll-maximum term
vanishes and :class:`~uowc.media.profiles.KamedaModel` returns a spatially uniform
chlorophyll concentration, so :meth:`InhomogeneousMedium.scenario_ii` and
:meth:`HomogeneousMedium.uniform` describe the *same* physical medium built through
two independent code paths. Same seed, same transport engine: their outputs must be
bit-identical, not merely statistically close - any difference is a depth-indexing
or evaluation-order bug, not Monte Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.core.rng import NumpyRng
from uowc.media import HomogeneousMedium, InhomogeneousMedium, KamedaModel, SurfaceValue
from uowc.optics import HaltrinModel
from uowc.transport import WoodcockDeltaTracker

pytestmark = pytest.mark.verification

_WAVELENGTH_NM = 520.0


def _flat_profile() -> KamedaModel:
    return KamedaModel(
        background_mg_m3=0.3, peak_integral_mg_m2=0.0, peak_depth_m=40.0, peak_width_m=10.0
    )


def _model() -> HaltrinModel:
    return HaltrinModel(
        wavelength_nm=_WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.15,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.3,
    )


def _run(medium, seed: int):
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=_WAVELENGTH_NM,
        divergence_rad=0.15,
    )
    receiver = Receiver(
        position=[0.0, 0.0, -8.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.5, fov_rad=np.pi / 3
    )
    config = SamplingConfig(n_photons=4000, estimator="analog", max_scatter_events=500)
    return WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(seed), config)


def _mediums():
    bounds = Region(lower=[-20.0, -20.0, -40.0], upper=[20.0, 20.0, 0.0])
    homogeneous = HomogeneousMedium.from_profile(
        profile=_flat_profile(),
        model=_model(),
        wavelength_nm=_WAVELENGTH_NM,
        bounds=bounds,
        rule=SurfaceValue(),
    )
    inhomogeneous = InhomogeneousMedium.scenario_ii(
        profile=_flat_profile(), model=_model(), wavelength_nm=_WAVELENGTH_NM, bounds=bounds
    )
    return homogeneous, inhomogeneous, bounds


def test_extinction_field_matches_at_arbitrary_depths() -> None:
    homogeneous, inhomogeneous, _ = _mediums()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -5.0], [0.0, 0.0, -39.9]])
    np.testing.assert_allclose(
        homogeneous.field.extinction(points), inhomogeneous.field.extinction(points)
    )


def test_majorants_match() -> None:
    homogeneous, inhomogeneous, bounds = _mediums()
    assert homogeneous.acceleration.majorant(bounds) == pytest.approx(
        inhomogeneous.acceleration.majorant(bounds), rel=1e-6
    )


def test_transport_output_is_bit_identical_for_the_same_seed() -> None:
    homogeneous, inhomogeneous, _ = _mediums()
    out_homogeneous = _run(homogeneous, seed=42)
    out_inhomogeneous = _run(inhomogeneous, seed=42)

    assert out_homogeneous.tallies.detected == out_inhomogeneous.tallies.detected
    assert out_homogeneous.tallies.absorbed_weight == pytest.approx(
        out_inhomogeneous.tallies.absorbed_weight
    )
    np.testing.assert_array_equal(
        out_homogeneous.photons.arrival_time_s, out_inhomogeneous.photons.arrival_time_s
    )
    np.testing.assert_array_equal(
        out_homogeneous.photons.n_scatters, out_inhomogeneous.photons.n_scatters
    )
