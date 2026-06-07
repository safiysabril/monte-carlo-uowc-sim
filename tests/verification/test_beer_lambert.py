"""Verification: a non-scattering homogeneous medium reproduces Beer-Lambert.

For a collimated beam in a purely absorbing homogeneous medium, the unscattered flux
reaching depth L is exp(-c L). With a wide on-axis detector this equals the detected
fraction. Uses the real HomogeneousMedium (Scenario I) - no scattering populations.
"""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core import IOP, Receiver, Region, SamplingConfig, Scattering, Source
from uowc.core.rng import NumpyRng
from uowc.media import HomogeneousMedium
from uowc.transport import WoodcockDeltaTracker

pytestmark = pytest.mark.verification


@pytest.mark.parametrize("c, length", [(0.1, 8.0), (0.25, 6.0), (0.05, 20.0)])
def test_beer_lambert_law(c: float, length: float) -> None:
    iop = IOP(absorption=c, scattering=Scattering(()), wavelength_nm=520.0)  # b = 0
    bounds = Region(lower=[-10.0, -10.0, -200.0], upper=[10.0, 10.0, 0.0])
    medium = HomogeneousMedium.from_iop(iop=iop, bounds=bounds)
    source = Source(position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], wavelength_nm=520.0, divergence_rad=0.0)
    receiver = Receiver(
        position=[0.0, 0.0, -length], normal=[0.0, 0.0, 1.0], aperture_radius_m=5.0, fov_rad=np.pi / 2
    )
    config = SamplingConfig(n_photons=200_000, estimator="analog")

    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(2024), config)
    fraction = out.tallies.detected / out.tallies.launched
    assert fraction == pytest.approx(np.exp(-c * length), rel=0.02)
