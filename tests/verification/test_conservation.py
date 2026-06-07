"""Verification: photon weight is conserved (absorbed + detected + escaped == launched).

Runs a scattering Scenario I medium (Haltrin model) and checks that every launched
photon ends in exactly one terminal state.
"""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.core.rng import NumpyRng
from uowc.media import HomogeneousMedium
from uowc.optics import HaltrinModel
from uowc.transport import WoodcockDeltaTracker

pytestmark = pytest.mark.verification


def test_weight_is_conserved() -> None:
    model = HaltrinModel(
        wavelength_nm=520.0,
        pure_water_absorption_m_inv=0.3,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.4,
    )
    bounds = Region(lower=[-5.0, -5.0, -8.0], upper=[5.0, 5.0, 0.0])
    medium = HomogeneousMedium.uniform(model=model, chlorophyll=0.5, wavelength_nm=520.0, bounds=bounds)
    source = Source(position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], wavelength_nm=520.0, divergence_rad=0.3)
    receiver = Receiver(position=[0.0, 0.0, -4.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=0.5, fov_rad=np.pi / 4)
    config = SamplingConfig(n_photons=3000, estimator="analog", max_scatter_events=1000)

    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(7), config)
    t = out.tallies
    total = t.detected_weight + t.absorbed_weight + t.escaped_weight + t.extra["killed_weight"]
    assert total == pytest.approx(t.launched)
    assert t.launched == 3000
