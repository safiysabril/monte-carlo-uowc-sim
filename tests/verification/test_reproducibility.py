"""Identical seed tree -> identical output (reference single-thread mode).

research-methodology.md lists "Reproducibility - identical seed and configuration
reproduce identical output" as a required property. Individual unit tests already
check this for narrow cases (e.g. ``test_woodcock.py``); this module is the dedicated,
discoverable verification-marked check across the pieces that matter for a real run:
a scattering Scenario III medium (profile + model + effects together), and the seed
tree recorded in run metadata, not just the raw photon arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Receiver, Region, RunMetadata, SamplingConfig, SeedTree, Source
from uowc.core.rng import NumpyRng
from uowc.effects import TurbulenceEffect
from uowc.media import InhomogeneousMedium, KamedaModel
from uowc.optics import HaltrinModel
from uowc.transport import WoodcockDeltaTracker

pytestmark = pytest.mark.verification

_WAVELENGTH_NM = 520.0


def _scenario_iii_medium() -> InhomogeneousMedium:
    profile = KamedaModel(
        background_mg_m3=0.1, peak_integral_mg_m2=30.0, peak_depth_m=25.0, peak_width_m=8.0
    )
    model = HaltrinModel(
        wavelength_nm=_WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.04,
        pure_water_scattering_m_inv=0.2,
    )
    bounds = Region(lower=[-15.0, -15.0, -30.0], upper=[15.0, 15.0, 0.0])
    turbulence = TurbulenceEffect.isotropic(
        rms_fluctuation=2e-3, correlation_length_m=4.0, n_modes=64, seed=3
    )
    return InhomogeneousMedium.scenario_iii(
        profile=profile,
        model=model,
        wavelength_nm=_WAVELENGTH_NM,
        bounds=bounds,
        effects=(turbulence,),
    )


def _run(seed: int, *, estimator: str = "analog"):
    medium = _scenario_iii_medium()
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=_WAVELENGTH_NM,
        divergence_rad=0.2,
    )
    receiver = Receiver(
        position=[0.0, 0.0, -10.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=np.pi / 3
    )
    config = SamplingConfig(n_photons=3000, estimator=estimator, max_scatter_events=500)
    return WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(seed), config)


def test_identical_seed_reproduces_identical_photon_arrays() -> None:
    first = _run(seed=2024)
    second = _run(seed=2024)

    np.testing.assert_array_equal(first.photons.arrival_time_s, second.photons.arrival_time_s)
    np.testing.assert_array_equal(first.photons.path_length_m, second.photons.path_length_m)
    np.testing.assert_array_equal(first.photons.weight, second.photons.weight)
    np.testing.assert_array_equal(first.photons.n_scatters, second.photons.n_scatters)
    np.testing.assert_array_equal(first.photons.incidence_rad, second.photons.incidence_rad)


def test_identical_seed_reproduces_identical_tallies() -> None:
    first, second = _run(seed=99), _run(seed=99)
    assert first.tallies == second.tallies


def test_identical_seed_reproduces_identical_next_event_output() -> None:
    # Next-event draws a data-dependent number of extra random numbers per collision
    # (ratio tracking) - a more demanding reproducibility case than the fixed, fully
    # vectorized draw pattern the analog estimator uses.
    first = _run(seed=55, estimator="next_event")
    second = _run(seed=55, estimator="next_event")
    np.testing.assert_array_equal(first.photons.weight, second.photons.weight)
    assert first.tallies.detected_weight == second.tallies.detected_weight


def test_different_seeds_produce_different_output() -> None:
    # The inverse check: reproducibility must not be an artifact of a broken RNG that
    # ignores the seed entirely.
    first, second = _run(seed=1), _run(seed=2)
    assert not np.array_equal(first.photons.arrival_time_s, second.photons.arrival_time_s)


def test_seed_tree_is_recorded_and_reproducible() -> None:
    rng_a = NumpyRng(123)
    rng_b = NumpyRng(123)
    tree_a = rng_a.seed_tree()
    tree_b = rng_b.seed_tree()
    assert tree_a == tree_b
    assert isinstance(tree_a.root_seed, int)


def test_run_metadata_is_a_frozen_reproducible_record() -> None:
    # RunMetadata itself (data.md's provenance record) must be a plain, comparable
    # value - two independently constructed records with the same inputs are equal.
    meta_kwargs = dict(
        scenario="III",
        medium_type="inhomogeneous",
        optical_model="haltrin",
        model_parameters={"wavelength_nm": 520.0},
        effects=("turbulence",),
        effect_parameters=({"rms_fluctuation": 0.001},),
        wavelength_nm=520.0,
        source=Source(
            position=[0.0, 0.0, 0.0],
            direction=[0.0, 0.0, -1.0],
            wavelength_nm=520.0,
            divergence_rad=0.2,
        ),
        receiver=Receiver(
            position=[0.0, 0.0, -10.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
        ),
        majorant=0.4,
        sampling=SamplingConfig(n_photons=100, estimator="analog"),
        seed_tree=SeedTree(root_seed=7),
        rng_impl="NumpyRng",
        timestamp_utc="2026-01-01T00:00:00Z",
        code_version="abc123",
    )
    assert RunMetadata(**meta_kwargs) == RunMetadata(**meta_kwargs)
