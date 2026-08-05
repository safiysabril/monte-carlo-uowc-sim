"""Result invariant to majorant choice and cell subdivision.

transport.md's *Majorant condition*: "c_max affects efficiency only, never the
answer. Doubling it doubles the null collisions and leaves every expectation
unchanged. This is the algorithm's sharpest verification test."

Uses ``InhomogeneousMedium``'s ``majorant_safety`` multiplier (a real, already-present
constructor argument - not a test double) to run the identical Scenario II medium at
c_max, 2*c_max and 10*c_max, and checks that both a proportion-valued metric
(detected fraction) and a continuous-valued metric (mean arrival time) agree across
all three within their combined Monte Carlo uncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.core.rng import NumpyRng
from uowc.media import InhomogeneousMedium, KamedaModel
from uowc.metrics.statistics import wilson_interval
from uowc.optics import HaltrinModel
from uowc.transport import WoodcockDeltaTracker

pytestmark = pytest.mark.verification

_WAVELENGTH_NM = 520.0
_N_PHOTONS = 50_000
_BOUNDS = Region(lower=[-30.0, -30.0, -60.0], upper=[30.0, 30.0, 0.0])
#: Generous multiple of the combined standard error - guards against a real bug (a
#: majorant-dependent bias) without flaking on ordinary Monte Carlo noise, since the
#: three runs are independent replicates (different seeds), not paired samples.
_SIGMA_TOLERANCE = 6.0


def _model() -> HaltrinModel:
    return HaltrinModel(
        wavelength_nm=_WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.2,
        chlorophyll_specific_absorption_m2_mg=0.04,
        pure_water_scattering_m_inv=0.05,
    )


def _profile() -> KamedaModel:
    return KamedaModel(
        background_mg_m3=0.1, peak_integral_mg_m2=40.0, peak_depth_m=40.0, peak_width_m=10.0
    )


def _medium(majorant_safety: float) -> InhomogeneousMedium:
    return InhomogeneousMedium.scenario_ii(
        profile=_profile(),
        model=_model(),
        wavelength_nm=_WAVELENGTH_NM,
        bounds=_BOUNDS,
        majorant_safety=majorant_safety,
    )


def _run(majorant_safety: float, seed: int):
    medium = _medium(majorant_safety)
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=_WAVELENGTH_NM,
        divergence_rad=0.1,
    )
    receiver = Receiver(
        position=[0.0, 0.0, -15.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=np.pi / 4
    )
    config = SamplingConfig(n_photons=_N_PHOTONS, estimator="analog", max_scatter_events=500)
    return WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(seed), config)


@pytest.fixture(scope="module")
def runs():
    return {
        1.0: _run(1.0, seed=100),
        2.0: _run(2.0, seed=200),
        10.0: _run(10.0, seed=300),
    }


def test_majorant_is_actually_scaled_between_runs() -> None:
    # Sanity check on the test setup itself: c_max must really scale linearly with
    # majorant_safety, or the invariance checks below would not be testing anything.
    baseline = _medium(1.0).acceleration.majorant(_BOUNDS)
    for safety in (2.0, 10.0):
        scaled = _medium(safety).acceleration.majorant(_BOUNDS)
        assert scaled == pytest.approx(baseline * safety)


def test_detected_fraction_is_invariant_to_the_majorant(runs) -> None:
    fractions_and_se = []
    for out in runs.values():
        p = out.tallies.detected / out.tallies.launched
        se, _, _ = wilson_interval(p, out.tallies.launched)
        fractions_and_se.append((p, se))

    baseline_p, baseline_se = fractions_and_se[0]
    for p, se in fractions_and_se[1:]:
        combined_se = np.sqrt(baseline_se**2 + se**2)
        assert abs(p - baseline_p) < _SIGMA_TOLERANCE * max(combined_se, 1e-12)


def test_mean_arrival_time_is_invariant_to_the_majorant(runs) -> None:
    # All weights are 1.0 under the analog estimator, so the weighted mean among
    # detected photons reduces to a plain sample mean, and its standard error to the
    # ordinary sample SE - no ratio-estimator subtlety to account for here.
    means_and_se = []
    for out in runs.values():
        arrivals = out.photons.arrival_time_s
        mean_t = float(np.mean(arrivals))
        se_t = float(np.std(arrivals, ddof=1) / np.sqrt(arrivals.size))
        means_and_se.append((mean_t, se_t))

    baseline_mean, baseline_se = means_and_se[0]
    for mean_t, se_t in means_and_se[1:]:
        combined_se = np.sqrt(baseline_se**2 + se_t**2)
        assert abs(mean_t - baseline_mean) < _SIGMA_TOLERANCE * combined_se
