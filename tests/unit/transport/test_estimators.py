"""Analog vs next-event unbiasedness on a known case.

transport.md: next-event estimation ("local estimate") must be validated against the
analog reference before being trusted. This module does that validation directly -
same stub medium style as test_woodcock.py, no optical-property model or metric
imported - plus a focused check of the ratio-tracking transmittance sub-estimator
that next-event's connecting-ray weighting depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import uowc.transport.estimators as estimators_module
from uowc.core import (
    IOP,
    LocalOpticalState,
    Receiver,
    Region,
    SamplingConfig,
    Scattering,
    ScatteringComponent,
    Source,
)
from uowc.core.rng import NumpyRng
from uowc.transport import WoodcockDeltaTracker
from uowc.transport.woodcock import _ratio_tracking_transmittance


def test_estimators_module_intentionally_exports_nothing() -> None:
    # Both estimators are implemented directly in WoodcockDeltaTracker - see
    # uowc.transport.estimators for why there is no separate Estimator abstraction.
    assert estimators_module.__all__ == []


class _Isotropic:
    asymmetry = 0.0

    def sample_cos_theta(self, u):
        return 2.0 * np.asarray(u, dtype=float) - 1.0

    def value(self, cos_theta):
        return np.full(np.asarray(cos_theta, dtype=float).shape, 1.0 / (4.0 * np.pi))


@dataclass
class _ConstField:
    a: float
    b: float
    n: float = 1.34

    def _iop(self) -> IOP:
        if self.b > 0.0:
            scattering = Scattering((ScatteringComponent(self.b, _Isotropic()),))
        else:
            scattering = Scattering(())
        return IOP(absorption=self.a, scattering=scattering, wavelength_nm=500.0)

    def extinction(self, positions, time_s=0.0):
        p = np.asarray(positions, dtype=float)
        return np.full(p.shape[:-1], self.a + self.b)

    def refractive_index(self, positions, time_s=0.0):
        p = np.asarray(positions, dtype=float)
        return np.full(p.shape[:-1], self.n)

    def refractive_index_gradient(self, positions, time_s=0.0):
        return np.zeros_like(np.asarray(positions, dtype=float))

    def local_state(self, position, time_s=0.0):
        return LocalOpticalState(iop=self._iop(), refractive_index=self.n)


@dataclass
class _StubDomain:
    region: Region

    def contains(self, positions):
        p = np.asarray(positions, dtype=float)
        return np.all((p >= self.region.lower) & (p <= self.region.upper), axis=-1)

    def bounds(self):
        return self.region


@dataclass
class _StubAccel:
    c_max: float

    def majorant(self, region):
        return self.c_max


@dataclass
class _StubMedium:
    field: object
    domain: object
    acceleration: object


def make_medium(field, half=20.0, depth=15.0) -> _StubMedium:
    region = Region(lower=[-half, -half, -depth], upper=[half, half, 0.0])
    return _StubMedium(
        field=field, domain=_StubDomain(region), acceleration=_StubAccel(field.a + field.b)
    )


def _weighted_mean_and_se(weights: np.ndarray, n_launched: int) -> tuple[float, float]:
    """research-methodology.md's weighted-estimator formula: variance over weights
    (zero for undetected photons), not over the count of nonzero entries."""
    s1 = float(weights.sum())
    s2 = float((weights**2).sum())
    mean = s1 / n_launched
    variance = (s2 - s1**2 / n_launched) / (n_launched * (n_launched - 1))
    return mean, float(np.sqrt(variance))


# --- _ratio_tracking_transmittance: Beer-Lambert cross-check ------------------------


class _HomogeneousExtinction:
    def __init__(self, c: float) -> None:
        self.c = c

    def extinction(self, positions):
        p = np.asarray(positions, dtype=float)
        return np.full(p.shape[:-1], self.c)


def test_ratio_tracking_matches_beer_lambert_in_a_homogeneous_medium() -> None:
    c = 0.2
    distance = 10.0
    field = _HomogeneousExtinction(c)
    rng = NumpyRng(42)
    n = 4000
    samples = np.array(
        [
            _ratio_tracking_transmittance(
                field, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), distance, c, rng
            )
            for _ in range(n)
        ]
    )
    mean = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(n)
    expected = np.exp(-c * distance)
    assert abs(mean - expected) < 4.0 * se


def test_ratio_tracking_transmittance_is_bounded_in_zero_one() -> None:
    field = _HomogeneousExtinction(0.5)
    rng = NumpyRng(1)
    for _ in range(200):
        t = _ratio_tracking_transmittance(
            field, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 5.0, 0.5, rng
        )
        assert 0.0 <= t <= 1.0


def test_ratio_tracking_transmittance_decreases_with_distance() -> None:
    field = _HomogeneousExtinction(0.3)
    n = 2000
    short = np.mean(
        [
            _ratio_tracking_transmittance(
                field, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 2.0, 0.3, NumpyRng(i)
            )
            for i in range(n)
        ]
    )
    long = np.mean(
        [
            _ratio_tracking_transmittance(
                field, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 20.0, 0.3, NumpyRng(i)
            )
            for i in range(n)
        ]
    )
    assert long < short


# --- WoodcockDeltaTracker: unbiasedness of the full next-event estimator ------------


def _offaxis_scattering_scenario():
    """A photon can only reach this receiver by scattering - the collimated beam's
    direct path does not pass through it - so the comparison below exercises the
    next-event contribution machinery, not the (unmodified) direct-path ballistic
    test that both estimators share."""
    field = _ConstField(a=0.05, b=0.3)
    medium = make_medium(field)
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    receiver = Receiver(
        position=[3.0, 0.0, -10.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=0.3, fov_rad=np.pi / 3
    )
    return medium, source, receiver


def test_next_event_agrees_with_analog_on_a_known_scattering_case() -> None:
    medium, source, receiver = _offaxis_scattering_scenario()
    n = 20_000

    analog_out = WoodcockDeltaTracker().run(
        medium,
        source,
        receiver,
        NumpyRng(1),
        SamplingConfig(n_photons=n, estimator="analog", max_scatter_events=200),
    )
    nee_out = WoodcockDeltaTracker().run(
        medium,
        source,
        receiver,
        NumpyRng(2),
        SamplingConfig(n_photons=n, estimator="next_event", max_scatter_events=200),
    )

    mean_analog, se_analog = _weighted_mean_and_se(analog_out.photons.weight, n)
    mean_nee, se_nee = _weighted_mean_and_se(nee_out.photons.weight, n)

    # Both estimate the same physical quantity (detected power fraction); a real bug
    # (wrong formula, missing factor) would show up as a large, systematic offset far
    # outside statistical noise - a generous multiple of the combined SE keeps this
    # from flaking while still catching that.
    combined_se = np.sqrt(se_analog**2 + se_nee**2)
    assert abs(mean_analog - mean_nee) < 5.0 * combined_se

    # The entire point of next-event estimation (transport.md): far lower variance
    # for the same photon budget, since it stops relying on rare lucky ballistic hits.
    assert se_nee < se_analog / 3.0


def test_next_event_reduces_to_ballistic_only_without_scattering() -> None:
    # b=0: no real collisions ever produce a scatter event, so next-event's only
    # source of detections is the same pre-scatter ballistic test analog uses -
    # the two estimators should therefore agree almost exactly, not just statistically.
    field = _ConstField(a=0.1, b=0.0)
    medium = make_medium(field)
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    receiver = Receiver(
        position=[0.0, 0.0, -10.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=2.0, fov_rad=np.pi / 2
    )
    cfg = SamplingConfig(n_photons=5000, estimator="analog", max_scatter_events=200)
    analog_out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(3), cfg)
    nee_out = WoodcockDeltaTracker().run(
        medium,
        source,
        receiver,
        NumpyRng(3),
        SamplingConfig(n_photons=5000, estimator="next_event", max_scatter_events=200),
    )
    assert nee_out.tallies.detected == analog_out.tallies.detected
    assert nee_out.tallies.detected_weight == pytest.approx(analog_out.tallies.detected_weight)


def test_next_event_contributes_nothing_outside_the_receiver_fov() -> None:
    field = _ConstField(a=0.05, b=0.3)
    medium = make_medium(field)
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    # An off-axis receiver with a near-zero FOV: no scattering direction can satisfy
    # the incidence-angle acceptance test, so every next-event contribution is zero.
    receiver = Receiver(
        position=[3.0, 0.0, -10.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=0.3, fov_rad=1e-6
    )
    cfg = SamplingConfig(n_photons=5000, estimator="next_event", max_scatter_events=200)
    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(4), cfg)
    assert out.tallies.detected_weight == pytest.approx(0.0)


def test_next_event_breaks_the_conservation_identity_by_design() -> None:
    # Documented in the module docstring: next-event adds a second, independent
    # estimate of detected power without removing weight from the main walk's own
    # conservation budget, so detected+absorbed+escaped+killed exceeds launched.
    medium, source, receiver = _offaxis_scattering_scenario()
    cfg = SamplingConfig(n_photons=5000, estimator="next_event", max_scatter_events=200)
    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(5), cfg)
    t = out.tallies
    total = t.detected_weight + t.absorbed_weight + t.escaped_weight + t.extra["killed_weight"]
    assert total > t.launched


def test_analog_conservation_identity_is_unaffected_by_next_event_support() -> None:
    medium, source, receiver = _offaxis_scattering_scenario()
    cfg = SamplingConfig(n_photons=5000, estimator="analog", max_scatter_events=200)
    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(6), cfg)
    t = out.tallies
    total = t.detected_weight + t.absorbed_weight + t.escaped_weight + t.extra["killed_weight"]
    assert total == pytest.approx(t.launched)


def test_next_event_is_reproducible_for_the_same_seed() -> None:
    medium, source, receiver = _offaxis_scattering_scenario()
    cfg = SamplingConfig(n_photons=3000, estimator="next_event", max_scatter_events=200)
    first = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(7), cfg)
    second = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(7), cfg)
    assert first.tallies.detected_weight == second.tallies.detected_weight
    np.testing.assert_array_equal(first.photons.weight, second.photons.weight)
