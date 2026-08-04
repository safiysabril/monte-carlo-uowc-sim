"""Tests for the Woodcock delta-tracking engine.

The engine is exercised against a hand-rolled stub ``Medium`` (constant or layered
extinction, isotropic scattering) - no optical-property model, medium implementation,
environmental effect, or metric is imported. This both tests the mechanics and
demonstrates that transport depends only on the ``Medium`` port.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from uowc.core import (
    IOP,
    LocalOpticalState,
    Region,
    SamplingConfig,
    Scattering,
    ScatteringComponent,
    Source,
    Receiver,
)
from uowc.core.ports import TransportEngine
from uowc.core.rng import NumpyRng
from uowc.transport import WoodcockDeltaTracker


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
class _LayeredField(_ConstField):
    """Extinction halved outside a band [-6, -4] m, to exercise null collisions."""

    def extinction(self, positions, time_s=0.0):
        p = np.asarray(positions, dtype=float)
        z = p[..., 2]
        c_max = self.a + self.b
        return np.where((z <= -4.0) & (z >= -6.0), c_max, 0.5 * c_max)


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


def make_medium(field, half=50.0, depth=200.0) -> _StubMedium:
    region = Region(lower=[-half, -half, -depth], upper=[half, half, 0.0])
    c_max = float(field.extinction(np.array([[0.0, 0.0, -5.0]]))[0])
    # for the layered field the band value is the true maximum
    c_max = max(c_max, field.a + field.b)
    return _StubMedium(field=field, domain=_StubDomain(region), acceleration=_StubAccel(c_max))


def collimated(depth_to_receiver: float, aperture=5.0, fov=np.pi / 2):
    source = Source(position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], wavelength_nm=500.0, divergence_rad=0.0)
    receiver = Receiver(
        position=[0.0, 0.0, -depth_to_receiver], normal=[0.0, 0.0, 1.0], aperture_radius_m=aperture, fov_rad=fov
    )
    return source, receiver


def test_conforms_to_transport_engine() -> None:
    assert isinstance(WoodcockDeltaTracker(), TransportEngine)
    assert WoodcockDeltaTracker().name == "woodcock"


def test_non_analog_estimator_not_implemented() -> None:
    medium = make_medium(_ConstField(0.1, 0.0))
    source, receiver = collimated(10.0)
    with pytest.raises(NotImplementedError):
        WoodcockDeltaTracker().run(
            medium, source, receiver, NumpyRng(0), SamplingConfig(n_photons=10, estimator="next_event")
        )


def test_beer_lambert_detected_fraction() -> None:
    a, length = 0.1, 10.0
    medium = make_medium(_ConstField(a, 0.0))
    source, receiver = collimated(length)
    out = WoodcockDeltaTracker().run(
        medium, source, receiver, NumpyRng(1), SamplingConfig(n_photons=100_000, estimator="analog")
    )
    frac = out.tallies.detected / out.tallies.launched
    assert frac == pytest.approx(np.exp(-a * length), abs=0.01)


def test_detected_arrival_time_is_optical_path() -> None:
    a, length, n = 0.05, 8.0, 1.34
    medium = make_medium(_ConstField(a, 0.0, n=n))
    source, receiver = collimated(length)
    out = WoodcockDeltaTracker().run(
        medium, source, receiver, NumpyRng(2), SamplingConfig(n_photons=2000, estimator="analog")
    )
    # collimated, non-scattering: every detection travels exactly `length` at speed c/n
    expected_time = length * n / 299_792_458.0
    assert np.allclose(out.photons.arrival_time_s, expected_time)
    assert np.allclose(out.photons.path_length_m, length)
    assert np.all(out.photons.n_scatters == 0)


def test_conservation_with_scattering() -> None:
    medium = make_medium(_ConstField(0.3, 0.7), half=5.0, depth=10.0)
    source, receiver = collimated(5.0, aperture=0.5, fov=np.pi / 4)
    out = WoodcockDeltaTracker().run(
        medium, source, receiver, NumpyRng(3), SamplingConfig(n_photons=3000, estimator="analog")
    )
    t = out.tallies
    total = t.detected_weight + t.absorbed_weight + t.escaped_weight + t.extra["killed_weight"]
    assert total == pytest.approx(t.launched)
    assert t.launched == 3000
    assert t.detected == len(out.photons.weight)


def test_scattering_produces_multiple_scatter_events() -> None:
    # thin, highly-scattering medium with a close, wide detector: most photons reach
    # the detector and many scatter at least once before crossing it.
    medium = make_medium(_ConstField(0.02, 0.5), half=100.0, depth=6.0)
    source, receiver = collimated(2.0, aperture=100.0, fov=np.pi / 2)
    out = WoodcockDeltaTracker().run(
        medium, source, receiver, NumpyRng(4), SamplingConfig(n_photons=4000, estimator="analog")
    )
    assert out.tallies.detected > 0
    assert out.photons.n_scatters.max() > 0  # some detected photons scattered first


def test_null_collisions_preserve_conservation() -> None:
    medium = make_medium(_LayeredField(0.2, 0.6), half=5.0, depth=10.0)
    source, receiver = collimated(5.0, aperture=0.5, fov=np.pi / 4)
    out = WoodcockDeltaTracker().run(
        medium, source, receiver, NumpyRng(5), SamplingConfig(n_photons=2000, estimator="analog")
    )
    t = out.tallies
    total = t.detected_weight + t.absorbed_weight + t.escaped_weight + t.extra["killed_weight"]
    assert total == pytest.approx(t.launched)


def test_no_binned_tallies_by_default() -> None:
    medium = make_medium(_ConstField(0.1, 0.0))
    source, receiver = collimated(10.0)
    out = WoodcockDeltaTracker().run(
        medium, source, receiver, NumpyRng(6), SamplingConfig(n_photons=10, estimator="analog")
    )
    assert dict(out.binned) == {}


def test_cir_and_angle_of_arrival_tallies_are_built_on_request() -> None:
    medium = make_medium(_ConstField(0.1, 0.0))
    source, receiver = collimated(10.0)
    cfg = SamplingConfig(n_photons=500, estimator="analog", tallies=("cir", "angle_of_arrival"))
    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(7), cfg)
    assert set(out.binned) == {"cir", "angle_of_arrival"}
    assert out.binned["cir"].values.sum() == pytest.approx(out.tallies.detected_weight)
    assert out.binned["angle_of_arrival"].values.sum() == pytest.approx(out.tallies.detected_weight)


def test_depth_deposition_tally_conserves_absorbed_weight() -> None:
    medium = make_medium(_ConstField(0.3, 0.0), half=5.0, depth=10.0)
    source, receiver = collimated(5.0, aperture=0.1, fov=np.pi / 8)
    cfg = SamplingConfig(n_photons=2000, estimator="analog", tallies=("depth_deposition",))
    out = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(8), cfg)
    assert set(out.binned) == {"depth_deposition"}
    assert out.binned["depth_deposition"].values.sum() == pytest.approx(out.tallies.absorbed_weight)


def test_reproducible_for_same_seed() -> None:
    medium = make_medium(_ConstField(0.3, 0.7), half=5.0, depth=10.0)
    source, receiver = collimated(5.0, aperture=0.5, fov=np.pi / 4)
    cfg = SamplingConfig(n_photons=1500, estimator="analog")
    first = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(9), cfg)
    second = WoodcockDeltaTracker().run(medium, source, receiver, NumpyRng(9), cfg)
    assert first.tallies.detected == second.tallies.detected
    assert first.tallies.absorbed_weight == second.tallies.absorbed_weight
    assert np.array_equal(first.photons.arrival_time_s, second.photons.arrival_time_s)
    assert np.array_equal(first.photons.n_scatters, second.photons.n_scatters)
