"""Fresnel coefficients; total-internal-reflection angle; bottom albedo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from uowc.core.ports import Boundary
from uowc.core.rng import NumpyRng
from uowc.media.boundaries import AirWaterSurface, LambertianBottom

# --- AirWaterSurface: construction ----------------------------------------------------


def test_conforms_to_boundary_port() -> None:
    assert isinstance(AirWaterSurface(), Boundary)
    assert AirWaterSurface().name == "air_water_surface"


def test_rejects_nonpositive_refractive_index() -> None:
    with pytest.raises(ValueError):
        AirWaterSurface(n_air=0.0, n_water=1.34)
    with pytest.raises(ValueError):
        AirWaterSurface(n_air=1.0, n_water=-1.0)


# --- Fresnel reflectance ---------------------------------------------------------------


def test_normal_incidence_reflectance_matches_the_textbook_two_percent() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    r0 = ((surface.n_water - surface.n_air) / (surface.n_water + surface.n_air)) ** 2
    assert surface.reflectance(1.0) == pytest.approx(r0)
    assert surface.reflectance(1.0) == pytest.approx(0.02, abs=0.002)


def test_reflectance_increases_toward_the_critical_angle() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    near_normal = surface.reflectance(math.cos(math.radians(10.0)))
    near_grazing = surface.reflectance(math.cos(math.radians(45.0)))
    assert near_grazing > near_normal


def test_reflectance_is_bounded_in_zero_one_below_critical_angle() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    for angle_deg in [0.0, 10.0, 20.0, 30.0, 40.0]:
        r = surface.reflectance(math.cos(math.radians(angle_deg)))
        assert 0.0 <= r <= 1.0


def test_reflectance_rejects_out_of_range_cosine() -> None:
    surface = AirWaterSurface()
    with pytest.raises(ValueError):
        surface.reflectance(-0.1)
    with pytest.raises(ValueError):
        surface.reflectance(1.1)


# --- Critical angle / total internal reflection -----------------------------------------


def test_critical_angle_is_in_the_documented_48_to_49_degree_range() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    degrees = math.degrees(surface.critical_angle_rad)
    assert 48.0 <= degrees <= 49.0


def test_critical_angle_matches_snells_law() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    expected = math.asin(surface.n_air / surface.n_water)
    assert surface.critical_angle_rad == pytest.approx(expected)


def test_beyond_critical_angle_reflectance_is_total() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    beyond = surface.critical_angle_rad + math.radians(5.0)
    assert surface.reflectance(math.cos(beyond)) == pytest.approx(1.0)


def test_just_below_critical_angle_reflectance_is_not_yet_total() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    just_below = surface.critical_angle_rad - math.radians(2.0)
    assert surface.reflectance(math.cos(just_below)) < 1.0


# --- interact(): probabilistic reflect/transmit ------------------------------------------


def test_interact_rejects_direction_not_pointing_at_the_surface() -> None:
    surface = AirWaterSurface()
    with pytest.raises(ValueError):
        surface.interact(
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 500.0, NumpyRng(0)
        )


def test_interact_beyond_critical_angle_always_reflects() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    beyond = surface.critical_angle_rad + math.radians(10.0)
    direction = np.array([math.sin(beyond), 0.0, math.cos(beyond)])
    rng = NumpyRng(1)
    for _ in range(200):
        outcome = surface.interact(np.array([0.0, 0.0, 0.0]), direction, 500.0, rng)
        assert not outcome.transmitted
        assert not outcome.absorbed
        assert outcome.direction[2] == pytest.approx(-direction[2])


def test_interact_at_normal_incidence_mostly_transmits() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    direction = np.array([0.0, 0.0, 1.0])
    rng = NumpyRng(2)
    n = 5000
    transmitted = sum(
        surface.interact(np.array([0.0, 0.0, 0.0]), direction, 500.0, rng).transmitted
        for _ in range(n)
    )
    # ~98% transmittance at normal incidence.
    assert transmitted / n == pytest.approx(0.98, abs=0.02)


def test_interact_reflection_flips_only_the_vertical_component() -> None:
    surface = AirWaterSurface(n_air=1.0, n_water=1.34)
    beyond = surface.critical_angle_rad + math.radians(10.0)
    direction = np.array([math.sin(beyond), 0.3, math.cos(beyond)])
    direction = direction / np.linalg.norm(direction)
    outcome = surface.interact(np.array([0.0, 0.0, 0.0]), direction, 500.0, NumpyRng(3))
    np.testing.assert_allclose(outcome.direction[:2], direction[:2])
    assert outcome.direction[2] == pytest.approx(-direction[2])


# --- LambertianBottom --------------------------------------------------------------------


def test_bottom_conforms_to_boundary_port() -> None:
    assert isinstance(LambertianBottom(albedo=0.5, depth_m=20.0), Boundary)
    assert LambertianBottom(albedo=0.5, depth_m=20.0).name == "lambertian_bottom"


def test_bottom_rejects_invalid_albedo() -> None:
    with pytest.raises(ValueError):
        LambertianBottom(albedo=1.5, depth_m=20.0)
    with pytest.raises(ValueError):
        LambertianBottom(albedo=-0.1, depth_m=20.0)


def test_bottom_rejects_nonpositive_depth() -> None:
    with pytest.raises(ValueError):
        LambertianBottom(albedo=0.5, depth_m=0.0)


def test_bottom_interact_rejects_direction_not_pointing_down() -> None:
    bottom = LambertianBottom(albedo=0.5, depth_m=20.0)
    with pytest.raises(ValueError):
        bottom.interact(
            np.array([0.0, 0.0, -20.0]), np.array([0.0, 0.0, 1.0]), 500.0, NumpyRng(4)
        )


def test_zero_albedo_always_absorbs() -> None:
    bottom = LambertianBottom(albedo=0.0, depth_m=20.0)
    rng = NumpyRng(5)
    direction = np.array([0.0, 0.0, -1.0])
    for _ in range(50):
        outcome = bottom.interact(np.array([0.0, 0.0, -20.0]), direction, 500.0, rng)
        assert outcome.absorbed
        assert not outcome.transmitted


def test_full_albedo_never_absorbs_and_reflects_upward() -> None:
    bottom = LambertianBottom(albedo=1.0, depth_m=20.0)
    rng = NumpyRng(6)
    direction = np.array([0.0, 0.0, -1.0])
    for _ in range(50):
        outcome = bottom.interact(np.array([0.0, 0.0, -20.0]), direction, 500.0, rng)
        assert not outcome.absorbed
        assert not outcome.transmitted
        assert outcome.direction[2] > 0.0  # reflected back up into the water column


def test_reflected_directions_are_unit_vectors() -> None:
    bottom = LambertianBottom(albedo=1.0, depth_m=20.0)
    rng = NumpyRng(7)
    direction = np.array([0.0, 0.0, -1.0])
    for _ in range(20):
        outcome = bottom.interact(np.array([0.0, 0.0, -20.0]), direction, 500.0, rng)
        assert np.linalg.norm(outcome.direction) == pytest.approx(1.0)


def test_diffuse_reflection_mean_cosine_matches_lambertian_theory() -> None:
    """A cosine-weighted hemisphere sample has E[cos theta] = 2/3 (Malley's method) -
    the defining statistical signature of ideal Lambertian reflection."""
    bottom = LambertianBottom(albedo=1.0, depth_m=20.0)
    rng = NumpyRng(8)
    direction = np.array([0.0, 0.0, -1.0])
    n = 20_000
    cosines = np.array(
        [
            bottom.interact(np.array([0.0, 0.0, -20.0]), direction, 500.0, rng).direction[2]
            for _ in range(n)
        ]
    )
    assert cosines.mean() == pytest.approx(2.0 / 3.0, abs=0.02)


def test_diffuse_reflection_independent_of_incoming_direction() -> None:
    """Ideal Lambertian reflection depends only on the surface normal, not the
    incoming ray - a steep and a shallow incoming angle must produce statistically
    identical outgoing distributions."""
    bottom = LambertianBottom(albedo=1.0, depth_m=20.0)
    n = 20_000
    steep_rng = NumpyRng(9)
    steep = np.array(
        [
            bottom.interact(
                np.array([0.0, 0.0, -20.0]), np.array([0.0, 0.0, -1.0]), 500.0, steep_rng
            ).direction[2]
            for _ in range(n)
        ]
    )
    shallow_direction = np.array([0.9, 0.0, -0.436])
    shallow_direction /= np.linalg.norm(shallow_direction)
    shallow_rng = NumpyRng(10)
    shallow = np.array(
        [
            bottom.interact(
                np.array([0.0, 0.0, -20.0]), shallow_direction, 500.0, shallow_rng
            ).direction[2]
            for _ in range(n)
        ]
    )
    assert steep.mean() == pytest.approx(shallow.mean(), abs=0.02)
