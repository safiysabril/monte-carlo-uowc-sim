"""Online tally accumulation and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import DetectedPhotons, Region
from uowc.transport.tallies import (
    available_tallies,
    build_angle_of_arrival_tally,
    build_cir_tally,
    build_depth_deposition_tally,
    build_requested_tallies,
)


def _photons(arrival_time_s, incidence_rad) -> DetectedPhotons:
    n = len(arrival_time_s)
    return DetectedPhotons(
        arrival_time_s=np.asarray(arrival_time_s, dtype=float),
        path_length_m=np.zeros(n),
        weight=np.ones(n),
        n_scatters=np.zeros(n, dtype=np.int64),
        incidence_rad=np.asarray(incidence_rad, dtype=float),
    )


# --- build_cir_tally -----------------------------------------------------------------


def test_cir_tally_conserves_total_weight() -> None:
    photons = _photons([1e-8, 2e-8, 2.5e-8, 9e-8], [0.0, 0.0, 0.0, 0.0])
    tally = build_cir_tally(photons, n_bins=10)
    assert tally.name == "cir"
    assert tally.values.sum() == pytest.approx(4.0)
    assert tally.axis_names == ("arrival_time_s",)
    assert tally.edges[0].shape == (11,)


def test_cir_tally_handles_no_detections() -> None:
    photons = _photons([], [])
    tally = build_cir_tally(photons, n_bins=5)
    assert tally.values.sum() == pytest.approx(0.0)
    assert tally.edges[0].shape == (6,)


def test_cir_tally_handles_a_single_arrival_time() -> None:
    photons = _photons([5e-8], [0.1])
    tally = build_cir_tally(photons, n_bins=4)
    assert tally.values.sum() == pytest.approx(1.0)


# --- build_angle_of_arrival_tally ------------------------------------------------------


def test_angle_of_arrival_tally_conserves_total_weight() -> None:
    photons = _photons([1e-8, 2e-8, 3e-8], [0.0, 0.3, 0.6])
    tally = build_angle_of_arrival_tally(photons, n_bins=10)
    assert tally.name == "angle_of_arrival"
    assert tally.values.sum() == pytest.approx(3.0)
    assert tally.axis_names == ("incidence_rad",)


# --- build_depth_deposition_tally -------------------------------------------------------


def test_depth_deposition_tally_conserves_total_weight() -> None:
    bounds = Region(lower=[-10.0, -10.0, -20.0], upper=[10.0, 10.0, 0.0])
    positions = np.array([[0.0, 0.0, -5.0], [1.0, 1.0, -10.0], [0.0, 0.0, -15.0]])
    weights = np.ones(3)
    tally = build_depth_deposition_tally(positions, weights, domain_bounds=bounds, n_bins=8)
    assert tally.name == "depth_deposition"
    assert tally.axis_names == ("depth_m",)
    assert tally.values.sum() == pytest.approx(3.0)
    # edges span the full domain depth extent [0, 20], not just the observed range.
    assert tally.edges[0][0] == pytest.approx(0.0)
    assert tally.edges[0][-1] == pytest.approx(20.0)


def test_depth_deposition_tally_handles_no_absorptions() -> None:
    bounds = Region(lower=[-10.0, -10.0, -20.0], upper=[10.0, 10.0, 0.0])
    tally = build_depth_deposition_tally(
        np.empty((0, 3)), np.empty(0), domain_bounds=bounds, n_bins=8
    )
    assert tally.values.sum() == pytest.approx(0.0)


def test_deeper_positions_map_to_larger_depth_bins() -> None:
    bounds = Region(lower=[-10.0, -10.0, -20.0], upper=[10.0, 10.0, 0.0])
    shallow = build_depth_deposition_tally(
        np.array([[0.0, 0.0, -1.0]]), np.ones(1), domain_bounds=bounds, n_bins=4
    )
    deep = build_depth_deposition_tally(
        np.array([[0.0, 0.0, -19.0]]), np.ones(1), domain_bounds=bounds, n_bins=4
    )
    shallow_bin = int(np.argmax(shallow.values))
    deep_bin = int(np.argmax(deep.values))
    assert deep_bin > shallow_bin


# --- build_requested_tallies / registry ------------------------------------------------


def test_available_tallies_lists_the_three_known_names() -> None:
    assert set(available_tallies()) == {"cir", "angle_of_arrival", "depth_deposition"}


def test_build_requested_tallies_builds_only_what_was_asked_for() -> None:
    photons = _photons([1e-8, 2e-8], [0.0, 0.1])
    bounds = Region(lower=[-1.0, -1.0, -5.0], upper=[1.0, 1.0, 0.0])
    result = build_requested_tallies(
        ["cir"],
        photons=photons,
        absorbed_positions=np.empty((0, 3)),
        absorbed_weights=np.empty(0),
        domain_bounds=bounds,
    )
    assert set(result) == {"cir"}


def test_build_requested_tallies_empty_names_returns_empty_dict() -> None:
    photons = _photons([1e-8], [0.0])
    bounds = Region(lower=[-1.0, -1.0, -5.0], upper=[1.0, 1.0, 0.0])
    result = build_requested_tallies(
        (),
        photons=photons,
        absorbed_positions=np.empty((0, 3)),
        absorbed_weights=np.empty(0),
        domain_bounds=bounds,
    )
    assert result == {}


def test_build_requested_tallies_rejects_unknown_name() -> None:
    photons = _photons([1e-8], [0.0])
    bounds = Region(lower=[-1.0, -1.0, -5.0], upper=[1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="unknown tallies"):
        build_requested_tallies(
            ["not_a_real_tally"],
            photons=photons,
            absorbed_positions=np.empty((0, 3)),
            absorbed_weights=np.empty(0),
            domain_bounds=bounds,
        )
