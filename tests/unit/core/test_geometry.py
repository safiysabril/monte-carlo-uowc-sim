"""Link-geometry value objects: construction, immutability, and equality.

Source and Receiver hold array-valued fields (position, direction/normal). The
default dataclass-generated ``__eq__`` compares fields with ``==``, which for a
multi-element numpy array returns an array rather than a bool and raises numpy's
"ambiguous truth value" error - so both classes define their own ``__eq__`` doing
elementwise comparison. This module pins that down directly, since it is easy to
silently regress (removing the custom ``__eq__`` still type-checks and imports fine).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from uowc.core.geometry import BoundaryOutcome, Receiver, Region, Source

# --- Source ----------------------------------------------------------------------------


def test_source_is_frozen() -> None:
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        source.wavelength_nm = 600.0  # type: ignore[misc]


def test_source_position_and_direction_are_read_only() -> None:
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    with pytest.raises(ValueError, match="read-only"):
        source.position[0] = 1.0


def test_source_equality_does_not_raise_for_multi_element_arrays() -> None:
    # Regression guard: the dataclass-default __eq__ would raise
    # "The truth value of an array with more than one element is ambiguous" here.
    a = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    b = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    assert a == b


def test_source_equality_detects_a_differing_array_field() -> None:
    a = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    b = Source(
        position=[1.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    assert a != b


def test_source_equality_detects_a_differing_scalar_field() -> None:
    a = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    b = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=600.0,
        divergence_rad=0.0,
    )
    assert a != b


def test_source_equality_against_a_different_type_is_not_equal() -> None:
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=500.0,
        divergence_rad=0.0,
    )
    assert source != "not a source"
    assert (source == "not a source") is False


# --- Receiver --------------------------------------------------------------------------


def test_receiver_is_frozen() -> None:
    receiver = Receiver(
        position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
    )
    with pytest.raises(FrozenInstanceError):
        receiver.fov_rad = 2.0  # type: ignore[misc]


def test_receiver_equality_does_not_raise_for_multi_element_arrays() -> None:
    a = Receiver(
        position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
    )
    b = Receiver(
        position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
    )
    assert a == b


def test_receiver_equality_detects_a_differing_array_field() -> None:
    a = Receiver(
        position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
    )
    b = Receiver(
        position=[0.0, 0.0, -2.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
    )
    assert a != b


def test_receiver_equality_against_a_different_type_is_not_equal() -> None:
    receiver = Receiver(
        position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
    )
    assert (receiver == 5) is False


# --- Region ------------------------------------------------------------------------------


def test_region_is_frozen() -> None:
    region = Region(lower=[-1.0, -1.0, -1.0], upper=[1.0, 1.0, 0.0])
    with pytest.raises(FrozenInstanceError):
        region.lower = np.zeros(3)  # type: ignore[misc]


def test_region_bounds_are_read_only() -> None:
    region = Region(lower=[-1.0, -1.0, -1.0], upper=[1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="read-only"):
        region.upper[0] = 99.0


# --- BoundaryOutcome -----------------------------------------------------------------------


def test_boundary_outcome_direction_is_read_only() -> None:
    outcome = BoundaryOutcome(direction=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="read-only"):
        outcome.direction[0] = 1.0


def test_boundary_outcome_defaults_to_reflected() -> None:
    outcome = BoundaryOutcome(direction=np.array([0.0, 0.0, 1.0]))
    assert not outcome.absorbed
    assert not outcome.transmitted


def test_boundary_outcome_rejects_absorbed_and_transmitted_together() -> None:
    with pytest.raises(ValueError, match="cannot be both"):
        BoundaryOutcome(direction=np.array([0.0, 0.0, 1.0]), absorbed=True, transmitted=True)
