"""EnvironmentalState construction; declared-parameter consumption."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from uowc.core.environment import EnvironmentalState


def test_chlorophyll_is_the_only_required_field() -> None:
    state = EnvironmentalState(chlorophyll=0.5)
    assert state.chlorophyll == 0.5
    assert state.cdom_440 is None
    assert state.nap is None
    assert state.temperature is None
    assert state.salinity is None


def test_extra_defaults_to_empty_mapping() -> None:
    state = EnvironmentalState(chlorophyll=0.5)
    assert dict(state.extra) == {}


def test_extra_carries_arbitrary_named_constituents() -> None:
    state = EnvironmentalState(chlorophyll=0.5, extra={"custom_field": 3.0})
    assert state.extra["custom_field"] == 3.0


def test_declared_constituents_are_independently_settable() -> None:
    state = EnvironmentalState(
        chlorophyll=0.5, cdom_440=0.1, nap=2.0, temperature=18.0, salinity=35.0
    )
    assert state.cdom_440 == 0.1
    assert state.nap == 2.0
    assert state.temperature == 18.0
    assert state.salinity == 35.0


def test_is_frozen() -> None:
    state = EnvironmentalState(chlorophyll=0.5)
    with pytest.raises(FrozenInstanceError):
        state.chlorophyll = 0.6  # type: ignore[misc]


def test_scalar_fields_stay_plain_python_scalars() -> None:
    # scalar_or_readonly must not silently promote a scalar to a 0-d array - a model
    # calling float(state.chlorophyll) or comparing state.chlorophyll == x must keep
    # working the same way whether or not the field ever holds a batch.
    state = EnvironmentalState(chlorophyll=0.5)
    assert isinstance(state.chlorophyll, float)


def test_array_valued_fields_support_a_batch_of_positions() -> None:
    chlorophyll = np.array([0.1, 0.2, 0.3])
    state = EnvironmentalState(chlorophyll=chlorophyll)
    np.testing.assert_array_equal(state.chlorophyll, chlorophyll)


def test_array_valued_fields_are_frozen_against_mutation() -> None:
    chlorophyll = np.array([0.1, 0.2, 0.3])
    state = EnvironmentalState(chlorophyll=chlorophyll)
    with pytest.raises(ValueError, match="read-only"):
        state.chlorophyll[0] = 99.0


def test_construction_freezes_the_callers_own_array_too() -> None:
    # units.scalar_or_readonly does not copy an already-array input - it freezes the
    # same buffer in place (its own docstring: "the value object... takes ownership;
    # callers must not mutate... after handing them to a core value object"). This is
    # the documented aliasing contract, not an accident, so it is worth pinning down.
    source = np.array([0.1, 0.2, 0.3])
    EnvironmentalState(chlorophyll=source)
    with pytest.raises(ValueError, match="read-only"):
        source[0] = 99.0
