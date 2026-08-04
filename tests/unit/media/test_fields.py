"""N-D field sampling; 1D depth specialization."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.media.fields import GriddedScalarField


def _linear_1d() -> GriddedScalarField:
    depth = np.array([0.0, 10.0, 20.0, 30.0])
    concentration = 2.0 * depth + 1.0  # C(d) = 2d + 1, exactly linear
    return GriddedScalarField(axes=(depth,), values=concentration)


# --- construction validation -------------------------------------------------------


def test_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        GriddedScalarField(axes=(), values=np.array(0.0))


def test_rejects_non_increasing_axis() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        GriddedScalarField(axes=(np.array([0.0, 2.0, 1.0]),), values=np.array([1.0, 2.0, 3.0]))


def test_rejects_axis_with_fewer_than_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        GriddedScalarField(axes=(np.array([0.0]),), values=np.array([1.0]))


def test_rejects_mismatched_values_shape() -> None:
    with pytest.raises(ValueError, match="does not match axes shape"):
        GriddedScalarField(axes=(np.array([0.0, 1.0, 2.0]),), values=np.array([1.0, 2.0]))


def test_n_dimensions_and_bounds() -> None:
    field = _linear_1d()
    assert field.n_dimensions == 1
    assert field.lower == pytest.approx([0.0])
    assert field.upper == pytest.approx([30.0])


# --- 1-D interpolation ---------------------------------------------------------------


def test_interpolates_exactly_at_grid_points() -> None:
    field = _linear_1d()
    value = field.at(np.array([[10.0]]))
    assert value == pytest.approx([21.0])


def test_interpolates_linearly_between_grid_points() -> None:
    field = _linear_1d()
    value = field.at(np.array([[5.0]]))
    assert value == pytest.approx([11.0])  # exact for a linear function


def test_batched_query_preserves_shape() -> None:
    field = _linear_1d()
    positions = np.array([[0.0], [10.0], [20.0]])
    values = field.at(positions)
    assert values.shape == (3,)
    np.testing.assert_allclose(values, [1.0, 21.0, 41.0])


def test_scalar_in_scalar_out() -> None:
    field = _linear_1d()
    value = field.at(np.array([15.0]))
    assert isinstance(value, float)
    assert value == pytest.approx(31.0)


# --- extrapolation policy ------------------------------------------------------------


def test_error_policy_is_the_default_and_rejects_out_of_range() -> None:
    field = _linear_1d()
    assert field.extrapolation == "error"
    with pytest.raises(ValueError, match="outside the sampled grid"):
        field.at(np.array([[35.0]]))


def test_clamp_policy_holds_the_edge_value() -> None:
    depth = np.array([0.0, 10.0, 20.0, 30.0])
    field = GriddedScalarField(
        axes=(depth,), values=2.0 * depth + 1.0, extrapolation="clamp"
    )
    below = field.at(np.array([[-100.0]]))
    above = field.at(np.array([[1000.0]]))
    assert below == pytest.approx([1.0])  # value at depth=0
    assert above == pytest.approx([61.0])  # value at depth=30


def test_constant_policy_returns_the_fill_value() -> None:
    depth = np.array([0.0, 10.0, 20.0, 30.0])
    field = GriddedScalarField(
        axes=(depth,),
        values=2.0 * depth + 1.0,
        extrapolation="constant",
        fill_value=-1.0,
    )
    assert field.at(np.array([[-5.0]])) == pytest.approx([-1.0])
    assert field.at(np.array([[35.0]])) == pytest.approx([-1.0])
    # in-range points are unaffected by fill_value
    assert field.at(np.array([[10.0]])) == pytest.approx([21.0])


# --- N-D (>1) interpolation ------------------------------------------------------------


def test_two_dimensional_field_interpolates_bilinearly() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    # f(x, y) = x + 2y: bilinear interpolation is exact for an affine function.
    values = np.array([[0.0, 2.0], [1.0, 3.0]])
    field = GriddedScalarField(axes=(x, y), values=values)
    result = field.at(np.array([[0.5, 0.5]]))
    assert result == pytest.approx([1.5])


def test_rejects_positions_with_wrong_trailing_dimension() -> None:
    field = _linear_1d()
    with pytest.raises(ValueError, match="last axis of positions"):
        field.at(np.array([[1.0, 2.0]]))
