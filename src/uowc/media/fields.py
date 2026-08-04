"""Generic N-D scalar fields over position: a grid of samples, linearly interpolated.

:class:`ChlorophyllProfile` (profiles.py) is the *interface* a depth-dependent medium
queries; it says nothing about how a concrete profile computes its values. Every
profile implemented so far is closed-form (the shifted-Gaussian in
:class:`~uowc.media.profiles.KamedaModel`). mediums.md's stated future work - 2D/3D
variation and measured oceanographic fields - has no closed form: a measured
temperature or chlorophyll field is a table of samples on a grid, not a formula.

:class:`GriddedScalarField` is that table, in any number of dimensions, with linear
interpolation between samples. It is deliberately *not* a :class:`ChlorophyllProfile`
itself - a 1-D field over depth would still need the ``depth = -z`` conversion at its
one seam, same as any other profile (mediums.md's coordinate convention) - but it is
the reusable piece a future gridded profile or gridded turbulence/temperature field
would wrap.

mediums.md requires the extrapolation policy for an out-of-range query to be
*explicit*, not silently chosen, because a fitted or measured field's edge behavior
is exactly where an off-by-one or an unconverted unit shows up as a runaway or
negative value. :attr:`GriddedScalarField.extrapolation` is not optional and has no
default other than the strict, fail-loud one (``"error"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from uowc.core.units import FloatArray, FloatOrArray, as_readonly

__all__ = ["ExtrapolationPolicy", "GriddedScalarField"]

#: How to evaluate a query point outside the sampled grid.
#:
#: * ``"error"``    - raise ``ValueError`` (the strict default; see module docstring).
#: * ``"clamp"``     - hold the nearest edge value constant beyond the grid.
#: * ``"constant"``  - return a fixed ``fill_value``.
ExtrapolationPolicy = Literal["error", "clamp", "constant"]


@dataclass(frozen=True, slots=True)
class GriddedScalarField:
    """An N-D axis-aligned grid of scalar samples, linearly interpolated.

    ``axes`` is one strictly increasing 1-D coordinate array per dimension (e.g. a
    single depth axis for a 1-D field, or ``(x, y, z)`` for a 3-D one); ``values`` has
    shape ``tuple(len(axis) for axis in axes)``. Queries via :meth:`at` take positions
    of shape ``(..., n_dimensions)`` and return values of shape ``(...)``, so the same
    field works for a single point or a batch.

    Units are whatever the caller's axes are in (metres for a spatial field, seconds
    for a time axis) - this class has no unit opinion of its own; the wrapping profile
    or effect states them.
    """

    axes: tuple[FloatArray, ...]
    values: FloatArray
    extrapolation: ExtrapolationPolicy = "error"
    fill_value: float = 0.0
    _interpolator: RegularGridInterpolator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.axes) == 0:
            raise ValueError("at least one axis is required")
        axes = tuple(np.asarray(a, dtype=np.float64) for a in self.axes)
        values = np.asarray(self.values, dtype=np.float64)
        for i, axis in enumerate(axes):
            if axis.ndim != 1:
                raise ValueError(f"axis {i} must be 1-D, got shape {axis.shape}")
            if axis.size < 2:
                raise ValueError(f"axis {i} must have at least 2 points, got {axis.size}")
            if not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"axis {i} must be strictly increasing")
        expected_shape = tuple(a.size for a in axes)
        if values.shape != expected_shape:
            raise ValueError(
                f"values shape {values.shape} does not match axes shape {expected_shape}"
            )
        object.__setattr__(self, "axes", tuple(as_readonly(a) for a in axes))
        object.__setattr__(self, "values", as_readonly(values))
        object.__setattr__(
            self,
            "_interpolator",
            RegularGridInterpolator(
                axes, values, method="linear", bounds_error=False, fill_value=None
            ),
        )

    @property
    def n_dimensions(self) -> int:
        """Number of spatial/parametric dimensions this field is sampled over."""
        return len(self.axes)

    @property
    def lower(self) -> FloatArray:
        """Lower grid bound along each axis, shape ``(n_dimensions,)``."""
        return np.array([axis[0] for axis in self.axes])

    @property
    def upper(self) -> FloatArray:
        """Upper grid bound along each axis, shape ``(n_dimensions,)``."""
        return np.array([axis[-1] for axis in self.axes])

    def at(self, positions: FloatArray) -> FloatOrArray:
        """Interpolated value at ``positions``, shape ``(..., n_dimensions)`` in ->
        shape ``(...)`` out. Out-of-range points are handled per
        :attr:`extrapolation`."""
        p = np.asarray(positions, dtype=np.float64)
        if p.shape[-1] != self.n_dimensions:
            raise ValueError(
                f"last axis of positions must have size {self.n_dimensions}, got {p.shape[-1]}"
            )
        batch_shape = p.shape[:-1]
        flat = p.reshape(-1, self.n_dimensions)

        lower, upper = self.lower, self.upper
        outside = np.any((flat < lower) | (flat > upper), axis=-1)
        if np.any(outside) and self.extrapolation == "error":
            bad = flat[outside][0]
            raise ValueError(
                f"position {tuple(bad)} lies outside the sampled grid "
                f"{tuple(lower)}..{tuple(upper)} and extrapolation='error'"
            )

        query = np.clip(flat, lower, upper) if self.extrapolation == "clamp" else flat
        result = np.asarray(self._interpolator(query), dtype=np.float64)
        if self.extrapolation == "constant":
            result = np.where(outside, self.fill_value, result)

        result = result.reshape(batch_shape)
        return result if result.ndim else float(result)
