"""SI unit conventions, array type aliases, and immutability helpers.

Coordinate convention (used throughout the framework):
    Right-handed Cartesian metres. The z-axis points UP; the water surface is at
    z = 0, and increasing depth means decreasing z (depth = -z, positive downward).
    All positions and directions are float64 arrays in this frame.

All quantities are SI unless a name says otherwise (e.g. ``*_nm``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

#: 1-D (or N-D) array of float64 values.
FloatArray = NDArray[np.float64]
#: 1-D (or N-D) array of int64 values.
IntArray = NDArray[np.int64]
#: 1-D (or N-D) array of complex128 values.
ComplexArray = NDArray[np.complex128]
#: 1-D (or N-D) array of booleans.
BoolArray = NDArray[np.bool_]
#: A 3-component float64 vector, shape ``(3,)``.
Vector3 = NDArray[np.float64]
#: A scalar or an array of float64 (for vectorized evaluation).
FloatOrArray = float | NDArray[np.float64]

__all__ = [
    "FloatArray",
    "IntArray",
    "ComplexArray",
    "BoolArray",
    "Vector3",
    "FloatOrArray",
    "as_readonly",
    "scalar_or_readonly",
    "freeze_value",
]


def as_readonly(a: Any, *, dtype: Any = np.float64) -> NDArray:
    """Return ``a`` as a read-only ndarray of ``dtype``.

    The value object that calls this takes ownership of the array; callers must not
    mutate arrays after handing them to a core value object. This prevents aliasing
    bugs when media or coefficient arrays are shared across scenarios (frozen
    dataclasses block attribute rebinding but not in-place array edits).
    """
    arr = np.asarray(a, dtype=dtype)
    arr.flags.writeable = False
    return arr


def scalar_or_readonly(a: Any, *, dtype: Any = np.float64) -> Any:
    """Pass ``None`` and scalars through (as Python scalars); freeze arrays.

    Used for ``FloatOrArray`` fields, which may hold a single value or a batch.
    """
    if a is None:
        return None
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 0:
        return arr.item()
    arr.flags.writeable = False
    return arr


def freeze_value(a: Any) -> Any:
    """Freeze array values in place (preserving dtype); pass scalars/None through."""
    if a is None or np.isscalar(a):
        return a
    arr = np.asarray(a)
    arr.flags.writeable = False
    return arr
