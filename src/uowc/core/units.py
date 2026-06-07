"""SI unit conventions and array type aliases.

All quantities are SI unless a field name says otherwise (e.g. ``*_nm``). These
aliases keep signatures vectorization-friendly without committing to a unit-
checking library. See docs/instructions/scientific-modelling.md.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: 1-D (or N-D) array of float64 values.
FloatArray = NDArray[np.float64]
#: 1-D (or N-D) array of int64 values.
IntArray = NDArray[np.int64]
#: A 3-component float64 vector, shape ``(3,)``.
Vector3 = NDArray[np.float64]
#: A scalar or an array of float64 (for vectorized evaluation).
FloatOrArray = float | NDArray[np.float64]

__all__ = ["FloatArray", "IntArray", "Vector3", "FloatOrArray"]
