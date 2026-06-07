"""Spatial domain (simulation extent) shared by mediums."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core import Region
from uowc.core.units import BoolArray, FloatArray

__all__ = ["BoxDomain"]


@dataclass(frozen=True, slots=True)
class BoxDomain:
    """Axis-aligned, inclusive ``[lower, upper]`` box (metres).

    Implements :class:`~uowc.core.ports.Domain`. Surface/bottom interaction physics is
    handled separately by a boundary, not here.
    """

    region: Region

    def contains(self, positions: FloatArray) -> BoolArray:
        p = np.asarray(positions, dtype=np.float64)
        return np.all((p >= self.region.lower) & (p <= self.region.upper), axis=-1)

    def bounds(self) -> Region:
        return self.region
