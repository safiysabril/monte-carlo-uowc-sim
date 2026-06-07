"""Homogenization rules: collapse a depth profile to a single concentration.

Used to derive the Scenario I homogeneous baseline from the *same* chlorophyll
profile that drives Scenario II, so the two scenarios differ only in medium
representation (see research-methodology.md). Which rule is "best" is itself a
research question.

:class:`SurfaceValue` and :class:`DepthAverage` operate in chlorophyll space.
Optical-depth-preserving homogenization is model-dependent (it must match the
path-integrated attenuation, which is non-linear in chlorophyll) and is therefore a
planned IOP-space variant rather than a chlorophyll-space rule here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from uowc.media.profiles import ChlorophyllProfile

__all__ = ["HomogenizationRule", "SurfaceValue", "DepthAverage"]

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


@runtime_checkable
class HomogenizationRule(Protocol):
    """Reduce a chlorophyll profile over a depth range to a single concentration.

    Depths are in metres, positive downward; ``depth_min_m`` is the shallowest
    (surface) depth and ``depth_max_m`` the deepest.
    """

    @property
    def name(self) -> str: ...

    def reduce(
        self, profile: ChlorophyllProfile, depth_min_m: float, depth_max_m: float
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class SurfaceValue:
    """Homogenize to the surface (shallowest) chlorophyll value, ``C(depth_min)``."""

    @property
    def name(self) -> str:
        return "surface"

    def reduce(
        self, profile: ChlorophyllProfile, depth_min_m: float, depth_max_m: float
    ) -> float:
        return float(np.asarray(profile.chlorophyll(np.asarray(depth_min_m, dtype=np.float64))))


@dataclass(frozen=True, slots=True)
class DepthAverage:
    """Homogenize to the depth-averaged chlorophyll, ``(1/D) * integral C(z) dz``."""

    samples: int = 1024

    @property
    def name(self) -> str:
        return "depth_average"

    def reduce(
        self, profile: ChlorophyllProfile, depth_min_m: float, depth_max_m: float
    ) -> float:
        if depth_max_m <= depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        depths = np.linspace(depth_min_m, depth_max_m, self.samples)
        column = profile.chlorophyll(depths)
        return float(_trapz(column, depths) / (depth_max_m - depth_min_m))
