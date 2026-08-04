"""Suspended-sediment effect: contributes non-algal particle (NAP) concentration to
the environmental state, before the optical-property model runs.

Implements :class:`~uowc.core.ports.ParameterEffect`. This is the correct layer for
sediment per mediums.md: a sediment load is a change to what is dissolved/suspended in
the water, not a directly-specified scattering population, so it is added to
:class:`~uowc.core.environment.EnvironmentalState` and left to the optical-property
model to convert into absorption/scattering - keeping whatever a/b relationship the
model defines for NAP internally consistent, rather than choosing that relationship
here by hand.

Model-scope caveat (scientific-modelling.md, "Model scope must be explicit")
------------------------------------------------------------------------------
This effect only ever *sets* ``EnvironmentalState.nap``; it has **no numerical effect
on transport unless the configured optical-property model reads that field**.
:class:`~uowc.optics.haltrin.HaltrinModel` is a Case-1 (chlorophyll-only) model and
does not consume ``nap`` - composing :class:`SedimentEffect` with it changes the
recorded environmental state but leaves every IOP, and therefore every transport
result, bit-for-bit identical. That is not a bug in this effect; it is Case-1 scope
Haltrin declares for itself. Using this effect for a quantitative result requires a
NAP-aware (Case-2) optical-property model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import Region
from uowc.core.units import FloatArray, FloatOrArray

__all__ = ["SedimentEffect"]


def _in_region(positions: FloatArray, region: Region) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(region.lower, dtype=np.float64)
    upper = np.asarray(region.upper, dtype=np.float64)
    return np.all((p >= lower) & (p <= upper), axis=-1)


@dataclass(frozen=True, slots=True)
class SedimentEffect:
    """Adds a suspended-sediment (non-algal particle) concentration to the state.

    ``concentration_g_m3`` is added to any NAP already present in the incoming state
    (background NAP plus a sediment contribution), not a replacement - two sediment
    sources in the same water are additive, not mutually exclusive.

    If ``region`` is given, the contribution is confined to it (e.g. a near-bottom
    resuspension layer or a river-plume box); outside the region the state passes
    through unmodified. If omitted, the contribution applies everywhere the effect is
    queried (a well-mixed water column).
    """

    concentration_g_m3: float
    region: Region | None = None

    def __post_init__(self) -> None:
        if self.concentration_g_m3 < 0.0:
            raise ValueError("concentration_g_m3 must be non-negative")

    @property
    def name(self) -> str:
        return "sediment"

    def apply(
        self, state: EnvironmentalState, positions: FloatArray, time_s: float = 0.0
    ) -> EnvironmentalState:
        p = np.asarray(positions, dtype=np.float64)
        background = np.asarray(state.nap, dtype=np.float64) if state.nap is not None else 0.0

        added: FloatOrArray
        if self.region is None:
            added = self.concentration_g_m3
        else:
            inside = _in_region(p, self.region)
            added = np.where(inside, self.concentration_g_m3, 0.0)

        nap = background + added
        return replace(state, nap=nap)
