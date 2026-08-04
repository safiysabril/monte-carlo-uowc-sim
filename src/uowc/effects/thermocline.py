"""Thermocline/halocline refractive-index layer, as a smooth (not discontinuous)
transition.

Implements :class:`~uowc.core.ports.RefractiveEffect`. A thermocline or halocline is a
depth range over which temperature or salinity - and therefore the refractive index -
changes relatively rapidly; physically it is a thin transition layer (typically tens
of centimetres to a few metres thick), not a mathematical discontinuity.

Why a smooth transition, not a Snell interface
------------------------------------------------
The :class:`RefractiveEffect` port and the transport engine's ray path (transport.md)
both operate on a continuous ``n(x)`` and its gradient - straight-line free flight,
with ray bending via ``grad n`` a *planned* extension, not yet applied. A literal
step-discontinuity model would need interface-crossing detection and a Snell's-law
deflection at the crossing, which nothing in the current transport engine performs;
supplying one here would silently do nothing (no consumer bends rays at an index
jump) while implying a level of physical fidelity the rest of the pipeline does not
provide.

A smooth transition is not a simplification bolted on for convenience - it is *more*
physically accurate than a true step, since a real thermocline has finite thickness -
and it composes correctly with today's engine: a continuous, analytically
differentiable ``n(x)`` and ``grad n`` (mediums.md: "the gradient must be the analytic
derivative of the same expression that produces the index"), ready for ray bending
once that extension lands.

The transition uses ``tanh`` for that differentiability; no specific paper's exact
transition-shape formula is claimed, only that it interpolates the two given index
contributions over the stated thickness. What *is* the physical part - the depth
convention and the fact that the index changes with depth, not east-west or
north-south position - follows mediums.md's coordinate convention (depth positive
downward, ``depth = -z``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core.units import FloatArray

__all__ = ["ThermoclineEffect"]


@dataclass(frozen=True, slots=True)
class ThermoclineEffect:
    """A depth-layered refractive-index contribution, smoothly transitioning from
    ``index_above`` to ``index_below`` across a layer centred at ``depth_m``.

    ``index_above``/``index_below`` are this effect's *contributions* to the index
    (added to the medium's base index, per the port), not absolute indices - typically
    one of them is 0 and the other is the index step ``delta_n`` across the layer.

    ``depth_m`` and ``transition_width_m`` are in metres, depth positive downward
    (mediums.md); position arrays are Cartesian with ``z`` pointing up, so depth is
    computed internally as ``-z``.
    """

    depth_m: float
    transition_width_m: float
    index_above: float
    index_below: float

    def __post_init__(self) -> None:
        if self.transition_width_m <= 0.0:
            raise ValueError("transition_width_m must be positive")

    @property
    def name(self) -> str:
        return "thermocline"

    def _fraction_below(self, depth_m: FloatArray) -> FloatArray:
        """``s(d)``: 0 well above the layer, 1 well below it, smooth in between."""
        return 0.5 * (1.0 + np.tanh((depth_m - self.depth_m) / self.transition_width_m))

    def index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        p = np.asarray(positions, dtype=np.float64)
        depth_m = -p[..., 2]
        fraction = self._fraction_below(depth_m)
        return self.index_above + (self.index_below - self.index_above) * fraction

    def gradient(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Analytic derivative of :meth:`index`; nonzero only in the vertical (z)."""
        p = np.asarray(positions, dtype=np.float64)
        depth_m = -p[..., 2]
        u = (depth_m - self.depth_m) / self.transition_width_m
        d_fraction_d_depth = 0.5 * (1.0 - np.tanh(u) ** 2) / self.transition_width_m
        d_index_d_depth = (self.index_below - self.index_above) * d_fraction_d_depth

        gradient = np.zeros_like(p)
        # depth = -z, so d(index)/dz = -d(index)/d(depth) (chain rule).
        gradient[..., 2] = -d_index_d_depth
        return gradient
