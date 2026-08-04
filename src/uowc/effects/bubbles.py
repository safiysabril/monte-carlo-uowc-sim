"""Bubble-layer effect: adds a localized scattering population from entrained air.

Implements :class:`~uowc.core.ports.OpticalEffect`. A bubble layer (breaking-wave
entrainment, a diver's exhaust, a thruster wake) is a distinct scatterer with its own
phase function, not a change in water composition - the correct layer per mediums.md
is *after* the optical-property model, adding a new
:class:`~uowc.core.state.ScatteringComponent` to the mixture, never folded into an
existing population's coefficient (which would discard its directionality).

Extinction magnitude: the geometric-optics limit
--------------------------------------------------
For a size parameter ``x = 2*pi*r/lambda >> 1`` (bubble radius much larger than the
wavelength), the extinction efficiency of a sphere approaches the classical
"extinction paradox" limit ``Q_ext -> 2`` (van de Hulst, *Light Scattering by Small
Particles*, 1957, Ch. 8) - a sphere removes twice the light incident on its geometric
cross-section, independent of composition, because diffraction removes as much energy
as geometric blocking does. Near-surface bubbles from breaking waves are typically tens
of micrometres to a few millimetres in radius; at visible wavelengths
(``lambda ~ 0.5 um``) this puts ``x`` from ~1e2 to ~1e4, safely in the geometric-optics
regime, so ``Q_ext = 2`` is used rather than a full Mie computation.

For a population of bubbles with void fraction ``f_v`` (volume of air per volume of
water) and (Sauter) mean radius ``r``, the number density is
``N = f_v / ((4/3) pi r^3)`` and the extinction coefficient is::

    b_bubble = N * Q_ext * pi * r^2 = (3/2) * f_v / r

Air's absorption at optical wavelengths over bubble-scale path lengths is negligible,
so a bubble population contributes scattering only.

What is *not* claimed here: the angular distribution (phase function) of geometric-
optics bubble scattering is a genuinely complex mix of diffraction, reflection and
refraction lobes (including glory/rainbow features for a low-index inclusion). No
single validated asymmetry parameter for it is asserted - ``phase`` is a required
argument, supplied by the caller, exactly so this module does not fabricate one (see
scientific-modelling.md, "Established form vs. supplied data").
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from uowc.core.geometry import Region
from uowc.core.ports import PhaseFunction
from uowc.core.state import LocalOpticalState, ScatteringComponent
from uowc.core.units import FloatArray, Vector3

__all__ = ["BubbleLayerEffect"]

#: Geometric-optics ("extinction paradox") limit, van de Hulst (1957).
_GEOMETRIC_OPTICS_Q_EXT = 2.0


def _in_region(positions: FloatArray, region: Region) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(region.lower, dtype=np.float64)
    upper = np.asarray(region.upper, dtype=np.float64)
    return np.all((p >= lower) & (p <= upper), axis=-1)


def _region_overlaps(a: Region, b: Region) -> bool:
    a_lower, a_upper = np.asarray(a.lower), np.asarray(a.upper)
    b_lower, b_upper = np.asarray(b.lower), np.asarray(b.upper)
    return bool(np.all(a_lower <= b_upper) and np.all(b_lower <= a_upper))


@dataclass(frozen=True, slots=True)
class BubbleLayerEffect:
    """A spatially bounded bubble population contributing one scattering component.

    ``void_fraction`` (dimensionless, air volume / water volume) and
    ``mean_radius_m`` (Sauter mean bubble radius, m) together set the population's
    extinction via the geometric-optics limit (module docstring); ``region`` bounds
    where the layer exists (e.g. a near-surface slab). ``phase`` is required - no
    default asymmetry is asserted for bubble scattering.
    """

    region: Region
    void_fraction: float
    mean_radius_m: float
    phase: PhaseFunction

    def __post_init__(self) -> None:
        if not (0.0 <= self.void_fraction < 1.0):
            raise ValueError("void_fraction must lie in [0, 1)")
        if self.mean_radius_m <= 0.0:
            raise ValueError("mean_radius_m must be positive")

    @property
    def name(self) -> str:
        return "bubbles"

    @property
    def _scattering_coefficient_m_inv(self) -> float:
        """``b_bubble = 1.5 * f_v / r`` (module docstring)."""
        return 1.5 * self.void_fraction / self.mean_radius_m

    def extinction_contribution(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        inside = _in_region(positions, self.region)
        return np.where(inside, self._scattering_coefficient_m_inv, 0.0)

    def extinction_bound(self, region: Region) -> float:
        """The layer's constant coefficient if ``region`` can reach it, else 0."""
        return self._scattering_coefficient_m_inv if _region_overlaps(self.region, region) else 0.0

    def apply(
        self, state: LocalOpticalState, position: Vector3, time_s: float = 0.0
    ) -> LocalOpticalState:
        p = np.asarray(position, dtype=np.float64)
        if not bool(_in_region(p[np.newaxis, :], self.region)[0]):
            return state

        bubble_component = ScatteringComponent(
            coefficient=self._scattering_coefficient_m_inv, phase=self.phase
        )
        scattering = replace(
            state.iop.scattering, components=state.iop.scattering.components + (bubble_component,)
        )
        iop = replace(state.iop, scattering=scattering)
        return replace(state, iop=iop)
