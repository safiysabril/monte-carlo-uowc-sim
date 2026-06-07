"""Optical-property value objects seen along a photon path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from uowc.core.units import FloatOrArray

if TYPE_CHECKING:
    # Imported for typing only; kept under TYPE_CHECKING to avoid a runtime
    # import cycle between state.py and ports.py.
    from uowc.core.ports import PhaseFunction

__all__ = ["IOP", "LocalOpticalState"]


@dataclass(frozen=True, slots=True)
class IOP:
    """Inherent optical properties (SI), scalar or array-valued.

    The scattering *shape* is deliberately not stored here: it is owned by a
    :class:`~uowc.core.ports.PhaseFunction` (carried on :class:`LocalOpticalState`)
    so models and effects can vary directionality independently of the bulk
    coefficients.

    Units:
        absorption (a):    m^-1
        scattering (b):    m^-1
        backscatter_ratio: dimensionless (b_b / b), in [0, 1]
    """

    absorption: FloatOrArray
    scattering: FloatOrArray
    backscatter_ratio: FloatOrArray

    @property
    def attenuation(self) -> FloatOrArray:
        """Beam attenuation coefficient ``c = a + b`` [m^-1] (definitional)."""
        return self.absorption + self.scattering

    @property
    def single_scattering_albedo(self) -> FloatOrArray:
        """Single-scattering albedo ``omega_0 = b / c`` [dimensionless]."""
        return self.scattering / self.attenuation


@dataclass(frozen=True, slots=True)
class LocalOpticalState:
    """Effective optical state at a location after profile and effects are applied.

    The object the transport engine inspects at an interaction point. The phase
    function is composed in (HAS-A), so a different scattering population (water,
    bubbles, sediment) is expressed by swapping ``phase`` rather than by subclassing.

    Units:
        refractive_index: dimensionless
    """

    iop: IOP
    refractive_index: FloatOrArray
    phase: "PhaseFunction"
