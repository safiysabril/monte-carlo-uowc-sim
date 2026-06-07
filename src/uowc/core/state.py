"""Optical-property value objects seen along a photon path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from uowc.core.units import FloatOrArray, Vector3, as_readonly, scalar_or_readonly

if TYPE_CHECKING:
    # Typing-only import to avoid a runtime cycle between state.py and ports.py.
    from uowc.core.ports import PhaseFunction

__all__ = ["ScatteringComponent", "Scattering", "IOP", "LocalOpticalState"]


@dataclass(frozen=True, slots=True)
class ScatteringComponent:
    """One scattering population: its partial coefficient and its phase function.

    Units:
        coefficient: m^-1 (b_i, the scattering coefficient of this population)
    """

    coefficient: FloatOrArray
    phase: "PhaseFunction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", scalar_or_readonly(self.coefficient))


@dataclass(frozen=True, slots=True)
class Scattering:
    """Volume scattering as a mixture of populations (water, bubbles, sediment...).

    Directionality lives entirely here - in each component's phase function - so it is
    never duplicated on :class:`IOP`. Multiple co-located populations are supported by
    holding more than one component (the fix for single-phase-per-location).
    """

    components: tuple[ScatteringComponent, ...]

    @property
    def coefficient(self) -> FloatOrArray:
        """Total scattering coefficient ``b = sum_i b_i`` [m^-1] (definitional)."""
        total: FloatOrArray = 0.0
        for comp in self.components:
            total = total + comp.coefficient
        return total


@dataclass(frozen=True, slots=True)
class IOP:
    """Inherent optical properties (SI): absorption plus the scattering mixture.

    Tagged with the wavelength it was evaluated at, so multi-wavelength results cannot
    be silently mixed. The scattering coefficient and its directionality both come
    from ``scattering`` - a single source of truth (no separate backscatter field).

    Units:
        absorption (a): m^-1
        wavelength_nm:  nm
    """

    absorption: FloatOrArray
    scattering: Scattering
    wavelength_nm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "absorption", scalar_or_readonly(self.absorption))

    @property
    def scattering_coefficient(self) -> FloatOrArray:
        """Total scattering coefficient ``b`` [m^-1]."""
        return self.scattering.coefficient

    @property
    def attenuation(self) -> FloatOrArray:
        """Beam attenuation ``c = a + b`` [m^-1] (definitional)."""
        return self.absorption + self.scattering.coefficient

    @property
    def single_scattering_albedo(self) -> FloatOrArray:
        """Single-scattering albedo ``omega_0 = b / c``; defined as 0 where c = 0."""
        c = np.asarray(self.attenuation, dtype=np.float64)
        b = np.asarray(self.scattering_coefficient, dtype=np.float64)
        omega = np.zeros_like(c)
        np.divide(b, c, out=omega, where=c > 0.0)
        return omega if omega.ndim else float(omega)


@dataclass(frozen=True, slots=True)
class LocalOpticalState:
    """Effective optical state at a location after profile and effects are applied.

    The scattering mixture (and thus all phase functions) lives inside ``iop``.
    Refraction carries both the index and, optionally, its spatial gradient, so a
    transport engine can bend rays through a thermocline.

    Units:
        refractive_index:          dimensionless
        refractive_index_gradient: m^-1 (d n / d x), shape (3,)
    """

    iop: IOP
    refractive_index: FloatOrArray
    refractive_index_gradient: Vector3 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "refractive_index", scalar_or_readonly(self.refractive_index)
        )
        if self.refractive_index_gradient is not None:
            object.__setattr__(
                self,
                "refractive_index_gradient",
                as_readonly(self.refractive_index_gradient),
            )
