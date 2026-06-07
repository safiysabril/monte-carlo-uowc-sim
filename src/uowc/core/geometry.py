"""Link-geometry value objects and spatial regions."""
from __future__ import annotations

from dataclasses import dataclass

from uowc.core.units import Vector3, as_readonly

__all__ = ["Source", "Receiver", "Region"]


@dataclass(frozen=True, slots=True)
class Source:
    """Photon source / transmitter - a physical description only.

    The statistical sample size lives in :class:`~uowc.core.config.SamplingConfig`,
    not here, so the same source can be reused across a convergence study (see
    research-methodology.md).

    Units:
        position:       m, shape (3,)
        direction:      unit vector, shape (3,)
        wavelength_nm:  nm
        divergence_rad: rad (beam half-angle)
    """

    position: Vector3
    direction: Vector3
    wavelength_nm: float
    divergence_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_readonly(self.position))
        object.__setattr__(self, "direction", as_readonly(self.direction))


@dataclass(frozen=True, slots=True)
class Receiver:
    """Receiver / detector configuration.

    Units:
        position:          m, shape (3,)
        normal:            inward-facing unit normal, shape (3,)
        aperture_radius_m: m
        fov_rad:           rad (acceptance half-angle / field of view)
    """

    position: Vector3
    normal: Vector3
    aperture_radius_m: float
    fov_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_readonly(self.position))
        object.__setattr__(self, "normal", as_readonly(self.normal))


@dataclass(frozen=True, slots=True)
class Region:
    """Axis-aligned bounding box ``[lower, upper]`` in metres (for regional majorants)."""

    lower: Vector3
    upper: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", as_readonly(self.lower))
        object.__setattr__(self, "upper", as_readonly(self.upper))
