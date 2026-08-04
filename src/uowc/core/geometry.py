"""Link-geometry value objects and spatial regions."""
from __future__ import annotations

from dataclasses import dataclass

from uowc.core.units import Vector3, as_readonly

__all__ = ["Source", "Receiver", "Region", "BoundaryOutcome"]


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


@dataclass(frozen=True, slots=True)
class BoundaryOutcome:
    """What happens when a photon reaches a :class:`~uowc.core.ports.Boundary`.

    Exactly one of ``absorbed`` / ``transmitted`` / (neither, i.e. reflected) applies.
    A reflected outcome carries the new outgoing ``direction``; an absorbed or
    transmitted outcome ends this photon's history at the boundary, so ``direction``
    is not meaningful and is not read.

    Units:
        direction: unit vector, shape (3,)
    """

    direction: Vector3
    absorbed: bool = False
    transmitted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", as_readonly(self.direction))
        if self.absorbed and self.transmitted:
            raise ValueError("a boundary outcome cannot be both absorbed and transmitted")
