"""Link-geometry value objects: source (transmitter) and receiver (detector)."""
from __future__ import annotations

from dataclasses import dataclass

from uowc.core.units import Vector3

__all__ = ["Source", "Receiver"]


@dataclass(frozen=True, slots=True)
class Source:
    """Photon source / transmitter configuration.

    Held constant across a scenario comparison (see research-methodology.md).

    Units:
        position:       m, shape (3,)
        direction:      unit vector, shape (3,)
        wavelength_nm:  nm
        divergence_rad: rad (beam half-angle)
        n_photons:      count of photons to launch
    """

    position: Vector3
    direction: Vector3
    wavelength_nm: float
    divergence_rad: float
    n_photons: int


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
