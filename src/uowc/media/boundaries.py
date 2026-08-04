"""Boundary models: air-water Fresnel/total-internal-reflection surface and bottom
albedo.

Implements :class:`~uowc.core.ports.Boundary`. mediums.md is explicit that ignoring
the air-water interface "silently converts a reflecting surface into a perfect
absorber and biases both received power and delay spread" - :class:`AirWaterSurface`
and :class:`LambertianBottom` are the real Fresnel/TIR and Lambertian-albedo physics
that a boundary-aware transport loop needs.

Not yet wired into transport
-----------------------------
transport.md lists boundary interaction as the *third* planned transport extension,
after next-event estimation and ray bending - deliberately, since correctly reflecting
a photon requires the delta-tracking loop to know *which* face of the domain a
free-flight segment crossed (to call the right boundary) and to resume tracking from
the crossing point with the remaining step budget, not just test containment at the
candidate endpoint the way :meth:`~uowc.core.ports.Domain.contains` does today. That
loop change is exactly the kind of correctness-critical, vectorized-hot-path surgery
this framework's working guidelines ask to be made as a small, independently
verifiable change (CLAUDE.md) - together with a majorant re-check, since neither
extinction bound above nor below the interface changes, but the *geometry* transport
tracks through does.

These classes are therefore usable and fully tested standalone physics today, ready
to be invoked by that future transport change, but no :class:`Medium` or
:class:`~uowc.core.ports.TransportEngine` in this codebase calls them yet - a domain
that spans the surface or floor still escapes there (mediums.md's stated default
policy) until this extension lands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uowc.core.geometry import BoundaryOutcome
from uowc.core.ports import Rng
from uowc.core.units import FloatArray, Vector3, as_readonly

__all__ = ["AirWaterSurface", "LambertianBottom"]


def _cosine_weighted_hemisphere(normal: FloatArray, rng: Rng) -> FloatArray:
    """Sample one direction from a cosine-weighted hemisphere about ``normal``
    (Malley's method): the correct distribution for ideal Lambertian reflection."""
    u1, u2 = float(rng.uniform(1)[0]), float(rng.uniform(1)[0])
    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    local = np.array([r * math.cos(phi), r * math.sin(phi), math.sqrt(max(0.0, 1.0 - u1))])

    n = normal / np.linalg.norm(normal)
    # An arbitrary vector not parallel to n, to build an orthonormal basis.
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_u = np.cross(helper, n)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(n, tangent_u)
    result = local[0] * tangent_u + local[1] * tangent_v + local[2] * n
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AirWaterSurface:
    """Flat air-water interface at ``z = 0`` (mediums.md coordinate convention):
    unpolarized Fresnel reflectance below the critical angle, total internal
    reflection beyond it.

    Polarization-averaged coefficients only, per mediums.md ("this framework does not
    track polarization"). A wind-roughened surface is a distinct model and is not
    implemented here (mediums.md: "must be declared when used" - a flat surface is the
    stated default).

    ``interact`` expects ``direction`` to point *toward* the surface from below, i.e.
    ``direction[2] > 0``; calling it with a downward-pointing direction is a caller
    error. The reflect/transmit choice is a probabilistic decision rather than a
    weight split, consistent with the analog estimator used elsewhere in this
    framework (transport.md).
    """

    n_air: float = 1.0
    n_water: float = 1.34

    def __post_init__(self) -> None:
        if self.n_air <= 0.0 or self.n_water <= 0.0:
            raise ValueError("refractive indices must be positive")

    @property
    def name(self) -> str:
        return "air_water_surface"

    @property
    def critical_angle_rad(self) -> float:
        """Angle beyond which a ray inside the water is totally internally
        reflected: ``arcsin(n_air / n_water)`` (mediums.md: ~48-49 deg for
        ``n_water ~ 1.34``)."""
        return math.asin(self.n_air / self.n_water)

    def reflectance(self, cos_incidence: float) -> float:
        """Unpolarized Fresnel reflectance for a ray in water incident on the
        surface at angle ``arccos(cos_incidence)`` from the normal.

        Returns 1.0 (total internal reflection) beyond the critical angle, where the
        Fresnel transmission angle is no longer real.
        """
        if not (0.0 <= cos_incidence <= 1.0):
            raise ValueError("cos_incidence must be in [0, 1]")
        sin_i = math.sqrt(max(0.0, 1.0 - cos_incidence * cos_incidence))
        sin_t = self.n_water / self.n_air * sin_i
        if sin_t >= 1.0:
            return 1.0
        cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))

        # Fresnel equations, ray going from the water (n_water) into air (n_air).
        r_s = (self.n_water * cos_incidence - self.n_air * cos_t) / (
            self.n_water * cos_incidence + self.n_air * cos_t
        )
        r_p = (self.n_water * cos_t - self.n_air * cos_incidence) / (
            self.n_water * cos_t + self.n_air * cos_incidence
        )
        return 0.5 * (r_s * r_s + r_p * r_p)

    def interact(
        self,
        position: Vector3,
        direction: FloatArray,
        wavelength_nm: float,
        rng: Rng,
    ) -> BoundaryOutcome:
        d = np.asarray(direction, dtype=np.float64)
        if d[2] <= 0.0:
            raise ValueError("direction must point toward the surface (direction[2] > 0)")

        cos_incidence = float(d[2])
        r = self.reflectance(cos_incidence)
        if float(rng.uniform(1)[0]) < r:
            reflected = np.array([d[0], d[1], -d[2]])
            return BoundaryOutcome(direction=as_readonly(reflected))
        return BoundaryOutcome(direction=d, transmitted=True)


@dataclass(frozen=True, slots=True)
class LambertianBottom:
    """Flat, ideal-diffuse (Lambertian) bottom reflector at a fixed depth.

    A photon reaching the bottom is diffusely reflected with probability ``albedo``
    (direction resampled from a cosine-weighted hemisphere about the upward bottom
    normal, independent of the incoming direction - the definition of an ideal
    diffuse reflector) and otherwise absorbed. mediums.md: "a bright sand bottom and
    a dark mud bottom are different channels" - ``albedo`` and ``depth_m`` must be
    recorded with the run, not hidden inside a shared default.

    A BRDF bottom (non-Lambertian) is future work; this is the stated default.
    """

    albedo: float
    depth_m: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.albedo <= 1.0):
            raise ValueError("albedo must be in [0, 1]")
        if self.depth_m <= 0.0:
            raise ValueError("depth_m must be positive (depth is positive downward)")

    @property
    def name(self) -> str:
        return "lambertian_bottom"

    def interact(
        self,
        position: Vector3,
        direction: FloatArray,
        wavelength_nm: float,
        rng: Rng,
    ) -> BoundaryOutcome:
        d = np.asarray(direction, dtype=np.float64)
        if d[2] >= 0.0:
            raise ValueError("direction must point toward the bottom (direction[2] < 0)")

        if float(rng.uniform(1)[0]) >= self.albedo:
            return BoundaryOutcome(direction=d, absorbed=True)
        normal = np.array([0.0, 0.0, 1.0])
        reflected = _cosine_weighted_hemisphere(normal, rng)
        return BoundaryOutcome(direction=as_readonly(reflected))
