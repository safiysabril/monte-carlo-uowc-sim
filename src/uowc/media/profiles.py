"""Vertical chlorophyll-profile models for depth-dependent (Scenario II) media.

A chlorophyll profile maps depth to chlorophyll-a concentration ``C(z)``. A medium
samples it to build an :class:`~uowc.core.environment.EnvironmentalState` at each
depth, which an :class:`~uowc.core.ports.OpticalPropertyModel` (e.g. Haltrin) then
converts to inherent optical properties. Profiles therefore live in the *media*
layer (spatial organisation), not in the optical-property layer.

Scientific basis
----------------
In stratified open ocean the vertical chlorophyll distribution is well described by
a **shifted-Gaussian** profile: a depth-uniform background plus a Gaussian subsurface
(deep) chlorophyll maximum (DCM)::

    C(z) = C_b  +  (h / (sigma * sqrt(2*pi))) * exp( -(z - z_max)^2 / (2 * sigma^2) )

with ``z`` the depth (m, positive downward). This is the form used by Kameda &
Matsumura (1998), who relate the four shape parameters to surface chlorophyll, and by
Johnson et al. (2013) for depth-dependent underwater optical attenuation.

References
----------
* Lewis, Cullen & Platt (1983), *J. Geophys. Res.* 88(C4):2565 - shifted-Gaussian
  deep-chlorophyll-maximum form.
* Kameda & Matsumura (1998), *J. Oceanogr.* 54:509-516 - parameterization of the
  vertical chlorophyll profile from surface chlorophyll.
* Johnson, Green & Leeson (2013), *Appl. Opt.* 52(33):7867-7873 - Gaussian
  chlorophyll-depth profile tied to surface chlorophyll for depth-dependent UOWC.

Note on coefficients
--------------------
The shifted-Gaussian *form* and its explicit parameters below are exact. The mapping
from surface chlorophyll to those parameters (:meth:`KamedaModel.from_surface_chlorophyll`)
uses the qualitative relationships reported in the references with *illustrative*
open-ocean default coefficients; calibrate them to the primary source or to local
data before quantitative use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from uowc.core.units import FloatArray

__all__ = ["ChlorophyllProfile", "KamedaModel"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


@runtime_checkable
class ChlorophyllProfile(Protocol):
    """Maps depth to chlorophyll-a concentration.

    Depth is in metres, positive downward (the framework convention; see
    :mod:`uowc.core.units`). Concentration is in mg m^-3. Implementations are
    vectorized and preserve the shape of the input depths.
    """

    def chlorophyll(self, depth_m: FloatArray) -> FloatArray:
        """Chlorophyll-a concentration [mg m^-3] at the given depth(s) [m]."""
        ...


@dataclass(frozen=True, slots=True)
class KamedaModel:
    """Shifted-Gaussian vertical chlorophyll profile (Kameda & Matsumura, 1998).

    Implements :class:`ChlorophyllProfile`::

        C(z) = background + peak_amplitude * exp( -(z - peak_depth)^2 / (2*width^2) )

    where ``peak_amplitude = peak_integral / (width * sqrt(2*pi))``.

    Parameters (all SI, depth positive downward):
        background_mg_m3 (C_b):    depth-uniform background concentration [mg m^-3]
        peak_integral_mg_m2 (h):   chlorophyll integrated over the Gaussian peak,
                                   above background [mg m^-2]
        peak_depth_m (z_max):      depth of the deep chlorophyll maximum [m]
        peak_width_m (sigma):      Gaussian standard-deviation width of the DCM [m]

    The four parameters are determined empirically; in Kameda & Matsumura (1998) they
    are functions of surface chlorophyll (see :meth:`from_surface_chlorophyll`).
    """

    background_mg_m3: float
    peak_integral_mg_m2: float
    peak_depth_m: float
    peak_width_m: float

    def __post_init__(self) -> None:
        if self.peak_width_m <= 0.0:
            raise ValueError("peak_width_m must be positive")
        if self.background_mg_m3 < 0.0:
            raise ValueError("background_mg_m3 must be non-negative")
        if self.peak_integral_mg_m2 < 0.0:
            raise ValueError("peak_integral_mg_m2 must be non-negative")
        if self.peak_depth_m < 0.0:
            raise ValueError("peak_depth_m must be non-negative (depth positive downward)")

    @property
    def peak_amplitude_mg_m3(self) -> float:
        """Peak concentration of the DCM above background [mg m^-3]."""
        return self.peak_integral_mg_m2 / (self.peak_width_m * _SQRT_2PI)

    def chlorophyll(self, depth_m: FloatArray) -> FloatArray:
        """Chlorophyll-a concentration ``C(z)`` [mg m^-3] at depth(s) ``depth_m`` [m].

        Vectorized; the returned array has the shape of ``depth_m``. The result is
        always non-negative because both the background and the Gaussian peak are.
        """
        z = np.asarray(depth_m, dtype=np.float64)
        reduced = (z - self.peak_depth_m) / self.peak_width_m
        peak = self.peak_amplitude_mg_m3 * np.exp(-0.5 * reduced * reduced)
        return self.background_mg_m3 + peak

    @classmethod
    def from_surface_chlorophyll(
        cls,
        surface_mg_m3: float,
        *,
        background_fraction: float = 0.3,
        peak_depth_ref_m: float = 120.0,
        peak_depth_decay_per_mg_m3: float = 0.6,
        peak_width_m: float = 25.0,
        peak_integral_coeff_mg_m2: float = 20.0,
    ) -> "KamedaModel":
        """Build a profile from surface chlorophyll, after Kameda & Matsumura (1998).

        The *trends* encoded here follow the literature: the background rises with
        surface chlorophyll, and the deep chlorophyll maximum shoals (moves toward the
        surface) as surface chlorophyll increases (oligotrophic waters have a deeper,
        more pronounced DCM; Johnson et al., 2013).

        The numeric coefficients are ILLUSTRATIVE open-ocean defaults, not the
        regional regressions of the primary source. Every coefficient is exposed as a
        keyword argument so it can be calibrated; do so before quantitative use.

        Mapping (``cs`` = surface chlorophyll, mg m^-3):
            background      = background_fraction * cs
            peak_depth      = peak_depth_ref_m / (1 + peak_depth_decay_per_mg_m3 * cs)
            peak_integral   = peak_integral_coeff_mg_m2 * cs
            peak_width      = peak_width_m
        """
        if surface_mg_m3 < 0.0:
            raise ValueError("surface_mg_m3 must be non-negative")
        cs = float(surface_mg_m3)
        return cls(
            background_mg_m3=background_fraction * cs,
            peak_integral_mg_m2=peak_integral_coeff_mg_m2 * cs,
            peak_depth_m=peak_depth_ref_m / (1.0 + peak_depth_decay_per_mg_m3 * cs),
            peak_width_m=peak_width_m,
        )
