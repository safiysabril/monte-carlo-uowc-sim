"""Scattering phase functions.

Currently provides the Henyey-Greenstein phase function; further forms (two-term HG,
Fournier-Forand, measured Petzold) implement the same :class:`~uowc.core.ports.PhaseFunction`
port and slot in as scattering-mixture components without touching transport.

Reference:
    Henyey, L. G. & Greenstein, J. L. (1941), *Astrophys. J.* 93:70-83.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core.units import FloatArray

__all__ = ["HenyeyGreenstein"]


@dataclass(frozen=True, slots=True)
class HenyeyGreenstein:
    """Henyey-Greenstein phase function parameterized by the asymmetry ``g``.

    Normalized over solid angle (the integral over 4pi steradian is 1)::

        p(cos t) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos t)^(3/2))

    ``g`` in (-1, 1): g -> 1 strongly forward-peaked, g = 0 isotropic, g < 0 backward.
    Typical open-ocean particles have g ~= 0.924 (Petzold-average; Mobley, 1994).
    """

    g: float

    def __post_init__(self) -> None:
        if not -1.0 < self.g < 1.0:
            raise ValueError("asymmetry g must lie in the open interval (-1, 1)")

    @property
    def asymmetry(self) -> float:
        """Asymmetry parameter ``g = <cos theta>`` [dimensionless]."""
        return self.g

    def sample_cos_theta(self, u: FloatArray) -> FloatArray:
        """Inverse-CDF sample of cos(theta) from uniform variates ``u`` in [0, 1).

        Exact for Henyey-Greenstein (Witt, 1977):
            cos t = (1 / 2g) * [1 + g^2 - ((1 - g^2) / (1 - g + 2*g*u))^2]
        with the isotropic limit ``cos t = 2u - 1`` for g = 0.
        """
        u = np.asarray(u, dtype=np.float64)
        g = self.g
        if g == 0.0:
            return 2.0 * u - 1.0
        bracket = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
        return (1.0 + g * g - bracket * bracket) / (2.0 * g)

    def value(self, cos_theta: FloatArray) -> FloatArray:
        """Phase-function value at ``cos_theta`` [sr^-1], normalized over 4pi."""
        cos_theta = np.asarray(cos_theta, dtype=np.float64)
        g = self.g
        denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
        return (1.0 - g * g) / (4.0 * np.pi * denom)
