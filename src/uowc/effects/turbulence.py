"""Refractive-index turbulence as a realized, spatially-correlated random field.

Implements :class:`~uowc.core.ports.RefractiveEffect`. The fluctuation is synthesized
as a sum of random Fourier modes (the randomization method for Gaussian random
fields), giving a continuous, analytically differentiable, zero-mean field n'(x) with
a prescribed RMS amplitude and correlation length::

    n'(x)      =  A * sum_m cos(k_m . x + phi_m),     A = sigma * sqrt(2 / M)
    grad n'(x) = -A * sum_m k_m sin(k_m . x + phi_m)

With wavevector components ``k_m ~ Normal(0, 1/L)`` the field has a Gaussian spatial
covariance with correlation length ``L``; phases ``phi_m ~ Uniform[0, 2*pi)``. This is
a *realized* field: it is fixed by a seed, so querying the same position always returns
the same value - genuine spatial correlation, not independent per-point noise.

References:
    Randomization / spectral synthesis of Gaussian random fields (e.g. Kraichnan,
    1970). A physical oceanic refractive spectrum (e.g. Nikishov & Nikishov, 2000) can
    be substituted by changing how ``k_m`` is drawn in :meth:`isotropic`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core.units import FloatArray, as_readonly

__all__ = ["TurbulenceEffect"]


@dataclass(frozen=True, slots=True)
class TurbulenceEffect:
    """Zero-mean refractive-index turbulence (a :class:`RefractiveEffect`).

    Construct with :meth:`isotropic` to draw the random modes from a seed. ``index``
    and ``gradient`` return this effect's *contribution* to the refractive-index field
    (a medium adds it to its base index).

    Units: ``rms_fluctuation`` and the returned index are dimensionless;
    ``correlation_length_m`` is in metres; wavevectors are in rad/m.
    """

    rms_fluctuation: float
    correlation_length_m: float
    seed: int
    wavevectors: FloatArray  # (M, 3) [rad/m]
    phases: FloatArray  # (M,) [rad]

    def __post_init__(self) -> None:
        if self.rms_fluctuation < 0.0:
            raise ValueError("rms_fluctuation must be non-negative")
        if self.correlation_length_m <= 0.0:
            raise ValueError("correlation_length_m must be positive")
        k = np.asarray(self.wavevectors, dtype=np.float64)
        phi = np.asarray(self.phases, dtype=np.float64)
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError("wavevectors must have shape (M, 3)")
        if phi.shape != (k.shape[0],):
            raise ValueError("phases must have shape (M,) matching the wavevectors")
        if k.shape[0] < 1:
            raise ValueError("at least one Fourier mode is required")
        object.__setattr__(self, "wavevectors", as_readonly(k))
        object.__setattr__(self, "phases", as_readonly(phi))

    @classmethod
    def isotropic(
        cls,
        *,
        rms_fluctuation: float,
        correlation_length_m: float,
        n_modes: int,
        seed: int,
    ) -> "TurbulenceEffect":
        """Draw an isotropic, Gaussian-covariance turbulence field from ``seed``.

        Wavevector components are drawn ``~ Normal(0, 1/L)`` (giving correlation length
        ``L``) and phases ``~ Uniform[0, 2*pi)``.
        """
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if correlation_length_m <= 0.0:
            raise ValueError("correlation_length_m must be positive")
        rng = np.random.default_rng(seed)
        wavevectors = rng.normal(0.0, 1.0 / correlation_length_m, size=(n_modes, 3))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        return cls(
            rms_fluctuation=rms_fluctuation,
            correlation_length_m=correlation_length_m,
            seed=seed,
            wavevectors=wavevectors,
            phases=phases,
        )

    @property
    def name(self) -> str:
        return "turbulence"

    @property
    def _amplitude(self) -> float:
        return float(self.rms_fluctuation * np.sqrt(2.0 / self.wavevectors.shape[0]))

    def index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Zero-mean refractive-index contribution n'(x) at the position(s)."""
        x = np.asarray(positions, dtype=np.float64)
        phase = x @ self.wavevectors.T + self.phases
        return self._amplitude * np.sum(np.cos(phase), axis=-1)

    def gradient(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Gradient of the index contribution, grad n'(x) [1/m]."""
        x = np.asarray(positions, dtype=np.float64)
        phase = x @ self.wavevectors.T + self.phases
        return -self._amplitude * (np.sin(phase) @ self.wavevectors)
