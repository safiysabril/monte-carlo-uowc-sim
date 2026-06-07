"""Haltrin chlorophyll-based bio-optical model: chlorophyll -> inherent optical
properties, with a two-population scattering mixture.

Scientific basis
----------------
A Case-1 (chlorophyll-parameterized) ocean model. Absorption and scattering are split
into a pure-water term and a biogenic (chlorophyll-dependent) term::

    a(lambda) = a_w(lambda) + a_c*(lambda) * C^m
    b(lambda) = b_w(lambda) + b_p(lambda, C)
    b_p(lambda, C) = beta * (lambda_ref / lambda) * C^n

with chlorophyll ``C`` in mg m^-3. The biogenic scattering ``b_p`` carries a strongly
forward-peaked phase function while pure-water (molecular) scattering ``b_w`` is
near-symmetric, so the result is naturally a two-component scattering mixture.

References
----------
* Haltrin, V. I. (1999), *Appl. Opt.* 38(33):6826-6832 - chlorophyll-based model of
  seawater optical properties.
* Morel, A. (1988), *J. Geophys. Res.* 93(C9):10749 - chlorophyll absorption exponent
  ``m = 0.602``.
* Gordon, H. R. & Morel, A. (1983), *Lecture Notes on Coastal and Estuarine Studies*
  4 - particle scattering ``b_p(550) = 0.30 * C^0.62``, ``b_p(lambda) ~ (550/lambda)``.

Note on coefficients
--------------------
The model *form* and the universal exponents/particle-scattering law above are
established. The wavelength-specific reference data (pure-water absorption and
scattering, chlorophyll-specific absorption) are inputs; the values from
:meth:`HaltrinModel.illustrative` are representative blue-green placeholders and
should be replaced with tabulated values (e.g. Pope & Fry 1997 absorption; Smith &
Baker 1981 / Morel scattering) for quantitative work. Web verification was
unavailable when this was written.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uowc.core.environment import EnvironmentalState
from uowc.core.ports import PhaseFunction
from uowc.core.state import IOP, Scattering, ScatteringComponent
from uowc.optics.phase import HenyeyGreenstein

__all__ = ["HaltrinModel"]

#: Tolerance for matching the configured wavelength in :meth:`HaltrinModel.evaluate`.
_WAVELENGTH_TOL_NM = 1.0

#: Petzold-average asymmetry for open-ocean particulate scattering (Mobley, 1994).
_PARTICLE_ASYMMETRY = 0.924


def _default_water_phase() -> PhaseFunction:
    # Molecular scattering is near-symmetric; isotropic (g=0) is used here as a
    # documented simplification (a Rayleigh phase function is a future refinement).
    return HenyeyGreenstein(0.0)


def _default_particle_phase() -> PhaseFunction:
    return HenyeyGreenstein(_PARTICLE_ASYMMETRY)


@dataclass(frozen=True, slots=True)
class HaltrinModel:
    """Chlorophyll-based bio-optical model implementing ``OpticalPropertyModel``.

    The wavelength-specific reference fields (``pure_water_absorption_m_inv``,
    ``chlorophyll_specific_absorption_m2_mg``, ``pure_water_scattering_m_inv``) are
    valid only at ``wavelength_nm``; :meth:`evaluate` enforces this so IOPs from
    different wavelengths cannot be silently mixed.

    Universal parameters carry established defaults:
        chlorophyll_absorption_exponent (m):     0.602  (Morel, 1988)
        particle_scattering_coeff (beta):        0.30   (Gordon & Morel, 1983, 550 nm)
        particle_scattering_exponent (n):        0.62   (Gordon & Morel, 1983)
        particle_scattering_ref_nm (lambda_ref): 550.0
    """

    wavelength_nm: float
    pure_water_absorption_m_inv: float
    chlorophyll_specific_absorption_m2_mg: float
    pure_water_scattering_m_inv: float
    chlorophyll_absorption_exponent: float = 0.602
    particle_scattering_coeff: float = 0.30
    particle_scattering_exponent: float = 0.62
    particle_scattering_ref_nm: float = 550.0
    water_phase: PhaseFunction = field(default_factory=_default_water_phase)
    particle_phase: PhaseFunction = field(default_factory=_default_particle_phase)

    def __post_init__(self) -> None:
        if self.wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be positive")
        if self.pure_water_absorption_m_inv < 0.0:
            raise ValueError("pure_water_absorption_m_inv must be non-negative")
        if self.chlorophyll_specific_absorption_m2_mg < 0.0:
            raise ValueError("chlorophyll_specific_absorption_m2_mg must be non-negative")
        if self.pure_water_scattering_m_inv < 0.0:
            raise ValueError("pure_water_scattering_m_inv must be non-negative")
        if self.particle_scattering_coeff < 0.0:
            raise ValueError("particle_scattering_coeff must be non-negative")

    @property
    def name(self) -> str:
        return "haltrin"

    def evaluate(self, state: EnvironmentalState, wavelength_nm: float) -> IOP:
        """Return IOPs (absorption + two-population scattering mixture) for ``state``.

        Raises ``ValueError`` if ``wavelength_nm`` differs from the configured
        wavelength (the reference coefficients are wavelength-specific).
        """
        if abs(wavelength_nm - self.wavelength_nm) > _WAVELENGTH_TOL_NM:
            raise ValueError(
                f"HaltrinModel configured for {self.wavelength_nm} nm "
                f"but evaluate() called at {wavelength_nm} nm"
            )
        chlorophyll = np.maximum(np.asarray(state.chlorophyll, dtype=np.float64), 0.0)

        absorption = (
            self.pure_water_absorption_m_inv
            + self.chlorophyll_specific_absorption_m2_mg
            * chlorophyll**self.chlorophyll_absorption_exponent
        )

        water_scattering = self.pure_water_scattering_m_inv
        particle_scattering = (
            self.particle_scattering_coeff
            * (self.particle_scattering_ref_nm / self.wavelength_nm)
            * chlorophyll**self.particle_scattering_exponent
        )

        scattering = Scattering(
            components=(
                ScatteringComponent(coefficient=water_scattering, phase=self.water_phase),
                ScatteringComponent(coefficient=particle_scattering, phase=self.particle_phase),
            )
        )
        return IOP(
            absorption=absorption,
            scattering=scattering,
            wavelength_nm=self.wavelength_nm,
        )

    @classmethod
    def illustrative(cls, wavelength_nm: float = 520.0) -> "HaltrinModel":
        """A blue-green configuration with ILLUSTRATIVE reference coefficients.

        Convenience for demos/tests. The reference values are representative
        order-of-magnitude placeholders, not calibrated data - replace them with
        tabulated pure-water and chlorophyll-specific absorption/scattering before
        quantitative use.
        """
        return cls(
            wavelength_nm=wavelength_nm,
            pure_water_absorption_m_inv=0.045,
            chlorophyll_specific_absorption_m2_mg=0.02,
            pure_water_scattering_m_inv=0.0017,
        )
