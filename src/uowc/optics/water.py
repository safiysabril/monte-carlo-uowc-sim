"""Pure-water optical baselines: building blocks for bio-optical models.

A chlorophyll-parameterized model (e.g. :class:`~uowc.optics.haltrin.HaltrinModel`)
splits absorption and scattering into a pure-water term plus a biogenic term. This
module supplies the pure-water term from tabulated/theoretical reference data instead
of the "illustrative blue-green placeholders" a model's convenience constructor may
ship with (see scientific-modelling.md, *Established form vs. supplied data*).

Two distinct kinds of number, deliberately kept separate:

* :class:`PureWaterAbsorption` wraps a **tabulated measurement**
  (Pope & Fry, 1997) - a fixed, independently verifiable table.
* :class:`PureWaterScattering` wraps an **established functional form**
  (Morel, 1974; refined by Zhang, Hu & He, 2009) - a power law in wavelength - but
  requires its absolute reference magnitude as an explicit argument rather than
  shipping a hardcoded default. A single number quoted from memory, with no primary
  citation attached, is exactly the unverifiable-placeholder failure mode this module
  exists to avoid (see :meth:`HaltrinModel.illustrative`'s own caveat).

References
----------
* Pope, R. M. & Fry, E. S. (1997), *Appl. Opt.* 36(33):8710-8723 - "Absorption
  spectrum (380-700 nm) of pure water. II. Integrating cavity measurements." Absorption
  minimum 0.0044 +/- 0.0006 m^-1 at 418 nm.
* Morel, A. (1974), in *Optical Aspects of Oceanography* (Jerlov & Nielsen, eds.),
  Academic Press, 1-24 - classical lambda^-4.32 wavelength dependence for pure-water
  molecular scattering.
* Zhang, X., Hu, L. & He, M.-X. (2009), *Opt. Express* 17(7):5698-5710 - theoretical
  re-derivation from density/concentration fluctuation theory; reports the power-law
  exponent as -4.286 at zero salinity, increasing (in magnitude) to -4.306 at 40 PSU
  (Table 1 and associated text), confirming Morel's classical exponent to within 0.5%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core.units import FloatOrArray

__all__ = ["PureWaterAbsorption", "PureWaterScattering"]

#: Pope & Fry (1997) Table: wavelength [nm] and absorption [1/cm], 400-600 nm.
#: Valid range is restricted to where the tabulated data was extracted; the
#: pure-water absorption minimum (0.0044 m^-1 at 418 nm) falls inside it, which is
#: this table's own internal consistency check (see test_water.py).
_POPE_FRY_WAVELENGTH_NM = np.array(
    [
        400.0, 402.5, 405.0, 407.5, 410.0, 412.5, 415.0, 417.5, 420.0, 422.5,
        425.0, 427.5, 430.0, 432.5, 435.0, 437.5, 440.0, 442.5, 445.0, 447.5,
        450.0, 452.5, 455.0, 457.5, 460.0, 462.5, 465.0, 467.5, 470.0, 472.5,
        475.0, 477.5, 480.0, 482.5, 485.0, 487.5, 490.0, 492.5, 495.0, 497.5,
        500.0, 502.5, 505.0, 507.5, 510.0, 512.5, 515.0, 517.5, 520.0, 522.5,
        525.0, 527.5, 530.0, 532.5, 535.0, 537.5, 540.0, 542.5, 545.0, 547.5,
        550.0, 552.5, 555.0, 557.5, 560.0, 562.5, 565.0, 567.5, 570.0, 572.5,
        575.0, 577.5, 580.0, 582.5, 585.0, 587.5, 590.0, 592.5, 595.0, 597.5,
        600.0,
    ]
)
_POPE_FRY_ABSORPTION_PER_CM = np.array(
    [
        0.0000663, 0.0000579, 0.000053, 0.0000503, 0.0000473, 0.0000452, 0.0000444,
        0.0000442, 0.0000454, 0.0000474, 0.0000478, 0.0000482, 0.0000495, 0.0000504,
        0.000053, 0.000058, 0.0000635, 0.0000696, 0.0000751, 0.000083, 0.0000922,
        0.0000969, 0.0000962, 0.0000957, 0.0000979, 0.0001005, 0.0001011, 0.000102,
        0.000106, 0.000109, 0.000114, 0.000121, 0.000127, 0.000131, 0.000136,
        0.000144, 0.00015, 0.000162, 0.000173, 0.000191, 0.000204, 0.000228,
        0.000256, 0.00028, 0.000325, 0.000372, 0.000396, 0.000399, 0.000409,
        0.000416, 0.000417, 0.000428, 0.000434, 0.000447, 0.000452, 0.000466,
        0.000474, 0.000489, 0.000511, 0.000537, 0.000565, 0.000593, 0.000596,
        0.000606, 0.000619, 0.00064, 0.000642, 0.000672, 0.000695, 0.000733,
        0.000772, 0.000836, 0.000896, 0.000989, 0.0011, 0.00122, 0.001351,
        0.001516, 0.001672, 0.001925, 0.002224,
    ]
)
#: 1/cm -> 1/m.
_PER_CM_TO_PER_M = 100.0

_WATER_MIN_NM = float(_POPE_FRY_WAVELENGTH_NM[0])
_WATER_MAX_NM = float(_POPE_FRY_WAVELENGTH_NM[-1])

#: Morel (1974) classical exponent, confirmed to within 0.5% by the theoretical
#: re-derivation in Zhang, Hu & He (2009) across 0-40 PSU salinity.
_MOREL_SCATTERING_EXPONENT = 4.32


@dataclass(frozen=True, slots=True)
class PureWaterAbsorption:
    """Pure-water absorption ``a_w(lambda)`` from Pope & Fry (1997), linearly
    interpolated over their tabulated 400-600 nm range.

    Extrapolation is refused rather than silently performed (mediums.md's
    extrapolation-policy principle, applied here): a wavelength outside
    [400, 600] nm raises, since Pope & Fry's integrating-cavity data does not
    cover the UV or near-IR pure-water absorption bands.
    """

    @property
    def valid_range_nm(self) -> tuple[float, float]:
        return (_WATER_MIN_NM, _WATER_MAX_NM)

    def at(self, wavelength_nm: FloatOrArray) -> FloatOrArray:
        """Pure-water absorption coefficient [m^-1] at ``wavelength_nm`` [nm]."""
        wl = np.asarray(wavelength_nm, dtype=np.float64)
        if np.any(wl < _WATER_MIN_NM) or np.any(wl > _WATER_MAX_NM):
            raise ValueError(
                f"wavelength_nm must lie in [{_WATER_MIN_NM}, {_WATER_MAX_NM}] nm "
                "(Pope & Fry 1997 tabulated range); refusing to extrapolate"
            )
        per_cm = np.interp(wl, _POPE_FRY_WAVELENGTH_NM, _POPE_FRY_ABSORPTION_PER_CM)
        result = per_cm * _PER_CM_TO_PER_M
        return result if result.ndim else float(result)


@dataclass(frozen=True, slots=True)
class PureWaterScattering:
    """Pure-water (molecular) scattering ``b_w(lambda)``, via the established
    inverse-fourth-power wavelength dependence::

        b_w(lambda) = b_w(lambda_ref) * (lambda_ref / lambda) ** exponent

    ``reference_scattering_m_inv`` at ``reference_wavelength_nm`` must be supplied by
    the caller from a primary source (e.g. Morel 1974 Table 2; Smith & Baker 1981;
    Zhang, Hu & He 2009) - this class intentionally ships no default magnitude, only
    the cited functional form and exponent (see module docstring).

    ``exponent`` defaults to the classical Morel (1974) value 4.32, which Zhang, Hu &
    He (2009) reproduce theoretically to within 0.5% (their exponent runs from 4.286
    at zero salinity to 4.306 at 40 PSU - closer to 4.29-4.31 for typical seawater,
    a small, well-characterized salinity dependence rather than a free parameter).
    """

    reference_scattering_m_inv: float
    reference_wavelength_nm: float
    exponent: float = _MOREL_SCATTERING_EXPONENT

    def __post_init__(self) -> None:
        if self.reference_scattering_m_inv < 0.0:
            raise ValueError("reference_scattering_m_inv must be non-negative")
        if self.reference_wavelength_nm <= 0.0:
            raise ValueError("reference_wavelength_nm must be positive")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive (scattering falls with lambda)")

    def at(self, wavelength_nm: FloatOrArray) -> FloatOrArray:
        """Pure-water scattering coefficient [m^-1] at ``wavelength_nm`` [nm]."""
        wl = np.asarray(wavelength_nm, dtype=np.float64)
        if np.any(wl <= 0.0):
            raise ValueError("wavelength_nm must be positive")
        result = self.reference_scattering_m_inv * (
            self.reference_wavelength_nm / wl
        ) ** self.exponent
        return result if result.ndim else float(result)
