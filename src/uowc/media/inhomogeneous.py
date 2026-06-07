"""Depth-dependent (Scenario II) medium.

Composes a :class:`~uowc.media.profiles.ChlorophyllProfile` with an
:class:`~uowc.core.ports.OpticalPropertyModel` to produce inherent optical
properties as a function of depth, IOP(z). The result is exposed through the three
decoupled medium capabilities - optical field, domain and Woodcock accelerator -
so the transport engine consumes it like any other :class:`~uowc.core.ports.Medium`.

Coordinate convention (see :mod:`uowc.core.units`): z points up, the surface is at
z = 0, and depth = -z (positive downward). Scenario II uses a depth-dependent IOP
field with a uniform refractive index (refractive structure is added later as a
Scenario III effect).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core import EnvironmentalState, LocalOpticalState, Region
from uowc.core.ports import Acceleration, Domain, OpticalField, OpticalPropertyModel
from uowc.core.units import FloatArray, Vector3
from uowc.media.profiles import ChlorophyllProfile

__all__ = ["InhomogeneousMedium"]


@dataclass(frozen=True, slots=True)
class _DepthOpticalField:
    """Optical field whose properties depend on depth via a chlorophyll profile."""

    profile: ChlorophyllProfile
    model: OpticalPropertyModel
    wavelength_nm: float
    refractive_index_value: float

    def _iop_at(self, positions: FloatArray):
        z = np.asarray(positions, dtype=np.float64)[..., 2]
        chlorophyll = self.profile.chlorophyll(-z)  # depth = -z (positive downward)
        return self.model.evaluate(
            EnvironmentalState(chlorophyll=chlorophyll), self.wavelength_nm
        )

    def extinction(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        return np.asarray(self._iop_at(positions).attenuation, dtype=np.float64)

    def refractive_index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        p = np.asarray(positions, dtype=np.float64)
        return np.full(p.shape[:-1], self.refractive_index_value, dtype=np.float64)

    def refractive_index_gradient(
        self, positions: FloatArray, time_s: float = 0.0
    ) -> FloatArray:
        return np.zeros_like(np.asarray(positions, dtype=np.float64))

    def local_state(self, position: Vector3, time_s: float = 0.0) -> LocalOpticalState:
        iop = self._iop_at(np.asarray(position, dtype=np.float64))
        return LocalOpticalState(iop=iop, refractive_index=self.refractive_index_value)


@dataclass(frozen=True, slots=True)
class _BoxDomain:
    """Axis-aligned box domain."""

    region: Region

    def contains(self, positions: FloatArray) -> FloatArray:
        p = np.asarray(positions, dtype=np.float64)
        return np.all((p >= self.region.lower) & (p <= self.region.upper), axis=-1)

    def bounds(self) -> Region:
        return self.region


@dataclass(frozen=True, slots=True)
class _SampledMajorant:
    """Woodcock majorant from dense depth sampling of the optical field.

    Extinction here depends only on depth, so the field is sampled along z across the
    region and the maximum is taken. ``safety`` (>= 1) inflates the bound; for sharp
    features increase ``samples`` and/or ``safety`` to guarantee an upper bound.
    """

    optical_field: OpticalField
    samples: int
    safety: float

    def majorant(self, region: Region) -> float:
        lower = np.asarray(region.lower, dtype=np.float64)
        upper = np.asarray(region.upper, dtype=np.float64)
        points = np.empty((self.samples, 3), dtype=np.float64)
        points[:, 0] = 0.5 * (lower[0] + upper[0])
        points[:, 1] = 0.5 * (lower[1] + upper[1])
        points[:, 2] = np.linspace(lower[2], upper[2], self.samples)
        return float(np.max(self.optical_field.extinction(points))) * self.safety


@dataclass(frozen=True, slots=True)
class InhomogeneousMedium:
    """Depth-dependent medium: a composition of the three medium capabilities.

    Build the Scenario II vertical slice with :meth:`scenario_ii`.
    """

    field: OpticalField
    domain: Domain
    acceleration: Acceleration

    @classmethod
    def scenario_ii(
        cls,
        *,
        profile: ChlorophyllProfile,
        model: OpticalPropertyModel,
        wavelength_nm: float,
        bounds: Region,
        refractive_index: float = 1.34,
        majorant_samples: int = 2048,
        majorant_safety: float = 1.0,
    ) -> "InhomogeneousMedium":
        """Compose a chlorophyll profile with an optical-property model into a
        depth-dependent medium (Scenario II): C(z) -> IOP(z).

        Parameters:
            profile:          chlorophyll C(z) (e.g. KamedaModel)
            model:            chlorophyll -> IOP model (e.g. HaltrinModel)
            wavelength_nm:    working wavelength (must match the model's)
            bounds:           simulation domain box (z in [-depth_max, 0])
            refractive_index: uniform seawater refractive index (~1.34)
            majorant_samples: depth samples used to estimate the Woodcock majorant
            majorant_safety:  multiplicative safety factor on the majorant (>= 1)
        """
        optical_field = _DepthOpticalField(
            profile=profile,
            model=model,
            wavelength_nm=wavelength_nm,
            refractive_index_value=refractive_index,
        )
        return cls(
            field=optical_field,
            domain=_BoxDomain(region=bounds),
            acceleration=_SampledMajorant(
                optical_field=optical_field,
                samples=majorant_samples,
                safety=majorant_safety,
            ),
        )
