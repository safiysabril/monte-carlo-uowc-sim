"""Homogeneous (Scenario I) medium: spatially constant optical properties.

A single set of inherent optical properties applies throughout the domain - the
baseline against which depth-dependent (Scenario II) results are compared (see
mediums.md). It is built from the *same* :class:`~uowc.core.ports.OpticalPropertyModel`
used in Scenario II so that, for a fair comparison, only the medium representation
differs. Because the medium is uniform, its Woodcock majorant is exactly the constant
attenuation - no sampling is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core import EnvironmentalState, LocalOpticalState, Region
from uowc.core.ports import Acceleration, Domain, OpticalField, OpticalPropertyModel
from uowc.core.state import IOP
from uowc.core.units import FloatArray, Vector3
from uowc.media.domain import BoxDomain
from uowc.media.homogenization import HomogenizationRule
from uowc.media.profiles import ChlorophyllProfile

__all__ = ["HomogeneousMedium"]


@dataclass(frozen=True, slots=True)
class _UniformOpticalField:
    """Optical field with spatially constant properties."""

    iop: IOP
    refractive_index_value: float

    @staticmethod
    def _batch_shape(positions: FloatArray) -> tuple[int, ...]:
        return np.asarray(positions, dtype=np.float64).shape[:-1]

    def extinction(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        return np.full(self._batch_shape(positions), float(self.iop.attenuation), dtype=np.float64)

    def refractive_index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        return np.full(self._batch_shape(positions), self.refractive_index_value, dtype=np.float64)

    def refractive_index_gradient(
        self, positions: FloatArray, time_s: float = 0.0
    ) -> FloatArray:
        return np.zeros_like(np.asarray(positions, dtype=np.float64))

    def local_state(self, position: Vector3, time_s: float = 0.0) -> LocalOpticalState:
        return LocalOpticalState(iop=self.iop, refractive_index=self.refractive_index_value)


@dataclass(frozen=True, slots=True)
class _ConstantMajorant:
    """Exact Woodcock majorant for a homogeneous medium: the constant extinction."""

    value: float

    def majorant(self, region: Region) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class HomogeneousMedium:
    """Spatially constant medium (Scenario I), composed of the three capabilities."""

    field: OpticalField
    domain: Domain
    acceleration: Acceleration

    @classmethod
    def from_iop(
        cls, *, iop: IOP, bounds: Region, refractive_index: float = 1.34
    ) -> "HomogeneousMedium":
        """Lowest-level constructor: a medium with a directly specified constant IOP."""
        if np.ndim(iop.attenuation) != 0:
            raise ValueError("HomogeneousMedium requires a scalar (spatially constant) IOP")
        return cls(
            field=_UniformOpticalField(iop=iop, refractive_index_value=refractive_index),
            domain=BoxDomain(region=bounds),
            acceleration=_ConstantMajorant(value=float(iop.attenuation)),
        )

    @classmethod
    def uniform(
        cls,
        *,
        model: OpticalPropertyModel,
        chlorophyll: float,
        wavelength_nm: float,
        bounds: Region,
        refractive_index: float = 1.34,
    ) -> "HomogeneousMedium":
        """Constant medium from a single chlorophyll value via an optical-property model."""
        if np.ndim(chlorophyll) != 0:
            raise ValueError("chlorophyll must be a single scalar for a homogeneous medium")
        iop = model.evaluate(EnvironmentalState(chlorophyll=float(chlorophyll)), wavelength_nm)
        return cls.from_iop(iop=iop, bounds=bounds, refractive_index=refractive_index)

    @classmethod
    def from_profile(
        cls,
        *,
        profile: ChlorophyllProfile,
        model: OpticalPropertyModel,
        wavelength_nm: float,
        bounds: Region,
        rule: HomogenizationRule,
        refractive_index: float = 1.34,
    ) -> "HomogeneousMedium":
        """Scenario I baseline from a chlorophyll profile, homogenized by ``rule``.

        The profile is collapsed to a single chlorophyll value over the domain's depth
        span (depth = -z), then fed through the same model used in Scenario II - so
        Scenario I and II differ only in how the medium is represented.
        """
        depth_min_m = -float(np.asarray(bounds.upper)[2])  # surface (z = 0 -> depth 0)
        depth_max_m = -float(np.asarray(bounds.lower)[2])  # bottom
        chlorophyll = rule.reduce(profile, depth_min_m, depth_max_m)
        return cls.uniform(
            model=model,
            chlorophyll=chlorophyll,
            wavelength_nm=wavelength_nm,
            bounds=bounds,
            refractive_index=refractive_index,
        )
