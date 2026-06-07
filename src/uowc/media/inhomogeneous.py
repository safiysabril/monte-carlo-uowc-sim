"""Depth-dependent medium (Scenario II) and effect composition (Scenario III).

Composes a :class:`~uowc.media.profiles.ChlorophyllProfile` with an
:class:`~uowc.core.ports.OpticalPropertyModel` to produce IOP(z), and optionally folds
in environmental effects. Effects come in three composable kinds (see
:mod:`uowc.core.ports`):

* ``ParameterEffect``  - modify environmental parameters *before* the optical model;
* ``OpticalEffect``    - modify local optical state / add extinction *after* the model;
* ``RefractiveEffect`` - contribute to the refractive-index field (e.g. turbulence).

Effects are passed as a single ``effects=[...]`` list and dispatched by kind, so they
are attachable/removable without changing transport (mediums.md).

Coordinate convention (see :mod:`uowc.core.units`): z up, surface at z = 0, depth = -z.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uowc.core import EnvironmentalState, LocalOpticalState, Region
from uowc.core.ports import (
    Acceleration,
    Domain,
    OpticalEffect,
    OpticalField,
    OpticalPropertyModel,
    ParameterEffect,
    RefractiveEffect,
)
from uowc.core.units import FloatArray, Vector3
from uowc.media.domain import BoxDomain
from uowc.media.profiles import ChlorophyllProfile

__all__ = ["InhomogeneousMedium"]

_EffectGroups = tuple[
    tuple[ParameterEffect, ...], tuple[OpticalEffect, ...], tuple[RefractiveEffect, ...]
]


def _classify_effects(effects: Sequence[object]) -> _EffectGroups:
    """Dispatch each effect to its kind.

    Checked most- to least-specific: a ``RefractiveEffect`` (has ``index``/``gradient``)
    and an ``OpticalEffect`` (has ``extinction_contribution``) are matched before
    ``ParameterEffect`` (only ``apply``/``name``), which they would otherwise also
    satisfy structurally.
    """
    parameter: list[ParameterEffect] = []
    optical: list[OpticalEffect] = []
    refractive: list[RefractiveEffect] = []
    for eff in effects:
        if isinstance(eff, RefractiveEffect):
            refractive.append(eff)
        elif isinstance(eff, OpticalEffect):
            optical.append(eff)
        elif isinstance(eff, ParameterEffect):
            parameter.append(eff)
        else:
            raise TypeError(
                f"{eff!r} does not implement ParameterEffect, OpticalEffect, or RefractiveEffect"
            )
    return tuple(parameter), tuple(optical), tuple(refractive)


@dataclass(frozen=True, slots=True)
class _DepthOpticalField:
    """Optical field that depends on depth via a chlorophyll profile, with effects."""

    profile: ChlorophyllProfile
    model: OpticalPropertyModel
    wavelength_nm: float
    refractive_index_value: float
    parameter_effects: tuple[ParameterEffect, ...] = ()
    optical_effects: tuple[OpticalEffect, ...] = ()
    refractive_effects: tuple[RefractiveEffect, ...] = ()

    def _environment_at(self, positions: FloatArray, time_s: float) -> EnvironmentalState:
        z = positions[..., 2]
        state = EnvironmentalState(chlorophyll=self.profile.chlorophyll(-z))
        for eff in self.parameter_effects:
            state = eff.apply(state, positions, time_s)
        return state

    def _iop_at(self, positions: FloatArray, time_s: float):
        return self.model.evaluate(self._environment_at(positions, time_s), self.wavelength_nm)

    def extinction(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        p = np.asarray(positions, dtype=np.float64)
        c = np.asarray(self._iop_at(p, time_s).attenuation, dtype=np.float64)
        for eff in self.optical_effects:
            c = c + np.asarray(eff.extinction_contribution(p, time_s), dtype=np.float64)
        return c

    def refractive_index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        p = np.asarray(positions, dtype=np.float64)
        n = np.full(p.shape[:-1], self.refractive_index_value, dtype=np.float64)
        for eff in self.refractive_effects:
            n = n + np.asarray(eff.index(p, time_s), dtype=np.float64)
        return n

    def refractive_index_gradient(
        self, positions: FloatArray, time_s: float = 0.0
    ) -> FloatArray:
        p = np.asarray(positions, dtype=np.float64)
        grad = np.zeros_like(p)
        for eff in self.refractive_effects:
            grad = grad + np.asarray(eff.gradient(p, time_s), dtype=np.float64)
        return grad

    def local_state(self, position: Vector3, time_s: float = 0.0) -> LocalOpticalState:
        p = np.asarray(position, dtype=np.float64)
        iop = self._iop_at(p, time_s)
        index: float = self.refractive_index_value
        gradient = None
        if self.refractive_effects:
            index = float(index) + float(
                np.sum([np.asarray(eff.index(p, time_s)) for eff in self.refractive_effects])
            )
            g = np.zeros(3, dtype=np.float64)
            for eff in self.refractive_effects:
                g = g + np.asarray(eff.gradient(p, time_s), dtype=np.float64)
            gradient = g
        state = LocalOpticalState(iop=iop, refractive_index=index, refractive_index_gradient=gradient)
        for eff in self.optical_effects:
            state = eff.apply(state, p, time_s)
        return state


@dataclass(frozen=True, slots=True)
class _SampledMajorant:
    """Woodcock majorant from dense depth sampling of the (effective) extinction.

    Because ``extinction`` already includes any OpticalEffect contributions, sampling it
    captures them; refractive effects do not change extinction. ``safety`` (>= 1)
    inflates the bound; for sharp optical features increase ``samples``/``safety``.
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

    Scenario II via :meth:`scenario_ii`; Scenario II + effects via :meth:`scenario_iii`.
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
        """Depth-dependent medium with no environmental effects (Scenario II)."""
        return cls._compose(
            profile=profile,
            model=model,
            wavelength_nm=wavelength_nm,
            bounds=bounds,
            effects=(),
            refractive_index=refractive_index,
            majorant_samples=majorant_samples,
            majorant_safety=majorant_safety,
        )

    @classmethod
    def scenario_iii(
        cls,
        *,
        profile: ChlorophyllProfile,
        model: OpticalPropertyModel,
        wavelength_nm: float,
        bounds: Region,
        effects: Sequence[object],
        refractive_index: float = 1.34,
        majorant_samples: int = 2048,
        majorant_safety: float = 1.0,
    ) -> "InhomogeneousMedium":
        """Depth-dependent medium with composed environmental effects (Scenario III)."""
        return cls._compose(
            profile=profile,
            model=model,
            wavelength_nm=wavelength_nm,
            bounds=bounds,
            effects=effects,
            refractive_index=refractive_index,
            majorant_samples=majorant_samples,
            majorant_safety=majorant_safety,
        )

    @classmethod
    def _compose(
        cls,
        *,
        profile: ChlorophyllProfile,
        model: OpticalPropertyModel,
        wavelength_nm: float,
        bounds: Region,
        effects: Sequence[object],
        refractive_index: float,
        majorant_samples: int,
        majorant_safety: float,
    ) -> "InhomogeneousMedium":
        parameter_effects, optical_effects, refractive_effects = _classify_effects(effects)
        field = _DepthOpticalField(
            profile=profile,
            model=model,
            wavelength_nm=wavelength_nm,
            refractive_index_value=refractive_index,
            parameter_effects=parameter_effects,
            optical_effects=optical_effects,
            refractive_effects=refractive_effects,
        )
        return cls(
            field=field,
            domain=BoxDomain(region=bounds),
            acceleration=_SampledMajorant(
                optical_field=field, samples=majorant_samples, safety=majorant_safety
            ),
        )
