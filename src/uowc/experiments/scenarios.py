"""Scenario definitions and medium builders (the fair-comparison core).

Each builder constructs a :class:`~uowc.core.ports.Medium` from the *shared*
:class:`~uowc.experiments.config.ExperimentConfig`, so:

* Scenario I  -> homogeneous medium (profile collapsed by the homogenization rule);
* Scenario II -> depth-dependent medium (same profile + model);
* Scenario III-> depth-dependent medium + the configured environmental effects.

Only the medium changes between scenarios; the optical model, profile, source,
receiver, domain and sampling are identical (research-methodology.md).
"""

from __future__ import annotations

from enum import StrEnum

from uowc.core.ports import Medium
from uowc.experiments.config import ExperimentConfig
from uowc.media import HomogeneousMedium, InhomogeneousMedium, OpticalDepthPreserving, depth_span_m

__all__ = ["Scenario", "build_medium", "medium_type", "effect_names"]


class Scenario(StrEnum):
    """The three research scenarios."""

    I = "I"
    II = "II"
    III = "III"


def build_medium(scenario: Scenario, config: ExperimentConfig) -> Medium:
    """Build the medium for ``scenario`` from the shared config."""
    if scenario is Scenario.I:
        if isinstance(config.homogenization, OpticalDepthPreserving):
            depth_min_m, depth_max_m = depth_span_m(config.bounds)
            iop = config.homogenization.reduce_iop(
                profile=config.profile,
                model=config.model,
                wavelength_nm=config.wavelength_nm,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            return HomogeneousMedium.from_iop(
                iop=iop, bounds=config.bounds, refractive_index=config.refractive_index
            )
        return HomogeneousMedium.from_profile(
            profile=config.profile,
            model=config.model,
            wavelength_nm=config.wavelength_nm,
            bounds=config.bounds,
            rule=config.homogenization,
            refractive_index=config.refractive_index,
        )
    if scenario is Scenario.II:
        return InhomogeneousMedium.scenario_ii(
            profile=config.profile,
            model=config.model,
            wavelength_nm=config.wavelength_nm,
            bounds=config.bounds,
            refractive_index=config.refractive_index,
        )
    if scenario is Scenario.III:
        return InhomogeneousMedium.scenario_iii(
            profile=config.profile,
            model=config.model,
            wavelength_nm=config.wavelength_nm,
            bounds=config.bounds,
            effects=config.effects,
            refractive_index=config.refractive_index,
        )
    raise ValueError(f"unknown scenario: {scenario!r}")


def medium_type(scenario: Scenario) -> str:
    """The medium-representation label recorded in run metadata."""
    return "homogeneous" if scenario is Scenario.I else "inhomogeneous"


def effect_names(scenario: Scenario, config: ExperimentConfig) -> tuple[str, ...]:
    """Names of the effects active in ``scenario`` (only Scenario III has any)."""
    if scenario is Scenario.III:
        return tuple(effect.name for effect in config.effects)
    return ()
