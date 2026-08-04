"""Name -> effect construction registry for config-driven assembly.

Lets Scenario III's effect list be specified declaratively (e.g. loaded from YAML,
per python.md's pyyaml dependency) as ``{"name": "bubbles", "params": {...}}`` rather
than importing and constructing effect classes by hand. A spec's ``name`` is exactly
the string :meth:`~uowc.core.ports.OpticalEffect.name` /
:meth:`~uowc.core.ports.ParameterEffect.name` / etc. would report, so a run built from
a spec and a run built by hand record the same effect name in
:class:`~uowc.core.results.RunMetadata`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from uowc.core.geometry import Region
from uowc.core.ports import PhaseFunction
from uowc.effects.bubbles import BubbleLayerEffect
from uowc.effects.sediment import SedimentEffect
from uowc.effects.thermocline import ThermoclineEffect
from uowc.effects.turbulence import TurbulenceEffect
from uowc.optics.phase import HenyeyGreenstein

__all__ = ["build_effect", "build_effects", "available_effects"]

EffectFactory = Callable[[Mapping[str, Any]], object]


def _build_region(spec: Mapping[str, Any]) -> Region:
    return Region(
        lower=np.asarray(spec["lower"], dtype=np.float64),
        upper=np.asarray(spec["upper"], dtype=np.float64),
    )


def _build_phase(spec: Mapping[str, Any]) -> PhaseFunction:
    kind = spec["type"]
    if kind == "henyey_greenstein":
        return HenyeyGreenstein(g=float(spec["g"]))
    raise ValueError(f"unknown phase-function type: {kind!r}; available: ['henyey_greenstein']")


def _build_turbulence(params: Mapping[str, Any]) -> TurbulenceEffect:
    return TurbulenceEffect.isotropic(
        rms_fluctuation=params["rms_fluctuation"],
        correlation_length_m=params["correlation_length_m"],
        n_modes=params["n_modes"],
        seed=params["seed"],
    )


def _build_sediment(params: Mapping[str, Any]) -> SedimentEffect:
    region = _build_region(params["region"]) if "region" in params else None
    return SedimentEffect(concentration_g_m3=params["concentration_g_m3"], region=region)


def _build_bubbles(params: Mapping[str, Any]) -> BubbleLayerEffect:
    return BubbleLayerEffect(
        region=_build_region(params["region"]),
        void_fraction=params["void_fraction"],
        mean_radius_m=params["mean_radius_m"],
        phase=_build_phase(params["phase"]),
    )


def _build_thermocline(params: Mapping[str, Any]) -> ThermoclineEffect:
    return ThermoclineEffect(
        depth_m=params["depth_m"],
        transition_width_m=params["transition_width_m"],
        index_above=params["index_above"],
        index_below=params["index_below"],
    )


_EFFECTS: dict[str, EffectFactory] = {
    "turbulence": _build_turbulence,
    "sediment": _build_sediment,
    "bubbles": _build_bubbles,
    "thermocline": _build_thermocline,
}


def build_effect(spec: Mapping[str, Any]) -> object:
    """Build one effect from ``{"name": ..., "params": {...}}``."""
    name = spec["name"]
    params = spec.get("params", {})
    try:
        factory = _EFFECTS[name]
    except KeyError:
        raise ValueError(f"unknown effect: {name!r}; available: {available_effects()}") from None
    return factory(params)


def build_effects(specs: Sequence[Mapping[str, Any]]) -> tuple[object, ...]:
    """Build an ordered tuple of effects - order matters where effects do not
    commute (mediums.md: composition order is a scientific parameter)."""
    return tuple(build_effect(spec) for spec in specs)


def available_effects() -> tuple[str, ...]:
    return tuple(sorted(_EFFECTS))
