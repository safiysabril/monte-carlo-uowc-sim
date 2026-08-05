"""YAML-driven :class:`~uowc.experiments.config.ExperimentConfig` construction.

Translates the declarative presets under ``configs/`` (water types, link
geometries, per-scenario medium specs) into the same typed objects
:mod:`uowc.experiments` already builds by hand, via the existing
:mod:`uowc.experiments.registry` and :mod:`uowc.effects.registry` factories. This
module adds no new physics and no new construction logic beyond that translation -
it is orchestration, not modelling.

Why some configs/ files raise instead of loading
--------------------------------------------------
``configs/scenarios/scenario_i.yaml`` and friends name an optical model
(``optical_model: haltrin``) and, for Scenario III, environmental effects
(``effects: [{type: turbulence}, {type: bubbles}]``) but do not give their numeric
coefficients - the files themselves say so ("parameters to be specified"). Filling
those in with plausible-looking numbers here would be exactly the fabricated-physics
failure mode CLAUDE.md and scientific-modelling.md exist to prevent. Instead,
:func:`build_experiment_config` requires the caller to supply ``model_params`` and
``effect_params`` explicitly, and raises :class:`ConfigError` naming precisely what is
missing when they are not - so a caller pointed at an incomplete config fails loudly
at the gap, not silently past it.

Scope
-----
Builds one :class:`~uowc.experiments.config.ExperimentConfig` per
(scenario, geometry, water_type[, range_m, seed]) combination. Expanding the full
``configs/campaigns/medium_representation_study.yaml`` axes (including the
per-scenario-I-only ``homogenization_rules`` list) into a single
:class:`~uowc.experiments.campaign.Campaign` is not yet implemented; see that file's
own comment for why it does not map onto the current ``Campaign``/``ParameterMatrix``
shape without redundant re-computation of Scenario II/III.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.effects.registry import build_effect
from uowc.experiments.config import ExperimentConfig
from uowc.experiments.registry import build_optical_model, build_profile
from uowc.media import DepthAverage, HomogenizationRule, OpticalDepthPreserving, SurfaceValue

__all__ = [
    "ConfigError",
    "load_yaml",
    "build_source_and_receiver",
    "build_homogenization_rule",
    "build_experiment_config",
]

_DEFAULT_CONFIG_ROOT = Path("configs")

#: configs/ names a profile by shape ("kameda_dcm"); the registry names it by model
#: ("kameda"). One profile shape exists today, so this is a fixed alias, not a
#: general lookup - extend it if a second named profile shape is added.
_PROFILE_ALIASES = {"kameda_dcm": "kameda"}

#: configs/campaigns/medium_representation_study.yaml's own comment calls
#: chlorophyll-space depth-averaging "path_average"; uowc.media.DepthAverage.name is
#: "depth_average". Both spellings are accepted so the configs/ files do not need to
#: be rewritten to match the code's internal name.
_HOMOGENIZATION_RULES: dict[str, type[HomogenizationRule] | type[OpticalDepthPreserving]] = {
    "surface": SurfaceValue,
    "path_average": DepthAverage,
    "depth_average": DepthAverage,
    "optical_depth": OpticalDepthPreserving,
}


class ConfigError(ValueError):
    """A configs/ YAML file (or a caller's override) is missing information needed
    to build a real object. Raised instead of silently defaulting a physical
    coefficient or an effect's parameters."""


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Parse one YAML file into a plain dict (empty dict for an empty file)."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_homogenization_rule(name: str) -> HomogenizationRule | OpticalDepthPreserving:
    """The rule named in a ``scenario_i.yaml``'s ``homogenization_rule`` key."""
    try:
        return _HOMOGENIZATION_RULES[name]()
    except KeyError:
        raise ConfigError(
            f"unknown homogenization_rule {name!r}; available: {sorted(_HOMOGENIZATION_RULES)}"
        ) from None


def build_source_and_receiver(
    geometry: Mapping[str, Any],
    *,
    wavelength_nm: float,
    divergence_rad: float,
    range_m: float | None = None,
) -> tuple[Source, Receiver]:
    """A geometry preset (``configs/geometries.yaml``) plus the shared source
    parameters (``configs/base.yaml``'s ``source:`` block).

    ``range_m``, when given, overrides the preset's own source-receiver distance:
    the receiver is placed ``range_m`` along the preset's (normalized) direction from
    the source, keeping the preset's normal/aperture/FOV - this is what lets a range
    sweep (e.g. the flagship campaign's ``range_m`` axis) reuse one geometry preset
    rather than needing one preset per range.
    """
    source_position = np.asarray(geometry["source_position_m"], dtype=np.float64)
    direction = np.asarray(geometry["source_direction"], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    if range_m is None:
        receiver_position = np.asarray(geometry["receiver_position_m"], dtype=np.float64)
    else:
        receiver_position = source_position + range_m * direction

    source = Source(
        position=source_position,
        direction=direction,
        wavelength_nm=wavelength_nm,
        divergence_rad=divergence_rad,
    )
    receiver = Receiver(
        position=receiver_position,
        normal=np.asarray(geometry["receiver_normal"], dtype=np.float64),
        aperture_radius_m=float(geometry["aperture_radius_m"]),
        fov_rad=float(geometry["fov_rad"]),
    )
    return source, receiver


def _domain_bounds(source: Source, receiver: Receiver) -> Region:
    """A domain generously larger than the link, so the boundary never clips a
    plausible photon path. This is a numerical-safety margin, not a physical
    parameter - unlike an IOP or an effect coefficient, its exact value does not
    change the result as long as it is "big enough" (mediums.md), so a fixed
    generous multiplier is a legitimate default rather than a fabricated input.
    """
    span = float(np.linalg.norm(np.asarray(receiver.position) - np.asarray(source.position)))
    lateral = max(5.0 * span, 20.0)
    depth = max(3.0 * span, 40.0)
    return Region(
        lower=np.array([-lateral, -lateral, -depth]), upper=np.array([lateral, lateral, 0.0])
    )


def build_experiment_config(
    *,
    scenario_name: str,
    geometry_name: str,
    water_type_name: str,
    config_root: Path | str = _DEFAULT_CONFIG_ROOT,
    range_m: float | None = None,
    seed: int | None = None,
    n_photons: int | None = None,
    estimator: str | None = None,
    model_params: Mapping[str, Any] | None = None,
    effect_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> ExperimentConfig:
    """Assemble one :class:`ExperimentConfig` from ``configs/``.

    ``model_params`` supplies the optical model's coefficients (configs/ names the
    model but not its numbers); ``effect_params`` maps each effect type named in the
    scenario's ``effects:`` list (e.g. ``"turbulence"``) to its own parameter dict.
    Both raise :class:`ConfigError` if required and omitted - see the module
    docstring for why nothing is defaulted here.
    """
    root = Path(config_root)
    base = load_yaml(root / "base.yaml")
    water_types = load_yaml(root / "water_types.yaml")
    geometries = load_yaml(root / "geometries.yaml")

    if water_type_name not in water_types:
        raise ConfigError(
            f"unknown water_type {water_type_name!r}; available: {sorted(water_types)}"
        )
    if geometry_name not in geometries:
        raise ConfigError(f"unknown geometry {geometry_name!r}; available: {sorted(geometries)}")

    scenario_path = root / "scenarios" / f"scenario_{scenario_name.lower()}.yaml"
    if not scenario_path.exists():
        raise ConfigError(f"no scenario config at {scenario_path}")
    scenario_spec = load_yaml(scenario_path)
    medium_spec = scenario_spec.get("medium", {})

    wavelength_nm = float(base["source"]["wavelength_nm"])
    divergence_rad = float(base["source"]["divergence_rad"])
    source, receiver = build_source_and_receiver(
        geometries[geometry_name],
        wavelength_nm=wavelength_nm,
        divergence_rad=divergence_rad,
        range_m=range_m,
    )

    optical_model_name = medium_spec.get("optical_model")
    if optical_model_name is None:
        raise ConfigError(f"{scenario_path} has no medium.optical_model")
    if model_params is None:
        raise ConfigError(
            f"{scenario_path} names optical_model={optical_model_name!r} but configs/ does not "
            "give its coefficients ('parameters to be specified') - pass model_params explicitly"
        )
    # wavelength_nm always comes from base.yaml's source block, never from
    # model_params - the two must agree (ExperimentConfig enforces this), so there is
    # exactly one place a caller can set it, not two that could silently diverge.
    model = build_optical_model(
        {"name": optical_model_name, "params": {**model_params, "wavelength_nm": wavelength_nm}}
    )

    profile_name = medium_spec.get("profile", "kameda_dcm")
    water_type = water_types[water_type_name]
    surface_chlorophyll = water_type.get("chlorophyll_mg_m3")
    if surface_chlorophyll is None:
        raise ConfigError(f"water_types.yaml[{water_type_name!r}] has no chlorophyll_mg_m3")
    profile = build_profile(
        {
            "name": _PROFILE_ALIASES.get(profile_name, profile_name),
            "params": {"surface_mg_m3": surface_chlorophyll},
        }
    )

    effects = []
    for effect_spec in medium_spec.get("effects", []):
        effect_type = effect_spec["type"]
        params = (effect_params or {}).get(effect_type)
        if params is None:
            raise ConfigError(
                f"{scenario_path} references effect {effect_type!r} with no parameters "
                "('parameters to be specified') - pass effect_params explicitly"
            )
        effects.append(build_effect({"name": effect_type, "params": dict(params)}))

    homogenization = (
        build_homogenization_rule(medium_spec["homogenization_rule"])
        if "homogenization_rule" in medium_spec
        else DepthAverage()
    )

    transport = base.get("transport", {})
    photons = base.get("photons", {})
    # int(float(...)): defensive against a YAML unsigned float exponent (e.g. "1e7"),
    # which PyYAML's SafeLoader parses as a *string*, not a number - see base.yaml.
    sampling = SamplingConfig(
        n_photons=int(
            float(n_photons if n_photons is not None else photons.get("production", 1_000_000))
        ),
        estimator=estimator or transport.get("estimator", "analog"),
        max_scatter_events=int(float(transport.get("max_scatter_events", 1000))),
    )

    return ExperimentConfig(
        profile=profile,
        model=model,
        source=source,
        receiver=receiver,
        bounds=_domain_bounds(source, receiver),
        sampling=sampling,
        wavelength_nm=wavelength_nm,
        seed=seed if seed is not None else int(base.get("seed", 20260607)),
        effects=tuple(effects),
        homogenization=homogenization,
    )
