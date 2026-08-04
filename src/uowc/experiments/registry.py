"""Name -> concrete registries (optical-property models, chlorophyll profiles,
environmental effects, metrics) for config-driven construction.

Lets an experiment be specified declaratively (e.g. loaded from YAML/JSON, per
python.md's pyyaml dependency) rather than importing and constructing concrete
classes by hand. Each registry maps a stable string name - the same name recorded in
:class:`~uowc.core.results.RunMetadata` - to a factory taking a plain parameter
mapping, so a run's provenance can name what was used without the reader needing to
import the class that built it.

Scope: only what has a real, usable implementation is registered
--------------------------------------------------------------------
* **Optical-property models**: ``"haltrin"``. ``"kameda"`` is deliberately absent -
  see :mod:`uowc.optics.kameda` for why Kameda is not an optical-property model.
* **Chlorophyll profiles**: ``"kameda"`` (:class:`~uowc.media.profiles.KamedaModel`) -
  the real, citable Kameda & Matsumura (1998) contribution, at the layer it actually
  belongs to (a profile, not an IOP model).
* **Environmental effects**: delegated to :mod:`uowc.effects.registry`.
* **Metrics**: the five :mod:`uowc.metrics` implementations, for assembling a custom
  pipeline from a config file rather than only :meth:`~uowc.metrics.MetricPipeline.default`.

Deliberately **not** registered:

* **Mediums** - fully determined by :class:`~uowc.experiments.scenarios.Scenario` and
  :func:`~uowc.experiments.scenarios.build_medium`; there is nothing to pick by name
  that isn't already picked by the scenario.
* **Estimators** - ``"analog"`` and ``"next_event"`` are implemented directly inside
  :class:`~uowc.transport.woodcock.WoodcockDeltaTracker` (see
  :mod:`uowc.transport.estimators` for why there is no separate estimator class);
  :attr:`~uowc.core.config.SamplingConfig.estimator` is consumed as a string by the
  transport engine, with nothing else to register.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from uowc.core.ports import Metric, OpticalPropertyModel
from uowc.effects.registry import available_effects, build_effect, build_effects
from uowc.media.profiles import ChlorophyllProfile, KamedaModel
from uowc.metrics import (
    Bandwidth3dB,
    FrequencyResponse,
    MeanArrivalTime,
    MetricPipeline,
    ReceivedPowerFraction,
    RmsDelaySpread,
)
from uowc.optics.haltrin import HaltrinModel

__all__ = [
    "build_optical_model",
    "available_optical_models",
    "build_profile",
    "available_profiles",
    "build_metric",
    "build_metrics",
    "build_metric_pipeline",
    "available_metrics",
    "build_effect",
    "build_effects",
    "available_effects",
]

ModelFactory = Callable[[Mapping[str, Any]], OpticalPropertyModel]
ProfileFactory = Callable[[Mapping[str, Any]], ChlorophyllProfile]
MetricFactory = Callable[[Mapping[str, Any]], Metric]


# --- optical-property models -------------------------------------------------------


def _build_haltrin(params: Mapping[str, Any]) -> HaltrinModel:
    return HaltrinModel(
        wavelength_nm=params["wavelength_nm"],
        pure_water_absorption_m_inv=params["pure_water_absorption_m_inv"],
        chlorophyll_specific_absorption_m2_mg=params["chlorophyll_specific_absorption_m2_mg"],
        pure_water_scattering_m_inv=params["pure_water_scattering_m_inv"],
        chlorophyll_absorption_exponent=params.get("chlorophyll_absorption_exponent", 0.602),
        particle_scattering_coeff=params.get("particle_scattering_coeff", 0.30),
        particle_scattering_exponent=params.get("particle_scattering_exponent", 0.62),
        particle_scattering_ref_nm=params.get("particle_scattering_ref_nm", 550.0),
    )


_MODELS: dict[str, ModelFactory] = {"haltrin": _build_haltrin}


def build_optical_model(spec: Mapping[str, Any]) -> OpticalPropertyModel:
    """Build one optical-property model from ``{"name": ..., "params": {...}}``."""
    name = spec["name"]
    params = spec.get("params", {})
    try:
        factory = _MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown optical-property model: {name!r}; available: {available_optical_models()}"
        ) from None
    return factory(params)


def available_optical_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))


# --- chlorophyll profiles -----------------------------------------------------------


def _build_kameda_profile(params: Mapping[str, Any]) -> KamedaModel:
    if "surface_mg_m3" in params:
        kwargs = {k: v for k, v in params.items() if k != "surface_mg_m3"}
        return KamedaModel.from_surface_chlorophyll(params["surface_mg_m3"], **kwargs)
    return KamedaModel(
        background_mg_m3=params["background_mg_m3"],
        peak_integral_mg_m2=params["peak_integral_mg_m2"],
        peak_depth_m=params["peak_depth_m"],
        peak_width_m=params["peak_width_m"],
    )


_PROFILES: dict[str, ProfileFactory] = {"kameda": _build_kameda_profile}


def build_profile(spec: Mapping[str, Any]) -> ChlorophyllProfile:
    """Build one chlorophyll profile from ``{"name": ..., "params": {...}}``.

    The ``"kameda"`` profile accepts either the four direct shifted-Gaussian
    parameters, or a single ``surface_mg_m3`` to derive them via
    :meth:`KamedaModel.from_surface_chlorophyll` (illustrative open-ocean trends;
    calibrate before quantitative use - see that method's docstring).
    """
    name = spec["name"]
    params = spec.get("params", {})
    try:
        factory = _PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown chlorophyll profile: {name!r}; available: {available_profiles()}"
        ) from None
    return factory(params)


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


# --- metrics -------------------------------------------------------------------------


def _build_frequency_response(params: Mapping[str, Any]) -> FrequencyResponse:
    return FrequencyResponse(frequencies_hz=np.asarray(params["frequencies_hz"], dtype=np.float64))


def _build_bandwidth_3db(params: Mapping[str, Any]) -> Bandwidth3dB:
    return Bandwidth3dB(
        max_frequency_hz=params["max_frequency_hz"],
        n_points=params.get("n_points", 2048),
        convention=params.get("convention", "electrical"),
        smoothing_window=params.get("smoothing_window", 1),
    )


_METRICS: dict[str, MetricFactory] = {
    "received_power_fraction": lambda params: ReceivedPowerFraction(),
    "mean_arrival_time": lambda params: MeanArrivalTime(),
    "rms_delay_spread": lambda params: RmsDelaySpread(),
    "frequency_response": _build_frequency_response,
    "bandwidth_3db": _build_bandwidth_3db,
}


def build_metric(spec: Mapping[str, Any]) -> Metric:
    """Build one metric from ``{"name": ..., "params": {...}}``."""
    name = spec["name"]
    params = spec.get("params", {})
    try:
        factory = _METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric: {name!r}; available: {available_metrics()}") from None
    return factory(params)


def build_metrics(specs: Sequence[Mapping[str, Any]]) -> tuple[Metric, ...]:
    return tuple(build_metric(spec) for spec in specs)


def build_metric_pipeline(specs: Sequence[Mapping[str, Any]]) -> MetricPipeline:
    """A :class:`~uowc.metrics.MetricPipeline` assembled from named metric specs,
    for a custom metric set loaded from config rather than ``MetricPipeline.default()``."""
    return MetricPipeline(metrics=build_metrics(specs))


def available_metrics() -> tuple[str, ...]:
    return tuple(sorted(_METRICS))
