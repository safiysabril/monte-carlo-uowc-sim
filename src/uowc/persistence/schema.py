"""Serialization between core result types and a Parquet-friendly representation.

The detected photons are stored as table columns; everything else (tallies, the full
run metadata, binned tallies and metrics) is JSON-encoded into the table's schema
metadata. Encoding is explicit (not ``asdict``) so it is dtype-preserving and stable
across schema evolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from uowc.core.config import SamplingConfig
from uowc.core.geometry import Receiver, Source
from uowc.core.results import (
    Axis,
    DetectedPhotons,
    MetricValue,
    RunMetadata,
    SeedTree,
    Tallies,
    TallyResult,
)

__all__ = [
    "PHOTON_COLUMNS",
    "photons_to_columns",
    "photons_from_columns",
    "encode_sidecar",
    "decode_sidecar",
]

#: Photon table columns and their numpy dtypes (dtype-preserving round-trip).
PHOTON_COLUMNS: dict[str, Any] = {
    "arrival_time_s": np.float64,
    "path_length_m": np.float64,
    "weight": np.float64,
    "n_scatters": np.int64,
    "incidence_rad": np.float64,
}


def photons_to_columns(photons: DetectedPhotons) -> dict[str, np.ndarray]:
    return {
        "arrival_time_s": np.asarray(photons.arrival_time_s, dtype=np.float64),
        "path_length_m": np.asarray(photons.path_length_m, dtype=np.float64),
        "weight": np.asarray(photons.weight, dtype=np.float64),
        "n_scatters": np.asarray(photons.n_scatters, dtype=np.int64),
        "incidence_rad": np.asarray(photons.incidence_rad, dtype=np.float64),
    }


def photons_from_columns(columns: Mapping[str, np.ndarray]) -> DetectedPhotons:
    return DetectedPhotons(
        arrival_time_s=np.asarray(columns["arrival_time_s"], dtype=np.float64),
        path_length_m=np.asarray(columns["path_length_m"], dtype=np.float64),
        weight=np.asarray(columns["weight"], dtype=np.float64),
        n_scatters=np.asarray(columns["n_scatters"], dtype=np.int64),
        incidence_rad=np.asarray(columns["incidence_rad"], dtype=np.float64),
    )


# --- scalar / array value encoding (for metric values) -----------------------


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    array = np.asarray(value)
    if array.ndim == 0:
        if np.iscomplexobj(array):
            return {"__complex__": [float(array.real), float(array.imag)]}
        return float(array)
    if np.iscomplexobj(array):
        return {
            "__ndarray__": {"real": array.real.tolist(), "imag": array.imag.tolist()},
            "dtype": str(array.dtype),
        }
    return {"__ndarray__": array.tolist(), "dtype": str(array.dtype)}


def _decode_value(payload: Any) -> Any:
    if payload is None or isinstance(payload, (bool, int, float)):
        return payload
    if "__complex__" in payload:
        real, imag = payload["__complex__"]
        return complex(real, imag)
    data = payload["__ndarray__"]
    if isinstance(data, dict):  # complex array
        return (np.asarray(data["real"]) + 1j * np.asarray(data["imag"])).astype(payload["dtype"])
    return np.asarray(data, dtype=payload["dtype"])


# --- value-object <-> dict ---------------------------------------------------


def _sampling_to_dict(sampling: SamplingConfig) -> dict[str, Any]:
    return {
        "n_photons": sampling.n_photons,
        "estimator": sampling.estimator,
        "max_scatter_events": sampling.max_scatter_events,
        "chunk_size": sampling.chunk_size,
        "tallies": list(sampling.tallies),
    }


def _sampling_from_dict(d: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        n_photons=d["n_photons"],
        estimator=d["estimator"],
        max_scatter_events=d["max_scatter_events"],
        chunk_size=d["chunk_size"],
        tallies=tuple(d["tallies"]),
    )


def _seed_tree_to_dict(seed_tree: SeedTree) -> dict[str, Any]:
    return {"root_seed": seed_tree.root_seed, "streams": dict(seed_tree.streams)}


def _seed_tree_from_dict(d: Mapping[str, Any]) -> SeedTree:
    return SeedTree(root_seed=d["root_seed"], streams=dict(d["streams"]))


def _source_to_dict(source: Source) -> dict[str, Any]:
    return {
        "position": np.asarray(source.position).tolist(),
        "direction": np.asarray(source.direction).tolist(),
        "wavelength_nm": source.wavelength_nm,
        "divergence_rad": source.divergence_rad,
    }


def _source_from_dict(d: Mapping[str, Any]) -> Source:
    return Source(
        position=np.asarray(d["position"], dtype=np.float64),
        direction=np.asarray(d["direction"], dtype=np.float64),
        wavelength_nm=d["wavelength_nm"],
        divergence_rad=d["divergence_rad"],
    )


def _receiver_to_dict(receiver: Receiver) -> dict[str, Any]:
    return {
        "position": np.asarray(receiver.position).tolist(),
        "normal": np.asarray(receiver.normal).tolist(),
        "aperture_radius_m": receiver.aperture_radius_m,
        "fov_rad": receiver.fov_rad,
    }


def _receiver_from_dict(d: Mapping[str, Any]) -> Receiver:
    return Receiver(
        position=np.asarray(d["position"], dtype=np.float64),
        normal=np.asarray(d["normal"], dtype=np.float64),
        aperture_radius_m=d["aperture_radius_m"],
        fov_rad=d["fov_rad"],
    )


def _metadata_to_dict(metadata: RunMetadata) -> dict[str, Any]:
    return {
        "scenario": metadata.scenario,
        "medium_type": metadata.medium_type,
        "optical_model": metadata.optical_model,
        "model_parameters": dict(metadata.model_parameters),
        "effects": list(metadata.effects),
        "effect_parameters": [dict(p) for p in metadata.effect_parameters],
        "wavelength_nm": metadata.wavelength_nm,
        "source": _source_to_dict(metadata.source),
        "receiver": _receiver_to_dict(metadata.receiver),
        "majorant": metadata.majorant,
        "sampling": _sampling_to_dict(metadata.sampling),
        "seed_tree": _seed_tree_to_dict(metadata.seed_tree),
        "rng_impl": metadata.rng_impl,
        "timestamp_utc": metadata.timestamp_utc,
        "code_version": metadata.code_version,
        "library_versions": dict(metadata.library_versions),
        "parameters": dict(metadata.parameters),
    }


def _metadata_from_dict(d: Mapping[str, Any]) -> RunMetadata:
    return RunMetadata(
        scenario=d["scenario"],
        medium_type=d["medium_type"],
        optical_model=d["optical_model"],
        model_parameters=dict(d["model_parameters"]),
        effects=tuple(d["effects"]),
        effect_parameters=tuple(dict(p) for p in d["effect_parameters"]),
        wavelength_nm=d["wavelength_nm"],
        source=_source_from_dict(d["source"]),
        receiver=_receiver_from_dict(d["receiver"]),
        majorant=d["majorant"],
        sampling=_sampling_from_dict(d["sampling"]),
        seed_tree=_seed_tree_from_dict(d["seed_tree"]),
        rng_impl=d["rng_impl"],
        timestamp_utc=d["timestamp_utc"],
        code_version=d["code_version"],
        library_versions=dict(d["library_versions"]),
        parameters=dict(d["parameters"]),
    )


def _tallies_to_dict(tallies: Tallies) -> dict[str, Any]:
    return {
        "launched": tallies.launched,
        "detected": tallies.detected,
        "detected_weight": tallies.detected_weight,
        "absorbed_weight": tallies.absorbed_weight,
        "escaped_weight": tallies.escaped_weight,
        "extra": dict(tallies.extra),
    }


def _tallies_from_dict(d: Mapping[str, Any]) -> Tallies:
    return Tallies(
        launched=d["launched"],
        detected=d["detected"],
        detected_weight=d["detected_weight"],
        absorbed_weight=d["absorbed_weight"],
        escaped_weight=d["escaped_weight"],
        extra=dict(d["extra"]),
    )


def _tally_result_to_dict(tally: TallyResult) -> dict[str, Any]:
    return {
        "name": tally.name,
        "values": np.asarray(tally.values).tolist(),
        "edges": [np.asarray(e).tolist() for e in tally.edges],
        "unit": tally.unit,
        "axis_names": list(tally.axis_names),
    }


def _tally_result_from_dict(d: Mapping[str, Any]) -> TallyResult:
    return TallyResult(
        name=d["name"],
        values=np.asarray(d["values"], dtype=np.float64),
        edges=tuple(np.asarray(e, dtype=np.float64) for e in d["edges"]),
        unit=d["unit"],
        axis_names=tuple(d["axis_names"]),
    )


def _metric_to_dict(metric: MetricValue) -> dict[str, Any]:
    return {
        "name": metric.name,
        "value": _encode_value(metric.value),
        "unit": metric.unit,
        "n_samples": metric.n_samples,
        "std": _encode_value(metric.std),
        "ci95_low": _encode_value(metric.ci95_low),
        "ci95_high": _encode_value(metric.ci95_high),
        "axes": [
            {"name": a.name, "values": np.asarray(a.values).tolist(), "unit": a.unit}
            for a in metric.axes
        ],
    }


def _metric_from_dict(d: Mapping[str, Any]) -> MetricValue:
    axes = tuple(
        Axis(name=a["name"], values=np.asarray(a["values"], dtype=np.float64), unit=a["unit"])
        for a in d.get("axes", [])
    )
    return MetricValue(
        name=d["name"],
        value=_decode_value(d["value"]),
        unit=d["unit"],
        n_samples=d["n_samples"],
        std=_decode_value(d["std"]),
        ci95_low=_decode_value(d["ci95_low"]),
        ci95_high=_decode_value(d["ci95_high"]),
        axes=axes,
    )


# --- the JSON sidecar (everything that is not photon columns) ----------------


def encode_sidecar(
    *,
    schema_version: int,
    metadata: RunMetadata,
    tallies: Tallies,
    binned: Mapping[str, TallyResult],
    metrics: Mapping[str, MetricValue],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "metadata": _metadata_to_dict(metadata),
        "tallies": _tallies_to_dict(tallies),
        "binned": {name: _tally_result_to_dict(t) for name, t in binned.items()},
        "metrics": {name: _metric_to_dict(m) for name, m in metrics.items()},
    }


def decode_sidecar(
    sidecar: Mapping[str, Any],
) -> tuple[int, RunMetadata, Tallies, dict[str, TallyResult], dict[str, MetricValue]]:
    return (
        sidecar["schema_version"],
        _metadata_from_dict(sidecar["metadata"]),
        _tallies_from_dict(sidecar["tallies"]),
        {name: _tally_result_from_dict(t) for name, t in sidecar.get("binned", {}).items()},
        {name: _metric_from_dict(m) for name, m in sidecar.get("metrics", {}).items()},
    )
