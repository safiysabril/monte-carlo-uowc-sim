"""Schema versioning and evolution.

Direct tests of the encode/decode functions in isolation - the JSON-shape contract
that :class:`~uowc.persistence.parquet_store.ParquetResultStore` depends on. Covers
the value shapes ``tests/unit/persistence/test_dataset.py`` doesn't reach: array- and
complex-valued metrics (the frequency response), multi-axis metrics, and binned
tallies.
"""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core.geometry import Receiver, Source
from uowc.core.results import (
    SCHEMA_VERSION,
    Axis,
    DetectedPhotons,
    MetricValue,
    RunMetadata,
    SamplingConfig,
    SeedTree,
    Tallies,
    TallyResult,
)
from uowc.persistence.schema import (
    PHOTON_COLUMNS,
    decode_sidecar,
    encode_sidecar,
    photons_from_columns,
    photons_to_columns,
)


def _metadata(**overrides) -> RunMetadata:
    kwargs = dict(
        scenario="II",
        medium_type="inhomogeneous",
        optical_model="haltrin",
        model_parameters={"wavelength_nm": 520.0, "pure_water_absorption_m_inv": 0.15},
        effects=("turbulence", "bubbles"),
        effect_parameters=(
            {"rms_fluctuation": 0.001, "correlation_length_m": 5.0},
            {"void_fraction": 1e-5, "mean_radius_m": 0.001},
        ),
        wavelength_nm=520.0,
        source=Source(
            position=[0.0, 0.0, 0.0],
            direction=[0.0, 0.0, -1.0],
            wavelength_nm=520.0,
            divergence_rad=0.1,
        ),
        receiver=Receiver(
            position=[0.0, 0.0, -10.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
        ),
        majorant=0.45,
        sampling=SamplingConfig(n_photons=1000, estimator="next_event", tallies=("cir",)),
        seed_tree=SeedTree(root_seed=42, streams={"transport": 1, "turbulence_field": 2}),
        rng_impl="NumpyRng",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        code_version="deadbeef",
        library_versions={"numpy": "1.26.0"},
        parameters={"homogenization": "surface", "range_m": 10.0, "flag": True},
    )
    kwargs.update(overrides)
    return RunMetadata(**kwargs)


# --- photon columns ------------------------------------------------------------------


def test_photon_columns_round_trip_preserves_dtypes() -> None:
    photons = DetectedPhotons(
        arrival_time_s=np.array([1e-7, 2e-7]),
        path_length_m=np.array([30.0, 45.0]),
        weight=np.array([1.0, 0.5]),
        n_scatters=np.array([0, 3], dtype=np.int64),
        incidence_rad=np.array([0.0, 0.2]),
    )
    columns = photons_to_columns(photons)
    assert set(columns) == set(PHOTON_COLUMNS)
    restored = photons_from_columns(columns)
    np.testing.assert_array_equal(restored.arrival_time_s, photons.arrival_time_s)
    np.testing.assert_array_equal(restored.n_scatters, photons.n_scatters)
    assert restored.n_scatters.dtype == np.int64


def test_photon_columns_round_trip_empty_arrays() -> None:
    photons = DetectedPhotons(
        arrival_time_s=np.empty(0),
        path_length_m=np.empty(0),
        weight=np.empty(0),
        n_scatters=np.empty(0, dtype=np.int64),
        incidence_rad=np.empty(0),
    )
    restored = photons_from_columns(photons_to_columns(photons))
    assert restored.arrival_time_s.size == 0


# --- metadata --------------------------------------------------------------------------


def test_metadata_round_trips_through_the_sidecar() -> None:
    metadata = _metadata()
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=metadata,
        tallies=Tallies(
            launched=10, detected=1, detected_weight=1.0, absorbed_weight=9.0, escaped_weight=0.0
        ),
        binned={},
        metrics={},
    )
    version, restored, *_ = decode_sidecar(sidecar)
    assert version == SCHEMA_VERSION
    assert restored == metadata


def test_metadata_parameters_preserve_mixed_types() -> None:
    metadata = _metadata(parameters={"range_m": 5.0, "count": 3, "label": "x", "on": False})
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=metadata,
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={},
        metrics={},
    )
    _, restored, *_ = decode_sidecar(sidecar)
    assert restored.parameters == metadata.parameters
    assert isinstance(restored.parameters["count"], int)
    assert isinstance(restored.parameters["on"], bool)


# --- tallies -----------------------------------------------------------------------------


def test_tallies_round_trip_including_extra() -> None:
    tallies = Tallies(
        launched=5000,
        detected=12,
        detected_weight=12.0,
        absorbed_weight=4980.0,
        escaped_weight=8.0,
        extra={"killed_weight": 0.0},
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION, metadata=_metadata(), tallies=tallies, binned={}, metrics={}
    )
    _, _, restored, _, _ = decode_sidecar(sidecar)
    assert restored == tallies


# --- binned tallies ------------------------------------------------------------------------


def test_binned_tally_round_trips() -> None:
    tally = TallyResult(
        name="cir",
        values=np.array([0.0, 3.0, 1.0]),
        edges=(np.array([0.0, 1.0, 2.0, 3.0]),),
        unit="dimensionless",
        axis_names=("arrival_time_s",),
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={"cir": tally},
        metrics={},
    )
    _, _, _, restored, _ = decode_sidecar(sidecar)
    assert set(restored) == {"cir"}
    np.testing.assert_array_equal(restored["cir"].values, tally.values)
    np.testing.assert_array_equal(restored["cir"].edges[0], tally.edges[0])
    assert restored["cir"].axis_names == tally.axis_names


def test_two_dimensional_binned_tally_round_trips() -> None:
    tally = TallyResult(
        name="joint",
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        edges=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])),
        unit="dimensionless",
        axis_names=("depth_m", "arrival_time_s"),
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={"joint": tally},
        metrics={},
    )
    _, _, _, restored, _ = decode_sidecar(sidecar)
    np.testing.assert_array_equal(restored["joint"].values, tally.values)
    assert len(restored["joint"].edges) == 2


# --- metrics: scalar, array, complex, multi-axis ------------------------------------------


def test_scalar_metric_round_trips() -> None:
    metric = MetricValue(
        name="received_power_fraction",
        value=0.0021,
        unit="fraction",
        n_samples=500,
        std=0.0001,
        ci95_low=0.0018,
        ci95_high=0.0024,
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={},
        metrics={"received_power_fraction": metric},
    )
    _, _, _, _, restored = decode_sidecar(sidecar)
    r = restored["received_power_fraction"]
    assert r.value == pytest.approx(metric.value)
    assert r.std == pytest.approx(metric.std)
    assert r.ci95_low == pytest.approx(metric.ci95_low)


def test_metric_with_none_uncertainty_round_trips() -> None:
    metric = MetricValue(name="mean_arrival_time", value=1.2e-7, unit="s", n_samples=10)
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={},
        metrics={"mean_arrival_time": metric},
    )
    _, _, _, _, restored = decode_sidecar(sidecar)
    assert restored["mean_arrival_time"].std is None
    assert restored["mean_arrival_time"].ci95_low is None


def test_real_array_metric_round_trips_with_axis() -> None:
    axis = Axis(name="frequency", values=np.array([0.0, 1e6, 2e6]), unit="Hz")
    metric = MetricValue(
        name="frequency_magnitude",
        value=np.array([1.0, 0.7, 0.3]),
        unit="dimensionless",
        n_samples=1000,
        axes=(axis,),
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={},
        metrics={"frequency_magnitude": metric},
    )
    _, _, _, _, restored = decode_sidecar(sidecar)
    r = restored["frequency_magnitude"]
    np.testing.assert_allclose(r.value, metric.value)
    assert len(r.axes) == 1
    assert r.axes[0].name == "frequency"
    np.testing.assert_allclose(r.axes[0].values, axis.values)


def test_complex_array_metric_round_trips() -> None:
    metric = MetricValue(
        name="frequency_response",
        value=np.array([1.0 + 0j, 0.7 - 0.2j, 0.3 + 0.1j]),
        unit="dimensionless",
        n_samples=1000,
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={},
        metrics={"frequency_response": metric},
    )
    _, _, _, _, restored = decode_sidecar(sidecar)
    r = restored["frequency_response"]
    np.testing.assert_allclose(r.value, metric.value)
    assert np.iscomplexobj(r.value)


def test_complex_scalar_metric_round_trips() -> None:
    metric = MetricValue(
        name="single_frequency", value=0.5 + 0.25j, unit="dimensionless", n_samples=1000
    )
    sidecar = encode_sidecar(
        schema_version=SCHEMA_VERSION,
        metadata=_metadata(),
        tallies=Tallies(
            launched=1, detected=0, detected_weight=0.0, absorbed_weight=1.0, escaped_weight=0.0
        ),
        binned={},
        metrics={"single_frequency": metric},
    )
    _, _, _, _, restored = decode_sidecar(sidecar)
    assert restored["single_frequency"].value == pytest.approx(metric.value)
