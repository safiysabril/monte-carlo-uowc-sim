"""Write/read RawResult; metadata persistence.

Real pyarrow file round-trips through :class:`ParquetResultStore` (not just the
in-memory encode/decode functions ``test_schema.py`` covers), including the shapes
``test_dataset.py`` doesn't exercise: binned tallies and a complex-array (frequency
response) metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uowc.core.geometry import Receiver, Source
from uowc.core.results import (
    DetectedPhotons,
    MetricValue,
    RawResult,
    RunMetadata,
    SamplingConfig,
    SeedTree,
    Tallies,
    TallyResult,
    TransportOutput,
)
from uowc.persistence.parquet_store import ParquetResultStore, ResultStore
from uowc.persistence.result import SimulationResult


def _result(*, binned=None, metrics=None) -> SimulationResult:
    photons = DetectedPhotons(
        arrival_time_s=np.array([1e-7, 1.05e-7, 1.2e-7]),
        path_length_m=np.array([30.0, 31.5, 36.0]),
        weight=np.array([1.0, 1.0, 0.6]),
        n_scatters=np.array([0, 1, 2], dtype=np.int64),
        incidence_rad=np.array([0.0, 0.1, 0.2]),
    )
    tallies = Tallies(
        launched=5000,
        detected=3,
        detected_weight=2.6,
        absorbed_weight=4990.0,
        escaped_weight=7.4,
        extra={"killed_weight": 0.0},
    )
    metadata = RunMetadata(
        scenario="III",
        medium_type="inhomogeneous",
        optical_model="haltrin",
        model_parameters={"wavelength_nm": 520.0, "pure_water_absorption_m_inv": 0.15},
        effects=("turbulence",),
        effect_parameters=({"rms_fluctuation": 0.001, "correlation_length_m": 5.0},),
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
        sampling=SamplingConfig(n_photons=5000, estimator="analog", tallies=("cir",)),
        seed_tree=SeedTree(root_seed=123, streams={"transport": 1}),
        rng_impl="NumpyRng",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        code_version="abc123",
        library_versions={"numpy": "1.26.0"},
        parameters={"homogenization": "surface"},
    )
    output = TransportOutput(photons=photons, tallies=tallies, binned=binned or {})
    raw = RawResult(output=output, metadata=metadata)
    return SimulationResult(raw=raw, metrics=metrics or {})


def test_conforms_to_result_store_protocol() -> None:
    assert isinstance(ParquetResultStore(), ResultStore)


def test_round_trip_preserves_photon_columns(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / "run.parquet"
    ParquetResultStore().write(result, path)
    restored = ParquetResultStore().read(path)

    np.testing.assert_array_equal(
        restored.raw.output.photons.arrival_time_s, result.raw.output.photons.arrival_time_s
    )
    np.testing.assert_array_equal(
        restored.raw.output.photons.n_scatters, result.raw.output.photons.n_scatters
    )
    assert restored.raw.output.photons.n_scatters.dtype == np.int64


def test_round_trip_preserves_metadata_and_tallies(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / "run.parquet"
    ParquetResultStore().write(result, path)
    restored = ParquetResultStore().read(path)

    assert restored.raw.metadata == result.raw.metadata
    assert restored.raw.output.tallies == result.raw.output.tallies
    assert restored.raw.schema_version == result.raw.schema_version


def test_round_trip_preserves_binned_tallies(tmp_path: Path) -> None:
    tally = TallyResult(
        name="cir",
        values=np.array([1.0, 2.0, 0.0]),
        edges=(np.array([0.0, 1.0, 2.0, 3.0]),),
        unit="dimensionless",
        axis_names=("arrival_time_s",),
    )
    result = _result(binned={"cir": tally})
    path = tmp_path / "run.parquet"
    ParquetResultStore().write(result, path)
    restored = ParquetResultStore().read(path)

    assert set(restored.raw.output.binned) == {"cir"}
    np.testing.assert_array_equal(restored.raw.output.binned["cir"].values, tally.values)


def test_round_trip_preserves_a_complex_array_metric(tmp_path: Path) -> None:
    metric = MetricValue(
        name="frequency_response",
        value=np.array([1.0 + 0j, 0.6 - 0.3j, 0.2 + 0.1j]),
        unit="dimensionless",
        n_samples=5000,
    )
    result = _result(metrics={"frequency_response": metric})
    path = tmp_path / "run.parquet"
    ParquetResultStore().write(result, path)
    restored = ParquetResultStore().read(path)

    np.testing.assert_allclose(restored.metrics["frequency_response"].value, metric.value)
    assert np.iscomplexobj(restored.metrics["frequency_response"].value)


def test_round_trip_preserves_scalar_metric_uncertainty(tmp_path: Path) -> None:
    metric = MetricValue(
        name="received_power_fraction",
        value=0.00052,
        unit="fraction",
        n_samples=5000,
        std=0.0001,
        ci95_low=0.0003,
        ci95_high=0.0008,
    )
    result = _result(metrics={"received_power_fraction": metric})
    path = tmp_path / "run.parquet"
    ParquetResultStore().write(result, path)
    restored = ParquetResultStore().read(path)

    r = restored.metrics["received_power_fraction"]
    assert r.value == pytest.approx(metric.value)
    assert r.ci95_low == pytest.approx(metric.ci95_low)
    assert r.ci95_high == pytest.approx(metric.ci95_high)


def test_read_rejects_a_parquet_file_without_uowc_metadata(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), str(path))
    with pytest.raises(ValueError, match="metadata"):
        ParquetResultStore().read(path)


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dirs" / "run.parquet"
    ParquetResultStore().write(_result(), path)
    assert path.exists()
