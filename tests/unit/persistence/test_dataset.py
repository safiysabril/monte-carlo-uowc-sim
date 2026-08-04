"""Partitioned dataset read/write and filtering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uowc.core.results import (
    DetectedPhotons,
    MetricValue,
    RawResult,
    RunMetadata,
    SamplingConfig,
    SeedTree,
    Tallies,
    TransportOutput,
)
from uowc.persistence.dataset import PartitionedResultDataset, default_partition
from uowc.persistence.result import SimulationResult


def _result(*, scenario: str, seed: int, wavelength_nm: float = 520.0) -> SimulationResult:
    photons = DetectedPhotons(
        arrival_time_s=np.array([1e-7, 1.1e-7]),
        path_length_m=np.array([30.0, 33.0]),
        weight=np.array([1.0, 1.0]),
        n_scatters=np.array([0, 1], dtype=np.int64),
        incidence_rad=np.array([0.0, 0.1]),
    )
    tallies = Tallies(
        launched=1000, detected=2, detected_weight=2.0, absorbed_weight=998.0, escaped_weight=0.0
    )
    meta = RunMetadata(
        scenario=scenario,
        medium_type="homogeneous",
        optical_model="haltrin",
        effects=(),
        wavelength_nm=wavelength_nm,
        sampling=SamplingConfig(n_photons=1000, estimator="analog"),
        seed_tree=SeedTree(root_seed=seed),
        rng_impl="NumpyRng",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        code_version="abc123",
    )
    raw = RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=meta)
    metrics = {
        "received_power_fraction": MetricValue(
            name="received_power_fraction", value=0.002, unit="fraction", n_samples=1000
        )
    }
    return SimulationResult(raw=raw, metrics=metrics)


def test_path_for_is_hive_style_and_sorted_by_key(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    path = dataset.path_for({"seed": 42, "scenario": "I"})
    assert path == tmp_path / "scenario=I" / "seed=42" / "result.parquet"


def test_path_for_is_independent_of_mapping_insertion_order(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    a = dataset.path_for({"scenario": "I", "seed": 42})
    b = dataset.path_for({"seed": 42, "scenario": "I"})
    assert a == b


def test_path_for_rejects_empty_partition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one key"):
        PartitionedResultDataset(root=tmp_path).path_for({})


def test_path_for_rejects_unsafe_partition_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        PartitionedResultDataset(root=tmp_path).path_for({"scenario": "I/II"})


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    original = _result(scenario="I", seed=42)
    written_path = dataset.write(original, {"scenario": "I", "seed": 42})

    assert written_path.exists()
    loaded = dataset.read({"scenario": "I", "seed": 42})
    np.testing.assert_allclose(
        loaded.raw.output.photons.arrival_time_s, original.raw.output.photons.arrival_time_s
    )
    assert loaded.raw.metadata.scenario == "I"
    assert loaded.raw.metadata.seed_tree.root_seed == 42
    assert loaded.metrics["received_power_fraction"].value == pytest.approx(0.002)


def test_list_partitions_discovers_every_written_run(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    dataset.write(_result(scenario="I", seed=1), {"scenario": "I", "seed": 1})
    dataset.write(_result(scenario="I", seed=2), {"scenario": "I", "seed": 2})
    dataset.write(_result(scenario="II", seed=1), {"scenario": "II", "seed": 1})

    partitions = dataset.list_partitions()
    assert len(partitions) == 3
    assert {"scenario": "I", "seed": "1"} in partitions
    assert {"scenario": "II", "seed": "1"} in partitions


def test_list_partitions_on_empty_dataset_returns_empty_tuple(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path / "does_not_exist_yet")
    assert dataset.list_partitions() == ()


def test_find_filters_by_partial_partition_match(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    dataset.write(_result(scenario="I", seed=1), {"scenario": "I", "seed": 1})
    dataset.write(_result(scenario="I", seed=2), {"scenario": "I", "seed": 2})
    dataset.write(_result(scenario="II", seed=1), {"scenario": "II", "seed": 1})

    matches = dataset.find(scenario="I")
    assert len(matches) == 2
    assert all(p["scenario"] == "I" for p in matches)

    exact = dataset.find(scenario="I", seed=1)
    assert len(exact) == 1


def test_read_matching_loads_only_filtered_results(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    dataset.write(_result(scenario="I", seed=1), {"scenario": "I", "seed": 1})
    dataset.write(_result(scenario="II", seed=1), {"scenario": "II", "seed": 1})

    loaded = dataset.read_matching(scenario="II")
    assert len(loaded) == 1
    assert loaded[0].raw.metadata.scenario == "II"


def test_default_partition_uses_scenario_model_wavelength_seed() -> None:
    result = _result(scenario="II", seed=99, wavelength_nm=500.0)
    partition = default_partition(result.raw.metadata)
    assert partition == {"scenario": "II", "model": "haltrin", "wavelength_nm": 500.0, "seed": 99}


def test_default_partition_round_trips_through_write_and_read(tmp_path: Path) -> None:
    dataset = PartitionedResultDataset(root=tmp_path)
    result = _result(scenario="I", seed=7)
    dataset.write(result, default_partition(result.raw.metadata))

    loaded = dataset.read(default_partition(result.raw.metadata))
    assert loaded.raw.metadata.seed_tree.root_seed == 7
