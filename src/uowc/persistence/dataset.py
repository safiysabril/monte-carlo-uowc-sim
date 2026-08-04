"""Partitioned Parquet dataset access (scenario/model/wavelength/seed/...) for
large-scale experiments.

A single :class:`~uowc.persistence.parquet_store.ResultStore` writes one run to one
file; this module decides *where* that file goes and how to find it again, using a
Hive-style layout::

    root/key1=value1/key2=value2/.../result.parquet

so a large campaign (data.md: "Data storage must support ... large-scale
experiments") can be organized, filtered and partially loaded without opening every
file. It does not read or write photon data itself - that stays entirely the
``ResultStore``'s job (data.md: "Avoid embedding analysis logic inside storage
layers"), and it never re-derives a partition key from a file's own contents: doing
so silently is exactly the kind of "data pipeline analysis logic" data.md keeps
separate from storage.

Partition keys are sorted before building a path, so the same partition (as a set of
key/value pairs) always maps to the same location regardless of the order the caller
happened to build the mapping in - there is deliberately no dependence on dict
insertion order.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from uowc.core.results import RunMetadata
from uowc.persistence.parquet_store import ParquetResultStore, ResultStore
from uowc.persistence.result import SimulationResult

__all__ = ["PartitionedResultDataset", "default_partition"]

PartitionValue = str | int | float
Partition = Mapping[str, PartitionValue]


def _sanitize(value: PartitionValue) -> str:
    text = str(value)
    if not text or "/" in text or "\\" in text:
        raise ValueError(f"partition value {value!r} is not filesystem-safe")
    return text


def default_partition(metadata: RunMetadata) -> dict[str, PartitionValue]:
    """A conventional partition (scenario/model/wavelength/seed) built directly from
    a run's own metadata, for the common case of not hand-picking partition keys."""
    return {
        "scenario": metadata.scenario,
        "model": metadata.optical_model,
        "wavelength_nm": metadata.wavelength_nm,
        "seed": metadata.seed_tree.root_seed,
    }


@dataclass(frozen=True, slots=True)
class PartitionedResultDataset:
    """A Hive-style partitioned collection of single-run Parquet files under
    ``root``, backed by a single :class:`~uowc.persistence.parquet_store.ResultStore`.
    """

    root: Path
    store: ResultStore = field(default_factory=ParquetResultStore)
    filename: str = "result.parquet"

    def path_for(self, partition: Partition) -> Path:
        """The file path for ``partition`` (sorted-key canonical, module docstring)."""
        if not partition:
            raise ValueError("partition must have at least one key")
        segments = [f"{key}={_sanitize(value)}" for key, value in sorted(partition.items())]
        return Path(self.root).joinpath(*segments, self.filename)

    def write(self, result: SimulationResult, partition: Partition) -> Path:
        """Write ``result`` under ``partition``; returns the path written to."""
        path = self.path_for(partition)
        self.store.write(result, path)
        return path

    def read(self, partition: Partition) -> SimulationResult:
        """Read the result stored under ``partition``."""
        return self.store.read(self.path_for(partition))

    def list_partitions(self) -> tuple[dict[str, str], ...]:
        """Every partition present on disk, discovered by walking directory names -
        no Parquet file is opened. Order is filesystem order, not guaranteed sorted;
        sort the result yourself if determinism matters.
        """
        root = Path(self.root)
        if not root.exists():
            return ()
        partitions = []
        for result_path in root.rglob(self.filename):
            key_value_parts = result_path.relative_to(root).parts[:-1]
            partitions.append(dict(part.split("=", 1) for part in key_value_parts))
        return tuple(partitions)

    def find(self, **filters: PartitionValue) -> tuple[dict[str, str], ...]:
        """Partitions matching every given filter (compared as strings, since
        partition keys are always strings once written to disk)."""
        wanted = {key: _sanitize(value) for key, value in filters.items()}
        return tuple(
            partition
            for partition in self.list_partitions()
            if wanted.items() <= partition.items()
        )

    def read_matching(self, **filters: PartitionValue) -> tuple[SimulationResult, ...]:
        """Load every result whose partition matches every given filter."""
        return tuple(self.read(partition) for partition in self.find(**filters))
