"""Parquet result store: write/read a SimulationResult with a round-trip guarantee.

The detected photons become Parquet columns; the run metadata (scenario, model,
parameters, seed, environmental effects), tallies, binned tallies and metrics are
JSON-encoded into the file's schema metadata. Reading reconstructs an equal
SimulationResult (dtype-preserving), so derived analysis is reproducible from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from uowc.core.results import RawResult, TransportOutput
from uowc.persistence.result import SimulationResult
from uowc.persistence.schema import (
    PHOTON_COLUMNS,
    decode_sidecar,
    encode_sidecar,
    photons_from_columns,
    photons_to_columns,
)

__all__ = ["ResultStore", "ParquetResultStore"]

_METADATA_KEY = b"uowc"


@runtime_checkable
class ResultStore(Protocol):
    """Persists and reloads a :class:`SimulationResult`."""

    def write(self, result: SimulationResult, path: Path) -> None: ...

    def read(self, path: Path) -> SimulationResult: ...


class ParquetResultStore:
    """A :class:`ResultStore` backed by a single Parquet file per run."""

    def write(self, result: SimulationResult, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        columns = photons_to_columns(result.raw.output.photons)
        table = pa.table({name: pa.array(columns[name]) for name in PHOTON_COLUMNS})

        sidecar = encode_sidecar(
            schema_version=result.raw.schema_version,
            metadata=result.raw.metadata,
            tallies=result.raw.output.tallies,
            binned=result.raw.output.binned,
            metrics=result.metrics,
        )
        table = table.replace_schema_metadata({_METADATA_KEY: json.dumps(sidecar).encode("utf-8")})
        pq.write_table(table, str(path))

    def read(self, path: Path) -> SimulationResult:
        table = pq.read_table(str(path))

        raw_metadata = table.schema.metadata or {}
        if _METADATA_KEY not in raw_metadata:
            raise ValueError(f"{path} is missing uowc result metadata")
        schema_version, metadata, tallies, binned, metrics = decode_sidecar(
            json.loads(raw_metadata[_METADATA_KEY].decode("utf-8"))
        )

        columns = {
            name: table.column(name).to_numpy(zero_copy_only=False) for name in PHOTON_COLUMNS
        }
        photons = photons_from_columns(columns)

        output = TransportOutput(photons=photons, tallies=tallies, binned=binned)
        raw = RawResult(output=output, metadata=metadata, schema_version=schema_version)
        return SimulationResult(raw=raw, metrics=metrics)
