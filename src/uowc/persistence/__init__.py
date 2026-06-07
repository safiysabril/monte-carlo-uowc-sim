"""Data persistence (adapter): Parquet round-trip. Downstream of simulation."""
from __future__ import annotations

from uowc.persistence.parquet_store import ParquetResultStore, ResultStore
from uowc.persistence.result import SimulationResult

__all__ = ["SimulationResult", "ResultStore", "ParquetResultStore"]
