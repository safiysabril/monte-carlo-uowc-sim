"""Run-level sampling configuration, kept separate from physical source config."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SamplingConfig"]


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How hard to simulate a link - distinct from *what* the link is.

    Separating the statistical sample budget from :class:`~uowc.core.geometry.Source`
    lets a convergence study vary ``n_photons`` without changing (or re-identifying)
    the physical source, preserving fair comparison and clean provenance.

    Fields:
        n_photons:          total photons to launch
        estimator:          detection estimator ("analog" | "next_event" | "forced")
        max_scatter_events: per-photon scatter cap (numerical safety)
        chunk_size:         photons per work unit; the chunk -> RNG-stream map is
                            fixed, so results are independent of worker count
        tallies:            names of online tallies to accumulate (e.g. "cir",
                            "depth_deposition", "angle_of_arrival")
    """

    n_photons: int
    estimator: str = "next_event"
    max_scatter_events: int = 1000
    chunk_size: int = 100_000
    tallies: tuple[str, ...] = ()
