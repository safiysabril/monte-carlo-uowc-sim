"""Concrete random source: a numpy PCG64-backed implementation of the Rng port.

Splittable via :meth:`NumpyRng.spawn`, which derives a child stream deterministically
from ``(root_seed, spawn_key)`` using a ``SeedSequence`` ``spawn_key`` - so a given
stream id always maps to the same stream regardless of execution order or worker
count (reproducible parallelism). The seed hierarchy is recoverable via
:meth:`NumpyRng.seed_tree`.
"""
from __future__ import annotations

import numpy as np

from uowc.core.results import SeedTree
from uowc.core.units import FloatArray, IntArray

__all__ = ["NumpyRng"]

_Shape = int | tuple[int, ...]


class NumpyRng:
    """Seeded, splittable RNG implementing :class:`~uowc.core.ports.Rng`."""

    def __init__(
        self,
        seed: int,
        *,
        _seed_sequence: np.random.SeedSequence | None = None,
        _spawn_key: tuple[int, ...] = (),
    ) -> None:
        self._root_seed = int(seed)
        self._spawn_key = tuple(_spawn_key)
        self._sequence = (
            _seed_sequence if _seed_sequence is not None else np.random.SeedSequence(self._root_seed)
        )
        self._generator = np.random.Generator(np.random.PCG64(self._sequence))
        self._spawned: dict[str, int] = {}

    @property
    def seed(self) -> int:
        return self._root_seed

    def uniform(self, shape: _Shape = 1) -> FloatArray:
        """Uniform variates in [0, 1)."""
        return self._generator.random(size=shape)

    def normal(self, shape: _Shape = 1) -> FloatArray:
        """Standard-normal variates."""
        return self._generator.standard_normal(size=shape)

    def integers(self, low: int, high: int, shape: _Shape = 1) -> IntArray:
        """Integers in [low, high)."""
        return self._generator.integers(low, high, size=shape, dtype=np.int64)

    def spawn(self, stream: int) -> "NumpyRng":
        """Return an independent child stream deterministically keyed by ``stream``."""
        key = self._spawn_key + (int(stream),)
        child_sequence = np.random.SeedSequence(entropy=self._root_seed, spawn_key=key)
        child = NumpyRng(self._root_seed, _seed_sequence=child_sequence, _spawn_key=key)
        self._spawned[str(int(stream))] = int(child_sequence.generate_state(1)[0])
        return child

    def seed_tree(self) -> SeedTree:
        """Recorded seed hierarchy: the root seed and any spawned child streams."""
        return SeedTree(root_seed=self._root_seed, streams=dict(self._spawned))
