"""Tests for the NumpyRng adapter."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core.ports import Rng
from uowc.core.results import SeedTree
from uowc.core.rng import NumpyRng


def test_conforms_to_rng_protocol() -> None:
    assert isinstance(NumpyRng(0), Rng)
    assert NumpyRng(123).seed == 123


def test_uniform_is_deterministic_for_same_seed() -> None:
    assert np.array_equal(NumpyRng(7).uniform(100), NumpyRng(7).uniform(100))


def test_shapes_and_ranges() -> None:
    rng = NumpyRng(1)
    u = rng.uniform(5)
    assert u.shape == (5,) and np.all((u >= 0.0) & (u < 1.0))
    assert rng.uniform((2, 3)).shape == (2, 3)
    assert rng.normal(10).shape == (10,)
    ints = rng.integers(0, 4, 1000)
    assert ints.min() >= 0 and ints.max() <= 3


def test_spawn_is_deterministic_and_order_independent() -> None:
    a = NumpyRng(42).spawn(3).uniform(50)
    b = NumpyRng(42).spawn(3).uniform(50)
    assert np.array_equal(a, b)


def test_distinct_streams_differ() -> None:
    s1 = NumpyRng(42).spawn(1).uniform(50)
    s2 = NumpyRng(42).spawn(2).uniform(50)
    assert not np.array_equal(s1, s2)


def test_seed_tree_records_spawned_streams() -> None:
    rng = NumpyRng(5)
    rng.spawn(0)
    rng.spawn(7)
    tree = rng.seed_tree()
    assert isinstance(tree, SeedTree)
    assert tree.root_seed == 5
    assert set(tree.streams) == {"0", "7"}
