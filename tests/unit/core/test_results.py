"""Result schema field/dtype contracts; labelled-axis integrity.

No ``LabelledCurve`` class exists in results.py - a curve-valued metric is a
:class:`MetricValue` with one or more :class:`Axis` entries. This module tests that
mechanism, plus the dtype/immutability contracts of the other schema types.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from uowc.core.config import SamplingConfig
from uowc.core.geometry import Receiver, Source
from uowc.core.results import (
    Axis,
    DetectedPhotons,
    MetricValue,
    RawResult,
    RunMetadata,
    SeedTree,
    Tallies,
    TallyResult,
    TransportOutput,
)

_SOURCE = Source(
    position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], wavelength_nm=520.0, divergence_rad=0.0
)
_RECEIVER = Receiver(
    position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
)


def _metadata() -> RunMetadata:
    return RunMetadata(
        scenario="I",
        medium_type="homogeneous",
        optical_model="haltrin",
        model_parameters={},
        effects=(),
        effect_parameters=(),
        wavelength_nm=520.0,
        source=_SOURCE,
        receiver=_RECEIVER,
        majorant=1.0,
        sampling=SamplingConfig(n_photons=100, estimator="analog"),
        seed_tree=SeedTree(root_seed=1),
        rng_impl="NumpyRng",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        code_version="abc123",
    )


# --- DetectedPhotons -------------------------------------------------------------------


def test_detected_photons_coerces_n_scatters_to_int64() -> None:
    photons = DetectedPhotons(
        arrival_time_s=np.array([1.0]),
        path_length_m=np.array([1.0]),
        weight=np.array([1.0]),
        n_scatters=np.array([2], dtype=np.int32),
        incidence_rad=np.array([0.0]),
    )
    assert photons.n_scatters.dtype == np.int64


def test_detected_photons_arrays_are_frozen() -> None:
    photons = DetectedPhotons(
        arrival_time_s=np.array([1.0]),
        path_length_m=np.array([1.0]),
        weight=np.array([1.0]),
        n_scatters=np.array([0], dtype=np.int64),
        incidence_rad=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="read-only"):
        photons.arrival_time_s[0] = 99.0


# --- Tallies / TallyResult -------------------------------------------------------------


def test_tallies_extra_defaults_to_empty() -> None:
    tallies = Tallies(
        launched=10, detected=1, detected_weight=1.0, absorbed_weight=9.0, escaped_weight=0.0
    )
    assert dict(tallies.extra) == {}


def test_tallies_is_frozen_and_comparable() -> None:
    a = Tallies(
        launched=10, detected=1, detected_weight=1.0, absorbed_weight=9.0, escaped_weight=0.0
    )
    b = Tallies(
        launched=10, detected=1, detected_weight=1.0, absorbed_weight=9.0, escaped_weight=0.0
    )
    assert a == b
    with pytest.raises(FrozenInstanceError):
        a.launched = 20  # type: ignore[misc]


def test_tally_result_values_shape_matches_edges_minus_one() -> None:
    tally = TallyResult(
        name="cir",
        values=np.array([1.0, 2.0, 3.0]),
        edges=(np.array([0.0, 1.0, 2.0, 3.0]),),
        unit="dimensionless",
        axis_names=("arrival_time_s",),
    )
    assert tally.values.shape[0] == tally.edges[0].shape[0] - 1


def test_tally_result_arrays_are_frozen() -> None:
    tally = TallyResult(
        name="cir",
        values=np.array([1.0, 2.0]),
        edges=(np.array([0.0, 1.0, 2.0]),),
        unit="dimensionless",
        axis_names=("t",),
    )
    with pytest.raises(ValueError, match="read-only"):
        tally.values[0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        tally.edges[0][0] = 5.0


# --- SeedTree / RunMetadata / RawResult -------------------------------------------------


def test_seed_tree_streams_default_to_empty() -> None:
    tree = SeedTree(root_seed=7)
    assert dict(tree.streams) == {}


def test_seed_tree_streams_are_recorded_by_name() -> None:
    tree = SeedTree(root_seed=7, streams={"transport": 1, "turbulence_field": 2})
    assert tree.streams["turbulence_field"] == 2


def test_run_metadata_parameters_and_library_versions_default_to_empty() -> None:
    meta = _metadata()
    assert dict(meta.parameters) == {}
    assert dict(meta.library_versions) == {}


def test_raw_result_defaults_to_the_current_schema_version() -> None:
    from uowc.core.results import SCHEMA_VERSION

    photons = DetectedPhotons(
        arrival_time_s=np.empty(0),
        path_length_m=np.empty(0),
        weight=np.empty(0),
        n_scatters=np.empty(0, dtype=np.int64),
        incidence_rad=np.empty(0),
    )
    tallies = Tallies(
        launched=0, detected=0, detected_weight=0.0, absorbed_weight=0.0, escaped_weight=0.0
    )
    raw = RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=_metadata())
    assert raw.schema_version == SCHEMA_VERSION


# --- Axis / MetricValue ------------------------------------------------------------------


def test_axis_values_are_frozen() -> None:
    axis = Axis(name="frequency", values=np.array([0.0, 1.0, 2.0]), unit="Hz")
    with pytest.raises(ValueError, match="read-only"):
        axis.values[0] = 99.0


def test_metric_value_scalar_passes_through_unfrozen() -> None:
    # freeze_value must not promote a plain float to a 0-d array - callers compare
    # metric.value == x and expect ordinary float semantics.
    metric = MetricValue(name="m", value=0.5, unit="fraction", n_samples=10)
    assert isinstance(metric.value, float)


def test_metric_value_array_is_frozen() -> None:
    metric = MetricValue(name="m", value=np.array([1.0, 2.0]), unit="dimensionless", n_samples=10)
    with pytest.raises(ValueError, match="read-only"):
        metric.value[0] = 5.0  # type: ignore[index]


def test_metric_value_none_uncertainty_is_the_default() -> None:
    metric = MetricValue(name="m", value=0.5, unit="fraction", n_samples=10)
    assert metric.std is None
    assert metric.ci95_low is None
    assert metric.ci95_high is None


def test_metric_value_carries_one_or_more_axes_for_a_curve() -> None:
    axis = Axis(name="frequency", values=np.array([0.0, 1e6, 2e6]), unit="Hz")
    metric = MetricValue(
        name="frequency_response",
        value=np.array([1.0, 0.7, 0.3]),
        unit="dimensionless",
        n_samples=10,
        axes=(axis,),
    )
    assert len(metric.axes) == 1
    assert metric.axes[0].unit == "Hz"
    assert metric.value.shape == metric.axes[0].values.shape


def test_metric_value_defaults_to_no_axes() -> None:
    metric = MetricValue(name="m", value=0.5, unit="fraction", n_samples=10)
    assert metric.axes == ()
