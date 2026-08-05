"""Parquet write/read equality (dtype preserving).

Unlike ``tests/unit/persistence/test_parquet_store.py`` (hand-built value objects),
this is the end-to-end capstone: a real transport run, scored by the real default
metrics pipeline, requesting a binned tally, persisted and reloaded - checking that
nothing in the full data.md pipeline (transport -> metrics -> storage) silently loses
or corrupts information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uowc.core import Receiver, Region, SamplingConfig, Source, provenance
from uowc.core.provenance import code_version, library_versions, utc_timestamp
from uowc.core.results import RawResult, RunMetadata
from uowc.core.rng import NumpyRng
from uowc.media import HomogeneousMedium
from uowc.metrics import MetricPipeline
from uowc.optics import HaltrinModel
from uowc.persistence.parquet_store import ParquetResultStore
from uowc.persistence.result import SimulationResult
from uowc.transport import WoodcockDeltaTracker

pytestmark = pytest.mark.verification

_WAVELENGTH_NM = 520.0


def _build_result() -> SimulationResult:
    model = HaltrinModel(
        wavelength_nm=_WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.15,
        chlorophyll_specific_absorption_m2_mg=0.05,
        pure_water_scattering_m_inv=0.3,
    )
    bounds = Region(lower=[-20.0, -20.0, -40.0], upper=[20.0, 20.0, 0.0])
    medium = HomogeneousMedium.uniform(
        model=model, chlorophyll=0.4, wavelength_nm=_WAVELENGTH_NM, bounds=bounds
    )
    source = Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        wavelength_nm=_WAVELENGTH_NM,
        divergence_rad=0.15,
    )
    receiver = Receiver(
        position=[0.0, 0.0, -8.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.5, fov_rad=np.pi / 3
    )
    sampling = SamplingConfig(
        n_photons=3000, estimator="analog", max_scatter_events=500, tallies=("cir",)
    )
    rng = NumpyRng(2024)
    output = WoodcockDeltaTracker().run(medium, source, receiver, rng, sampling)

    metadata = RunMetadata(
        scenario="I",
        medium_type="homogeneous",
        optical_model="haltrin",
        model_parameters=provenance.scalar_fields(model),
        effects=(),
        effect_parameters=(),
        wavelength_nm=_WAVELENGTH_NM,
        source=source,
        receiver=receiver,
        majorant=medium.acceleration.majorant(bounds),
        sampling=sampling,
        seed_tree=rng.seed_tree(),
        rng_impl=type(rng).__name__,
        timestamp_utc=utc_timestamp(),
        code_version=code_version(),
        library_versions=library_versions(),
    )
    raw = RawResult(output=output, metadata=metadata)
    metrics = MetricPipeline.default().run(raw)
    return SimulationResult(raw=raw, metrics=dict(metrics))


def test_full_pipeline_round_trips_exactly(tmp_path: Path) -> None:
    result = _build_result()
    path = tmp_path / "run.parquet"
    ParquetResultStore().write(result, path)
    restored = ParquetResultStore().read(path)

    # Photons: dtype-preserving array equality.
    np.testing.assert_array_equal(
        restored.raw.output.photons.arrival_time_s, result.raw.output.photons.arrival_time_s
    )
    np.testing.assert_array_equal(
        restored.raw.output.photons.n_scatters, result.raw.output.photons.n_scatters
    )
    assert (
        restored.raw.output.photons.n_scatters.dtype == result.raw.output.photons.n_scatters.dtype
    )

    # Tallies and full provenance record.
    assert restored.raw.output.tallies == result.raw.output.tallies
    assert restored.raw.metadata == result.raw.metadata

    # The binned CIR tally requested via sampling.tallies.
    assert set(restored.raw.output.binned) == set(result.raw.output.binned)
    for name in result.raw.output.binned:
        np.testing.assert_array_equal(
            restored.raw.output.binned[name].values, result.raw.output.binned[name].values
        )

    # Every metric the default pipeline produced, including its uncertainty.
    assert set(restored.metrics) == set(result.metrics)
    for name, metric in result.metrics.items():
        restored_metric = restored.metrics[name]
        np.testing.assert_allclose(np.asarray(restored_metric.value), np.asarray(metric.value))
        if metric.std is not None:
            assert restored_metric.std == pytest.approx(metric.std)
