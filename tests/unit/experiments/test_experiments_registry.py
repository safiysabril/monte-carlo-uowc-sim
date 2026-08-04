"""Config-driven construction of optical models, profiles, and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.experiments.registry import (
    available_metrics,
    available_optical_models,
    available_profiles,
    build_metric,
    build_metric_pipeline,
    build_optical_model,
    build_profile,
)
from uowc.media.profiles import KamedaModel
from uowc.metrics import Bandwidth3dB, FrequencyResponse, MeanArrivalTime, ReceivedPowerFraction
from uowc.optics.haltrin import HaltrinModel


def test_available_optical_models_is_haltrin_only() -> None:
    """kameda is deliberately absent - see uowc.optics.kameda."""
    assert available_optical_models() == ("haltrin",)


def test_build_haltrin_with_required_params() -> None:
    model = build_optical_model(
        {
            "name": "haltrin",
            "params": {
                "wavelength_nm": 520.0,
                "pure_water_absorption_m_inv": 0.04,
                "chlorophyll_specific_absorption_m2_mg": 0.02,
                "pure_water_scattering_m_inv": 0.002,
            },
        }
    )
    assert isinstance(model, HaltrinModel)
    assert model.wavelength_nm == pytest.approx(520.0)
    assert model.chlorophyll_absorption_exponent == pytest.approx(0.602)  # documented default


def test_build_haltrin_respects_optional_overrides() -> None:
    model = build_optical_model(
        {
            "name": "haltrin",
            "params": {
                "wavelength_nm": 500.0,
                "pure_water_absorption_m_inv": 0.04,
                "chlorophyll_specific_absorption_m2_mg": 0.02,
                "pure_water_scattering_m_inv": 0.002,
                "chlorophyll_absorption_exponent": 0.7,
            },
        }
    )
    assert model.chlorophyll_absorption_exponent == pytest.approx(0.7)


def test_unknown_optical_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown optical-property model"):
        build_optical_model({"name": "kameda", "params": {}})


def test_available_profiles_is_kameda_only() -> None:
    assert available_profiles() == ("kameda",)


def test_build_kameda_profile_direct_parameters() -> None:
    profile = build_profile(
        {
            "name": "kameda",
            "params": {
                "background_mg_m3": 0.05,
                "peak_integral_mg_m2": 20.0,
                "peak_depth_m": 80.0,
                "peak_width_m": 20.0,
            },
        }
    )
    assert isinstance(profile, KamedaModel)
    assert profile.peak_depth_m == pytest.approx(80.0)


def test_build_kameda_profile_from_surface_chlorophyll() -> None:
    profile = build_profile({"name": "kameda", "params": {"surface_mg_m3": 0.3}})
    assert isinstance(profile, KamedaModel)
    direct = KamedaModel.from_surface_chlorophyll(0.3)
    assert profile.background_mg_m3 == pytest.approx(direct.background_mg_m3)
    assert profile.peak_depth_m == pytest.approx(direct.peak_depth_m)


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown chlorophyll profile"):
        build_profile({"name": "smith_baker", "params": {}})


def test_available_metrics_lists_all_five() -> None:
    assert available_metrics() == (
        "bandwidth_3db",
        "frequency_response",
        "mean_arrival_time",
        "received_power_fraction",
        "rms_delay_spread",
    )


def test_build_zero_arg_metrics() -> None:
    assert isinstance(build_metric({"name": "received_power_fraction"}), ReceivedPowerFraction)
    assert isinstance(build_metric({"name": "mean_arrival_time"}), MeanArrivalTime)


def test_build_frequency_response_metric() -> None:
    metric = build_metric(
        {"name": "frequency_response", "params": {"frequencies_hz": [0.0, 1e6, 2e6]}}
    )
    assert isinstance(metric, FrequencyResponse)
    np.testing.assert_allclose(metric.frequencies_hz, [0.0, 1e6, 2e6])


def test_build_bandwidth_metric_with_defaults_and_overrides() -> None:
    default = build_metric({"name": "bandwidth_3db", "params": {"max_frequency_hz": 1e8}})
    assert isinstance(default, Bandwidth3dB)
    assert default.convention == "electrical"
    assert default.n_points == 2048

    custom = build_metric(
        {"name": "bandwidth_3db", "params": {"max_frequency_hz": 1e8, "convention": "optical"}}
    )
    assert custom.convention == "optical"


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        build_metric({"name": "coherence_bandwidth"})


def test_build_metric_pipeline_runs_end_to_end() -> None:
    from uowc.core import (
        DetectedPhotons,
        RawResult,
        RunMetadata,
        SamplingConfig,
        SeedTree,
        Tallies,
        TransportOutput,
    )

    pipeline = build_metric_pipeline(
        [{"name": "received_power_fraction"}, {"name": "mean_arrival_time"}]
    )
    photons = DetectedPhotons(
        arrival_time_s=np.array([1e-7]),
        path_length_m=np.array([30.0]),
        weight=np.array([1.0]),
        n_scatters=np.array([0], dtype=np.int64),
        incidence_rad=np.array([0.0]),
    )
    tallies = Tallies(
        launched=10, detected=1, detected_weight=1.0, absorbed_weight=9.0, escaped_weight=0.0
    )
    meta = RunMetadata(
        scenario="T",
        medium_type="t",
        optical_model="t",
        effects=(),
        wavelength_nm=520.0,
        sampling=SamplingConfig(n_photons=10, estimator="analog"),
        seed_tree=SeedTree(root_seed=0),
        rng_impl="t",
        timestamp_utc="t",
        code_version="t",
    )
    raw = RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=meta)
    results = pipeline.run(raw)
    assert set(results) == {"received_power_fraction", "mean_arrival_time"}
