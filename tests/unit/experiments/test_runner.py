"""Tests for scenario orchestration: shared engine + metrics, fair comparison."""
from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.effects import TurbulenceEffect
from uowc.experiments import ExperimentConfig, Scenario, ScenarioRunner
from uowc.media import KamedaModel, SurfaceValue
from uowc.optics import HaltrinModel

WAVELENGTH_NM = 520.0


def make_config(*, profile, effects=(), homogenization=None, seed=1) -> ExperimentConfig:
    model = HaltrinModel(
        wavelength_nm=WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.2,
        chlorophyll_specific_absorption_m2_mg=0.04,
        pure_water_scattering_m_inv=0.05,
    )
    kwargs = {}
    if homogenization is not None:
        kwargs["homogenization"] = homogenization
    return ExperimentConfig(
        profile=profile,
        model=model,
        source=Source(position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], wavelength_nm=WAVELENGTH_NM, divergence_rad=0.1),
        receiver=Receiver(position=[0.0, 0.0, -15.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=np.pi / 4),
        bounds=Region(lower=[-30.0, -30.0, -60.0], upper=[30.0, 30.0, 0.0]),
        sampling=SamplingConfig(n_photons=3000, estimator="analog", max_scatter_events=500),
        wavelength_nm=WAVELENGTH_NM,
        effects=effects,
        seed=seed,
        **kwargs,
    )


def dcm_profile() -> KamedaModel:
    return KamedaModel(background_mg_m3=0.1, peak_integral_mg_m2=40.0, peak_depth_m=40.0, peak_width_m=10.0)


def flat_profile() -> KamedaModel:
    return KamedaModel(background_mg_m3=0.3, peak_integral_mg_m2=0.0, peak_depth_m=40.0, peak_width_m=10.0)


def power(scenario_result) -> float:
    return float(scenario_result.metrics["received_power_fraction"].value)


# --- execution modes ---------------------------------------------------------


def test_single_scenario_execution() -> None:
    runner = ScenarioRunner.default()
    res = runner.run(Scenario.I, make_config(profile=dcm_profile()))
    assert res.scenario is Scenario.I
    assert res.result.metadata.scenario == "I"
    assert res.result.metadata.medium_type == "homogeneous"
    assert set(res.metrics) == {"received_power_fraction", "mean_arrival_time", "rms_delay_spread"}


def test_selected_scenarios_execution() -> None:
    out = ScenarioRunner.default().run_selected([Scenario.I, Scenario.III], make_config(profile=dcm_profile()))
    assert set(out) == {Scenario.I, Scenario.III}


def test_all_scenarios_execution() -> None:
    out = ScenarioRunner.default().run_all(make_config(profile=dcm_profile()))
    assert set(out) == {Scenario.I, Scenario.II, Scenario.III}
    assert out[Scenario.I].result.metadata.medium_type == "homogeneous"
    assert out[Scenario.II].result.metadata.medium_type == "inhomogeneous"
    assert out[Scenario.III].result.metadata.medium_type == "inhomogeneous"


def test_shared_metrics_pipeline_keys_identical() -> None:
    out = ScenarioRunner.default().run_all(make_config(profile=dcm_profile()))
    keys = [set(r.metrics) for r in out.values()]
    assert keys[0] == keys[1] == keys[2]


# --- fair comparison: differences originate only from medium / effects -------


def test_constant_profile_makes_scenarios_I_and_II_identical() -> None:
    # With a constant profile, the homogeneous and depth-dependent media coincide,
    # so (same seed, same everything else) the results must be bit-identical.
    config = make_config(profile=flat_profile(), homogenization=SurfaceValue())
    runner = ScenarioRunner.default()
    res_i = runner.run(Scenario.I, config)
    res_ii = runner.run(Scenario.II, config)
    assert res_i.result.output.tallies.detected == res_ii.result.output.tallies.detected
    assert power(res_i) == power(res_ii)
    assert np.array_equal(
        res_i.result.output.photons.arrival_time_s, res_ii.result.output.photons.arrival_time_s
    )


def test_depth_structure_makes_I_and_II_differ() -> None:
    # A deep chlorophyll maximum below the receiver: the homogeneous baseline averages
    # in the DCM and over-attenuates the clearer upper layer the photons traverse.
    out = ScenarioRunner.default().run_all(make_config(profile=dcm_profile()))
    assert power(out[Scenario.II]) > power(out[Scenario.I])


def test_scenario_iii_without_effects_equals_scenario_ii() -> None:
    config = make_config(profile=dcm_profile(), effects=())
    runner = ScenarioRunner.default()
    res_ii = runner.run(Scenario.II, config)
    res_iii = runner.run(Scenario.III, config)
    assert res_ii.result.output.tallies.detected == res_iii.result.output.tallies.detected
    assert np.array_equal(
        res_ii.result.output.photons.arrival_time_s, res_iii.result.output.photons.arrival_time_s
    )


def test_scenario_iii_turbulence_shifts_time_only() -> None:
    # Turbulence is refractive: it perturbs n (hence arrival time) but not extinction,
    # so the same photons are detected via the same paths with shifted arrival times.
    turbulence = TurbulenceEffect.isotropic(rms_fluctuation=5e-3, correlation_length_m=5.0, n_modes=128, seed=11)
    config = make_config(profile=dcm_profile(), effects=(turbulence,))
    runner = ScenarioRunner.default()
    res_ii = runner.run(Scenario.II, config)
    res_iii = runner.run(Scenario.III, config)

    p_ii = res_ii.result.output.photons
    p_iii = res_iii.result.output.photons
    assert res_iii.result.metadata.effects == ("turbulence",)
    assert res_ii.result.output.tallies.detected == res_iii.result.output.tallies.detected
    assert np.array_equal(p_ii.path_length_m, p_iii.path_length_m)   # identical trajectories
    assert np.array_equal(p_ii.n_scatters, p_iii.n_scatters)
    # arrival times shift (relatively); absolute times are ~1e-7 s so compare with atol=0
    assert not np.allclose(p_ii.arrival_time_s, p_iii.arrival_time_s, rtol=1e-7, atol=0.0)


def test_metadata_records_provenance() -> None:
    config = make_config(profile=dcm_profile(), homogenization=SurfaceValue())
    meta = ScenarioRunner.default().run(Scenario.I, config).result.metadata
    assert meta.optical_model == "haltrin"
    assert meta.rng_impl == "NumpyRng"
    assert meta.parameters["homogenization"] == "surface"
    assert meta.seed_tree.root_seed == config.seed
    assert "numpy" in meta.library_versions
