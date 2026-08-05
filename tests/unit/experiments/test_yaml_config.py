"""YAML-driven ExperimentConfig construction, tested against the real configs/ tree.

Deliberately not hand-rolled fixture YAML: configs/ is a real, checked-in artifact
this loader must keep working against, so a change to either side that breaks the
contract is caught here rather than only in a manual run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uowc.experiments import ConfigError, Scenario, ScenarioRunner, build_experiment_config
from uowc.experiments.yaml_config import (
    build_homogenization_rule,
    build_source_and_receiver,
    load_yaml,
)
from uowc.media import DepthAverage, OpticalDepthPreserving, SurfaceValue

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

_MODEL_PARAMS = {
    "pure_water_absorption_m_inv": 0.05,
    "chlorophyll_specific_absorption_m2_mg": 0.05,
    "pure_water_scattering_m_inv": 0.003,
}


def _skip_if_no_configs() -> None:
    if not _CONFIG_ROOT.exists():
        pytest.skip(f"configs/ not found at {_CONFIG_ROOT}")


# --- load_yaml -------------------------------------------------------------------------


def test_base_yaml_photon_counts_parse_as_numbers_not_strings() -> None:
    _skip_if_no_configs()
    # Regression guard: PyYAML's SafeLoader parses an unsigned float exponent
    # ("1.0e7") as a *string*; base.yaml must use the signed form ("1.0e+7").
    base = load_yaml(_CONFIG_ROOT / "base.yaml")
    assert isinstance(base["photons"]["production"], float)
    assert all(isinstance(v, float) for v in base["photons"]["ladder"])


def test_load_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_yaml(path) == {}


# --- build_homogenization_rule ----------------------------------------------------------


def test_build_homogenization_rule_surface() -> None:
    assert isinstance(build_homogenization_rule("surface"), SurfaceValue)


def test_build_homogenization_rule_accepts_both_depth_average_spellings() -> None:
    assert isinstance(build_homogenization_rule("path_average"), DepthAverage)
    assert isinstance(build_homogenization_rule("depth_average"), DepthAverage)


def test_build_homogenization_rule_optical_depth() -> None:
    assert isinstance(build_homogenization_rule("optical_depth"), OpticalDepthPreserving)


def test_build_homogenization_rule_rejects_unknown_name() -> None:
    with pytest.raises(ConfigError, match="unknown homogenization_rule"):
        build_homogenization_rule("not_a_real_rule")


# --- build_source_and_receiver ----------------------------------------------------------


def test_build_source_and_receiver_uses_the_presets_own_positions_by_default() -> None:
    geometry = {
        "source_position_m": [0.0, 0.0, 0.0],
        "source_direction": [0.0, 0.0, -1.0],
        "receiver_position_m": [0.0, 0.0, -10.0],
        "receiver_normal": [0.0, 0.0, 1.0],
        "aperture_radius_m": 0.1,
        "fov_rad": 0.35,
    }
    source, receiver = build_source_and_receiver(geometry, wavelength_nm=520.0, divergence_rad=0.01)
    np.testing.assert_allclose(receiver.position, [0.0, 0.0, -10.0])
    assert source.wavelength_nm == 520.0
    assert source.divergence_rad == 0.01


def test_build_source_and_receiver_range_m_overrides_distance_along_direction() -> None:
    geometry = {
        "source_position_m": [0.0, 0.0, 0.0],
        "source_direction": [0.0, 0.0, -1.0],
        "receiver_position_m": [0.0, 0.0, -10.0],
        "receiver_normal": [0.0, 0.0, 1.0],
        "aperture_radius_m": 0.1,
        "fov_rad": 0.35,
    }
    _, receiver = build_source_and_receiver(
        geometry, wavelength_nm=520.0, divergence_rad=0.01, range_m=25.0
    )
    np.testing.assert_allclose(receiver.position, [0.0, 0.0, -25.0])


def test_build_source_and_receiver_normalizes_a_non_unit_direction() -> None:
    geometry = {
        "source_position_m": [0.0, 0.0, 0.0],
        "source_direction": [0.0, 0.0, -2.0],  # not a unit vector
        "receiver_position_m": [0.0, 0.0, -10.0],
        "receiver_normal": [0.0, 0.0, 1.0],
        "aperture_radius_m": 0.1,
        "fov_rad": 0.35,
    }
    _, receiver = build_source_and_receiver(
        geometry, wavelength_nm=520.0, divergence_rad=0.01, range_m=10.0
    )
    np.testing.assert_allclose(receiver.position, [0.0, 0.0, -10.0])


# --- build_experiment_config, against the real configs/ tree ----------------------------


def test_scenario_i_builds_with_explicit_model_params() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="I",
        geometry_name="vertical",
        water_type_name="clear_ocean",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        n_photons=500,
    )
    assert config.model.name == "haltrin"
    assert config.homogenization.name == "optical_depth"  # scenario_i.yaml's own choice
    assert config.sampling.n_photons == 500
    assert config.effects == ()


def test_scenario_ii_builds_with_the_kameda_profile() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="II",
        geometry_name="vertical",
        water_type_name="coastal",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        n_photons=500,
    )
    assert config.profile.__class__.__name__ == "KamedaModel"


def test_scenario_iii_requires_effect_params() -> None:
    _skip_if_no_configs()
    with pytest.raises(ConfigError, match="turbulence"):
        build_experiment_config(
            scenario_name="III",
            geometry_name="vertical",
            water_type_name="clear_ocean",
            config_root=_CONFIG_ROOT,
            model_params=_MODEL_PARAMS,
        )


def test_scenario_iii_builds_once_effect_params_are_supplied() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="III",
        geometry_name="vertical",
        water_type_name="clear_ocean",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        n_photons=500,
        effect_params={
            "turbulence": {
                "rms_fluctuation": 2e-3,
                "correlation_length_m": 5.0,
                "n_modes": 32,
                "seed": 3,
            },
            "bubbles": {
                "region": {"lower": [-50.0, -50.0, -6.0], "upper": [50.0, 50.0, -4.0]},
                "void_fraction": 1e-6,
                "mean_radius_m": 1e-3,
                "phase": {"type": "henyey_greenstein", "g": 0.9},
            },
        },
    )
    assert {e.name for e in config.effects} == {"turbulence", "bubbles"}


def test_missing_optical_model_params_raises_a_clear_error() -> None:
    _skip_if_no_configs()
    with pytest.raises(ConfigError, match="model_params"):
        build_experiment_config(
            scenario_name="I",
            geometry_name="vertical",
            water_type_name="clear_ocean",
            config_root=_CONFIG_ROOT,
        )


def test_unknown_water_type_raises() -> None:
    _skip_if_no_configs()
    with pytest.raises(ConfigError, match="water_type"):
        build_experiment_config(
            scenario_name="I",
            geometry_name="vertical",
            water_type_name="not_a_real_water_type",
            config_root=_CONFIG_ROOT,
            model_params=_MODEL_PARAMS,
        )


def test_unknown_geometry_raises() -> None:
    _skip_if_no_configs()
    with pytest.raises(ConfigError, match="geometry"):
        build_experiment_config(
            scenario_name="I",
            geometry_name="not_a_real_geometry",
            water_type_name="clear_ocean",
            config_root=_CONFIG_ROOT,
            model_params=_MODEL_PARAMS,
        )


def test_range_m_is_threaded_through_to_the_receiver() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="II",
        geometry_name="vertical",
        water_type_name="clear_ocean",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        range_m=40.0,
        n_photons=500,
    )
    distance = float(
        np.linalg.norm(np.asarray(config.receiver.position) - np.asarray(config.source.position))
    )
    assert distance == pytest.approx(40.0)


def test_domain_bounds_contain_both_source_and_receiver() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="II",
        geometry_name="horizontal",
        water_type_name="clear_ocean",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        range_m=40.0,
        n_photons=500,
    )
    lower, upper = np.asarray(config.bounds.lower), np.asarray(config.bounds.upper)
    for point in (np.asarray(config.source.position), np.asarray(config.receiver.position)):
        assert np.all(point >= lower) and np.all(point <= upper)


def test_estimator_and_n_photons_are_overridable() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="I",
        geometry_name="vertical",
        water_type_name="clear_ocean",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        n_photons=42,
        estimator="analog",
    )
    assert config.sampling.n_photons == 42
    assert config.sampling.estimator == "analog"


# --- end-to-end: a loaded config actually runs -------------------------------------------


def test_loaded_scenario_actually_runs() -> None:
    _skip_if_no_configs()
    config = build_experiment_config(
        scenario_name="II",
        geometry_name="vertical",
        water_type_name="clear_ocean",
        config_root=_CONFIG_ROOT,
        model_params=_MODEL_PARAMS,
        n_photons=500,
        estimator="analog",
    )
    result = ScenarioRunner.default().run(Scenario.II, config)
    assert result.result.metadata.optical_model == "haltrin"
    assert "received_power_fraction" in result.metrics
