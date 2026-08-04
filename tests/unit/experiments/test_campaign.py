"""Campaign: parameter matrices, seed ensembles, provenance, and the paired
independent-replication procedure (research-methodology.md)."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.experiments import ExperimentConfig, Scenario, ScenarioRunner
from uowc.experiments.campaign import Campaign, ParameterMatrix, ParameterPoint, replicate_seeds
from uowc.media import KamedaModel
from uowc.optics import HaltrinModel

WAVELENGTH_NM = 520.0


def make_config(*, seed=1) -> ExperimentConfig:
    model = HaltrinModel(
        wavelength_nm=WAVELENGTH_NM,
        pure_water_absorption_m_inv=0.2,
        chlorophyll_specific_absorption_m2_mg=0.04,
        pure_water_scattering_m_inv=0.05,
    )
    profile = KamedaModel(
        background_mg_m3=0.1, peak_integral_mg_m2=40.0, peak_depth_m=40.0, peak_width_m=10.0
    )
    return ExperimentConfig(
        profile=profile,
        model=model,
        source=Source(
            position=[0.0, 0.0, 0.0],
            direction=[0.0, 0.0, -1.0],
            wavelength_nm=WAVELENGTH_NM,
            divergence_rad=0.1,
        ),
        receiver=Receiver(
            position=[0.0, 0.0, -15.0],
            normal=[0.0, 0.0, 1.0],
            aperture_radius_m=1.0,
            fov_rad=np.pi / 4,
        ),
        bounds=Region(lower=[-30.0, -30.0, -60.0], upper=[30.0, 30.0, 0.0]),
        sampling=SamplingConfig(n_photons=500, estimator="analog", max_scatter_events=500),
        wavelength_nm=WAVELENGTH_NM,
        seed=seed,
    )


# --- ParameterPoint / ParameterMatrix -------------------------------------------------


def test_parameter_point_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="not an ExperimentConfig field"):
        ParameterPoint(label="bad", overrides={"not_a_real_field": 1})


def test_parameter_point_apply_overrides_a_top_level_field() -> None:
    point = ParameterPoint(label="wl=500", overrides={"wavelength_nm": 500.0})
    config = point.apply(make_config())
    assert config.wavelength_nm == 500.0
    assert config.seed == make_config().seed  # untouched fields pass through


def test_empty_matrix_yields_a_single_default_point() -> None:
    points = ParameterMatrix().points()
    assert len(points) == 1
    assert points[0].label == "default"
    assert points[0].overrides == {}


def test_matrix_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="not an ExperimentConfig field"):
        ParameterMatrix(axes={"not_a_real_field": [1, 2]})


def test_matrix_rejects_empty_axis_values() -> None:
    with pytest.raises(ValueError, match="no values"):
        ParameterMatrix(axes={"wavelength_nm": []})


def test_matrix_expands_a_single_axis() -> None:
    points = ParameterMatrix(axes={"wavelength_nm": [450.0, 500.0, 550.0]}).points()
    assert len(points) == 3
    assert {p.overrides["wavelength_nm"] for p in points} == {450.0, 500.0, 550.0}


def test_matrix_expands_the_cartesian_product_of_two_axes() -> None:
    points = ParameterMatrix(axes={"wavelength_nm": [450.0, 550.0], "seed": [1, 2, 3]}).points()
    assert len(points) == 6
    combos = {(p.overrides["wavelength_nm"], p.overrides["seed"]) for p in points}
    assert combos == {(w, s) for w in (450.0, 550.0) for s in (1, 2, 3)}


def test_matrix_point_labels_are_unique() -> None:
    points = ParameterMatrix(axes={"wavelength_nm": [450.0, 500.0]}).points()
    assert len({p.label for p in points}) == len(points)


# --- replicate_seeds -------------------------------------------------------------------


def test_replicate_seeds_are_deterministic_for_the_same_root() -> None:
    assert replicate_seeds(42, 5) == replicate_seeds(42, 5)


def test_replicate_seeds_differ_from_each_other() -> None:
    seeds = replicate_seeds(42, 10)
    assert len(set(seeds)) == 10


def test_replicate_seeds_prefix_is_stable_regardless_of_how_many_are_requested() -> None:
    # A seed's identity depends only on its own index, not on n_replicates - so
    # extending a campaign from 5 to 10 replicates does not retroactively change the
    # first 5 seeds (and therefore does not invalidate their already-computed results).
    five = replicate_seeds(42, 5)
    ten = replicate_seeds(42, 10)
    assert ten[:5] == five


def test_replicate_seeds_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError):
        replicate_seeds(42, 0)


# --- Campaign.run() ----------------------------------------------------------------------


def test_campaign_rejects_too_few_replicates() -> None:
    with pytest.raises(ValueError, match="n_replicates"):
        Campaign(base_config=make_config(), runner=ScenarioRunner.default(), n_replicates=1)


def test_campaign_rejects_no_scenarios() -> None:
    with pytest.raises(ValueError, match="at least one scenario"):
        Campaign(base_config=make_config(), runner=ScenarioRunner.default(), scenarios=())


def test_campaign_runs_every_combination() -> None:
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I, Scenario.II),
        n_replicates=3,
    )
    result = campaign.run()
    assert set(result.results) == {
        ("default", r, s) for r in range(3) for s in (Scenario.I, Scenario.II)
    }
    assert len(result.nodes) == 6


def test_campaign_replicates_use_independent_seeds_shared_across_scenarios() -> None:
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I, Scenario.II),
        n_replicates=3,
        root_seed=7,
    )
    result = campaign.run()
    by_replicate = {node.replicate_index: node.seed for node in result.nodes}
    # exactly one seed per replicate index (shared across both scenarios that replicate ran)
    assert len(by_replicate) == 3
    seeds_per_replicate = {
        r: {node.seed for node in result.nodes if node.replicate_index == r} for r in range(3)
    }
    assert all(len(s) == 1 for s in seeds_per_replicate.values())
    # replicate seeds themselves are pairwise distinct
    assert len({s for (s,) in seeds_per_replicate.values()}) == 3


def test_campaign_expands_the_parameter_matrix() -> None:
    # code_version is an inert top-level field (no coupling to the optical model),
    # used here to exercise the sweep machinery without a physics side effect - unlike
    # wavelength_nm, which must match the model's own configured wavelength
    # (ExperimentConfig's docstring) and so cannot be swept without also rebuilding it.
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I,),
        parameter_matrix=ParameterMatrix(axes={"code_version": ["v1", "v2"]}),
        n_replicates=2,
    )
    result = campaign.run()
    assert result.parameter_labels() == ("code_version=v1", "code_version=v2")


def test_campaign_provenance_nodes_carry_content_hashes() -> None:
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I,),
        n_replicates=2,
    )
    result = campaign.run()
    hashes = {node.content_hash for node in result.nodes}
    assert len(hashes) == 2  # distinct seeds -> distinct content hashes
    assert all(isinstance(h, str) and len(h) > 0 for h in hashes)


# --- CampaignResult.replicate_metric_values / compare -------------------------------------


def test_replicate_metric_values_are_ordered_by_replicate_index() -> None:
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I,),
        n_replicates=4,
    )
    result = campaign.run()
    values = result.replicate_metric_values(
        parameter_label="default", scenario=Scenario.I, metric_name="received_power_fraction"
    )
    assert len(values) == 4
    assert all(isinstance(v, float) for v in values)


def test_replicate_metric_values_raises_for_unknown_key() -> None:
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I,),
        n_replicates=2,
    )
    result = campaign.run()
    with pytest.raises(KeyError):
        result.replicate_metric_values(
            parameter_label="not_a_point",
            scenario=Scenario.I,
            metric_name="received_power_fraction",
        )


def test_compare_produces_a_scenario_difference_from_campaign_replicates() -> None:
    campaign = Campaign(
        base_config=make_config(),
        runner=ScenarioRunner.default(),
        scenarios=(Scenario.I, Scenario.II),
        n_replicates=10,
    )
    result = campaign.run()
    difference = result.compare(
        parameter_label="default",
        scenario_a=Scenario.I,
        scenario_b=Scenario.II,
        metric_name="received_power_fraction",
    )
    assert difference.n_replicates == 10
    assert difference.scenario_a == "I"
    assert difference.scenario_b == "II"
    assert difference.meets_recommended_replicate_count
