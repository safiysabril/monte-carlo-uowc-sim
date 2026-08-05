"""Entry point for the medium-representation study.

Loads ``configs/campaigns/medium_representation_study.yaml`` (via ``config.yaml``'s
``extends``), builds one :class:`~uowc.experiments.ExperimentConfig` per axis
combination with :func:`uowc.experiments.build_experiment_config`, runs each through
:class:`~uowc.experiments.ScenarioRunner`, and persists raw results to
``data/raw`` (partitioned as ``base.yaml``'s ``output.partition_by`` declares:
scenario/water_type/geometry/range_m/seed, plus ``homogenization`` where it applies).

Scenario I is run once per entry in the campaign's ``homogenization_rules`` list -
research-methodology.md: "Results should be reported across all implemented rules,
not a single favoured one."

Scenario III needs numeric parameters for its ``turbulence``/``bubbles`` effects that
``configs/scenarios/scenario_iii.yaml`` does not give ("parameters to be specified");
this script does not invent them (see ``uowc.experiments.yaml_config``'s module
docstring for why) - pass ``effect_params`` to :func:`main` to include it, or accept
the default of running Scenario I/II only.

The full campaign grid (3 geometries x 3 water types x 4 ranges x 5 seeds, plus the
homogenization sweep, at the production photon count) is a long batch job, not
something to run by invoking this file directly without thought. ``main(quick=True)``
(the default when run as a script) exercises one geometry, one water type, one range
and one seed at a reduced photon count, to smoke-test the whole pipeline quickly;
pass ``quick=False`` for the real grid.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

from uowc.experiments import (
    ConfigError,
    Scenario,
    ScenarioRunner,
    build_experiment_config,
    build_homogenization_rule,
    load_yaml,
)
from uowc.persistence.dataset import PartitionedResultDataset
from uowc.persistence.result import SimulationResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _REPO_ROOT / "configs"

#: Illustrative, not calibrated - see the module docstring on why nothing more
#: specific is invented here. Replace before any quantitative use.
_DEFAULT_MODEL_PARAMS = {
    "pure_water_absorption_m_inv": 0.05,
    "chlorophyll_specific_absorption_m2_mg": 0.05,
    "pure_water_scattering_m_inv": 0.003,
}


def _load_campaign_spec() -> dict:
    instance = load_yaml(_REPO_ROOT / "experiments" / "medium_representation" / "config.yaml")
    campaign_path = _REPO_ROOT / instance["extends"]
    return load_yaml(campaign_path)


def main(
    *,
    quick: bool = True,
    n_photons: int | None = None,
    model_params: dict | None = None,
    effect_params: dict | None = None,
    data_root: Path | None = None,
) -> list[Path]:
    """Run the medium-representation study; return the paths written."""
    spec = _load_campaign_spec()
    axes = spec["axes"]
    homogenization_rules = spec["homogenization_rules"]
    model_params = model_params if model_params is not None else _DEFAULT_MODEL_PARAMS

    geometries = axes["geometry"][:1] if quick else axes["geometry"]
    water_types = axes["water_type"][:1] if quick else axes["water_type"]
    ranges = axes["range_m"][:1] if quick else axes["range_m"]
    seeds = axes["seed"][:1] if quick else axes["seed"]
    quick_n_photons = 2_000
    n_photons = n_photons if n_photons is not None else (quick_n_photons if quick else None)

    dataset = PartitionedResultDataset(
        root=Path(data_root) if data_root else _REPO_ROOT / "data" / "raw"
    )
    runner = ScenarioRunner.default()
    written: list[Path] = []

    for geometry, water_type, range_m, seed in itertools.product(
        geometries, water_types, ranges, seeds
    ):
        base_partition = {
            "geometry": geometry,
            "water_type": water_type,
            "range_m": float(range_m),
            "seed": int(seed),
        }

        for scenario_name in ("I", "II", "III"):
            if scenario_name == "III" and effect_params is None:
                print(f"Skipping Scenario III at {base_partition}: no effect_params supplied")
                print("(see this module's docstring)")
                continue
            try:
                config = build_experiment_config(
                    scenario_name=scenario_name,
                    geometry_name=geometry,
                    water_type_name=water_type,
                    config_root=_CONFIG_ROOT,
                    range_m=float(range_m),
                    seed=int(seed),
                    n_photons=n_photons,
                    model_params=model_params,
                    effect_params=effect_params,
                )
            except ConfigError as exc:
                print(f"Skipping scenario {scenario_name} at {base_partition}: {exc}")
                continue

            rule_names = homogenization_rules if scenario_name == "I" else [None]
            for rule_name in rule_names:
                run_config = config
                partition = {"scenario": scenario_name, **base_partition}
                if rule_name is not None:
                    run_config = replace(
                        config, homogenization=build_homogenization_rule(rule_name)
                    )
                    partition = {**partition, "homogenization": rule_name}

                result = runner.run(Scenario(scenario_name), run_config)
                sim_result = SimulationResult(raw=result.result, metrics=dict(result.metrics))
                path = dataset.write(sim_result, partition)
                written.append(path)
                print(f"Wrote {path}")

    return written


if __name__ == "__main__":
    main()
