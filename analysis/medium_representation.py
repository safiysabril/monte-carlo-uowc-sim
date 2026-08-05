"""Analysis for the medium-representation study.

Consumes the Parquet dataset ``experiments/medium_representation/run.py`` writes
under ``data/raw`` (Simulation -> Parquet -> Analysis -> Visualization; data.md) and
reports the study's two headline quantities:

* **Homogenization spread** (research-methodology.md: "The spread across rules is
  itself a headline result: it bounds how much of the reported 'difference' is
  physics and how much is bookkeeping.") - the received-power-fraction range across
  every implemented Scenario I homogenization rule, at each (geometry, water_type,
  range_m) point that has one.
* **Scenario I vs II difference** - the paired-replicate procedure
  (research-methodology.md) where enough seeds are present to run it (>= 2), and a
  clearly-labelled raw point difference (not a fabricated interval) where only one
  seed is available, e.g. after ``run.py``'s quick smoke mode.

Never re-runs a simulation; every number here comes from what is already on disk.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from uowc.analysis.comparison import paired_scenario_difference
from uowc.analysis.uncertainty import spread_across_alternatives
from uowc.persistence.dataset import PartitionedResultDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_METRIC_NAME = "received_power_fraction"


def _combo_key(partition: dict[str, str]) -> tuple[str, str, str]:
    return (partition["geometry"], partition["water_type"], partition["range_m"])


def _power(dataset: PartitionedResultDataset, partition: dict[str, str]) -> float:
    result = dataset.read(partition)
    return float(result.metrics[_METRIC_NAME].value)


def homogenization_spread_report(
    dataset: PartitionedResultDataset,
) -> dict[tuple[str, str, str], float]:
    """Received-power-fraction spread across homogenization rules, per
    (geometry, water_type, range_m) point that has more than one rule on disk."""
    scenario_i = [p for p in dataset.find(scenario="I") if "homogenization" in p]
    by_combo: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for partition in scenario_i:
        by_combo[_combo_key(partition)].append(_power(dataset, partition))

    spreads = {}
    for combo, values in by_combo.items():
        if len(values) >= 2:
            spreads[combo] = spread_across_alternatives(values)
    return spreads


def scenario_i_vs_ii_report(
    dataset: PartitionedResultDataset, *, reference_rule: str = "optical_depth"
):
    """Scenario I (at ``reference_rule``) vs Scenario II, per
    (geometry, water_type, range_m) point present for both.

    Returns ``(paired, raw)``: ``paired`` holds a proper
    :class:`~uowc.analysis.comparison.ScenarioDifference` for combos with >= 2 seeds;
    ``raw`` holds a plain point difference (no interval - not enough replicates to
    compute one) for combos with exactly one.
    """
    scenario_i = [
        p for p in dataset.find(scenario="I") if p.get("homogenization") == reference_rule
    ]
    scenario_ii = dataset.find(scenario="II")

    values_i: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    values_ii: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for partition in scenario_i:
        values_i[_combo_key(partition)][partition["seed"]] = _power(dataset, partition)
    for partition in scenario_ii:
        values_ii[_combo_key(partition)][partition["seed"]] = _power(dataset, partition)

    paired = {}
    raw = {}
    for combo, i_by_seed in values_i.items():
        ii_by_seed = values_ii.get(combo)
        if not ii_by_seed:
            continue
        shared_seeds = sorted(set(i_by_seed) & set(ii_by_seed))
        if len(shared_seeds) >= 2:
            paired[combo] = paired_scenario_difference(
                metric_name=_METRIC_NAME,
                scenario_a="I",
                scenario_b="II",
                values_a=[i_by_seed[s] for s in shared_seeds],
                values_b=[ii_by_seed[s] for s in shared_seeds],
            )
        elif len(shared_seeds) == 1:
            s = shared_seeds[0]
            raw[combo] = i_by_seed[s] - ii_by_seed[s]
    return paired, raw


def main(data_root: Path | None = None) -> None:
    dataset = PartitionedResultDataset(
        root=Path(data_root) if data_root else _REPO_ROOT / "data" / "raw"
    )
    if not dataset.list_partitions():
        print(f"No results found under {dataset.root}")
        print("Run experiments/medium_representation/run.py first.")
        return

    print("=== Homogenization spread (Scenario I, across implemented rules) ===")
    spreads = homogenization_spread_report(dataset)
    if not spreads:
        print("(need >= 2 homogenization rules on disk for the same geometry/water_type/range_m)")
    for (geometry, water_type, range_m), spread in sorted(spreads.items()):
        print(f"  {geometry:<12}{water_type:<14}range={range_m:<8}spread={spread:.6g}")

    print("\n=== Scenario I (optical_depth) vs Scenario II ===")
    paired, raw = scenario_i_vs_ii_report(dataset)
    for (geometry, water_type, range_m), diff in sorted(paired.items()):
        print(
            f"  {geometry:<12}{water_type:<14}range={range_m:<8}"
            f"diff={diff.mean_difference:.6g} CI=[{diff.ci95_low:.6g}, {diff.ci95_high:.6g}] "
            f"(n={diff.n_replicates})"
        )
    for (geometry, water_type, range_m), delta in sorted(raw.items()):
        print(
            f"  {geometry:<12}{water_type:<14}range={range_m:<8}"
            f"diff={delta:.6g} (single seed - no interval)"
        )
    if not paired and not raw:
        print("(no matching Scenario I/II pair on disk)")


if __name__ == "__main__":
    main()
