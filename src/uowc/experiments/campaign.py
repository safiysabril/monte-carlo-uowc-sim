"""Experiment campaigns: parameter matrices, seed ensembles, and the provenance
graph linking related runs.

research-methodology.md's *independent-replication* procedure for comparing
scenarios is explicit about what "same seed" does and does not buy under delta
tracking: common random numbers do not fully correlate Scenario I/II/III (they
desynchronize at the first null collision), so a scenario comparison must be built
from ``R >= 10`` independent replicates - each replicate its own independent seed
set, shared *within* the replicate across scenarios - with the per-replicate
difference fed to a paired statistic (:func:`~uowc.analysis.comparison.paired_scenario_difference`).
A :class:`Campaign` is the orchestration for exactly that procedure, generalized
across an optional parameter sweep.

Everything here is composition over :class:`~uowc.experiments.runner.ScenarioRunner`
and :class:`~uowc.experiments.config.ExperimentConfig` - no new physics, no new
statistics. The one new idea is the provenance link: every produced
:class:`~uowc.experiments.runner.ScenarioResult` is tagged with the parameter point,
replicate index and content hash that produced it
(:func:`~uowc.core.provenance.content_hash`), so a downstream comparison or plot can
always be traced back to exactly which run it came from.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from uowc.analysis.comparison import ScenarioDifference, paired_scenario_difference
from uowc.core import provenance
from uowc.experiments.config import ExperimentConfig
from uowc.experiments.runner import ScenarioResult, ScenarioRunner
from uowc.experiments.scenarios import Scenario

__all__ = [
    "ParameterPoint",
    "ParameterMatrix",
    "replicate_seeds",
    "ProvenanceNode",
    "CampaignResult",
    "Campaign",
]

#: Fields of ExperimentConfig that a parameter sweep or replicate may legally override.
_OVERRIDABLE_FIELDS = frozenset(f.name for f in dataclasses.fields(ExperimentConfig))


@dataclass(frozen=True, slots=True)
class ParameterPoint:
    """One point in a parameter sweep: named top-level overrides applied to a base
    :class:`~uowc.experiments.config.ExperimentConfig` via :func:`dataclasses.replace`.

    Restricted to top-level fields (e.g. ``bounds``, ``seed``) - overriding a field
    nested inside ``source`` or ``receiver`` means constructing a new
    :class:`~uowc.core.geometry.Source`/:class:`~uowc.core.geometry.Receiver` and
    passing that as the override value; this class does not reach inside them.
    Sweeping ``wavelength_nm`` needs the same care: it must also rebuild ``model``
    with a matching wavelength (ExperimentConfig's docstring - the two are required to
    agree), so a bare ``{"wavelength_nm": ...}`` override alone is not enough for a
    wavelength-locked model such as :class:`~uowc.optics.haltrin.HaltrinModel`.
    """

    label: str
    overrides: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.overrides) - _OVERRIDABLE_FIELDS
        if unknown:
            raise ValueError(f"not an ExperimentConfig field: {sorted(unknown)}")

    def apply(self, base: ExperimentConfig) -> ExperimentConfig:
        """The config for this point: ``base`` with :attr:`overrides` applied."""
        # A field-name-keyed override cannot be typed per-field; __post_init__
        # already checked every key is a real ExperimentConfig field, and
        # dataclasses.replace raises TypeError at runtime for a genuinely wrong value.
        return dataclasses.replace(base, **self.overrides)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ParameterMatrix:
    """A Cartesian product of named axis values, expanded into
    :class:`ParameterPoint` instances.

    research-methodology.md's "isolate the variable under investigation" applies here
    too: a matrix with several axes explores their *combination*, which is a
    different (and usually less interpretable) experiment than one axis at a time.
    An empty matrix (the default) is a single, unmodified parameter point.
    """

    axes: Mapping[str, Sequence[object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.axes) - _OVERRIDABLE_FIELDS
        if unknown:
            raise ValueError(f"not an ExperimentConfig field: {sorted(unknown)}")
        for name, values in self.axes.items():
            if len(values) == 0:
                raise ValueError(f"axis {name!r} has no values")

    def points(self) -> tuple[ParameterPoint, ...]:
        """Every combination of axis values, one :class:`ParameterPoint` each."""
        if not self.axes:
            return (ParameterPoint(label="default"),)
        names = sorted(self.axes)
        combinations = itertools.product(*(self.axes[name] for name in names))
        result = []
        for combo in combinations:
            overrides = dict(zip(names, combo, strict=True))
            label = ",".join(f"{name}={value}" for name, value in overrides.items())
            result.append(ParameterPoint(label=label, overrides=overrides))
        return tuple(result)


def replicate_seeds(root_seed: int, n_replicates: int) -> tuple[int, ...]:
    """``n_replicates`` seeds deterministically derived from ``root_seed``.

    Uses ``numpy.random.SeedSequence.spawn``, the same mechanism
    :class:`~uowc.core.rng.NumpyRng` uses for its own stream splitting - so a
    replicate's seed does not depend on how many replicates were requested before it,
    only on its own index (research-methodology.md: "each with its own independent
    seed set").
    """
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1")
    children = np.random.SeedSequence(root_seed).spawn(n_replicates)
    return tuple(int(child.generate_state(1)[0]) for child in children)


@dataclass(frozen=True, slots=True)
class ProvenanceNode:
    """One produced run, tagged with what produced it - the edge from a
    :class:`ScenarioResult` back to the campaign, parameter point and replicate that
    generated it.
    """

    parameter_label: str
    replicate_index: int
    scenario: Scenario
    seed: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """Every :class:`ScenarioResult` produced by a :class:`Campaign` run, indexed by
    ``(parameter_label, replicate_index, scenario)``, plus the provenance node for
    each.
    """

    results: Mapping[tuple[str, int, Scenario], ScenarioResult]
    nodes: tuple[ProvenanceNode, ...]

    def parameter_labels(self) -> tuple[str, ...]:
        """Distinct parameter-point labels present in this result, in first-seen order."""
        seen: dict[str, None] = {}
        for label, _, _ in self.results:
            seen[label] = None
        return tuple(seen)

    def replicate_metric_values(
        self, *, parameter_label: str, scenario: Scenario, metric_name: str
    ) -> tuple[float, ...]:
        """The scalar value of ``metric_name`` for one (parameter point, scenario),
        one entry per replicate, ordered by replicate index - the input
        :func:`~uowc.analysis.comparison.paired_scenario_difference` expects.
        """
        indices = sorted(
            replicate_index
            for (label, replicate_index, sc) in self.results
            if label == parameter_label and sc == scenario
        )
        if not indices:
            raise KeyError(
                f"no results for parameter_label={parameter_label!r}, scenario={scenario!r}"
            )
        values = []
        for replicate_index in indices:
            metric = self.results[(parameter_label, replicate_index, scenario)].metrics[metric_name]
            if not isinstance(metric.value, (int, float)):
                raise TypeError(
                    f"{metric_name!r} is not scalar-valued; cannot form a replicate series"
                )
            values.append(float(metric.value))
        return tuple(values)

    def compare(
        self,
        *,
        parameter_label: str,
        scenario_a: Scenario,
        scenario_b: Scenario,
        metric_name: str,
        confidence: float = 0.95,
    ) -> ScenarioDifference:
        """The paired-replicate scenario difference for ``metric_name`` at one
        parameter point (research-methodology.md's independent-replication
        procedure), computed directly from this campaign's own replicates.
        """
        values_a = self.replicate_metric_values(
            parameter_label=parameter_label, scenario=scenario_a, metric_name=metric_name
        )
        values_b = self.replicate_metric_values(
            parameter_label=parameter_label, scenario=scenario_b, metric_name=metric_name
        )
        return paired_scenario_difference(
            metric_name=metric_name,
            scenario_a=str(scenario_a),
            scenario_b=str(scenario_b),
            values_a=values_a,
            values_b=values_b,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class Campaign:
    """A batch of :class:`ScenarioRunner` executions across a parameter matrix and a
    replicate ensemble.

    Every scenario at a given (parameter point, replicate) shares that replicate's
    seed - "all scenarios sharing the seed set within a replicate"
    (research-methodology.md) - while different replicates get independent seeds via
    :func:`replicate_seeds`. ``n_replicates`` has a hard floor of 2 (the minimum
    :func:`~uowc.analysis.comparison.paired_scenario_difference` can compute a
    variance from); research-methodology.md's *recommended* floor of 10 is checked
    downstream by
    :attr:`~uowc.analysis.comparison.ScenarioDifference.meets_recommended_replicate_count`,
    not enforced here, so a deliberately small pilot campaign is not blocked.
    """

    base_config: ExperimentConfig
    runner: ScenarioRunner
    scenarios: tuple[Scenario, ...] = tuple(Scenario)
    parameter_matrix: ParameterMatrix = field(default_factory=ParameterMatrix)
    n_replicates: int = 10
    root_seed: int = 20260607

    def __post_init__(self) -> None:
        if self.n_replicates < 2:
            raise ValueError(
                "n_replicates must be at least 2 (a paired difference needs a variance)"
            )
        if not self.scenarios:
            raise ValueError("at least one scenario is required")

    def run(self) -> CampaignResult:
        """Execute every (parameter point, replicate, scenario) combination."""
        seeds = replicate_seeds(self.root_seed, self.n_replicates)
        results: dict[tuple[str, int, Scenario], ScenarioResult] = {}
        nodes: list[ProvenanceNode] = []

        for point in self.parameter_matrix.points():
            point_config = point.apply(self.base_config)
            for replicate_index, seed in enumerate(seeds):
                run_config = dataclasses.replace(point_config, seed=seed)
                for scenario in self.scenarios:
                    result = self.runner.run(scenario, run_config)
                    key = (point.label, replicate_index, scenario)
                    results[key] = result
                    nodes.append(
                        ProvenanceNode(
                            parameter_label=point.label,
                            replicate_index=replicate_index,
                            scenario=scenario,
                            seed=seed,
                            content_hash=provenance.content_hash(
                                dict(point.overrides) | {"seed": seed}
                            ),
                        )
                    )

        return CampaignResult(results=results, nodes=tuple(nodes))
