"""Scenario execution: shared transport engine + shared metrics pipeline.

A single :class:`~uowc.transport.WoodcockDeltaTracker` and a single
:class:`~uowc.metrics.MetricPipeline` are reused for every scenario, and each scenario
is run with the same seed (common random numbers). Differences between scenario
results therefore originate only from the medium representation and the environmental
effects - never from transport or metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from uowc.core import RawResult, RunMetadata, provenance
from uowc.core.ports import TransportEngine
from uowc.core.results import MetricValue
from uowc.core.rng import NumpyRng
from uowc.experiments.config import ExperimentConfig
from uowc.experiments.scenarios import Scenario, build_medium, effect_names, medium_type
from uowc.metrics import MetricPipeline
from uowc.transport import WoodcockDeltaTracker

__all__ = ["ScenarioResult", "ScenarioRunner"]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """A scenario's raw transport result plus its computed metrics."""

    scenario: Scenario
    result: RawResult
    metrics: Mapping[str, MetricValue]


@dataclass(frozen=True, slots=True)
class ScenarioRunner:
    """Runs scenarios with a shared engine and a shared metrics pipeline."""

    engine: TransportEngine
    pipeline: MetricPipeline

    @classmethod
    def default(cls) -> ScenarioRunner:
        """Runner with the Woodcock engine and the default metrics pipeline."""
        return cls(engine=WoodcockDeltaTracker(), pipeline=MetricPipeline.default())

    def run(self, scenario: Scenario, config: ExperimentConfig) -> ScenarioResult:
        """Execute a single scenario."""
        medium = build_medium(scenario, config)
        rng = NumpyRng(config.seed)
        output = self.engine.run(medium, config.source, config.receiver, rng, config.sampling)
        raw = RawResult(output=output, metadata=self._metadata(scenario, config, rng))
        return ScenarioResult(scenario=scenario, result=raw, metrics=self.pipeline.run(raw))

    def run_selected(
        self, scenarios: Sequence[Scenario], config: ExperimentConfig
    ) -> dict[Scenario, ScenarioResult]:
        """Execute a chosen set of scenarios."""
        return {scenario: self.run(scenario, config) for scenario in scenarios}

    def run_all(self, config: ExperimentConfig) -> dict[Scenario, ScenarioResult]:
        """Execute all three scenarios."""
        return self.run_selected(tuple(Scenario), config)

    def _metadata(self, scenario: Scenario, config: ExperimentConfig, rng: NumpyRng) -> RunMetadata:
        parameters: dict[str, float | int | str | bool] = {"seed": config.seed}
        if scenario is Scenario.I:
            parameters["homogenization"] = config.homogenization.name
        # "unknown" is ExperimentConfig's unset sentinel: auto-detect from git unless
        # the caller pinned an explicit version (e.g. a frozen release tag).
        code_version = config.code_version
        if code_version == "unknown":
            code_version = provenance.code_version()
        return RunMetadata(
            scenario=scenario.value,
            medium_type=medium_type(scenario),
            optical_model=config.model.name,
            effects=effect_names(scenario, config),
            wavelength_nm=config.wavelength_nm,
            sampling=config.sampling,
            seed_tree=rng.seed_tree(),
            rng_impl=type(rng).__name__,
            timestamp_utc=provenance.utc_timestamp(),
            code_version=code_version,
            library_versions=provenance.library_versions(),
            parameters=parameters,
        )
