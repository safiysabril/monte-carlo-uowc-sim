"""Uncertainty decomposition: Monte Carlo vs model vs environmental vs measurement.

research-methodology.md, "Uncertainty Quantification":

    Distinguish, and never sum as though equivalent:
    * Monte Carlo variance - reducible by more photons.
    * Model uncertainty - phase function, Case-1 applicability, placeholder
      coefficients. Typically larger than Monte Carlo variance and not reducible by
      more photons. Bound it by re-running under alternative model choices.
    * Environmental uncertainty - the profile itself is uncertain.
    * Measurement uncertainty - in any reference data used for validation.

    A tight Monte Carlo interval around a result computed from placeholder
    coefficients is precision without accuracy. Report both, and do not let the
    former imply the latter.

Why model/environmental uncertainty is a *range*, not a standard deviation
--------------------------------------------------------------------------
Monte Carlo variance shrinks as more photons are launched, because more photons are
independent samples from a well-defined distribution. Re-running under a different
phase function, a different homogenization rule, or a different chlorophyll profile is
not that: each alternative is one deliberate, named choice, not a draw from a
probability distribution over "the correct model." There is no law of large numbers to
invoke, and computing a standard deviation across a handful of such alternatives would
misleadingly imply the spread shrinks as more alternatives are tried - it does not.
The honest statistic is the **range** across the alternatives actually run: a bound,
not a variance estimate.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uowc.core.results import MetricValue

__all__ = ["UncertaintyBudget", "spread_across_alternatives", "build_uncertainty_budget"]


def spread_across_alternatives(values: Sequence[float]) -> float:
    """Range (max - min) across a small, non-random set of alternative-assumption
    runs (module docstring). Requires at least 2 alternatives to have a range."""
    if len(values) < 2:
        raise ValueError("at least 2 alternative-assumption runs are required to bound a spread")
    return float(max(values) - min(values))


@dataclass(frozen=True, slots=True)
class UncertaintyBudget:
    """A decomposition of total uncertainty into its distinct, non-summable sources.

    Any component may be ``None`` if it was not assessed for this result - a reduced
    budget that says so is honest; treating an unassessed source as zero is not.
    """

    metric_name: str
    monte_carlo_std: float | None
    model_spread: float | None
    environmental_spread: float | None
    measurement_std: float | None

    @property
    def dominant_source(self) -> str | None:
        """Name of the largest-magnitude *assessed* component.

        Monte Carlo error is usually not the binding constraint (module docstring),
        so this is often a different answer than "how many photons were launched"
        would suggest - that is exactly why it is surfaced explicitly rather than
        left for a reader to infer from a single reported number.
        """
        candidates = {
            "monte_carlo": self.monte_carlo_std,
            "model": self.model_spread,
            "environmental": self.environmental_spread,
            "measurement": self.measurement_std,
        }
        assessed = {name: value for name, value in candidates.items() if value is not None}
        if not assessed:
            return None
        return max(assessed, key=lambda name: assessed[name])

    @property
    def monte_carlo_dominates_model_uncertainty(self) -> bool | None:
        """Whether the Monte Carlo interval is narrower than the model-choice spread
        - the "precision without accuracy" trap the module docstring warns against.

        ``None`` when not assessable (model uncertainty was never bounded), and a
        caller must not read ``None`` as "no trap": an unassessed model uncertainty
        cannot be ruled safe, and is itself the finding to report.
        """
        if self.monte_carlo_std is None or self.model_spread is None:
            return None
        return self.monte_carlo_std < self.model_spread


def build_uncertainty_budget(
    *,
    metric_name: str,
    monte_carlo: MetricValue | None = None,
    model_alternatives: Sequence[float] | None = None,
    environmental_alternatives: Sequence[float] | None = None,
    measurement_std: float | None = None,
) -> UncertaintyBudget:
    """Assemble an :class:`UncertaintyBudget` from whichever sources were assessed.

    ``monte_carlo`` is the :class:`~uowc.core.results.MetricValue` computed from the
    production run (its ``std`` is taken directly - see metrics.md for how it was
    derived per metric type). ``model_alternatives``/``environmental_alternatives``
    are point estimates of the same metric from re-runs under different model choices
    or environmental realizations (module docstring); their spread, not their
    standard deviation, is reported. ``measurement_std`` is a fixed value taken from a
    cited source's stated measurement precision (e.g. Pope & Fry 1997's
    +/- 0.0006 m^-1), not computed here.
    """
    monte_carlo_std: float | None = None
    if monte_carlo is not None:
        if monte_carlo.std is None or not isinstance(monte_carlo.std, (int, float)):
            raise ValueError(f"{metric_name!r}: monte_carlo MetricValue must carry a scalar std")
        monte_carlo_std = float(monte_carlo.std)

    model_spread = (
        spread_across_alternatives(model_alternatives) if model_alternatives is not None else None
    )
    environmental_spread = (
        spread_across_alternatives(environmental_alternatives)
        if environmental_alternatives is not None
        else None
    )

    if measurement_std is not None and measurement_std < 0.0:
        raise ValueError("measurement_std must be non-negative")

    return UncertaintyBudget(
        metric_name=metric_name,
        monte_carlo_std=monte_carlo_std,
        model_spread=model_spread,
        environmental_spread=environmental_spread,
        measurement_std=measurement_std,
    )
