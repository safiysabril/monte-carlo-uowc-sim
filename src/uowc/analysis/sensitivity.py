"""Sensitivity analysis: one-at-a-time (OFAT) parameter sweeps; influence ranking.

research-methodology.md, "Sensitivity Analysis":

    Sensitivity studies should identify which parameters most influence channel
    behavior... Unless studying interactions, vary one parameter at a time.

Vary model inputs, not model outputs
-------------------------------------
``a`` and ``b`` are **not** independent parameters under a chlorophyll-parameterized
model - both are derived from the same environmental state by the same model.
Sweeping ``a`` while holding ``b`` fixed describes no water the model can produce, and
breaks the causal chain in the scientific hierarchy (scientific-modelling.md).
Sensitivity must be run over model **inputs** (chlorophyll, CDOM, NAP, wavelength,
geometry, depth, effect strength, homogenization rule) - direct sweeps of a derived
IOP are permitted only as an explicitly labelled, model-free "what-if" probe, and this
module refuses to compute one silently: sweeping a name it recognizes as a derived
coefficient requires an explicit opt-in.

OFAT is a screening tool, not a complete design
--------------------------------------------------
One-at-a-time sweeps explore a cross-shaped slice through parameter space and cannot
detect interactions between parameters. That is an accepted limitation for screening,
not a defect in this module - but it means a :class:`SensitivityResult` here says
nothing about interaction effects, and research-methodology.md is explicit that a
factorial or variance-based (Sobol) design is needed where interactions are suspected.
This module implements OFAT only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uowc.core.results import MetricValue

__all__ = ["SensitivityResult", "compute_sensitivity", "rank_by_influence"]

#: Names this module recognizes as model-*derived* IOPs rather than independent
#: inputs (research-methodology.md). Not exhaustive - a caller sweeping a
#: differently-named derived quantity is still responsible for judging whether it is
#: an input or an output of their model.
_DERIVED_COEFFICIENT_NAMES = frozenset(
    {
        "a",
        "b",
        "c",
        "absorption",
        "scattering",
        "attenuation",
        "scattering_coefficient",
        "single_scattering_albedo",
        "omega_0",
        "omega0",
    }
)


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """OFAT sensitivity of one metric to one input parameter, between a baseline and
    one perturbed value.

    ``elasticity`` is the normalized sensitivity coefficient
    ``(dY/Y0) / (dX/X0)`` - the percent change in the metric per percent change in the
    parameter. It is dimensionless, so parameters in unrelated units (chlorophyll in
    mg/m^3, field of view in radians, aperture in metres, ...) can be ranked against
    each other directly by ``|elasticity|`` (:func:`rank_by_influence`).
    """

    metric_name: str
    parameter_name: str
    baseline_value: float
    baseline_metric: float
    perturbed_value: float
    perturbed_metric: float
    is_model_free_probe: bool

    @property
    def relative_input_change(self) -> float:
        return (self.perturbed_value - self.baseline_value) / self.baseline_value

    @property
    def relative_output_change(self) -> float:
        return (self.perturbed_metric - self.baseline_metric) / self.baseline_metric

    @property
    def elasticity(self) -> float:
        return self.relative_output_change / self.relative_input_change


def _scalar(metric: MetricValue) -> float:
    if not isinstance(metric.value, (int, float)):
        raise TypeError(
            f"{metric.name!r}: sensitivity analysis requires a scalar metric value, "
            f"got {type(metric.value).__name__}"
        )
    return float(metric.value)


def compute_sensitivity(
    *,
    metric_name: str,
    parameter_name: str,
    baseline_value: float,
    baseline_metric: float | MetricValue,
    perturbed_value: float,
    perturbed_metric: float | MetricValue,
    allow_derived_coefficient: bool = False,
) -> SensitivityResult:
    """Elasticity of ``metric_name`` with respect to ``parameter_name`` between a
    baseline run and one perturbed run (module docstring).

    ``baseline_metric``/``perturbed_metric`` accept either a raw scalar or a
    :class:`~uowc.core.results.MetricValue` (its ``.value`` is used; an array-valued
    metric is rejected). Raises if ``parameter_name`` names a model-derived
    coefficient (see ``_DERIVED_COEFFICIENT_NAMES``) unless
    ``allow_derived_coefficient=True``, and if either baseline is zero (elasticity, a
    ratio of percent changes, is undefined at a zero baseline) or the two parameter
    values are equal (no variation to attribute a sensitivity to).
    """
    is_model_free_probe = parameter_name.lower() in _DERIVED_COEFFICIENT_NAMES
    if is_model_free_probe and not allow_derived_coefficient:
        raise ValueError(
            f"{parameter_name!r} is a model-derived IOP, not an independent input "
            "(research-methodology.md: 'vary model inputs, not model outputs' - under "
            "a chlorophyll-parameterized model, a and b are both derived from the same "
            "environmental state, so varying one while holding the other fixed "
            "describes no water the model can produce). Pass "
            "allow_derived_coefficient=True to run this as an explicitly labelled, "
            "model-free what-if probe, and report it as such."
        )

    baseline_metric_value = (
        _scalar(baseline_metric)
        if isinstance(baseline_metric, MetricValue)
        else float(baseline_metric)
    )
    perturbed_metric_value = (
        _scalar(perturbed_metric)
        if isinstance(perturbed_metric, MetricValue)
        else float(perturbed_metric)
    )

    if baseline_value == 0.0:
        raise ZeroDivisionError(
            f"{parameter_name!r}: baseline_value is 0; elasticity is undefined at a zero baseline"
        )
    if baseline_metric_value == 0.0:
        raise ZeroDivisionError(
            f"{metric_name!r}: baseline_metric is 0; elasticity is undefined at a zero baseline"
        )
    if perturbed_value == baseline_value:
        raise ValueError(f"{parameter_name!r}: perturbed_value must differ from baseline_value")

    return SensitivityResult(
        metric_name=metric_name,
        parameter_name=parameter_name,
        baseline_value=baseline_value,
        baseline_metric=baseline_metric_value,
        perturbed_value=perturbed_value,
        perturbed_metric=perturbed_metric_value,
        is_model_free_probe=is_model_free_probe,
    )


def rank_by_influence(results: Sequence[SensitivityResult]) -> tuple[SensitivityResult, ...]:
    """``results`` sorted by ``|elasticity|`` descending - most influential first
    (research-methodology.md: "identify which parameters most influence channel
    behavior"). Comparing across ``metric_name`` values is meaningless; callers should
    rank one metric's sensitivities at a time.
    """
    return tuple(sorted(results, key=lambda r: abs(r.elasticity), reverse=True))
