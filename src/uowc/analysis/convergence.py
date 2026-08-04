"""Convergence studies: is the standard error falling as N^-1/2, not just "is the
estimate stable" (research-methodology.md, "Convergence Studies").

Stability of a point estimate is not evidence of convergence: a biased estimator is
perfectly stable, and in the rare-event regime this channel operates in, a metric can
sit flat at zero simply because nothing has been detected yet. The only criterion that
actually distinguishes convergence from a stalled estimator is the **error**, not the
estimate:

* the standard error must fall as ``N^-1/2`` - equivalently, ``SE * sqrt(N)`` must be
  flat against ``N`` on a log-log plot; a *rising* trend (slope of ``log(SE)`` vs
  ``log(N)`` shallower than -0.5) indicates a heavy-tailed estimator that has not
  reached its asymptotic regime, and a *falling* trend (steeper than -0.5) indicates a
  bug, since Monte Carlo error cannot fall faster than the central-limit rate;
* successive estimates must lie within each other's confidence intervals.

Convergence is assessed **per metric** (research-methodology.md's table of relative
convergence rates: received power converges fastest, RMS delay spread and 3 dB
bandwidth much slower) - never once for the cheapest metric and assumed to hold for
the rest.

This module fits the log-log slope and checks interval overlap; it does not recompute
uncertainty from scratch. Each point's confidence interval is taken directly from the
:class:`~uowc.core.results.MetricValue` that produced it (Wilson for a proportion,
weighted-mean SE for others - see metrics.md), so the check stays valid regardless of
which estimator built it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats

from uowc.core.results import MetricValue

__all__ = ["ConvergencePoint", "ConvergenceReport", "assess_convergence"]

#: Central-limit-theorem rate: Monte Carlo standard error falls as N^-1/2.
_EXPECTED_SLOPE = -0.5


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    """One point of a photon-count sweep for a single metric."""

    n_photons: int
    estimate: float
    standard_error: float

    @property
    def normalized_standard_error(self) -> float:
        """``SE * sqrt(N)``; research-methodology.md's flatness diagnostic."""
        return float(self.standard_error * np.sqrt(self.n_photons))


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """Convergence assessment for one metric across an increasing-N photon sweep."""

    metric_name: str
    points: tuple[ConvergencePoint, ...]
    fitted_slope: float
    fitted_slope_stderr: float
    slope_tolerance: float
    consecutive_overlaps: tuple[bool, ...]

    @property
    def slope_within_tolerance(self) -> bool:
        """Whether the fitted ``log(SE)`` vs ``log(N)`` slope is close enough to the
        CLT rate of -0.5 (within ``slope_tolerance``) to call this Monte Carlo error
        rather than something else (bias, a heavy tail, a bug)."""
        return abs(self.fitted_slope - _EXPECTED_SLOPE) <= self.slope_tolerance

    @property
    def all_consecutive_estimates_overlap(self) -> bool:
        return all(self.consecutive_overlaps)

    @property
    def converged(self) -> bool:
        """Both criteria from research-methodology.md hold: the error is falling at
        the expected rate, and successive estimates are consistent with each other."""
        return self.slope_within_tolerance and self.all_consecutive_estimates_overlap


def _intervals_overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return low_a <= high_b and low_b <= high_a


def assess_convergence(
    *,
    metric_name: str,
    n_photons: Sequence[int],
    values: Sequence[MetricValue],
    slope_tolerance: float = 0.15,
) -> ConvergenceReport:
    """Assess convergence of one metric across a photon-count sweep.

    ``n_photons[i]`` is the *launched* count for the run that produced ``values[i]``
    (``RunMetadata.sampling.n_photons``, not ``MetricValue.n_samples`` - the latter is
    the detected/effective sample count the metric happened to have, a different and
    usually much smaller number). Every ``MetricValue`` must carry ``std``,
    ``ci95_low`` and ``ci95_high`` - a convergence claim with no uncertainty attached
    is not a convergence claim.

    Requires at least 3 distinct, positive photon counts to fit a trend.
    """
    if len(n_photons) != len(values):
        raise ValueError("n_photons and values must have the same length")
    if len(n_photons) < 3:
        raise ValueError("at least 3 photon counts are required to fit a convergence trend")
    if len(set(n_photons)) != len(n_photons):
        raise ValueError("n_photons must be distinct (one run per photon count)")
    if any(n <= 0 for n in n_photons):
        raise ValueError("n_photons must be positive")

    order = [int(i) for i in np.argsort(np.asarray(n_photons, dtype=np.int64))]
    n_sorted = [int(n_photons[i]) for i in order]
    values_sorted = [values[i] for i in order]

    estimates: list[float] = []
    stds: list[float] = []
    ci_los: list[float] = []
    ci_his: list[float] = []
    for mv in values_sorted:
        if not isinstance(mv.value, (int, float)):
            raise TypeError(f"{metric_name!r}: convergence check requires scalar metric values")
        if mv.std is None or not isinstance(mv.std, (int, float)):
            raise ValueError(
                f"{metric_name!r}: MetricValue.std is required for a convergence check"
            )
        if mv.std <= 0.0:
            raise ValueError(
                f"{metric_name!r}: standard error must be positive to fit a log-log slope "
                f"(got {mv.std!r}; a metric with std == 0, e.g. from a run with zero "
                "detections, cannot be placed on a log-log convergence plot)"
            )
        if (
            mv.ci95_low is None
            or mv.ci95_high is None
            or not isinstance(mv.ci95_low, (int, float))
            or not isinstance(mv.ci95_high, (int, float))
        ):
            raise ValueError(
                f"{metric_name!r}: scalar ci95_low/ci95_high are required for a convergence check"
            )
        estimates.append(float(mv.value))
        stds.append(float(mv.std))
        ci_los.append(float(mv.ci95_low))
        ci_his.append(float(mv.ci95_high))

    log_n = np.log(np.asarray(n_sorted, dtype=np.float64))
    log_se = np.log(np.asarray(stds, dtype=np.float64))
    regression = scipy.stats.linregress(log_n, log_se)

    overlaps: list[bool] = [
        _intervals_overlap(ci_los[i], ci_his[i], ci_los[i + 1], ci_his[i + 1])
        for i in range(len(values_sorted) - 1)
    ]

    points = tuple(
        ConvergencePoint(n_photons=n, estimate=e, standard_error=s)
        for n, e, s in zip(n_sorted, estimates, stds, strict=True)
    )

    return ConvergenceReport(
        metric_name=metric_name,
        points=points,
        fitted_slope=float(regression.slope),
        fitted_slope_stderr=float(regression.stderr),
        slope_tolerance=slope_tolerance,
        consecutive_overlaps=tuple(overlaps),
    )
