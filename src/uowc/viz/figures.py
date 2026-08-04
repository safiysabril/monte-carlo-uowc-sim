"""Figure builders: power vs a swept parameter, CIR, frequency response, scenario
comparisons, convergence - reproducible from stored Parquet outputs (visualization.md).

Each builder takes already-computed values (a :class:`~uowc.core.results.MetricValue`,
a :class:`~uowc.analysis.comparison.ScenarioDifference`, a
:class:`~uowc.analysis.convergence.ConvergenceReport`, or raw detected-photon arrays)
and an optional matplotlib ``Axes`` to draw on; none of them run a simulation, compute
a metric, or fit a model - visualization.md: "Visualization must not control
transport, generate optical coefficients, or modify simulation outputs." Every
builder returns the ``Axes`` so the caller decides layout, saving and display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from uowc.core.results import MetricValue
from uowc.viz.style import axis_label, bandwidth_label, scenario_color

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from uowc.analysis.comparison import ScenarioDifference
    from uowc.analysis.convergence import ConvergenceReport

__all__ = [
    "plot_metric_vs_parameter",
    "plot_cir",
    "plot_frequency_response",
    "plot_scenario_comparison",
    "plot_convergence",
]


def _scalar_value(metric: MetricValue) -> float:
    if not isinstance(metric.value, (int, float)):
        raise TypeError(f"{metric.name!r}: this figure requires a scalar metric value")
    return float(metric.value)


def _error_bounds(metric: MetricValue) -> tuple[float, float]:
    """(lower_error, upper_error) for an errorbar: from the CI if present, else from
    ``std``, else ``(0, 0)`` - a point with no assessed uncertainty is drawn without a
    fabricated error bar, not with one invented to avoid looking bare."""
    value = _scalar_value(metric)
    if metric.ci95_low is not None and metric.ci95_high is not None:
        return value - float(metric.ci95_low), float(metric.ci95_high) - value
    if metric.std is not None and isinstance(metric.std, (int, float)):
        return float(metric.std), float(metric.std)
    return 0.0, 0.0


def plot_metric_vs_parameter(
    *,
    parameter_values: Sequence[float],
    metric_values: Sequence[MetricValue],
    parameter_name: str,
    parameter_unit: str,
    scenario: str | None = None,
    log_y: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """A metric (e.g. received power fraction) against a swept parameter (range,
    depth, ...), with its own confidence interval as error bars.

    A zero-valued point is drawn as a downward-pointing upper-limit marker at
    ``ci95_high``, never as a point sitting at exactly zero (metrics.md: a
    non-detection is an upper bound, not a measured zero) - and never silently on a
    log axis, where zero cannot even be placed.
    """
    import matplotlib.pyplot as plt

    if len(parameter_values) != len(metric_values):
        raise ValueError("parameter_values and metric_values must have the same length")
    if not metric_values:
        raise ValueError("at least one point is required")
    if ax is None:
        _, ax = plt.subplots()

    x = np.asarray(parameter_values, dtype=np.float64)
    values = np.array([_scalar_value(mv) for mv in metric_values])
    lower = np.array([_error_bounds(mv)[0] for mv in metric_values])
    upper = np.array([_error_bounds(mv)[1] for mv in metric_values])
    color = scenario_color(scenario) if scenario is not None else None

    detected = values > 0.0
    if np.any(detected):
        ax.errorbar(
            x[detected],
            values[detected],
            yerr=[lower[detected], upper[detected]],
            fmt="o-",
            color=color,
            label=scenario,
            capsize=3,
        )
    if np.any(~detected):
        bounds = []
        for mv in (metric_values[int(i)] for i in np.flatnonzero(~detected)):
            if mv.ci95_high is None:
                raise ValueError(
                    f"{mv.name!r}: a zero-valued point with no ci95_high cannot be "
                    "plotted as an upper bound"
                )
            bounds.append(float(mv.ci95_high))
        bounds_arr = np.asarray(bounds)
        ax.errorbar(
            x[~detected],
            bounds_arr,
            yerr=bounds_arr * 0.25,
            uplims=True,
            fmt="none",
            color=color or "black",
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(axis_label(parameter_name, parameter_unit))
    metric_display_name = metric_values[0].name.replace("_", " ").capitalize()
    ax.set_ylabel(axis_label(metric_display_name, metric_values[0].unit))
    if scenario is not None:
        ax.legend()
    return ax


def plot_cir(
    *,
    arrival_time_s: Sequence[float],
    weight: Sequence[float],
    t0_s: float,
    n_bins: int = 200,
    scenario: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Channel impulse response: log-intensity vs excess delay ``t - t0_s``
    (visualization.md).

    This histogram is for **display only**. It must never be reused to compute a
    frequency response or bandwidth - binning before transforming multiplies the true
    ``H(f)`` by ``sinc(f * bin_width)``, a bias comparable in size to the 3 dB point
    itself (metrics.md). Frequency-domain metrics are computed directly from
    unbinned arrival times.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    times = np.asarray(arrival_time_s, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    excess_delay_s = times - t0_s
    counts, edges = np.histogram(excess_delay_s, bins=n_bins, weights=weights)
    centers_ns = 0.5 * (edges[:-1] + edges[1:]) * 1e9

    positive = counts > 0.0
    color = scenario_color(scenario) if scenario is not None else None
    ax.semilogy(centers_ns[positive], counts[positive], color=color, label=scenario)
    ax.set_xlabel(axis_label("Excess delay", "ns"))
    ax.set_ylabel(axis_label("Received weight per bin", "dimensionless"))
    if scenario is not None:
        ax.legend()
    return ax


def plot_frequency_response(
    *,
    frequencies_hz: Sequence[float],
    response: Sequence[complex],
    bandwidth_hz: float | None = None,
    convention: str = "electrical",
    scenario: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """``|H(f)|`` against frequency, with the 3 dB crossing marked if
    ``bandwidth_hz`` is given (visualization.md: "3 dB crossing marked")."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    freqs_mhz = np.asarray(frequencies_hz, dtype=np.float64) / 1.0e6
    magnitude = np.abs(np.asarray(response, dtype=np.complex128))
    color = scenario_color(scenario) if scenario is not None else None

    ax.plot(freqs_mhz, magnitude, color=color, label=scenario)
    if bandwidth_hz is not None:
        line_color = color or "black"
        ax.axvline(bandwidth_hz / 1.0e6, linestyle="--", color=line_color, linewidth=1.0)
        ax.annotate(
            bandwidth_label(convention),
            xy=(bandwidth_hz / 1.0e6, 0.5),
            rotation=90,
            va="center",
            fontsize=8,
            color=line_color,
        )
    ax.set_xlabel(axis_label("Frequency", "MHz"))
    ax.set_ylabel("|H(f)| (dimensionless)")
    if scenario is not None:
        ax.legend()
    return ax


def plot_scenario_comparison(
    differences: Mapping[str, ScenarioDifference], *, ax: Axes | None = None
) -> Axes:
    """One row per metric: the paired-replicate scenario difference and its
    confidence interval, with a zero-reference line.

    visualization.md: "plotting the difference directly, with its paired interval and
    a zero reference line, is usually clearer than overlaying two curves and inviting
    the reader to subtract by eye" - this is that plot, not a substitute for it.
    """
    import matplotlib.pyplot as plt

    if not differences:
        raise ValueError("at least one ScenarioDifference is required")
    if ax is None:
        _, ax = plt.subplots()

    names = list(differences)
    y_positions = np.arange(len(names))
    means = np.array([differences[name].mean_difference for name in names])
    lower = means - np.array([differences[name].ci95_low for name in names])
    upper = np.array([differences[name].ci95_high for name in names]) - means

    ax.errorbar(means, y_positions, xerr=[lower, upper], fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    first = differences[names[0]]
    ax.set_xlabel(axis_label(f"{first.scenario_a} - {first.scenario_b}", ""))
    return ax


def plot_convergence(report: ConvergenceReport, *, ax: Axes | None = None) -> Axes:
    """Standard error vs launched photon count on log-log axes, with the fitted slope
    and the Monte Carlo (N^-1/2) reference slope both drawn for direct visual
    comparison (visualization.md; research-methodology.md's convergence criterion).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    n_photons = np.array([point.n_photons for point in report.points], dtype=np.float64)
    standard_errors = np.array([point.standard_error for point in report.points], dtype=np.float64)
    log_n = np.log(n_photons)

    ax.loglog(n_photons, standard_errors, "o-", color="#0072B2", label="Standard error")

    fitted = np.exp(np.log(standard_errors[0]) + report.fitted_slope * (log_n - log_n[0]))
    ax.loglog(
        n_photons,
        fitted,
        "--",
        color="#0072B2",
        alpha=0.5,
        label=f"Fitted slope ({report.fitted_slope:.2f})",
    )

    reference = np.exp(np.log(standard_errors[0]) - 0.5 * (log_n - log_n[0]))
    ax.loglog(n_photons, reference, ":", color="gray", label="N⁻¹ᐟ² (Monte Carlo)")

    ax.set_xlabel(axis_label("Launched photon count", "count"))
    metric_display_name = report.metric_name.replace("_", " ").capitalize()
    ax.set_ylabel(axis_label(f"{metric_display_name} standard error", ""))
    ax.legend()
    return ax
