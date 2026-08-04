"""Figure builders draw the expected data without touching simulation/analysis logic."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from uowc.analysis.comparison import paired_scenario_difference
from uowc.analysis.convergence import assess_convergence
from uowc.core.results import MetricValue
from uowc.viz.figures import (
    plot_cir,
    plot_convergence,
    plot_frequency_response,
    plot_metric_vs_parameter,
    plot_scenario_comparison,
)


def _mv(value: float, *, ci_low: float | None = None, ci_high: float | None = None) -> MetricValue:
    return MetricValue(
        name="received_power_fraction",
        value=value,
        unit="fraction",
        n_samples=1000,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )


# --- plot_metric_vs_parameter -------------------------------------------------------


def test_plots_one_point_per_parameter_value() -> None:
    values = [_mv(0.3, ci_low=0.28, ci_high=0.32), _mv(0.2, ci_low=0.18, ci_high=0.22)]
    ax = plot_metric_vs_parameter(
        parameter_values=[10.0, 20.0],
        metric_values=values,
        parameter_name="Range",
        parameter_unit="m",
    )
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), [10.0, 20.0])
    np.testing.assert_allclose(line.get_ydata(), [0.3, 0.2])
    assert ax.get_yscale() == "log"
    plt.close(ax.figure)


def test_labels_include_units() -> None:
    values = [_mv(0.3, ci_low=0.28, ci_high=0.32), _mv(0.2, ci_low=0.18, ci_high=0.22)]
    ax = plot_metric_vs_parameter(
        parameter_values=[10.0, 20.0],
        metric_values=values,
        parameter_name="Range",
        parameter_unit="m",
    )
    assert ax.get_xlabel() == "Range (m)"
    assert "fraction" in ax.get_ylabel()
    plt.close(ax.figure)


def test_scenario_sets_color_and_legend() -> None:
    values = [_mv(0.3, ci_low=0.28, ci_high=0.32), _mv(0.2, ci_low=0.18, ci_high=0.22)]
    ax = plot_metric_vs_parameter(
        parameter_values=[10.0, 20.0],
        metric_values=values,
        parameter_name="Range",
        parameter_unit="m",
        scenario="I",
    )
    assert ax.get_legend() is not None
    plt.close(ax.figure)


def test_zero_detection_point_is_plotted_at_its_upper_bound_not_zero() -> None:
    values = [_mv(0.3, ci_low=0.28, ci_high=0.32), _mv(0.0, ci_low=0.0, ci_high=0.003)]
    ax = plot_metric_vs_parameter(
        parameter_values=[10.0, 200.0],
        metric_values=values,
        parameter_name="Range",
        parameter_unit="m",
    )
    # The zero-detection point must appear as an errorbar container with an
    # upper-limit marker, plotted at the ci95_high bound, not at y = 0.
    assert len(ax.containers) >= 2
    plt.close(ax.figure)


def test_zero_detection_without_ci95_high_raises() -> None:
    values = [_mv(0.3, ci_low=0.28, ci_high=0.32), _mv(0.0)]
    with pytest.raises(ValueError, match="ci95_high"):
        plot_metric_vs_parameter(
            parameter_values=[10.0, 200.0],
            metric_values=values,
            parameter_name="Range",
            parameter_unit="m",
        )


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_metric_vs_parameter(
            parameter_values=[10.0],
            metric_values=[_mv(0.3), _mv(0.2)],
            parameter_name="Range",
            parameter_unit="m",
        )


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        plot_metric_vs_parameter(
            parameter_values=[], metric_values=[], parameter_name="Range", parameter_unit="m"
        )


# --- plot_cir ------------------------------------------------------------------------


def test_cir_plots_against_excess_delay_in_nanoseconds() -> None:
    times = [1.0e-7, 1.05e-7, 1.1e-7, 1.5e-7]
    weights = [1.0, 1.0, 1.0, 1.0]
    ax = plot_cir(arrival_time_s=times, weight=weights, t0_s=1.0e-7, n_bins=10)
    assert ax.get_yscale() == "log"
    assert "ns" in ax.get_xlabel()
    x_data = ax.lines[0].get_xdata()
    assert np.all(x_data >= 0.0)  # all bins are at or after t0
    plt.close(ax.figure)


# --- plot_frequency_response ----------------------------------------------------------


def test_frequency_response_plots_magnitude() -> None:
    freqs = np.array([0.0, 1e6, 2e6])
    response = np.array([1.0 + 0j, 0.7 + 0j, 0.3 + 0j])
    ax = plot_frequency_response(frequencies_hz=freqs, response=response)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 0.7, 0.3])
    plt.close(ax.figure)


def test_frequency_response_marks_the_bandwidth_crossing() -> None:
    freqs = np.array([0.0, 1e6, 2e6])
    response = np.array([1.0 + 0j, 0.7 + 0j, 0.3 + 0j])
    ax = plot_frequency_response(frequencies_hz=freqs, response=response, bandwidth_hz=1.5e6)
    vlines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(vlines) == 1
    plt.close(ax.figure)


# --- plot_scenario_comparison ----------------------------------------------------------


def test_scenario_comparison_draws_one_row_per_metric() -> None:
    diff = paired_scenario_difference(
        metric_name="received_power_fraction",
        scenario_a="I",
        scenario_b="II",
        values_a=[0.30, 0.31, 0.29],
        values_b=[0.25, 0.26, 0.24],
    )
    ax = plot_scenario_comparison({"received_power_fraction": diff})
    assert len(ax.get_yticklabels()) == 1
    # A zero-reference vertical line must be present.
    assert any(line.get_xdata()[0] == 0.0 for line in ax.lines if len(set(line.get_xdata())) == 1)
    plt.close(ax.figure)


def test_scenario_comparison_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_scenario_comparison({})


# --- plot_convergence ------------------------------------------------------------------


def test_convergence_plot_draws_the_se_curve_and_reference_lines() -> None:
    n_photons = [10_000, 40_000, 160_000, 640_000]
    k = 0.01
    values = [
        MetricValue(
            name="m",
            value=0.3,
            unit="fraction",
            n_samples=1000,
            std=k / np.sqrt(n),
            ci95_low=0.3 - k / np.sqrt(n),
            ci95_high=0.3 + k / np.sqrt(n),
        )
        for n in n_photons
    ]
    report = assess_convergence(
        metric_name="received_power_fraction", n_photons=n_photons, values=values
    )
    ax = plot_convergence(report)

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 3  # SE curve, fitted-slope line, N^-1/2 reference line
    plt.close(ax.figure)
