"""Visualization (downstream renderer). Reads data; never controls transport."""

from __future__ import annotations

from uowc.viz.figures import (
    plot_cir,
    plot_convergence,
    plot_frequency_response,
    plot_metric_vs_parameter,
    plot_scenario_comparison,
)
from uowc.viz.style import (
    SCENARIO_COLORS,
    SEQUENTIAL_COLORMAP,
    apply_style,
    axis_label,
    bandwidth_label,
    path_loss_label,
    scenario_color,
)

__all__ = [
    "SCENARIO_COLORS",
    "SEQUENTIAL_COLORMAP",
    "scenario_color",
    "axis_label",
    "path_loss_label",
    "bandwidth_label",
    "apply_style",
    "plot_metric_vs_parameter",
    "plot_cir",
    "plot_frequency_response",
    "plot_scenario_comparison",
    "plot_convergence",
]
