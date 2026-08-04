"""Unit-aware labels, fixed scenario colors, dB/bandwidth convention labels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from uowc.viz.style import (
    SCENARIO_COLORS,
    apply_style,
    axis_label,
    bandwidth_label,
    path_loss_label,
    scenario_color,
)


def test_every_scenario_has_a_distinct_fixed_color() -> None:
    assert set(SCENARIO_COLORS) == {"I", "II", "III"}
    assert len(set(SCENARIO_COLORS.values())) == 3


def test_scenario_color_matches_the_fixed_table() -> None:
    for name, color in SCENARIO_COLORS.items():
        assert scenario_color(name) == color


def test_scenario_color_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="no assigned color"):
        scenario_color("IV")


def test_axis_label_includes_unit() -> None:
    assert axis_label("Wavelength", "nm") == "Wavelength (nm)"


def test_axis_label_omits_parentheses_for_empty_unit() -> None:
    assert axis_label("Elasticity", "") == "Elasticity"


def test_path_loss_label_states_the_convention() -> None:
    assert "optical" in path_loss_label("optical").lower()
    assert "electrical" in path_loss_label("electrical").lower()
    assert path_loss_label("optical") != path_loss_label("electrical")


def test_path_loss_label_rejects_unknown_convention() -> None:
    with pytest.raises(ValueError, match="convention"):
        path_loss_label("linear")


def test_bandwidth_label_states_the_convention() -> None:
    assert "electrical" in bandwidth_label("electrical").lower()
    assert "optical" in bandwidth_label("optical").lower()
    assert bandwidth_label("optical") != bandwidth_label("electrical")


def test_bandwidth_label_rejects_unknown_convention() -> None:
    with pytest.raises(ValueError, match="convention"):
        bandwidth_label("acoustic")


def test_apply_style_sets_expected_rcparams() -> None:
    apply_style()
    import matplotlib as mpl

    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["legend.frameon"] is False


def test_apply_style_is_idempotent() -> None:
    apply_style()
    apply_style()  # must not raise or accumulate state
