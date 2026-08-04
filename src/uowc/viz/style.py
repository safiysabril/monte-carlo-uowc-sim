"""Publication figure style: unit-aware axes, a fixed categorical palette, and
consistent labels (visualization.md).

Color choices
-------------
Scenario I/II/III are assigned a **fixed** color by identity, never by plot order or
by which scenarios happen to appear in a given figure - the same scenario must be the
same color in every figure in a paper, so a reader's legend memory transfers across
figures. The three colors are drawn from the Okabe-Ito color-universal-design set
(Okabe, M. & Ito, K. (2008), "Color Universal Design (CUD)"), chosen for
discriminability under the common forms of color vision deficiency; it is the same
qualitative palette behind, e.g., matplotlib's/seaborn's "colorblind" cycle.

Magnitude fields (depth, delay, frequency) use a single perceptually-uniform
sequential colormap (``viridis``) rather than a rainbow - light-to-dark in one hue
reads as an ordered magnitude; a rainbow does not.
"""

from __future__ import annotations

__all__ = [
    "SCENARIO_COLORS",
    "SEQUENTIAL_COLORMAP",
    "scenario_color",
    "axis_label",
    "path_loss_label",
    "bandwidth_label",
    "apply_style",
]

#: Fixed scenario -> color assignment (Okabe & Ito 2008; module docstring).
#: Never reassign or cycle these - a scenario keeps its color across every figure.
SCENARIO_COLORS: dict[str, str] = {
    "I": "#0072B2",  # blue
    "II": "#D55E00",  # vermillion
    "III": "#009E73",  # bluish green
}

#: Perceptually-uniform sequential colormap for magnitude fields (depth, delay, ...).
SEQUENTIAL_COLORMAP = "viridis"

_PATH_LOSS_LABELS = {
    "optical": "Path loss (optical dB, 10 log₁₀)",
    "electrical": "Path loss (electrical dB, 20 log₁₀)",
}
_BANDWIDTH_LABELS = {
    "electrical": "Electrical 3 dB bandwidth (Hz)",
    "optical": "Optical 3 dB bandwidth (Hz)",
}


def scenario_color(scenario: str) -> str:
    """The fixed color for ``scenario`` ("I", "II" or "III")."""
    try:
        return SCENARIO_COLORS[scenario]
    except KeyError:
        raise ValueError(
            f"no assigned color for scenario {scenario!r}; expected one of {tuple(SCENARIO_COLORS)}"
        ) from None


def axis_label(name: str, unit: str) -> str:
    """``"name (unit)"``, or bare ``name`` if ``unit`` is empty.

    Structurally enforces visualization.md's "axes must carry units" rather than
    relying on every call site to remember to append one by hand.
    """
    return f"{name} ({unit})" if unit else name


def path_loss_label(convention: str = "optical") -> str:
    """Axis label naming the optical/electrical dB convention (metrics.md) - a path-
    loss axis must never be labelled with the ambiguous bare word "dB"."""
    try:
        return _PATH_LOSS_LABELS[convention]
    except KeyError:
        raise ValueError(
            f"convention must be 'optical' or 'electrical', got {convention!r}"
        ) from None


def bandwidth_label(convention: str = "electrical") -> str:
    """Axis label naming the optical/electrical 3 dB bandwidth convention (metrics.md)."""
    try:
        return _BANDWIDTH_LABELS[convention]
    except KeyError:
        raise ValueError(
            f"convention must be 'optical' or 'electrical', got {convention!r}"
        ) from None


def apply_style() -> None:
    """Apply a consistent, publication-appropriate matplotlib style once per process:
    thin lines, recessive gridlines, no top/right spines, a legible base font size.

    Idempotent - safe to call more than once (e.g. once per figure module) without
    accumulating state, since it only assigns fixed rcParams values.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.0,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
