"""Online tallies: channel-impulse-response bins, depth deposition, angle-of-arrival.

transport.md lists "tallies sufficient to close the weight balance" as a core
transport responsibility; the scalar conservation tally
(:class:`~uowc.core.results.Tallies`) already satisfies that. This module supplies
the *named*, binned diagnostics :class:`~uowc.core.config.SamplingConfig.tallies`
requests by name (``"cir"``, ``"angle_of_arrival"``, ``"depth_deposition"``), each
returned as a :class:`~uowc.core.results.TallyResult` in
:attr:`~uowc.core.results.TransportOutput.binned`.

The CIR and angle-of-arrival tallies are built from data the transport engine already
returns per detected photon (arrival time, incidence angle) - binning them is pure
post-processing, not a change to the tracked physics. Depth deposition needs one
additional thing the analog loop does not otherwise keep: *where* each absorption
happened, so :class:`~uowc.transport.woodcock.WoodcockDeltaTracker` records absorption
positions only when ``"depth_deposition"`` is actually requested.

These are diagnostics, not the estimator: they must never feed back into weight
updates or termination decisions (transport.md's requirement that tallies stay
downstream of the transport loop, not part of it).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from uowc.core.geometry import Region
from uowc.core.results import DetectedPhotons, TallyResult
from uowc.core.units import FloatArray

__all__ = [
    "available_tallies",
    "build_cir_tally",
    "build_angle_of_arrival_tally",
    "build_depth_deposition_tally",
    "build_requested_tallies",
]

_DEFAULT_N_BINS = 50


def _bin_edges(values: FloatArray, n_bins: int) -> FloatArray:
    """Evenly spaced edges spanning the observed range of ``values``.

    An empty or degenerate (single-valued) input has no natural range; both are
    padded to a single unit-width bin rather than raising, since "nothing was
    detected/absorbed" is itself a valid, reportable outcome for a tally.
    """
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def build_cir_tally(
    photons: DetectedPhotons, *, n_bins: int = _DEFAULT_N_BINS
) -> TallyResult:
    """Detected weight binned by absolute arrival time (a discretized CIR).

    metrics.md warns that a binned histogram must never be *reused* to compute a
    frequency-domain metric (it multiplies the true ``H(f)`` by a sinc bias) - this
    tally is for storage/display only, same restriction as
    :func:`~uowc.viz.figures.plot_cir`.
    """
    edges = _bin_edges(photons.arrival_time_s, n_bins)
    counts, edges = np.histogram(photons.arrival_time_s, bins=edges, weights=photons.weight)
    return TallyResult(
        name="cir",
        values=counts,
        edges=(edges,),
        unit="dimensionless",
        axis_names=("arrival_time_s",),
    )


def build_angle_of_arrival_tally(
    photons: DetectedPhotons, *, n_bins: int = _DEFAULT_N_BINS
) -> TallyResult:
    """Detected weight binned by incidence angle at the receiver."""
    edges = _bin_edges(photons.incidence_rad, n_bins)
    counts, edges = np.histogram(photons.incidence_rad, bins=edges, weights=photons.weight)
    return TallyResult(
        name="angle_of_arrival",
        values=counts,
        edges=(edges,),
        unit="dimensionless",
        axis_names=("incidence_rad",),
    )


def build_depth_deposition_tally(
    absorbed_positions: FloatArray,
    absorbed_weights: FloatArray,
    *,
    domain_bounds: Region,
    n_bins: int = _DEFAULT_N_BINS,
) -> TallyResult:
    """Absorbed weight binned by depth (mediums.md: depth positive downward, ``-z``).

    Bin edges span the domain's full vertical extent (not just the observed range of
    absorption events), so a depth-deposition profile is comparable across runs with
    different absorption patterns but the same domain.
    """
    depth_lo = -float(domain_bounds.upper[2])
    depth_hi = -float(domain_bounds.lower[2])
    if depth_hi <= depth_lo:
        depth_hi = depth_lo + 1.0
    edges = np.linspace(depth_lo, depth_hi, n_bins + 1)
    positions = np.asarray(absorbed_positions, dtype=np.float64)
    depths = -positions[..., 2] if positions.size else np.empty(0)
    counts, edges = np.histogram(depths, bins=edges, weights=absorbed_weights)
    return TallyResult(
        name="depth_deposition",
        values=counts,
        edges=(edges,),
        unit="dimensionless",
        axis_names=("depth_m",),
    )


def available_tallies() -> tuple[str, ...]:
    """Names accepted by :attr:`~uowc.core.config.SamplingConfig.tallies`."""
    return ("cir", "angle_of_arrival", "depth_deposition")


def build_requested_tallies(
    names: Sequence[str],
    *,
    photons: DetectedPhotons,
    absorbed_positions: FloatArray,
    absorbed_weights: FloatArray,
    domain_bounds: Region,
    n_bins: int = _DEFAULT_N_BINS,
) -> dict[str, TallyResult]:
    """Build every requested named tally; unknown names raise rather than being
    silently dropped, since a misspelled tally name would otherwise look like an
    empty-but-valid result."""
    unknown = set(names) - set(available_tallies())
    if unknown:
        raise ValueError(f"unknown tallies {sorted(unknown)}; available: {available_tallies()}")

    built: dict[str, TallyResult] = {}
    for tally_name in names:
        if tally_name == "cir":
            built[tally_name] = build_cir_tally(photons, n_bins=n_bins)
        elif tally_name == "angle_of_arrival":
            built[tally_name] = build_angle_of_arrival_tally(photons, n_bins=n_bins)
        elif tally_name == "depth_deposition":
            built[tally_name] = build_depth_deposition_tally(
                absorbed_positions, absorbed_weights, domain_bounds=domain_bounds, n_bins=n_bins
            )
    return built
