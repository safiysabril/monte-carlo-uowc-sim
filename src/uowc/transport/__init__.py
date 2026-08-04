"""Photon transport engine (adapter).

Depends only on the core ``Medium`` port and value objects - independent of any
optical-property model, medium implementation, environmental effect, or metric.
"""
from __future__ import annotations

from uowc.transport.tallies import (
    available_tallies,
    build_angle_of_arrival_tally,
    build_cir_tally,
    build_depth_deposition_tally,
    build_requested_tallies,
)
from uowc.transport.woodcock import WoodcockDeltaTracker

__all__ = [
    "WoodcockDeltaTracker",
    "available_tallies",
    "build_cir_tally",
    "build_angle_of_arrival_tally",
    "build_depth_deposition_tally",
    "build_requested_tallies",
]
