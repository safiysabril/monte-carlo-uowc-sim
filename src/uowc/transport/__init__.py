"""Photon transport engine (adapter).

Depends only on the core ``Medium`` port and value objects - independent of any
optical-property model, medium implementation, environmental effect, or metric.
"""
from __future__ import annotations

from uowc.transport.woodcock import WoodcockDeltaTracker

__all__ = ["WoodcockDeltaTracker"]
