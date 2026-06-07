"""Environmental parameter bundle consumed by optical-property models."""
from __future__ import annotations

from dataclasses import dataclass

from uowc.core.units import FloatOrArray

__all__ = ["EnvironmentalState"]


@dataclass(frozen=True, slots=True)
class EnvironmentalState:
    """Environmental constituents at a point, or vectorized over many points.

    The single input every :class:`~uowc.core.ports.OpticalPropertyModel` consumes.
    Fields may be scalars (one state) or equal-length arrays (a batch, or a sampled
    depth profile), enabling vectorized evaluation by broadcasting. Constituents a
    given model does not use are left ``None``.

    Using a parameter bundle (rather than a fixed argument list) lets new models
    declare additional constituents without breaking existing signatures.

    Units:
        chlorophyll: mg m^-3
        cdom_440:    m^-1   (CDOM absorption coefficient at 440 nm)
        nap:         g m^-3 (non-algal / mineral particle concentration)
        temperature: degrees Celsius
        salinity:    PSU
    """

    chlorophyll: FloatOrArray
    cdom_440: FloatOrArray | None = None
    nap: FloatOrArray | None = None
    temperature: FloatOrArray | None = None
    salinity: FloatOrArray | None = None
