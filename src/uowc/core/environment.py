"""Environmental parameter bundle consumed by optical-property models."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from uowc.core.units import FloatOrArray, scalar_or_readonly

__all__ = ["EnvironmentalState"]

_EMPTY: Mapping[str, FloatOrArray] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class EnvironmentalState:
    """Environmental constituents at a point, or vectorized over many points.

    The single input every :class:`~uowc.core.ports.OpticalPropertyModel` consumes.
    Fields may be scalars or equal-length arrays (a batch or a sampled depth
    profile). Constituents a model does not use are left ``None``. The ``extra``
    mapping is an escape hatch so new constituents can be carried without editing
    this core type.

    Units:
        chlorophyll: mg m^-3
        cdom_440:    m^-1   (CDOM absorption at 440 nm; the model owns the spectral
                            slope used to extrapolate to other wavelengths)
        nap:         g m^-3 (non-algal / mineral particle concentration)
        temperature: degrees Celsius
        salinity:    PSU
    """

    chlorophyll: FloatOrArray
    cdom_440: FloatOrArray | None = None
    nap: FloatOrArray | None = None
    temperature: FloatOrArray | None = None
    salinity: FloatOrArray | None = None
    extra: Mapping[str, FloatOrArray] = _EMPTY

    def __post_init__(self) -> None:
        for name in ("chlorophyll", "cdom_440", "nap", "temperature", "salinity"):
            object.__setattr__(self, name, scalar_or_readonly(getattr(self, name)))
