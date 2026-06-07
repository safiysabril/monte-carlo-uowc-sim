"""Medium representations and spatial fields (adapters).

Profiles map depth to environmental parameters; mediums organize optical properties
in space. Depends on the core ports and value objects only.
"""
from __future__ import annotations

from uowc.media.domain import BoxDomain
from uowc.media.homogeneous import HomogeneousMedium
from uowc.media.homogenization import DepthAverage, HomogenizationRule, SurfaceValue
from uowc.media.inhomogeneous import InhomogeneousMedium
from uowc.media.profiles import ChlorophyllProfile, KamedaModel

__all__ = [
    "ChlorophyllProfile",
    "KamedaModel",
    "BoxDomain",
    "HomogenizationRule",
    "SurfaceValue",
    "DepthAverage",
    "HomogeneousMedium",
    "InhomogeneousMedium",
]
