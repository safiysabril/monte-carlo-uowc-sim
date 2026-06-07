"""Medium representations and spatial fields (adapters).

Profiles map depth to environmental parameters; mediums organize optical properties
in space. Depends on the core ports and value objects only.
"""
from __future__ import annotations

from uowc.media.inhomogeneous import InhomogeneousMedium
from uowc.media.profiles import ChlorophyllProfile, KamedaModel

__all__ = ["ChlorophyllProfile", "KamedaModel", "InhomogeneousMedium"]
