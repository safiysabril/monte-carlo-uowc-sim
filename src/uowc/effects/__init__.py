"""Environmental effects (adapters), composed into a medium.

Effects come in three composable kinds (see :mod:`uowc.core.ports`):
``ParameterEffect``, ``OpticalEffect`` and ``RefractiveEffect``. Turbulence is a
refractive-index effect.
"""
from __future__ import annotations

from uowc.effects.turbulence import TurbulenceEffect

__all__ = ["TurbulenceEffect"]
