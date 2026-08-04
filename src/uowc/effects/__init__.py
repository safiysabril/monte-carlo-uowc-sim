"""Environmental effects (adapters), composed into a medium.

Effects come in three composable kinds (see :mod:`uowc.core.ports`):
``ParameterEffect``, ``OpticalEffect`` and ``RefractiveEffect``. Turbulence is a
refractive-index effect.
"""
from __future__ import annotations

from uowc.effects.bubbles import BubbleLayerEffect
from uowc.effects.registry import available_effects, build_effect, build_effects
from uowc.effects.sediment import SedimentEffect
from uowc.effects.thermocline import ThermoclineEffect
from uowc.effects.turbulence import TurbulenceEffect

__all__ = [
    "TurbulenceEffect",
    "SedimentEffect",
    "BubbleLayerEffect",
    "ThermoclineEffect",
    "build_effect",
    "build_effects",
    "available_effects",
]
