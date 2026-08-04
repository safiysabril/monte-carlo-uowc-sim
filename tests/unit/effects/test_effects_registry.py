"""Config-driven effect construction (name -> factory)."""

from __future__ import annotations

import numpy as np
import pytest

from uowc.effects.bubbles import BubbleLayerEffect
from uowc.effects.registry import available_effects, build_effect, build_effects
from uowc.effects.sediment import SedimentEffect
from uowc.effects.thermocline import ThermoclineEffect
from uowc.effects.turbulence import TurbulenceEffect


def test_available_effects_lists_all_four() -> None:
    assert available_effects() == ("bubbles", "sediment", "thermocline", "turbulence")


def test_build_turbulence() -> None:
    effect = build_effect(
        {
            "name": "turbulence",
            "params": {
                "rms_fluctuation": 1e-4,
                "correlation_length_m": 5.0,
                "n_modes": 64,
                "seed": 3,
            },
        }
    )
    assert isinstance(effect, TurbulenceEffect)
    assert effect.rms_fluctuation == pytest.approx(1e-4)
    assert effect.wavevectors.shape == (64, 3)


def test_build_sediment_without_region() -> None:
    effect = build_effect({"name": "sediment", "params": {"concentration_g_m3": 2.0}})
    assert isinstance(effect, SedimentEffect)
    assert effect.region is None
    assert effect.concentration_g_m3 == pytest.approx(2.0)


def test_build_sediment_with_region() -> None:
    spec = {
        "name": "sediment",
        "params": {
            "concentration_g_m3": 1.0,
            "region": {"lower": [-1.0, -1.0, -5.0], "upper": [1.0, 1.0, -2.0]},
        },
    }
    effect = build_effect(spec)
    assert isinstance(effect, SedimentEffect)
    assert effect.region is not None
    np.testing.assert_allclose(effect.region.lower, [-1.0, -1.0, -5.0])
    np.testing.assert_allclose(effect.region.upper, [1.0, 1.0, -2.0])


def test_build_bubbles_with_henyey_greenstein_phase() -> None:
    spec = {
        "name": "bubbles",
        "params": {
            "region": {"lower": [-10.0, -10.0, -1.0], "upper": [10.0, 10.0, 0.0]},
            "void_fraction": 1e-4,
            "mean_radius_m": 1e-4,
            "phase": {"type": "henyey_greenstein", "g": 0.85},
        },
    }
    effect = build_effect(spec)
    assert isinstance(effect, BubbleLayerEffect)
    assert effect.phase.asymmetry == pytest.approx(0.85)


def test_build_bubbles_rejects_unknown_phase_type() -> None:
    spec = {
        "name": "bubbles",
        "params": {
            "region": {"lower": [0, 0, -1], "upper": [1, 1, 0]},
            "void_fraction": 1e-4,
            "mean_radius_m": 1e-4,
            "phase": {"type": "fournier_forand"},
        },
    }
    with pytest.raises(ValueError, match="unknown phase-function type"):
        build_effect(spec)


def test_build_thermocline() -> None:
    spec = {
        "name": "thermocline",
        "params": {
            "depth_m": 20.0,
            "transition_width_m": 2.0,
            "index_above": 0.0,
            "index_below": 0.001,
        },
    }
    effect = build_effect(spec)
    assert isinstance(effect, ThermoclineEffect)
    assert effect.depth_m == pytest.approx(20.0)


def test_unknown_effect_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError, match="unknown effect"):
        build_effect({"name": "nonexistent", "params": {}})


def test_build_effects_preserves_order() -> None:
    specs = [
        {"name": "sediment", "params": {"concentration_g_m3": 1.0}},
        {
            "name": "thermocline",
            "params": {
                "depth_m": 5.0,
                "transition_width_m": 1.0,
                "index_above": 0.0,
                "index_below": 0.001,
            },
        },
    ]
    effects = build_effects(specs)
    assert isinstance(effects[0], SedimentEffect)
    assert isinstance(effects[1], ThermoclineEffect)


def test_params_default_to_empty_mapping_when_omitted() -> None:
    with pytest.raises(KeyError):
        build_effect({"name": "sediment"})
