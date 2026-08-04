"""No Kameda IOP model exists here - see uowc.optics.kameda for why.

Regression guard only: the profile model (the real, citable Kameda & Matsumura
contribution) must keep living in the media layer, and optics.kameda must keep
exporting nothing rather than silently growing a fabricated IOP formula.
"""
from __future__ import annotations

import uowc.optics.kameda as kameda_module
from uowc.media.profiles import KamedaModel


def test_optics_kameda_intentionally_exports_nothing() -> None:
    assert kameda_module.__all__ == []


def test_kameda_profile_lives_in_media_not_optics() -> None:
    assert not hasattr(kameda_module, "KamedaModel")
    assert KamedaModel is not None
