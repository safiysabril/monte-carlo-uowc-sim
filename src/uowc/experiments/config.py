"""Typed configuration shared by every scenario in an experiment.

The whole point of holding these in one object is the fair-comparison guarantee: the
optical model, profile, source, receiver, domain, sampling budget and seed are the
*same* across scenarios, so the only things that can differ are the medium
representation (Scenario I vs II) and the environmental effects (Scenario II vs III).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from uowc.core import Receiver, Region, SamplingConfig, Source
from uowc.core.ports import OpticalPropertyModel
from uowc.media import ChlorophyllProfile, DepthAverage, HomogenizationRule, OpticalDepthPreserving

__all__ = ["ExperimentConfig"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Shared inputs for a Scenario I/II/III comparison.

    ``effects`` apply only to Scenario III; ``homogenization`` defines how the profile
    is collapsed to build the Scenario I baseline - either a chlorophyll-space
    :class:`~uowc.media.HomogenizationRule` (:class:`~uowc.media.SurfaceValue`,
    :class:`~uowc.media.DepthAverage`) or the IOP-space
    :class:`~uowc.media.OpticalDepthPreserving` rule (research-methodology.md's
    preferred reference; a distinct interface, not a variant of the former).
    ``wavelength_nm`` must match the optical model's wavelength.
    """

    profile: ChlorophyllProfile
    model: OpticalPropertyModel
    source: Source
    receiver: Receiver
    bounds: Region
    sampling: SamplingConfig
    wavelength_nm: float
    seed: int = 20260607
    effects: Sequence[object] = ()
    homogenization: HomogenizationRule | OpticalDepthPreserving = field(
        default_factory=DepthAverage
    )
    refractive_index: float = 1.34
    code_version: str = "unknown"
