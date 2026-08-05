"""Homogenization rules: collapse a depth profile to a single concentration, or a
single IOP set.

Used to derive the Scenario I homogeneous baseline from the *same* chlorophyll
profile that drives Scenario II, so the two scenarios differ only in medium
representation (see research-methodology.md). Which rule is "best" is itself a
research question: research-methodology.md requires results to be reported across
*all* implemented rules, not one favoured choice, since the spread between them
bounds how much of a reported difference is physics versus bookkeeping.

:class:`SurfaceValue` and :class:`DepthAverage` operate in chlorophyll space: reduce
the profile to one concentration, then let the optical-property model compute IOPs
from it as usual. :class:`OpticalDepthPreserving` is a *distinct interface*
(research-methodology.md: "not a variant of" the chlorophyll-space rule) - it
evaluates the model at each depth first and averages the resulting IOPs, because
IOPs are concave in chlorophyll (Jensen's inequality) and averaging chlorophyll
first systematically overestimates attenuation relative to averaging IOPs. Its
method is deliberately not named ``reduce`` (unlike the chlorophyll-space rules), so
it does not accidentally satisfy the ``@runtime_checkable`` :class:`HomogenizationRule`
check by having a same-named but differently-shaped method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import Region
from uowc.core.ports import OpticalPropertyModel
from uowc.core.state import IOP, Scattering, ScatteringComponent
from uowc.core.units import FloatArray
from uowc.media.profiles import ChlorophyllProfile

__all__ = [
    "HomogenizationRule",
    "SurfaceValue",
    "DepthAverage",
    "OpticalDepthPreserving",
    "depth_span_m",
]

def _trapz(y: FloatArray, x: FloatArray) -> FloatArray:
    """``numpy.trapezoid`` (numpy>=2.0) or its pre-2.0 name ``numpy.trapz`` -
    whichever this numpy has - given a definite signature so callers do not each
    need to guard against the ``None`` a bare ``getattr(..., None)`` could return."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    assert fn is not None  # one of the two names exists in every supported numpy
    return fn(y, x)  # type: ignore[no-any-return]


def depth_span_m(bounds: Region) -> tuple[float, float]:
    """``(depth_min_m, depth_max_m)`` spanned by ``bounds`` (mediums.md: depth
    positive downward, ``depth = -z``; the domain's shallowest depth is at its
    highest ``z``, i.e. ``bounds.upper``)."""
    depth_min_m = -float(np.asarray(bounds.upper)[2])
    depth_max_m = -float(np.asarray(bounds.lower)[2])
    return depth_min_m, depth_max_m


@runtime_checkable
class HomogenizationRule(Protocol):
    """Reduce a chlorophyll profile over a depth range to a single concentration.

    Depths are in metres, positive downward; ``depth_min_m`` is the shallowest
    (surface) depth and ``depth_max_m`` the deepest.
    """

    @property
    def name(self) -> str: ...

    def reduce(
        self, profile: ChlorophyllProfile, depth_min_m: float, depth_max_m: float
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class SurfaceValue:
    """Homogenize to the surface (shallowest) chlorophyll value, ``C(depth_min)``."""

    @property
    def name(self) -> str:
        return "surface"

    def reduce(self, profile: ChlorophyllProfile, depth_min_m: float, depth_max_m: float) -> float:
        return float(np.asarray(profile.chlorophyll(np.asarray(depth_min_m, dtype=np.float64))))


@dataclass(frozen=True, slots=True)
class DepthAverage:
    """Homogenize to the depth-averaged chlorophyll, ``(1/D) * integral C(z) dz``."""

    samples: int = 1024

    @property
    def name(self) -> str:
        return "depth_average"

    def reduce(self, profile: ChlorophyllProfile, depth_min_m: float, depth_max_m: float) -> float:
        if depth_max_m <= depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        depths = np.linspace(depth_min_m, depth_max_m, self.samples)
        column = profile.chlorophyll(depths)
        return float(_trapz(column, depths) / (depth_max_m - depth_min_m))


@dataclass(frozen=True, slots=True)
class OpticalDepthPreserving:
    """IOP-space homogenization: preserves ``integral a ds`` and ``integral b ds``
    exactly along the nominal line of sight (research-methodology.md's "preferred
    reference" rule)::

        a_bar = (1/D) * integral(0, D) a(s) ds
        b_bar = (1/D) * integral(0, D) b(s) ds

    This reproduces the true medium's unscattered (ballistic) transmittance
    ``exp(-tau)`` exactly, and preserves the path-averaged single-scattering albedo -
    unlike :class:`SurfaceValue`/:class:`DepthAverage`, which average chlorophyll
    *before* the model's nonlinear ``C -> IOP`` mapping and so carry a Jensen-inequality
    bias (research-methodology.md, "Averaging space matters").

    Limits (state these alongside any result using this rule): it does not preserve
    the multiply-scattered field, since that depends on *where* along the path
    scattering occurred, not only on the path integral; and it is defined along a
    single nominal straight path, so its quality degrades as the true link's
    single-scattering albedo approaches 1 and photons wander far from that path.

    Each scattering population is averaged separately (not just the total ``b``), so
    a two-population mixture (e.g. Haltrin's water + particle terms) keeps its two
    phase functions rather than collapsing to one - the same population structure the
    model would report at any single depth, since population identity (which phase
    function belongs to which term) does not vary with chlorophyll in these models.
    """

    samples: int = 1024

    @property
    def name(self) -> str:
        return "optical_depth"

    def reduce_iop(
        self,
        *,
        profile: ChlorophyllProfile,
        model: OpticalPropertyModel,
        wavelength_nm: float,
        depth_min_m: float,
        depth_max_m: float,
    ) -> IOP:
        if depth_max_m <= depth_min_m:
            raise ValueError("depth_max_m must exceed depth_min_m")
        depths = np.linspace(depth_min_m, depth_max_m, self.samples)
        chlorophyll = profile.chlorophyll(depths)
        column = model.evaluate(EnvironmentalState(chlorophyll=chlorophyll), wavelength_nm)
        span = depth_max_m - depth_min_m

        mean_absorption = self._path_mean(column.absorption, depths, span)
        mean_components = tuple(
            ScatteringComponent(
                coefficient=self._path_mean(component.coefficient, depths, span),
                phase=component.phase,
            )
            for component in column.scattering.components
        )
        return IOP(
            absorption=mean_absorption,
            scattering=Scattering(mean_components),
            wavelength_nm=column.wavelength_nm,
        )

    @staticmethod
    def _path_mean(values: FloatArray | float, depths: FloatArray, span: float) -> float:
        # A population that does not depend on chlorophyll (e.g. a constant
        # pure-water term) evaluates to a bare scalar rather than one value per
        # depth; broadcasting makes the same integration formula handle both.
        broadcast = np.broadcast_to(np.asarray(values, dtype=np.float64), depths.shape)
        return float(_trapz(broadcast, depths) / span)
