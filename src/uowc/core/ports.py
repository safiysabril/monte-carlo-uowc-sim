"""Core ports: the Protocol interfaces every adapter implements.

Structural ``typing.Protocol`` types (prefer Protocols, per python.md); contracts
only - no algorithms. ``@runtime_checkable`` checks member *names* only; rely on mypy
for full conformance.

Headline interfaces: :class:`OpticalPropertyModel`; :class:`Medium`, composed from
:class:`OpticalField`, :class:`Domain` and :class:`Acceleration`; the effect family
:class:`ParameterEffect`, :class:`OpticalEffect` and :class:`RefractiveEffect`;
:class:`TransportEngine`; and :class:`Metric` / :class:`ComparativeMetric`.
:class:`PhaseFunction` and :class:`Rng` are supporting ports.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from uowc.core.config import SamplingConfig
from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import BoundaryOutcome, Receiver, Region, Source
from uowc.core.results import MetricValue, RawResult, SeedTree, TransportOutput
from uowc.core.state import IOP, LocalOpticalState
from uowc.core.units import BoolArray, FloatArray, IntArray, Vector3

__all__ = [
    "PhaseFunction",
    "OpticalPropertyModel",
    "ParameterEffect",
    "OpticalEffect",
    "RefractiveEffect",
    "OpticalField",
    "Domain",
    "Acceleration",
    "Medium",
    "Boundary",
    "Rng",
    "TransportEngine",
    "Metric",
    "ComparativeMetric",
]


@runtime_checkable
class PhaseFunction(Protocol):
    """Scattering phase function for one population: sampler plus evaluator.

    Sampling must be deterministic given the supplied uniform random numbers.
    """

    @property
    def asymmetry(self) -> float:
        """Asymmetry parameter ``g = <cos theta>`` [dimensionless]."""
        ...

    def sample_cos_theta(self, u: FloatArray) -> FloatArray:
        """Map uniform randoms ``u`` in [0, 1) to scattering-angle cosines."""
        ...

    def value(self, cos_theta: FloatArray) -> FloatArray:
        """Phase-function value at ``cos_theta``, normalized over solid angle."""
        ...


@runtime_checkable
class OpticalPropertyModel(Protocol):
    """Environmental parameters -> inherent optical properties (incl. the scattering
    mixture). Spatially agnostic; depends only on an :class:`EnvironmentalState` and
    wavelength (see scientific-modelling.md). Should be vectorized.
    """

    @property
    def name(self) -> str:
        """Stable identifier recorded in run metadata (e.g. ``"haltrin"``)."""
        ...

    def evaluate(self, state: EnvironmentalState, wavelength_nm: float) -> IOP:
        """Return the IOPs (absorption + scattering mixture) for ``state``."""
        ...


@runtime_checkable
class ParameterEffect(Protocol):
    """Effect that modifies environmental parameters *before* the optical model.

    The right layer for constituent changes (e.g. suspended sediment adding NAP,
    chlorophyll variation), so the bio-optical coupling stays consistent. Batched:
    operates on a (possibly array-valued) :class:`EnvironmentalState`.
    """

    @property
    def name(self) -> str: ...

    def apply(
        self, state: EnvironmentalState, positions: FloatArray, time_s: float = 0.0
    ) -> EnvironmentalState:
        """Return modified environmental parameters at ``positions``."""
        ...


@runtime_checkable
class OpticalEffect(Protocol):
    """Effect that modifies local optical state *after* the optical model.

    E.g. a bubble layer adding a scattering population. Provides a batched extinction
    contribution for the transport hot path and an additive, *regional* majorant bound
    (not a global scalar), so the medium's accelerator stays tight.
    """

    @property
    def name(self) -> str: ...

    def apply(
        self, state: LocalOpticalState, position: Vector3, time_s: float = 0.0
    ) -> LocalOpticalState:
        """Return the effect-modified optical state at one point."""
        ...

    def extinction_contribution(
        self, positions: FloatArray, time_s: float = 0.0
    ) -> FloatArray:
        """Additive extinction [m^-1] contributed at each position, shape ``(N,)``."""
        ...

    def extinction_bound(self, region: Region) -> float:
        """Upper bound on the additive extinction introduced anywhere in ``region``."""
        ...


@runtime_checkable
class RefractiveEffect(Protocol):
    """Effect that contributes to the refractive-index field (e.g. turbulence,
    thermocline/halocline).

    Supplies this effect's *additive contribution* to the refractive index (around the
    medium's base index) and that contribution's spatial gradient, so a medium composes
    ``n(x) = n0 + sum_i`` and the transport engine can bend rays during free flight. The
    ``index`` and ``gradient`` methods return this contribution, not the absolute field.
    Batched.
    """

    @property
    def name(self) -> str: ...

    def index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Refractive index [dimensionless] at each position, shape ``(N,)``."""
        ...

    def gradient(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Refractive-index gradient [m^-1], shape ``(N, 3)``."""
        ...


@runtime_checkable
class OpticalField(Protocol):
    """Spatial query of optical properties (one of the three medium capabilities).

    Scalar fields are batched for the delta-tracking hot path; the full
    :class:`LocalOpticalState` (including the scattering mixture) is returned per point
    at the rarer real-interaction events.
    """

    def extinction(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Beam attenuation ``c(x)`` [m^-1], shape ``(N,)`` for positions ``(N, 3)``."""
        ...

    def refractive_index(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Refractive index, shape ``(N,)`` (for arrival time and ray bending)."""
        ...

    def refractive_index_gradient(
        self, positions: FloatArray, time_s: float = 0.0
    ) -> FloatArray:
        """Refractive-index gradient [m^-1], shape ``(N, 3)``."""
        ...

    def local_state(self, position: Vector3, time_s: float = 0.0) -> LocalOpticalState:
        """Full effective optical state at a single interaction point."""
        ...


@runtime_checkable
class Domain(Protocol):
    """Spatial extent of the simulation (one of the three medium capabilities).

    Surface/bottom interaction physics is handled separately by a boundary, not here.
    """

    def contains(self, positions: FloatArray) -> BoolArray:
        """Whether each position lies inside the domain, shape ``(N,)``."""
        ...

    def bounds(self) -> Region:
        """Axis-aligned bounds of the domain."""
        ...


@runtime_checkable
class Acceleration(Protocol):
    """Woodcock acceleration structure (one of the three medium capabilities).

    Kept separate so a non-Woodcock transport need not provide a majorant.
    """

    def majorant(self, region: Region) -> float:
        """Upper bound ``c_max >= c(x)`` within ``region`` [m^-1]; regional, so a
        localized high-extinction layer does not inflate the bound everywhere."""
        ...


@runtime_checkable
class Medium(Protocol):
    """The transport-facing medium: a composition of three decoupled capabilities.

    Concrete mediums compose an :class:`OpticalPropertyModel`, a spatial parameter
    field, and the effect family, exposing the result as a field, a domain and an
    accelerator (see mediums.md).
    """

    @property
    def field(self) -> OpticalField: ...

    @property
    def domain(self) -> Domain: ...

    @property
    def acceleration(self) -> Acceleration: ...


@runtime_checkable
class Boundary(Protocol):
    """What happens when a photon's path reaches a domain boundary (mediums.md:
    "the domain answers 'is this position inside'; the boundary answers 'what
    happens on contact'").

    Distinct from :class:`Domain`, which only tests containment. A domain that spans
    ``z = 0`` has an air-water surface; a domain with a lower bound has a bottom. Not
    every domain edge needs a ``Boundary`` - an edge with no stated boundary is a pure
    escape (mediums.md's default domain-edge policy), which is what every current
    :class:`~uowc.transport.TransportEngine` implements today. A ``Boundary`` is
    supplied only where photon-boundary interaction (reflection, absorption) is
    intended, and is not yet consumed by the transport engine (transport.md's
    "Future Extensions": boundary interaction is planned, after next-event
    estimation and ray bending).
    """

    @property
    def name(self) -> str: ...

    def interact(
        self,
        position: Vector3,
        direction: FloatArray,
        wavelength_nm: float,
        rng: Rng,
    ) -> BoundaryOutcome:
        """Resolve one photon's contact with this boundary at ``position``, arriving
        along ``direction`` (unit vector, pointing *toward* the boundary)."""
        ...


@runtime_checkable
class Rng(Protocol):
    """Seeded, splittable random source for reproducible, parallel-safe sampling.

    Stream assignment is independent of execution order and worker count; the full
    seed hierarchy is recoverable via :meth:`seed_tree`.
    """

    @property
    def seed(self) -> int: ...

    def uniform(self, shape: int | tuple[int, ...] = 1) -> FloatArray: ...

    def normal(self, shape: int | tuple[int, ...] = 1) -> FloatArray: ...

    def integers(
        self, low: int, high: int, shape: int | tuple[int, ...] = 1
    ) -> IntArray: ...

    def spawn(self, stream: int) -> Rng: ...

    def seed_tree(self) -> SeedTree: ...


@runtime_checkable
class TransportEngine(Protocol):
    """Propagates photons through a medium and records detections plus tallies.

    The same engine serves every scenario; differences come only from the injected
    medium (see transport.md). It returns a :class:`TransportOutput` - provenance is
    attached by orchestration, not by transport.
    """

    @property
    def name(self) -> str: ...

    def run(
        self,
        medium: Medium,
        source: Source,
        receiver: Receiver,
        rng: Rng,
        config: SamplingConfig,
    ) -> TransportOutput: ...


@runtime_checkable
class Metric(Protocol):
    """Single-run metric: :class:`RawResult` -> :class:`MetricValue` with uncertainty
    (see metrics.md). Must not launch photons, modify transport, or import plotting.
    """

    @property
    def name(self) -> str: ...

    def compute(self, result: RawResult) -> MetricValue: ...


@runtime_checkable
class ComparativeMetric(Protocol):
    """Multi-run metric over an ordered set of results - the home for the headline
    research outputs: scenario differences (I vs II, II vs III) and convergence /
    sensitivity curves.
    """

    @property
    def name(self) -> str: ...

    def compare(self, results: Sequence[RawResult]) -> MetricValue: ...
