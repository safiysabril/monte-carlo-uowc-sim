"""Core ports: the Protocol interfaces every adapter implements.

Per docs/instructions/python.md these are structural ``typing.Protocol`` types
(prefer Protocols over ABCs). They define contracts only - no algorithms.

Note: ``@runtime_checkable`` enables ``isinstance`` checks against these Protocols,
but it verifies *member names only*, not signatures or types. Rely on static type
checking (mypy) for full conformance; use ``isinstance`` only for coarse
plugin/registry guards.

The five headline interfaces are :class:`OpticalPropertyModel`, :class:`Medium`,
:class:`EnvironmentalEffect`, :class:`TransportEngine` and :class:`Metric`.
:class:`PhaseFunction` and :class:`Rng` are supporting ports they reference.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from uowc.core.environment import EnvironmentalState
from uowc.core.geometry import Receiver, Source
from uowc.core.results import MetricValue, RawResult
from uowc.core.state import IOP, LocalOpticalState
from uowc.core.units import FloatArray, Vector3

__all__ = [
    "PhaseFunction",
    "OpticalPropertyModel",
    "EnvironmentalEffect",
    "Medium",
    "Rng",
    "TransportEngine",
    "Metric",
]


@runtime_checkable
class PhaseFunction(Protocol):
    """Scattering phase function: angle sampler plus evaluator.

    Owns scattering directionality so it can be swapped per population without
    subclassing. Sampling must be deterministic given the supplied uniform random
    numbers (reproducibility).
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
    """Environmental parameters -> inherent optical properties.

    Spatially agnostic: depends only on an :class:`EnvironmentalState` and the
    wavelength, never on position, transport, metrics, or storage (see
    scientific-modelling.md). Implementations should be vectorized so array-valued
    inputs yield array-valued :class:`IOP` fields.
    """

    @property
    def name(self) -> str:
        """Stable identifier recorded in run metadata (e.g. ``"haltrin"``)."""
        ...

    def evaluate(self, state: EnvironmentalState, wavelength_nm: float) -> IOP:
        """Return the IOPs for ``state`` at the given wavelength."""
        ...


@runtime_checkable
class EnvironmentalEffect(Protocol):
    """Composable modifier of local optical state.

    A pure transformation of :class:`LocalOpticalState`, plus a bound on the
    extinction it may introduce (so a medium can build a valid Woodcock majorant).
    Effects are composed as an ordered sequence by a medium and must not depend on
    transport, metrics, storage, or plotting.
    """

    @property
    def name(self) -> str:
        """Stable identifier recorded in run metadata (e.g. ``"bubbles"``)."""
        ...

    def apply(
        self,
        state: LocalOpticalState,
        position: Vector3,
        time_s: float = 0.0,
    ) -> LocalOpticalState:
        """Return the effect-modified optical state at ``position`` and ``time_s``."""
        ...

    def max_extinction_factor(self) -> float:
        """Upper bound (>= 1) on the multiplicative increase this effect can apply
        to the local attenuation coefficient anywhere in the domain."""
        ...


@runtime_checkable
class Medium(Protocol):
    """Spatial distribution of optical properties - the transport-facing query.

    A medium composes an :class:`OpticalPropertyModel` with a spatial parameter
    field and an ordered list of :class:`EnvironmentalEffect`; it organizes
    coefficients in space but does not generate them (see mediums.md). It must
    expose a majorant that bounds local extinction everywhere - including effect
    contributions - for Woodcock delta tracking.
    """

    def extinction(self, positions: FloatArray, time_s: float = 0.0) -> FloatArray:
        """Beam-attenuation coefficient ``c(x)`` [m^-1] for a batch of positions of
        shape ``(N, 3)``, returning shape ``(N,)``. Hot path for delta-tracking
        accept/reject; implementations must be vectorized."""
        ...

    def local_state(self, position: Vector3, time_s: float = 0.0) -> LocalOpticalState:
        """Full effective optical state at a single interaction point (phase function
        and refractive index included), after profile and effects."""
        ...

    def majorant_extinction(self) -> float:
        """Domain-wide upper bound ``c_max >= c(x)`` used to sample Woodcock free
        paths [m^-1]. (A regional majorant is a planned efficiency refinement.)"""
        ...

    def contains(self, position: Vector3) -> bool:
        """Whether ``position`` lies within the simulation domain. Surface/bottom
        interaction physics is handled by a separate boundary, not here."""
        ...


@runtime_checkable
class Rng(Protocol):
    """Seeded, splittable random source for reproducible, parallel-safe sampling.

    Stream assignment must be independent of execution order and worker count, so
    results reproduce regardless of parallel layout (see research-methodology.md).
    The full seed tree is recorded in run metadata.
    """

    @property
    def seed(self) -> int:
        """Root seed of this stream."""
        ...

    def uniform(self, size: int) -> FloatArray:
        """Draw ``size`` i.i.d. uniform variates in [0, 1)."""
        ...

    def spawn(self, stream: int) -> "Rng":
        """Return an independent child stream deterministically keyed by ``stream``."""
        ...


@runtime_checkable
class TransportEngine(Protocol):
    """Propagates photons through a medium and records detections.

    The same engine is used for every scenario; scenario differences arise only from
    the injected medium (see transport.md). It must not generate optical
    coefficients, compute metrics, write files, or plot.
    """

    @property
    def name(self) -> str:
        """Stable identifier recorded in run metadata (e.g. ``"woodcock"``)."""
        ...

    def run(
        self,
        medium: Medium,
        source: Source,
        receiver: Receiver,
        rng: Rng,
    ) -> RawResult:
        """Launch ``source.n_photons`` photons and return detections plus tallies."""
        ...


@runtime_checkable
class Metric(Protocol):
    """Transforms raw transport results into a reported value with uncertainty.

    Operates only on a :class:`RawResult`; must not launch photons, modify transport,
    or import plotting (see metrics.md). Identical code runs on in-memory results or
    results read back from Parquet.
    """

    @property
    def name(self) -> str:
        """Stable identifier (e.g. ``"rms_delay_spread"``)."""
        ...

    def compute(self, result: RawResult) -> MetricValue:
        """Compute the metric, with uncertainty, from a single run."""
        ...
