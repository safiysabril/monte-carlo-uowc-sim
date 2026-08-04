"""Why there is no separate Estimator abstraction here.

This module's name once implied three interchangeable ``Estimator`` classes -
``Analog``, ``NextEvent``, ``ForcedDetection`` - selected behind a small port,
mirroring how :class:`~uowc.core.ports.OpticalPropertyModel` or
:class:`~uowc.core.ports.PhaseFunction` are pluggable.

That shape does not fit what an estimator actually needs here. Both implemented
estimators are variations on *how a single free-flight segment is scored*, and
scoring needs almost everything the delta-tracking loop already has in scope: the
current position and direction, the medium's field and majorant, the receiver
geometry, the shared RNG stream, and (for next-event) the in-progress
``has_scattered`` state that decides whether the ballistic receiver test still
applies to a given photon this iteration. A port that exposed all of that would be a
large, leaky interface written for exactly one caller - the premature abstraction
CLAUDE.md asks to avoid ("don't design for hypothetical future requirements").

So the two implemented estimators live directly inside
:class:`~uowc.transport.woodcock.WoodcockDeltaTracker`, selected by
:attr:`~uowc.core.config.SamplingConfig.estimator`:

* **``"analog"``** - the unbiased reference: ray/aperture intersection tested on
  every free-flight segment, absorb-or-scatter sampled by the single-scattering
  albedo. See transport.md, *Photon Lifecycle*.
* **``"next_event"``** - local-estimate / next-event estimation: at every real
  collision, an analytic scatter-toward-the-receiver contribution (single-scattering
  albedo x phase function x subtended solid angle x ratio-tracked transmittance) is
  added on top of the same analog walk, restricted to collisions the ballistic test
  has not already covered. See :mod:`uowc.transport.woodcock`'s module docstring for
  the full formula, why it does not double-count against the ballistic test, and why
  it breaks the conservation identity by design; see
  ``tests/unit/transport/test_estimators.py`` for its validation against analog on a
  known case (transport.md's requirement before trusting it).

**``"forced"`` (forced detection) remains unimplemented.** It is a distinct
variance-reduction technique (deterministically forcing the *next* real collision to
occur before the receiver, then compensating with a transmittance weight - useful for
optically thin media where next-event's ratio tracking gains little) and would need
its own validation against analog the same way next-event required. Left for when a
scenario actually needs it, rather than spending validation effort on a technique
transport.md does not currently identify as high-priority.

If a future transport algorithm genuinely needs to share estimator logic with
Woodcock delta tracking, extract a port *then*, informed by what that second engine
actually requires - not speculatively now.
"""

from __future__ import annotations

__all__: list[str] = []
