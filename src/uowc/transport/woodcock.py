"""Woodcock delta-tracking transport engine.

Implements :class:`~uowc.core.ports.TransportEngine`. Propagates photons through *any*
:class:`~uowc.core.ports.Medium` by null-collision (delta) tracking: free paths are
sampled with the medium's majorant ``c_max`` and a candidate collision is accepted as
real with probability ``c(x) / c_max`` (otherwise it is a null collision and flight
continues). The same engine therefore serves Scenario I/II/III - only the injected
medium differs.

Dependencies: this module imports only :mod:`uowc.core` (the ``Medium`` port and the
core value objects) and numpy. It is independent of any optical-property model,
medium implementation, environmental effect, or metric.

Scope of this version:
    * analog estimator (the unbiased verification reference);
    * next-event estimation ("local estimate"): a deterministic scatter-toward-the-
      receiver contribution at every real collision, weighted by the single-
      scattering albedo, the local phase function, the receiver's subtended solid
      angle, and the connecting ray's transmittance (via ratio tracking) -
      transport.md's first planned variance-reduction extension. Must (and is)
      validated against the analog reference on a known case
      (``tests/unit/transport/test_estimators.py``);
    * straight-line free flight (refractive ray-bending via the refractive-index
      gradient is a planned extension; arrival time already uses the local index);
    * a flat-disk receiver with aperture radius and field-of-view acceptance.

Next-event estimation and the conservation identity
-----------------------------------------------------
Under the analog estimator, every launched photon's fate is exactly one of
detected / absorbed / escaped / killed, so ``detected_weight + absorbed_weight +
escaped_weight + killed_weight == launched`` exactly (transport.md's conservation
verification hook). Next-event estimation adds a *second*, independent estimator of
detected power on top of that same walk - a deterministic contribution extracted at
every real collision, without removing weight from the walk's own conservation
budget. That contribution is real and unbiased, but it is not accounted for anywhere
else in the ledger, so the conservation identity is **not** expected to hold, and
does not, under ``estimator="next_event"``; it remains exact for ``"analog"``. This
is why the two estimators are verified differently: analog by conservation, next-event
by agreement with analog on a known case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from uowc.core import (
    DetectedPhotons,
    Receiver,
    SamplingConfig,
    Source,
    Tallies,
    TransportOutput,
)
from uowc.core.ports import Medium, OpticalField, Rng
from uowc.core.units import FloatArray, IntArray
from uowc.transport.tallies import build_requested_tallies

__all__ = ["WoodcockDeltaTracker"]

#: Speed of light in vacuum [m/s], for optical-path arrival time.
_C_VACUUM = 299_792_458.0

#: Hard cap on delta-tracking iterations per chunk (guards against infinite loops).
_MAX_ITERATIONS = 100_000

_EMPTY_F = np.empty(0)
_EMPTY_I = np.empty(0, dtype=np.int64)
_EMPTY_POS = np.empty((0, 3))


@dataclass
class _ChunkResult:
    """One chunk's contribution to the run, named instead of positional - this
    function's output grew past what a plain tuple can hold safely (next-event
    contributions, absorption positions) without index-transcription mistakes."""

    arrival_time_s: FloatArray = dc_field(default_factory=lambda: _EMPTY_F)
    path_length_m: FloatArray = dc_field(default_factory=lambda: _EMPTY_F)
    weight: FloatArray = dc_field(default_factory=lambda: _EMPTY_F)
    n_scatters: IntArray = dc_field(default_factory=lambda: _EMPTY_I)
    incidence_rad: FloatArray = dc_field(default_factory=lambda: _EMPTY_F)
    absorbed_weight: float = 0.0
    escaped_weight: float = 0.0
    killed_weight: float = 0.0
    absorbed_positions: FloatArray = dc_field(default_factory=lambda: _EMPTY_POS)
    absorbed_position_weights: FloatArray = dc_field(default_factory=lambda: _EMPTY_F)


def _scatter_into_direction(
    directions: FloatArray, cos_theta: FloatArray, phi: FloatArray
) -> FloatArray:
    """Rotate unit ``directions`` by polar angle (cos ``theta``) and azimuth ``phi``.

    Vectorized direction-cosine update (after MCML / PBRT), with the pole special case.
    """
    directions = np.asarray(directions, dtype=np.float64)
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, None))
    dx, dy, dz = directions[..., 0], directions[..., 1], directions[..., 2]
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    denom = np.sqrt(np.clip(1.0 - dz * dz, 1e-300, None))
    ux = sin_theta * (dx * dz * cos_phi - dy * sin_phi) / denom + dx * cos_theta
    uy = sin_theta * (dy * dz * cos_phi + dx * sin_phi) / denom + dy * cos_theta
    uz = -sin_theta * cos_phi * denom + dz * cos_theta

    near_pole = np.abs(dz) > 1.0 - 1e-12
    ux = np.where(near_pole, sin_theta * cos_phi, ux)
    uy = np.where(near_pole, sin_theta * sin_phi, uy)
    uz = np.where(near_pole, np.sign(dz) * cos_theta, uz)

    out = np.stack([ux, uy, uz], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


#: Hard cap on ratio-tracking steps for one transmittance estimate (safety guard only;
#: the expected step count is ~c_max * distance, so this should never be reached for a
#: sanely chosen majorant and a connecting distance within the domain).
_MAX_TRANSMITTANCE_STEPS = 1_000_000


def _ratio_tracking_transmittance(
    field: OpticalField,
    origin: FloatArray,
    direction: FloatArray,
    distance: float,
    c_max: float,
    rng: Rng,
) -> float:
    """Unbiased estimate of the beam transmittance ``exp(-integral c ds)`` from
    ``origin`` to ``origin + distance * direction``, via ratio tracking (Novak et al.):
    sample free flights at the majorant, and at each null-collision point multiply a
    running weight by ``1 - c(x)/c_max`` instead of classifying real-vs-null.

    Used only by the next-event estimator's connecting-ray transmittance; never by the
    main photon random walk (which still classifies real/null collisions as usual).
    """
    weight = 1.0
    travelled = 0.0
    position = np.asarray(origin, dtype=np.float64)
    for _ in range(_MAX_TRANSMITTANCE_STEPS):
        step = -math.log1p(-float(rng.uniform(1)[0])) / c_max
        travelled += step
        if travelled >= distance:
            return weight
        position = position + step * direction
        c_local = float(np.asarray(field.extinction(position[None, :]))[0])
        weight *= 1.0 - c_local / c_max
        if weight <= 0.0:
            return 0.0
    return weight


class WoodcockDeltaTracker:
    """Woodcock delta-tracking engine (analog and next-event estimators)."""

    @property
    def name(self) -> str:
        return "woodcock"

    def run(
        self,
        medium: Medium,
        source: Source,
        receiver: Receiver,
        rng: Rng,
        config: SamplingConfig,
    ) -> TransportOutput:
        if config.estimator not in ("analog", "next_event"):
            raise NotImplementedError(
                "WoodcockDeltaTracker implements only the 'analog' and 'next_event' "
                f"estimators, not {config.estimator!r}"
            )
        next_event = config.estimator == "next_event"

        n_total = int(config.n_photons)
        chunk_size = max(1, int(config.chunk_size))

        want_depth_deposition = "depth_deposition" in config.tallies

        chunks: list[_ChunkResult] = []
        start = 0
        chunk_index = 0
        while start < n_total:
            n_chunk = min(chunk_size, n_total - start)
            chunks.append(
                self._track_chunk(
                    medium,
                    source,
                    receiver,
                    config,
                    rng.spawn(chunk_index),
                    n_chunk,
                    want_depth_deposition,
                    next_event,
                )
            )
            start += n_chunk
            chunk_index += 1

        photons = DetectedPhotons(
            arrival_time_s=np.concatenate([c.arrival_time_s for c in chunks])
            if chunks
            else _EMPTY_F,
            path_length_m=np.concatenate([c.path_length_m for c in chunks]) if chunks else _EMPTY_F,
            weight=np.concatenate([c.weight for c in chunks]) if chunks else _EMPTY_F,
            n_scatters=np.concatenate([c.n_scatters for c in chunks]) if chunks else _EMPTY_I,
            incidence_rad=np.concatenate([c.incidence_rad for c in chunks]) if chunks else _EMPTY_F,
        )
        absorbed = sum(c.absorbed_weight for c in chunks)
        escaped = sum(c.escaped_weight for c in chunks)
        killed = sum(c.killed_weight for c in chunks)
        tallies = Tallies(
            launched=n_total,
            detected=int(photons.weight.size),
            detected_weight=float(np.sum(photons.weight)),
            absorbed_weight=float(absorbed),
            escaped_weight=float(escaped),
            extra={"killed_weight": float(killed)},
        )
        binned = build_requested_tallies(
            config.tallies,
            photons=photons,
            absorbed_positions=(
                np.concatenate([c.absorbed_positions for c in chunks]) if chunks else _EMPTY_POS
            ),
            absorbed_weights=(
                np.concatenate([c.absorbed_position_weights for c in chunks])
                if chunks
                else _EMPTY_F
            ),
            domain_bounds=medium.domain.bounds(),
        )
        return TransportOutput(photons=photons, tallies=tallies, binned=binned)

    @staticmethod
    def _launch(source: Source, n: int, rng: Rng) -> tuple[FloatArray, FloatArray]:
        position = np.tile(np.asarray(source.position, dtype=np.float64), (n, 1))
        base = np.asarray(source.direction, dtype=np.float64)
        if source.divergence_rad <= 0.0:
            direction = np.tile(base, (n, 1))
        else:
            cos_min = np.cos(source.divergence_rad)
            cos_theta = cos_min + (1.0 - cos_min) * rng.uniform(n)
            phi = 2.0 * np.pi * rng.uniform(n)
            direction = _scatter_into_direction(np.tile(base, (n, 1)), cos_theta, phi)
        return position, direction

    def _track_chunk(
        self,
        medium: Medium,
        source: Source,
        receiver: Receiver,
        config: SamplingConfig,
        rng: Rng,
        n: int,
        want_depth_deposition: bool = False,
        next_event: bool = False,
    ) -> _ChunkResult:
        field = medium.field
        c_max = float(medium.acceleration.majorant(medium.domain.bounds()))

        position, direction = self._launch(source, n, rng)
        time = np.zeros(n)
        path = np.zeros(n)
        n_scatters = np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=bool)
        has_scattered = np.zeros(n, dtype=bool)

        detected_mask = np.zeros(n, dtype=bool)
        det_time = np.zeros(n)
        det_path = np.zeros(n)
        det_incidence = np.zeros(n)
        absorbed = escaped = killed = 0.0
        absorbed_positions: list[FloatArray] = []

        # Next-event ("local estimate") contributions: one entry per real collision
        # within the receiver's field of view, in addition to (never overlapping
        # with) the ballistic ray/aperture test below - see the module docstring.
        nee_time: list[float] = []
        nee_path: list[float] = []
        nee_weight: list[float] = []
        nee_scatters: list[int] = []
        nee_incidence: list[float] = []

        normal = np.asarray(receiver.normal, dtype=np.float64)
        centre = np.asarray(receiver.position, dtype=np.float64)
        radius = float(receiver.aperture_radius_m)
        aperture_area = math.pi * radius * radius
        cos_fov = float(np.cos(receiver.fov_rad))

        for _ in range(_MAX_ITERATIONS):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break

            here = position[idx]
            heading = direction[idx]
            step = -np.log1p(-rng.uniform(idx.size)) / c_max
            n_local = np.asarray(field.refractive_index(here), dtype=np.float64)

            # --- detection: ray-disk intersection within the free-flight segment ---
            # Under next-event estimation this ballistic test is restricted to
            # photons that have not yet had a real collision: once a photon
            # scatters, its post-scatter segments are no longer geometrically
            # tested (that would double-count against the deterministic
            # per-collision contribution computed below) - see the module docstring.
            approach = heading @ normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t_hit = np.where(approach < 0.0, ((centre - here) @ normal) / approach, np.inf)
            hit = here + t_hit[:, None] * heading
            radial = np.linalg.norm(hit - centre, axis=1)
            cos_incidence = -approach
            detect = (
                (approach < 0.0)
                & (t_hit > 0.0)
                & (t_hit <= step)
                & (radial <= radius)
                & (cos_incidence >= cos_fov)
            )
            if next_event:
                detect &= ~has_scattered[idx]
            det_global = idx[detect]
            det_time[det_global] = time[idx][detect] + t_hit[detect] * n_local[detect] / _C_VACUUM
            det_path[det_global] = path[idx][detect] + t_hit[detect]
            det_incidence[det_global] = np.arccos(np.clip(cos_incidence[detect], -1.0, 1.0))
            detected_mask[det_global] = True
            alive[det_global] = False

            survivors = ~detect
            sidx = idx[survivors]
            if sidx.size == 0:
                continue
            s_step = step[survivors]
            s_nlocal = n_local[survivors]
            candidate = here[survivors] + s_step[:, None] * heading[survivors]

            inside = np.asarray(medium.domain.contains(candidate), dtype=bool)
            escaped += int(np.count_nonzero(~inside))
            alive[sidx[~inside]] = False

            in_idx = sidx[inside]
            if in_idx.size == 0:
                continue
            cand_in = candidate[inside]
            position[in_idx] = cand_in
            time[in_idx] += s_step[inside] * s_nlocal[inside] / _C_VACUUM
            path[in_idx] += s_step[inside]

            c_local = np.asarray(field.extinction(cand_in), dtype=np.float64)
            real = rng.uniform(in_idx.size) < (c_local / c_max)
            real_idx = in_idx[real]
            if real_idx.size == 0:
                continue

            # --- real collision: absorb vs scatter (analog), then sample direction ---
            u_absorb = rng.uniform(real_idx.size)
            u_pop = rng.uniform(real_idx.size)
            u_cos = rng.uniform(real_idx.size)
            u_phi = rng.uniform(real_idx.size)
            cos_theta = np.empty(real_idx.size)
            scatter = np.zeros(real_idx.size, dtype=bool)
            for j, gi in enumerate(real_idx):
                state = field.local_state(position[gi])
                albedo = float(state.iop.single_scattering_albedo)
                components = state.iop.scattering.components

                if next_event:
                    # Local-estimate contribution: analytic scatter-toward-the-
                    # receiver term, evaluated at *every* real collision regardless
                    # of the random absorb/scatter draw below (transport.md: the
                    # albedo already accounts for the scatter-vs-absorb split
                    # analytically here, so this must not be gated on u_absorb).
                    has_scattered[gi] = True
                    to_receiver = centre - position[gi]
                    r = float(np.linalg.norm(to_receiver))
                    if r > 0.0:
                        dir_to_receiver = to_receiver / r
                        cos_incidence_nee = float(-(dir_to_receiver @ normal))
                        if cos_incidence_nee >= cos_fov:
                            cos_theta_nee = float(
                                np.clip(direction[gi] @ dir_to_receiver, -1.0, 1.0)
                            )
                            b_total = sum(float(np.asarray(c.coefficient)) for c in components)
                            phase_value = 0.0
                            if b_total > 0.0:
                                phase_value = sum(
                                    float(np.asarray(c.coefficient))
                                    / b_total
                                    * float(np.asarray(c.phase.value(np.array([cos_theta_nee])))[0])
                                    for c in components
                                )
                            solid_angle = aperture_area * cos_incidence_nee / (r * r)
                            transmittance = _ratio_tracking_transmittance(
                                field, position[gi], dir_to_receiver, r, c_max, rng
                            )
                            contribution = albedo * phase_value * solid_angle * transmittance
                            if contribution > 0.0:
                                nee_time.append(
                                    time[gi]
                                    + r
                                    * float(
                                        np.asarray(field.refractive_index(position[gi][None, :]))[0]
                                    )
                                    / _C_VACUUM
                                )
                                nee_path.append(path[gi] + r)
                                nee_weight.append(contribution)
                                nee_scatters.append(int(n_scatters[gi]) + 1)
                                nee_incidence.append(
                                    float(np.arccos(np.clip(cos_incidence_nee, -1.0, 1.0)))
                                )

                if u_absorb[j] >= albedo:
                    alive[gi] = False
                    absorbed += 1.0
                    if want_depth_deposition:
                        absorbed_positions.append(position[gi].copy())
                    continue
                scatter[j] = True
                coeffs = np.array([float(np.asarray(c.coefficient)) for c in components])
                cumulative = np.cumsum(coeffs)
                pick = int(np.searchsorted(cumulative, u_pop[j] * cumulative[-1]))
                pick = min(pick, len(components) - 1)
                cos_theta[j] = float(
                    np.asarray(components[pick].phase.sample_cos_theta(np.array([u_cos[j]])))[0]
                )

            scatter_idx = real_idx[scatter]
            if scatter_idx.size:
                direction[scatter_idx] = _scatter_into_direction(
                    direction[scatter_idx], cos_theta[scatter], 2.0 * np.pi * u_phi[scatter]
                )
                n_scatters[scatter_idx] += 1
                over = n_scatters[scatter_idx] >= config.max_scatter_events
                killed += int(np.count_nonzero(over))
                alive[scatter_idx[over]] = False

        killed += float(np.count_nonzero(alive))  # any still-live photons hit the iteration cap

        keep = detected_mask
        n_ballistic = int(keep.sum())
        return _ChunkResult(
            arrival_time_s=np.concatenate([det_time[keep], np.asarray(nee_time)]),
            path_length_m=np.concatenate([det_path[keep], np.asarray(nee_path)]),
            weight=np.concatenate([np.ones(n_ballistic), np.asarray(nee_weight)]),
            n_scatters=np.concatenate([n_scatters[keep], np.asarray(nee_scatters, dtype=np.int64)]),
            incidence_rad=np.concatenate([det_incidence[keep], np.asarray(nee_incidence)]),
            absorbed_weight=absorbed,
            escaped_weight=escaped,
            killed_weight=killed,
            absorbed_positions=(np.stack(absorbed_positions) if absorbed_positions else _EMPTY_POS),
            absorbed_position_weights=np.ones(len(absorbed_positions)),
        )
