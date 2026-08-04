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
    * straight-line free flight (refractive ray-bending via the refractive-index
      gradient is a planned extension; arrival time already uses the local index);
    * a flat-disk receiver with aperture radius and field-of-view acceptance.
"""
from __future__ import annotations

import numpy as np

from uowc.core import (
    DetectedPhotons,
    Receiver,
    SamplingConfig,
    Source,
    Tallies,
    TransportOutput,
)
from uowc.core.ports import Medium, Rng
from uowc.core.units import FloatArray
from uowc.transport.tallies import build_requested_tallies

__all__ = ["WoodcockDeltaTracker"]

#: Speed of light in vacuum [m/s], for optical-path arrival time.
_C_VACUUM = 299_792_458.0

#: Hard cap on delta-tracking iterations per chunk (guards against infinite loops).
_MAX_ITERATIONS = 100_000


def _scatter_into_direction(directions: FloatArray, cos_theta: FloatArray, phi: FloatArray) -> FloatArray:
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


class WoodcockDeltaTracker:
    """Woodcock delta-tracking engine (analog estimator)."""

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
        if config.estimator != "analog":
            raise NotImplementedError(
                "WoodcockDeltaTracker currently implements only the 'analog' estimator"
            )

        n_total = int(config.n_photons)
        chunk_size = max(1, int(config.chunk_size))

        want_depth_deposition = "depth_deposition" in config.tallies

        times: list[FloatArray] = []
        paths: list[FloatArray] = []
        weights: list[FloatArray] = []
        scatters: list[FloatArray] = []
        incidences: list[FloatArray] = []
        absorbed_positions: list[FloatArray] = []
        absorbed_weights: list[FloatArray] = []
        detected = absorbed = escaped = killed = 0.0

        start = 0
        chunk_index = 0
        while start < n_total:
            n_chunk = min(chunk_size, n_total - start)
            result = self._track_chunk(
                medium,
                source,
                receiver,
                config,
                rng.spawn(chunk_index),
                n_chunk,
                want_depth_deposition,
            )
            times.append(result[0])
            paths.append(result[1])
            weights.append(result[2])
            scatters.append(result[3])
            incidences.append(result[4])
            detected += result[5]
            absorbed += result[6]
            escaped += result[7]
            killed += result[8]
            absorbed_positions.append(result[9])
            absorbed_weights.append(result[10])
            start += n_chunk
            chunk_index += 1

        photons = DetectedPhotons(
            arrival_time_s=np.concatenate(times) if times else np.empty(0),
            path_length_m=np.concatenate(paths) if paths else np.empty(0),
            weight=np.concatenate(weights) if weights else np.empty(0),
            n_scatters=np.concatenate(scatters) if scatters else np.empty(0, dtype=np.int64),
            incidence_rad=np.concatenate(incidences) if incidences else np.empty(0),
        )
        tallies = Tallies(
            launched=n_total,
            detected=int(detected),
            detected_weight=float(detected),
            absorbed_weight=float(absorbed),
            escaped_weight=float(escaped),
            extra={"killed_weight": float(killed)},
        )
        binned = build_requested_tallies(
            config.tallies,
            photons=photons,
            absorbed_positions=(
                np.concatenate(absorbed_positions) if absorbed_positions else np.empty((0, 3))
            ),
            absorbed_weights=(
                np.concatenate(absorbed_weights) if absorbed_weights else np.empty(0)
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
    ) -> tuple[
        FloatArray, FloatArray, FloatArray, FloatArray, FloatArray,
        float, float, float, float, FloatArray, FloatArray,
    ]:
        field = medium.field
        c_max = float(medium.acceleration.majorant(medium.domain.bounds()))

        position, direction = self._launch(source, n, rng)
        time = np.zeros(n)
        path = np.zeros(n)
        n_scatters = np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=bool)

        detected_mask = np.zeros(n, dtype=bool)
        det_time = np.zeros(n)
        det_path = np.zeros(n)
        det_incidence = np.zeros(n)
        absorbed = escaped = killed = 0.0
        absorbed_positions: list[FloatArray] = []

        normal = np.asarray(receiver.normal, dtype=np.float64)
        centre = np.asarray(receiver.position, dtype=np.float64)
        radius = float(receiver.aperture_radius_m)
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
                if u_absorb[j] >= albedo:
                    alive[gi] = False
                    absorbed += 1.0
                    if want_depth_deposition:
                        absorbed_positions.append(position[gi].copy())
                    continue
                scatter[j] = True
                components = state.iop.scattering.components
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
        positions_out = (
            np.stack(absorbed_positions) if absorbed_positions else np.empty((0, 3))
        )
        weights_out = np.ones(len(absorbed_positions))
        return (
            det_time[keep],
            det_path[keep],
            np.ones(int(keep.sum())),
            n_scatters[keep],
            det_incidence[keep],
            float(int(keep.sum())),
            absorbed,
            escaped,
            killed,
            positions_out,
            weights_out,
        )
