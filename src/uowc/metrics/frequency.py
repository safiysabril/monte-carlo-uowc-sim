"""Frequency metrics: frequency response H(f) and 3 dB bandwidth (metrics.md).

``H(f)`` is the discrete-time Fourier transform of the *unbinned* weighted impulse
response, evaluated directly from photon arrival times::

    H(f) = sum_i w_i * exp(-j*2*pi*f*t_i) / sum_i w_i           H(0) = 1

It is deliberately **not** computed by histogramming the CIR and transforming the
histogram. Binning at width ``Delta`` convolves the true response with a rectangle of
that width, multiplying ``H(f)`` by ``sinc(f*Delta)``. At ``f = 1/(2*Delta)`` that
factor is ``2/pi ~= 0.637`` - an attenuation of the same order as the 3 dB point being
measured, so the "bandwidth" would become a property of the bin width rather than of
the channel (metrics.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uowc.core.results import Axis, MetricValue, RawResult
from uowc.core.units import ComplexArray, FloatArray

__all__ = ["FrequencyResponse", "Bandwidth3dB"]

#: |H(f)|^2 threshold for the *electrical* 3 dB point (power in an IM/DD photocurrent
#: is proportional to optical power squared, so the half-power point in H is 1/sqrt(2)).
_ELECTRICAL_THRESHOLD = 0.5
#: |H(f)| threshold for the *optical* 3 dB point.
_OPTICAL_THRESHOLD = 0.5

_CONVENTIONS = ("electrical", "optical")


def _frequency_response(
    times_s: FloatArray, weights: FloatArray, freqs_hz: FloatArray
) -> ComplexArray:
    """Direct DTFT of the weighted, unbinned impulse response; NaN if undefined."""
    total = float(np.sum(weights))
    if not (total > 0.0):
        return np.full(freqs_hz.shape, np.nan + 0j, dtype=np.complex128)
    phase = -2.0j * np.pi * np.outer(freqs_hz, times_s)
    weights_c = np.asarray(weights, dtype=np.complex128)
    return (np.exp(phase) @ weights_c) / total


def _moving_average(x: FloatArray, window: int) -> FloatArray:
    """Centered moving average with edge-replicated padding; ``window`` odd, >= 1."""
    if window <= 1:
        return x
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def _find_crossing(freqs_hz: FloatArray, power_ratio: FloatArray, threshold: float) -> float:
    """First frequency (linearly interpolated) at which ``power_ratio`` falls to
    ``threshold``, starting from DC. NaN if the grid never reaches it - the grid was
    too narrow, or the channel has no such rolloff within it."""
    below = power_ratio <= threshold
    idx = np.argmax(below) if np.any(below) else -1
    if idx <= 0:
        return float("nan")
    f0, f1 = freqs_hz[idx - 1], freqs_hz[idx]
    p0, p1 = power_ratio[idx - 1], power_ratio[idx]
    if p0 == p1:
        return float(f0)
    frac = (p0 - threshold) / (p0 - p1)
    return float(f0 + frac * (f1 - f0))


@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    """Complex baseband frequency response ``H(f)``, normalized ``H(0) = 1``.

    Computed directly from arrival times (see module docstring), not from a
    histogrammed CIR. ``frequencies_hz`` is the caller-supplied grid to evaluate on;
    it is recorded via the returned :class:`~uowc.core.results.Axis` so the grid used
    is always traceable alongside the values (metrics.md, data.md).
    """

    frequencies_hz: FloatArray

    def __post_init__(self) -> None:
        freqs = np.asarray(self.frequencies_hz, dtype=np.float64)
        if freqs.ndim != 1 or freqs.size < 1:
            raise ValueError("frequencies_hz must be a non-empty 1-D array")
        if np.any(freqs < 0.0):
            raise ValueError("frequencies_hz must be non-negative (baseband)")
        object.__setattr__(self, "frequencies_hz", freqs)

    @property
    def name(self) -> str:
        return "frequency_response"

    def compute(self, result: RawResult) -> MetricValue:
        photons = result.output.photons
        n = int(photons.weight.size)
        h = _frequency_response(photons.arrival_time_s, photons.weight, self.frequencies_hz)
        return MetricValue(
            name=self.name,
            value=h,
            unit="dimensionless",
            n_samples=n,
            axes=(Axis(name="frequency", values=self.frequencies_hz, unit="Hz"),),
        )


@dataclass(frozen=True, slots=True)
class Bandwidth3dB:
    """3 dB bandwidth of the frequency response.

    Reports the **electrical** 3 dB bandwidth by default (``|H(f)|^2 = 0.5``), the
    quantity that bounds achievable data rate in an IM/DD link; pass
    ``convention="optical"`` for ``|H(f)| = 0.5`` and label it as such wherever it is
    reported (metrics.md) - the two differ, and ``B_electrical < B_optical``.

    ``H(f)`` from a finite photon sample is noisy and not guaranteed monotone, so the
    crossing is located on a moving-average-smoothed ``|H(f)|`` (``smoothing_window``,
    odd, ``1`` = no smoothing) rather than a single raw sample. The frequency grid is
    ``n_points`` points linearly spaced over ``[0, max_frequency_hz]``; both must be
    chosen wide/fine enough to resolve the rolloff and are recorded in run metadata as
    part of this metric's configuration (data.md).

    Undefined (``NaN``) if no photons were detected, or if the response never falls to
    the threshold within ``max_frequency_hz`` - report the grid bound in that case
    rather than treating the channel as unlimited.
    """

    max_frequency_hz: float
    n_points: int = 2048
    convention: str = "electrical"
    smoothing_window: int = 1

    def __post_init__(self) -> None:
        if not (self.max_frequency_hz > 0.0):
            raise ValueError("max_frequency_hz must be positive")
        if self.n_points < 2:
            raise ValueError("n_points must be >= 2")
        if self.convention not in _CONVENTIONS:
            raise ValueError(f"convention must be one of {_CONVENTIONS}")
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")

    @property
    def name(self) -> str:
        return "bandwidth_3db"

    @property
    def _threshold(self) -> float:
        return _ELECTRICAL_THRESHOLD if self.convention == "electrical" else _OPTICAL_THRESHOLD

    @property
    def _power_exponent(self) -> int:
        """1 for |H| directly (optical), 2 for |H|^2 (electrical)."""
        return 2 if self.convention == "electrical" else 1

    def compute(self, result: RawResult) -> MetricValue:
        photons = result.output.photons
        n = int(photons.weight.size)
        freqs = np.linspace(0.0, self.max_frequency_hz, self.n_points)
        h = _frequency_response(photons.arrival_time_s, photons.weight, freqs)

        if not np.isfinite(h[0]):
            return MetricValue(name=self.name, value=float("nan"), unit="Hz", n_samples=n)

        magnitude = np.abs(h)
        power_ratio = magnitude**self._power_exponent
        smoothed = _moving_average(power_ratio, self.smoothing_window)
        bandwidth = _find_crossing(freqs, smoothed, self._threshold)
        return MetricValue(name=self.name, value=bandwidth, unit="Hz", n_samples=n)
