"""Frequency response and 3 dB bandwidth.

Uses the closed-form two-impulse channel as the analytic reference throughout: for
two equal-weight arrivals separated by ``dt``,

    H(f)    = 0.5 * (1 + exp(-j*2*pi*f*dt))
    |H(f)|  = |cos(pi*f*dt)|
    |H(f)|^2 = cos^2(pi*f*dt)

so the electrical (|H|^2 = 0.5) and optical (|H| = 0.5) 3 dB points have exact,
independently-derivable values: f = 1/(4*dt) and f = 1/(3*dt) respectively.
"""

from __future__ import annotations

import numpy as np
import pytest

from uowc.core import (
    DetectedPhotons,
    RawResult,
    Receiver,
    RunMetadata,
    SamplingConfig,
    SeedTree,
    Source,
    Tallies,
    TransportOutput,
)
from uowc.core.ports import Metric
from uowc.metrics import Bandwidth3dB, FrequencyResponse

_SOURCE = Source(
    position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], wavelength_nm=500.0, divergence_rad=0.0
)
_RECEIVER = Receiver(
    position=[0.0, 0.0, -1.0], normal=[0.0, 0.0, 1.0], aperture_radius_m=1.0, fov_rad=1.0
)


def _raw(times, weights) -> RawResult:
    times = np.asarray(times, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = times.size
    photons = DetectedPhotons(
        arrival_time_s=times,
        path_length_m=np.zeros(n),
        weight=weights,
        n_scatters=np.zeros(n, dtype=np.int64),
        incidence_rad=np.zeros(n),
    )
    tallies = Tallies(
        launched=max(n, 1),
        detected=n,
        detected_weight=float(weights.sum()),
        absorbed_weight=0.0,
        escaped_weight=0.0,
    )
    meta = RunMetadata(
        scenario="T",
        medium_type="t",
        optical_model="t",
        model_parameters={},
        effects=(),
        effect_parameters=(),
        wavelength_nm=500.0,
        source=_SOURCE,
        receiver=_RECEIVER,
        majorant=1.0,
        sampling=SamplingConfig(n_photons=0, estimator="analog"),
        seed_tree=SeedTree(root_seed=0),
        rng_impl="t",
        timestamp_utc="t",
        code_version="t",
    )
    return RawResult(output=TransportOutput(photons=photons, tallies=tallies), metadata=meta)


_DT = 1.0e-8  # 10 ns impulse separation


def _two_impulse_result() -> RawResult:
    return _raw(times=[0.0, _DT], weights=[0.5, 0.5])


# --- FrequencyResponse ------------------------------------------------------------


def test_frequency_response_conforms_to_metric() -> None:
    assert isinstance(FrequencyResponse(frequencies_hz=np.array([0.0])), Metric)
    assert FrequencyResponse(frequencies_hz=np.array([0.0])).name == "frequency_response"


def test_dc_response_is_unity() -> None:
    freqs = np.array([0.0])
    metric = FrequencyResponse(frequencies_hz=freqs).compute(_two_impulse_result())
    assert metric.value[0] == pytest.approx(1.0 + 0.0j)


def test_matches_two_impulse_closed_form() -> None:
    freqs = np.array([0.0, 1.0e7, 2.5e7, 5.0e7])
    metric = FrequencyResponse(frequencies_hz=freqs).compute(_two_impulse_result())
    expected = 0.5 * (1.0 + np.exp(-2.0j * np.pi * freqs * _DT))
    np.testing.assert_allclose(metric.value, expected, atol=1e-12)
    assert metric.axes[0].name == "frequency"
    assert metric.axes[0].unit == "Hz"
    np.testing.assert_allclose(metric.axes[0].values, freqs)


def test_response_is_nan_with_no_detections() -> None:
    metric = FrequencyResponse(frequencies_hz=np.array([0.0, 1e7])).compute(_raw([], []))
    assert np.all(np.isnan(metric.value))


def test_frequencies_must_be_nonnegative_baseband() -> None:
    with pytest.raises(ValueError):
        FrequencyResponse(frequencies_hz=np.array([-1.0, 0.0]))


def test_frequencies_must_not_be_empty() -> None:
    with pytest.raises(ValueError):
        FrequencyResponse(frequencies_hz=np.array([]))


# --- Bandwidth3dB -------------------------------------------------------------------


def test_bandwidth_conforms_to_metric() -> None:
    assert isinstance(Bandwidth3dB(max_frequency_hz=1e8), Metric)
    assert Bandwidth3dB(max_frequency_hz=1e8).name == "bandwidth_3db"


def test_electrical_bandwidth_matches_closed_form() -> None:
    """|H|^2 = 0.5 at f = 1/(4*dt) for the two-impulse channel."""
    metric = Bandwidth3dB(max_frequency_hz=1e8, n_points=200_001, convention="electrical").compute(
        _two_impulse_result()
    )
    expected = 1.0 / (4.0 * _DT)
    assert metric.value == pytest.approx(expected, rel=1e-4)
    assert metric.unit == "Hz"


def test_optical_bandwidth_matches_closed_form() -> None:
    """|H| = 0.5 at f = 1/(3*dt) for the two-impulse channel."""
    metric = Bandwidth3dB(max_frequency_hz=1e8, n_points=200_001, convention="optical").compute(
        _two_impulse_result()
    )
    expected = 1.0 / (3.0 * _DT)
    assert metric.value == pytest.approx(expected, rel=1e-4)


def test_electrical_bandwidth_is_narrower_than_optical() -> None:
    """B_electrical < B_optical always (metrics.md) - not an artifact of this channel."""
    common = dict(max_frequency_hz=1e8, n_points=200_001)
    b_elec = Bandwidth3dB(**common, convention="electrical").compute(_two_impulse_result()).value
    b_opt = Bandwidth3dB(**common, convention="optical").compute(_two_impulse_result()).value
    assert b_elec < b_opt


def test_bandwidth_is_nan_with_no_detections() -> None:
    metric = Bandwidth3dB(max_frequency_hz=1e8).compute(_raw([], []))
    assert np.isnan(metric.value)


def test_bandwidth_is_nan_when_grid_never_reaches_threshold() -> None:
    """A channel with negligible spread has |H| ~= 1 across a narrow grid: no crossing."""
    result = _raw(times=[0.0, 1e-12], weights=[0.5, 0.5])
    metric = Bandwidth3dB(max_frequency_hz=1.0).compute(result)
    assert np.isnan(metric.value)


def test_smoothing_window_must_be_odd() -> None:
    with pytest.raises(ValueError):
        Bandwidth3dB(max_frequency_hz=1e8, smoothing_window=2)


def test_invalid_convention_rejected() -> None:
    with pytest.raises(ValueError):
        Bandwidth3dB(max_frequency_hz=1e8, convention="power")


def test_nonpositive_max_frequency_rejected() -> None:
    with pytest.raises(ValueError):
        Bandwidth3dB(max_frequency_hz=0.0)


def test_too_few_points_rejected() -> None:
    with pytest.raises(ValueError):
        Bandwidth3dB(max_frequency_hz=1e8, n_points=1)
