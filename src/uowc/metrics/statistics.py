"""Uncertainty helpers shared by metrics."""
from __future__ import annotations

import numpy as np

from uowc.core.units import FloatArray

__all__ = [
    "weighted_mean",
    "weighted_rms_spread",
    "wilson_interval",
    "effective_sample_size",
    "weighted_standard_error",
]


def weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    """Weighted mean; NaN if total weight is zero."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0.0 else float("nan")


def weighted_rms_spread(values: FloatArray, weights: FloatArray) -> float:
    """Weighted RMS spread about the weighted mean; NaN if total weight is zero."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    mean = np.dot(values, weights) / total
    variance = np.dot(weights, (values - mean) ** 2) / total
    return float(np.sqrt(max(variance, 0.0)))


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Binomial standard error and 95% *Wilson score* interval for proportion ``p``.

    Returns ``(se, low, high)``. The Wald interval ``p +- z*se`` is **not** used: UOWC
    capture probabilities are routinely 1e-6 or smaller, and in that regime Wald's
    coverage collapses - at ``p = 0`` it degenerates to ``[0, 0]``, asserting certainty
    from no observations (see metrics.md).

    The Wilson interval inverts the score test and stays well behaved for small ``p``
    and small counts::

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/(4n^2) )

    At ``p = 0`` it yields ``[0, z^2/(n + z^2)]`` (~3.84/n for large ``n``) - the
    non-degenerate upper bound that a non-detection actually licenses, comparable to
    the exact one-sided "rule of three" bound ``3/n``.

    ``se`` remains the ordinary binomial standard error ``sqrt(p(1-p)/n)``; it
    describes the estimator's spread and is reported separately from the interval.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = min(max(p, 0.0), 1.0)
    se = float(np.sqrt(p * (1.0 - p) / n))
    z2_n = z * z / n
    centre = (p + 0.5 * z2_n) / (1.0 + z2_n)
    half = (z / (1.0 + z2_n)) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return se, float(max(0.0, centre - half)), float(min(1.0, centre + half))


def effective_sample_size(weights: FloatArray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``; 0.0 if there is no weight.

    Far below the detected count means a few heavy photons dominate the estimate and
    its interval is optimistic (metrics.md).
    """
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.dot(w, w))
    return float(np.square(w.sum()) / denom) if denom > 0.0 else 0.0


def weighted_standard_error(scores: FloatArray, n_trials: int) -> float:
    """Monte Carlo standard error of the mean of per-trial ``scores``.

    ``scores`` holds the score of every *detected* photon; the ``n_trials - size``
    undetected photons score zero and are accounted for implicitly, so ``n_trials`` is
    the launched count. Variance is taken over weights, not counts::

        SE = sqrt( ( sum w^2 - (sum w)^2 / N ) / ( N (N - 1) ) )

    NaN if fewer than two trials.
    """
    w = np.asarray(scores, dtype=np.float64)
    n = int(n_trials)
    if n < 2:
        return float("nan")
    total = float(w.sum())
    sum_sq = float(np.dot(w, w))
    variance = (sum_sq - total * total / n) / (n * (n - 1))
    return float(np.sqrt(max(variance, 0.0)))
