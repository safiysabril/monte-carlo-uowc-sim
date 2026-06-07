"""Uncertainty helpers shared by metrics."""
from __future__ import annotations

import numpy as np

from uowc.core.units import FloatArray

__all__ = ["weighted_mean", "weighted_rms_spread", "wald_interval"]


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


def wald_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Standard error and (clamped) 95% Wald interval for a proportion ``p`` from ``n``."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
    return se, max(0.0, p - z * se), min(1.0, p + z * se)
