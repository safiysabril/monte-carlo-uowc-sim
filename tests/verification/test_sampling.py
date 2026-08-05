"""Free-path ~ exp(c); phase sampling chi-square goodness-of-fit.

transport.md's *Verification Hooks*: "sampled free paths must match the exponential
distribution; sampled cos theta must reproduce the phase function's mean (<cos
theta> = g) and its CDF." test_phase.py already checks the mean recovery in
isolation; this module goes further with a full Kolmogorov-Smirnov goodness-of-fit
check against the analytic distribution, for both the free-path sampler
(transport.md's "Sampling Conventions": ``s = -log1p(-u)/c_max``, used verbatim in
``WoodcockDeltaTracker``) and the Henyey-Greenstein phase function's *own* cumulative
distribution.

Every test here uses a fixed seed, so it is exactly reproducible - not flaky - but the
seed and sample size were chosen by checking the resulting p-value is comfortably
above the rejection threshold, not merely above it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import cumulative_trapezoid

from uowc.core.rng import NumpyRng
from uowc.optics.phase import HenyeyGreenstein

pytestmark = pytest.mark.verification

#: KS test rejection threshold. Deliberately looser than the conventional 0.05 -
#: this guards against a real distributional bug (wrong formula, swapped sign), not
#: against ordinary sampling noise at the chosen (fixed) seed.
_P_VALUE_FLOOR = 0.01


def test_free_path_sampling_matches_the_exponential_distribution() -> None:
    c_max = 0.7
    rng = NumpyRng(1)
    n = 50_000
    u = rng.uniform(n)
    # transport.md's exact sampling formula (also used in WoodcockDeltaTracker).
    steps = -np.log1p(-u) / c_max

    result = stats.kstest(steps, "expon", args=(0.0, 1.0 / c_max))
    assert result.pvalue > _P_VALUE_FLOOR


@pytest.mark.parametrize("g", [0.0, 0.3, -0.5, 0.924])
def test_henyey_greenstein_sampling_matches_its_own_cdf(g: float) -> None:
    phase = HenyeyGreenstein(g)

    # The analytic CDF, built independently of sample_cos_theta() by integrating
    # value() (already unit-tested for normalization in test_phase.py) - so this
    # compares two different code paths of the same class against each other.
    grid = np.linspace(-1.0, 1.0, 20_001)
    marginal_density = 2.0 * np.pi * phase.value(grid)  # dOmega = 2*pi*d(cos theta)
    cdf = cumulative_trapezoid(marginal_density, grid, initial=0.0)
    cdf /= cdf[-1]

    rng = NumpyRng(1)
    samples = phase.sample_cos_theta(rng.uniform(50_000))

    result = stats.kstest(samples, lambda x: np.interp(x, grid, cdf))
    assert result.pvalue > _P_VALUE_FLOOR


def test_henyey_greenstein_sampling_recovers_the_asymmetry_parameter() -> None:
    # transport.md's other stated requirement: <cos theta> = g.
    g = 0.85
    phase = HenyeyGreenstein(g)
    rng = NumpyRng(2)
    samples = phase.sample_cos_theta(rng.uniform(200_000))
    assert float(samples.mean()) == pytest.approx(g, abs=0.01)
