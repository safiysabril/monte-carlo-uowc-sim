"""Path loss vs range matches published Monte Carlo within tolerance.

Not implemented - and deliberately not stubbed silently, so the gap stays visible
rather than reading as "zero tests, nothing to check here."

Reproducing a specific published UOWC path-loss-vs-range curve requires matching the
source paper's *full* simulation setup, not just its headline IOPs. For the one
candidate this repository has actually inspected (Geldard, Thompson & Popoola 2020,
arXiv:2008.01152 - see ``test_iop_reference.py`` for the IOPs it does supply), the
scattering mixture is a two-population split (sea water Mie scattering + particle
Henyey-Greenstein, their Eq. 7-9) with the *per-population* coefficients ``b_sw`` and
``b_p`` used only implicitly ("in line with [Eq. 5]", deferring to other cited
papers for the split) - the paper's own text does not restate the numeric split
alongside the total ``b_Petzold`` values it does give. Guessing that split, the
receiver FOV (not stated in the excerpt available), or the source's exact angular
profile to force a match would be fabricating agreement, not verifying it
(scientific-modelling.md's standard for citing a source).

To implement this properly: find a source that publishes path-loss-vs-range together
with *every* parameter needed to reconstruct its :class:`~uowc.experiments.ExperimentConfig`
(full scattering mixture per population, source/receiver geometry, boundary
conditions if any) - then build that config here and compare directly.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.validation


def test_literature_pathloss_reference_not_yet_available() -> None:
    pytest.skip(
        "no source found with a complete, reproducible simulation setup alongside "
        "its path-loss-vs-range results - see module docstring"
    )
