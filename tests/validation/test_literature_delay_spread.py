"""RMS delay spread vs range vs literature.

Not implemented - and deliberately not stubbed silently, so the gap stays visible
rather than reading as "zero tests, nothing to check here."

Geldard, Thompson & Popoola (2020), arXiv:2008.01152 (see ``test_iop_reference.py``)
reports RMS delay spread only as fitted double-gamma-function (DGF) coefficients
(their Table I: ``h(t) = C1*t*exp(-C2*t) + C3*t*exp(-C4*t)``) and as points read off
a log-scale figure, not as directly stated numeric values. Two ways to use this were
considered and rejected:

* Reading ``D_rms`` off the figure by eye is not a citable number - a plot
  description is not a measurement, and encodes whatever error this session's
  reading of it introduces.
* Integrating their fitted ``h(t)`` analytically to get an exact ``D_rms``, then
  comparing against *our own* simulation, requires reproducing their exact source/
  receiver geometry and full scattering-mixture split - the same missing
  information that blocks ``test_literature_pathloss.py``.

To implement this properly: find a source that tabulates RMS delay spread as an
explicit number (not only a fitted curve or a figure) alongside a complete,
reproducible simulation setup.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.validation


def test_literature_delay_spread_reference_not_yet_available() -> None:
    pytest.skip(
        "no source found with an explicit numeric RMS delay spread alongside a "
        "complete, reproducible simulation setup - see module docstring"
    )
