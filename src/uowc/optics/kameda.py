"""Why there is no Kameda optical-property model here.

Kameda & Matsumura (1998), *J. Oceanogr.* 54:509-516, is frequently cited alongside
Haltrin in UOWC literature, which invites the assumption that it is a second
IOP model (a(lambda), b(lambda) from chlorophyll) - the ``OpticalPropertyModel``
placeholder this module name once implied.

It is not. Kameda & Matsumura's published contribution is the vertical
**chlorophyll-profile** parameterization C(z) - a shifted-Gaussian deep chlorophyll
maximum - not a set of absorption/scattering coefficients. That belongs to the
*medium* layer, not optics, and is implemented as
:class:`uowc.media.profiles.KamedaModel`. A Kameda profile still needs an actual
optical-property model (Haltrin, here) to turn C(z) into IOPs at each depth - see
:class:`uowc.media.inhomogeneous.InhomogeneousMedium`.

No independently published a(lambda)/b(lambda) formulation attributed to Kameda was
found (see scientific-modelling.md, "Established form vs. supplied data"). Inventing
one and shipping it under a real researcher's name would be exactly the unverifiable,
unreferenced physics that document exists to prevent, so this module intentionally
implements nothing. If a genuine Kameda IOP formulation is later located in the
primary literature, implement it here against that citation - not from memory.
"""
from __future__ import annotations

__all__: list[str] = []
