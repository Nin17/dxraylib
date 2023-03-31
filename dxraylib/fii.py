"""
Anomalous Scattering Factor Δf''
"""

from __future__ import annotations

from ._interpolators import _interpolate
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_FII = _load("fii")


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def Fii(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Anomalous scattering factor Δf''.

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        anomalous scattering factor Δf''

    """
    return _interpolate(_FII, Z, E, E)
