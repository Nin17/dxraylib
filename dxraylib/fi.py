"""
Anomalous Scattering Factor Δf'
"""

from __future__ import annotations

from ._interpolate import interpolate1d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_FI = _load("fi")


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def Fi(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Anomalous scattering factor Δf'.

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        anomalous scattering factor Δf'
    """
    return interpolate1d(_FI, Z, E, E)
