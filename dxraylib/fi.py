"""
Anomalous Scattering Factor Δf'
"""

from __future__ import annotations
import os

from ._interpolators import _interpolate
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_FI_PATH = os.path.join(_DIRPATH, "data/fi.npy")
_FI = xp.load(_FI_PATH)


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
    return _interpolate(_FI, Z, E, E)
