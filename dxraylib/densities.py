"""
Element Densities
"""

from __future__ import annotations
import os

from ._indexors import _index1d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_DEN_PATH = os.path.join(_DIRPATH, "data/densities.npy")
_DEN = xp.load(_DEN_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def ElementDensity(Z: ArrayLike) -> Array:
    """
    Element density (g/cm³) at room temperature.

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    array
        element density (g/cm³)
    """
    return _index1d(_DEN, Z-1)
