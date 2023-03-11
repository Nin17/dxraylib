"""
Element Densities
"""

from __future__ import annotations
import os

from ._indexors import _index1d
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)
_DEN_PATH = os.path.join(_DIRPATH, "data/densities.npy")
_DEN = xp.load(_DEN_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def ElementDensity(Z: ArrayLike) -> NDArray:
    """
    Element Density

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    NDArray
        element density
    """
    return _index1d(_DEN, Z-1)
