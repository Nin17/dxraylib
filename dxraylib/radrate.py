"""
Radiative rates.
"""

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_RAD_PATH = os.path.join(_DIRPATH, "data/rad_rate.npy")
_RAD = xp.load(_RAD_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def RadRate(Z: ArrayLike, line: ArrayLike) -> Array:
    """
    Radiative rate.

    Parameters
    ----------
    Z : array_like
        atomic number
    line : array_like
        line-type macro

    Returns
    -------
    array
        radiative rate
    """
    return _index2d(_RAD, Z - 5, line)
