"""
Fluorescence yield.
"""

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_FY_PATH = os.path.join(_DIRPATH, "data/fluor_yield.npy")
_FY = xp.load(_FY_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def FluorYield(Z: ArrayLike, shell: ArrayLike) -> Array:
    """
    Fluoresence yield.

    Parameters
    ----------
    Z : array_like
        atomic number
    shell : array_like
        shell-type macro

    Returns
    -------
    array
        fluorescence yield
    """
    return _index2d(_FY, Z - 3, shell)
