"""
Atomic level widths
"""

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_ALW_PATH = os.path.join(_DIRPATH, "data/atomic_level_width.npy")
_ALW = xp.load(_ALW_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def AtomicLevelWidth(Z: ArrayLike, shell: ArrayLike) -> Array:
    """
    Atomic level width (keV).

    Parameters
    ----------
    Z : array_like
        atomic number
    shell : array_like
        shell-type macro

    Returns
    -------
    array
        atomic level width (keV)
    """

    return _index2d(_ALW, Z - 10, shell)
