"""
Standard Atomic Weight
"""

from __future__ import annotations
import os

from ._indexors import _index1d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_AW_PATH = os.path.join(_DIRPATH, "data/atomic_weight.npy")
_AW = xp.load(_AW_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def AtomicWeight(Z: ArrayLike) -> Array:
    """
    Standard atomic weight (g/mol).

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    array
        standard atomic weight (g/mol)
    """
    return _index1d(_AW, Z - 1)
