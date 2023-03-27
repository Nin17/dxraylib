"""_summary_
"""
# TODO docstring

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_CK_PATH = os.path.join(_DIRPATH, "data/coskron.npy")
_CK = xp.load(_CK_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CosKronTransProb(Z: ArrayLike, trans: ArrayLike) -> Array:
    """
    Coster-Kronig transition probability.

    Parameters
    ----------
    Z : array_like
        atomic number
    trans : array_like
        Coster-Kronig transition macro

    Returns
    -------
    array
        Coster-Kronig transition probability
    """

    return _index2d(_CK, Z - 1, trans)
