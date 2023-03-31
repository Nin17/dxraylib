"""
Coster-Kronig transition probabilities.
"""

from __future__ import annotations

from ._indexors import _index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_CK = _load("coskron")


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
