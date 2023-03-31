"""
Standard Atomic Weight
"""

from __future__ import annotations

from ._indexors import _index1d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_AW = _load("atomic_weight")


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
