"""
Radiative rates.
"""

from __future__ import annotations

from ._index import index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_RAD = _load("rad_rate")


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
    return index2d(_RAD, Z - 5, line)
