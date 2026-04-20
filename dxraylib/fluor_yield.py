"""
Fluorescence yield.
"""

from __future__ import annotations

from ._index import index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_FY = _load("fluor_yield")


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
    return index2d(_FY, Z - 3, shell)
