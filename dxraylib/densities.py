"""
Element Densities
"""

from __future__ import annotations

from ._indexors import _index1d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_DEN = _load("densities")


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def ElementDensity(Z: ArrayLike) -> Array:
    """
    Element density (g/cm³) at room temperature.

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    array
        element density (g/cm³)
    """
    return _index1d(_DEN, Z-1)
