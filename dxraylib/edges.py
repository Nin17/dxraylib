"""
Absorption edge energies.
"""

from __future__ import annotations
import os

from ._indexors import _index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_ED = _load("edges")


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def EdgeEnergy(Z: ArrayLike, shell: ArrayLike) -> Array:
    """
    Absorption edge energy (keV).

    Parameters
    ----------
    Z : array_like
        atomic number
    shell : array_like
        shell-type macro

    Returns
    -------
    array
        absorption edge energy (keV)
    """
    return _index2d(_ED, Z - 1, shell)
