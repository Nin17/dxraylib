"""
Atomic level widths
"""

from __future__ import annotations

from ._indexors import _index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_ALW = _load("atomic_level_width")


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
