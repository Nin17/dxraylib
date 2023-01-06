"""
Standard Atomic Weight
"""

from __future__ import annotations
import functools
import os

from .config import jit, jit_kwargs, xp, NDArray
from ._utilities import raise_errors

DIRPATH = os.path.dirname(__file__)
AW_PATH = os.path.join(DIRPATH, "data/atomic_weight.npy")
AW = xp.load(AW_PATH)


@functools.partial(jit, **jit_kwargs)
def _AtomicWeight(Z: int | NDArray) -> tuple[NDArray, bool]:
    """
    Standard atomic weight

    Parameters
    ----------
    Z : int | Array
        _description_

    Returns
    -------
    tuple[Array, bool]
        _description_
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    output = xp.where((Z >= 1) & (Z <= AW.shape[0]), AW[Z - 1], xp.nan)
    return output, xp.isnan(output).any()


@raise_errors(f"Z out of range: 1 to {AW.shape[0]}")
def AtomicWeight(Z: int | NDArray) -> NDArray:
    """
    Standard atomic weight

    Parameters
    ----------
    Z : int | Array
        atomic number

    Returns
    -------
    Array
        standard atomic weight

    Raises
    ------
    ValueError
        if atomic number < 1 or > 103
    """
    return _AtomicWeight(Z)
