"""
Standard Atomic Weight
"""

from __future__ import annotations
import os
from typing import overload

from .config import jit, jit_kwargs, xp, NDArray
from ._utilities import value_error, wrapped_partial, output_type

DIRPATH = os.path.dirname(__file__)
AW_PATH = os.path.join(DIRPATH, "data/atomic_weight.npy")
AW = xp.load(AW_PATH)


@wrapped_partial(jit, **jit_kwargs)
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


@overload
def AtomicWeight(Z: int) -> float:
    ...


@overload
def AtomicWeight(Z: NDArray) -> NDArray:
    ...


@output_type
@value_error(f"Z out of range: 1 to {AW.shape[0]}")
def AtomicWeight(Z):
    """
    Standard atomic weight

    Parameters
    ----------
    Z : int | Array
        atomic number

    Returns
    -------
    float | Array
        standard atomic weight

    Raises
    ------
    ValueError
        if atomic number < 1 or > 103
    """
    return _AtomicWeight(Z)
