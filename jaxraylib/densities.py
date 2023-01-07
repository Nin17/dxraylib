"""
Element Densities
"""

from __future__ import annotations
import os
from typing import overload

from .config import jit, jit_kwargs, xp, NDArray
from ._utilities import value_error, wrapped_partial, output_type

DIRPATH = os.path.dirname(__file__)
DEN_PATH = os.path.join(DIRPATH, "data/densities.npy")
DEN = xp.load(DEN_PATH)


@wrapped_partial(jit, **jit_kwargs)
def _ElementDensity(Z: int | NDArray[int]) -> tuple[NDArray[float], bool]:
    """
    Element Density

    Parameters
    ----------
    Z : int | Array
        atomic number

    Returns
    -------
    tuple[Array, bool]
        element density
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    output = xp.where((Z >= 1) & (Z <= DEN.shape[0]), DEN[Z - 1], xp.nan)
    return output, xp.isnan(output).any()


@overload
def ElementDensity(Z: int) -> float:
    ...


@overload
def ElementDensity(Z: NDArray[int]) -> NDArray[float]:
    ...


@output_type
@value_error(f"Z out of range: 1 to {DEN.shape[0]}")
def ElementDensity(Z):
    """
    Element Density

    Parameters
    ----------
    Z : int | Array
        atomic number

    Returns
    -------
    float | Array
        element density

    Raises
    ------
    ValueError
        if atomic number < 1 or > 98
    """
    return _ElementDensity(Z)
