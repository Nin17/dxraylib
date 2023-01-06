"""
Element Densities
"""

from __future__ import annotations
import functools
import os

from .config import jit, jit_kwargs, xp, NDArray
from ._utilities import raise_errors

DIRPATH = os.path.dirname(__file__)
DEN_PATH = os.path.join(DIRPATH, "data/densities.npy")
DEN = xp.load(DEN_PATH)


@functools.partial(jit, **jit_kwargs)
def _ElementDensity(Z: int | NDArray) -> tuple[NDArray, bool]:
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


@raise_errors(f"Z out of range: 1 to {DEN.shape[0]}")
def ElementDensity(Z: int | NDArray) -> NDArray:
    """
    Element Density

    Parameters
    ----------
    Z : int | Array
        atomic number

    Returns
    -------
    Array
        element density

    Raises
    ------
    ValueError
        if atomic number < 1 or > 98
    """
    return _ElementDensity(Z)
