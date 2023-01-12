"""
Element Densities
"""

from __future__ import annotations
import os

from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from ._utilities import wrapped_partial, xrl_xrlnp

DIRPATH = os.path.dirname(__file__)
DEN_PATH = os.path.join(DIRPATH, "data/densities.npy")
DEN = xp.load(DEN_PATH)


@wrapped_partial(jit, **jit_kwargs)
def _ElementDensity(Z: ArrayLike) -> tuple[NDArray[float], bool]:
    """
    Element Density

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    tuple[Array, bool]
        element density
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    output = xp.where((Z >= 1) & (Z <= DEN.shape[0]), DEN[Z - 1], xp.nan)
    return output, xp.isnan(output).any()


# TODO another version with jax jitable decorator
@xrl_xrlnp(f"Z out of range: 1 to {DEN.shape[0]}")
def ElementDensity(Z):
    """
    Element Density

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    Array
        element density

    Raises
    ------
    ValueError
        if atomic number < 1 or > 98

    Notes
    ------
    Z must be <= 1D if jaxraylib.config.jit == numba.njit

    """
    return _ElementDensity(Z)
