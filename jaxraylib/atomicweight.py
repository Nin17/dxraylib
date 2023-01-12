"""
Standard Atomic Weight
"""

from __future__ import annotations
import os

from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from ._utilities import wrapped_partial, xrl_xrlnp

DIRPATH = os.path.dirname(__file__)
AW_PATH = os.path.join(DIRPATH, "data/atomic_weight.npy")
AW = xp.load(AW_PATH)


@wrapped_partial(jit, **jit_kwargs)
def _AtomicWeight(Z: ArrayLike) -> tuple[NDArray[float], bool]:
    """
    Standard atomic weight

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    tuple[Array[float], bool]



    """
    Z = xp.atleast_1d(xp.asarray(Z))
    output = xp.where((Z >= 1) & (Z <= AW.shape[0]), AW[Z - 1], xp.nan)
    return output, xp.isnan(output).any()


# TODO another version with jax jitable decorator
@xrl_xrlnp(f"Z out of range: 1 to {AW.shape[0]}")
def AtomicWeight(Z: ArrayLike) -> NDArray[float]:
    """
    Standard atomic weight

    Parameters
    ----------
    Z : array_like
        atomic number

    Returns
    -------
    Array
        standard atomic weight

    Raises
    ------
    ValueError
        if atomic number < 1 or > 103

    Notes
    ------
    Z must be <= 1D if jaxraylib.config.jit == numba.njit


    """
    return _AtomicWeight(Z)
