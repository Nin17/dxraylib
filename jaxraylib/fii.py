"""
Anomalous Scattering Factor Fii
"""

from __future__ import annotations
import os
from typing import overload

from .config import jit, jit_kwargs, xp, NDArray
from ._splint import _splint
from ._utilities import raise_errors, wrapped_partial, output_type

DIRPATH = os.path.dirname(__file__)
FII_PATH = os.path.join(DIRPATH, "data/fii.npy")
FII = xp.load(FII_PATH)


# TODO what is going on with numba njit and this???
# FIXME


@wrapped_partial(jit, **jit_kwargs)
def _Fii(Z: int | NDArray, E: float | NDArray) -> tuple[NDArray, bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to FII[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z <= FII.shape[0]) & (E > 0),
        _splint(FII[Z[0] - 1], E),
        xp.nan,
    )
    return output, xp.isnan(output).any()


@overload
def Fii(Z: int, E: float) -> float:
    ...


@overload
def Fii(Z: NDArray, E: NDArray) -> NDArray:
    ...


@output_type
@raise_errors(
    f"Z out of range: 1 to {FII.shape[0]} | Energy must be strictly positive"
)
def Fii(Z, E):
    """
    Anomalous Scattering Factor Fii

    Parameters
    ----------
    Z : int | Array
        atomic number
    E : float | Array
        Energy (keV)

    Returns
    -------
    float | Array
        Anomalous Scattering Factor Fii

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fii(Z, E)
