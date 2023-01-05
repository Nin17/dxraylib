"""
Anomalous Scattering Factor Fii
"""

from __future__ import annotations
import os

from .config import jit, xp, NDArray
from ._splint import _splint
from ._utilities import raise_errors

DIRPATH = os.path.dirname(__file__)
FII_PATH = os.path.join(DIRPATH, "data/fii.npy")
FII = xp.load(FII_PATH)


# TODO what is going on with numba jit and this???
# FIXME


@jit
def _Fii(Z: int | NDArray, E: float | NDArray) -> tuple[NDArray, bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to FI[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z <= FII.shape[0]), _splint(FII[Z[0] - 1], E), xp.nan
    )
    return output, xp.isnan(output).any()


@raise_errors(
    f"Z out of range: 1 to {FII.shape[0]} | Energy must be strictly positive"
)
def Fii(Z: int | NDArray, E: float | NDArray) -> NDArray:
    """
    Anomalous Scattering Factor Fii

    Parameters
    ----------
    Z : int
        atomic number
    E : float
        Energy (keV)

    Returns
    -------
    Array | float
        Anomalous Scattering Factor Fii

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fii(Z, E)
