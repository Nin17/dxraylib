"""
Anomalous Scattering Factor Fi
"""

from __future__ import annotations
import functools
import os

from .config import jit, jit_kwargs, xp, NDArray
from ._splint import _splint
from ._utilities import raise_errors

DIRPATH = os.path.dirname(__file__)
FI_PATH = os.path.join(DIRPATH, "data/fi.npy")
FI = xp.load(FI_PATH)


# TODO what is going on with numba jit and this???
# FIXME


@functools.partial(jit, **jit_kwargs)
def _Fi(Z: int | NDArray, E: float | NDArray) -> tuple[NDArray, bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to FI[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z <= FI.shape[0]), _splint(FI[Z[0] - 1], E), xp.nan
    )
    return output, xp.isnan(output).any()


@raise_errors(
    f"Z out of range: 1 to {FI.shape[0]} | Energy must be strictly positive"
)
def Fi(Z: int | NDArray, E: float | NDArray) -> NDArray:
    """
    Anomalous Scattering Factor Fi

    Parameters
    ----------
    Z : int | Array
        atomic number
    E : float | Array
        Energy (keV)

    Returns
    -------
    Array
        Anomalous Scattering Factor Fi

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fi(Z, E)
