"""
Anomalous Scattering Factor Fi
"""

from __future__ import annotations
import os
from typing import overload

from .config import jit, jit_kwargs, xp, NDArray
from ._splint import _splint
from ._utilities import value_error, wrapped_partial, output_type

DIRPATH = os.path.dirname(__file__)
FI_PATH = os.path.join(DIRPATH, "data/fi.npy")
FI = xp.load(FI_PATH)


# TODO what is going on with numba njit and this???
# FIXME


@wrapped_partial(jit, **jit_kwargs)
def _Fi(Z: int | NDArray, E: float | NDArray) -> tuple[NDArray, bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to FI[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z <= FI.shape[0]) & (E > 0),
        _splint(FI[Z[0] - 1], E),
        xp.nan,
    )
    return output, xp.isnan(output).any()


@overload
def Fi(Z: int, E: float) -> float:
    ...


@overload
def Fi(Z: NDArray, E: NDArray) -> NDArray:
    ...


@output_type
@value_error(
    f"Z out of range: 1 to {FI.shape[0]} | Energy must be strictly positive"
)
def Fi(Z, E):
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
    float | Array
        Anomalous Scattering Factor Fi

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fi(Z, E)
