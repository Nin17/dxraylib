"""
Anomalous Scattering Factor Fi
"""

from __future__ import annotations
import os

from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from ._splint import _splint
from ._utilities import wrapped_partial, xrl_xrlnp

DIRPATH = os.path.dirname(__file__)
FI_PATH = os.path.join(DIRPATH, "data/fi.npy")
FI = xp.load(FI_PATH)


# TODO what is going on with numba njit and this???
# FIXME


@wrapped_partial(jit, **jit_kwargs)
def _Fi(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray[float], bool]:
    """
    Anomalous Scattering Factor Fi

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    tuple[NDArray, bool]
        _description_
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to FI[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z <= FI.shape[0]) & (E > 0),
        _splint(FI[Z[0] - 1], E),
        xp.nan,
    )
    return output, xp.isnan(output).any()


# TODO another version with jax jitable decorator
@xrl_xrlnp(
    f"Z out of range: 1 to {FI.shape[0]} | Energy must be strictly positive"
)
def Fi(Z: ArrayLike, E: ArrayLike) -> NDArray[float]:
    """
    Anomalous Scattering Factor Fi

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        Energy (keV)

    Returns
    -------
    NDArray
        Anomalous Scattering Factor Fi

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fi(Z, E)
