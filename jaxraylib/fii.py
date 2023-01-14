"""
Anomalous Scattering Factor Fii
"""

from __future__ import annotations
import os

from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from ._splint import _splint
from ._utilities import wrapped_partial, xrl_xrlnp

DIRPATH = os.path.dirname(__file__)
FII_PATH = os.path.join(DIRPATH, "data/fii.npy")
FII = xp.load(FII_PATH)


@wrapped_partial(jit, **jit_kwargs)
def _Fii(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray[float], bool]:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    tuple[NDArray[float], bool]
        _description_
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to FII[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z <= FII.shape[0]) & (E > 0),
        _splint(FII[Z[0] - 1], E),
        xp.nan,
    )
    return output, xp.isnan(output).any()


# TODO with jax jitable decorator
@xrl_xrlnp(
    f"Z out of range: 1 to {FII.shape[0]} | Energy must be strictly positive"
)
def Fii(Z: ArrayLike, E: ArrayLike) -> NDArray[float]:
    """
    Anomalous Scattering Factor Fii

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        Energy (keV)

    Returns
    -------
    Array
        Anomalous Scattering Factor Fii

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fii(Z, E)
