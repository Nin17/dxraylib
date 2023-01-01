"""
Anomalous Scattering Factor Fii
"""

from __future__ import annotations
import functools
import os

from jax._src.typing import Array

from .config import jit, xp
from ._splint import _splint

DIRPATH = os.path.dirname(__file__)
FII_PATH = os.path.join(DIRPATH, "data/fii.npy")
FII = xp.load(FII_PATH)
shape = FII.shape[0]
VALUE_ERROR = f"Z out of range: 1 to {shape}"


@functools.partial(
    jit, **({"static_argnums": (0,)} if jit.__name__ == "jit" else {})
)
def _Fii(Z: int, E: float) -> Array | float:
    if Z < 1 or Z > 99:
        raise ValueError(VALUE_ERROR)
    if E < 0:
        raise ValueError("negative energy")
    return _splint(FII[Z - 1], E)


def Fii(Z: int, E: float) -> Array | float:
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
