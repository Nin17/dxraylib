"""
Anomalous Scattering Factor Fi
"""

from __future__ import annotations
import functools
import os

from jax._src.typing import Array

from .config import jit, xp
from ._splint import _splint

DIRPATH = os.path.dirname(__file__)
FI_PATH = os.path.join(DIRPATH, "data/fi.npy")
FI = xp.load(FI_PATH)
shape = FI.shape
VALUE_ERROR = f"Z out of range: 1 to {shape}"


@functools.partial(
    jit, **({"static_argnums": (0, 1)} if jit.__name__ == "jit" else {})
)
def _Fi(Z: int, E: float) -> Array | float:
    if Z < 1 or Z > 99:
        raise ValueError(VALUE_ERROR)
    if E < 0:
        raise ValueError("negative energy")
    return _splint(FI[Z - 1], E)


def Fi(Z: int, E: Array | float) -> Array | float:
    """
    Anomalous Scattering Factor Fi

    Parameters
    ----------
    Z : int
        atomic number
    E : float
        Energy (keV)

    Returns
    -------
    Array | float
        Anomalous Scattering Factor Fi

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    """
    return _Fi(Z, E)
