"""
Anomalous Scattering Factor Fii
"""
# TODO document shapes in docstring
# TODO document error raising

from __future__ import annotations
import os

from ._interpolators import _interpolate
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)
_FII_PATH = os.path.join(_DIRPATH, "data/fii.npy")
_FII = xp.load(_FII_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def Fii(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Anomalous Scattering Factor Fii

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        anomalous scattering factor fii

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    ValueError
        if E results in extrapolation
    """
    return _interpolate(_FII, Z, E, E)
