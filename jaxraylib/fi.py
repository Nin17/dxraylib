"""
Anomalous Scattering Factor Fi
"""
# TODO document shapes in docstring
# TODO document error raising xrl xrl_np

from __future__ import annotations
import os

from ._interpolators import _interpolate
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)
_FI_PATH = os.path.join(_DIRPATH, "data/fi.npy")
_FI = xp.load(_FI_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def Fi(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Anomalous Scattering Factor Fi

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        anomalous scattering factor fi

    Raises
    ------
    ValueError
        if Z < 1 or Z > 99
    ValueError
        if E < 0
    ValueError
        if E results in extrapolation
    """
    return _interpolate(_FI, Z, E, E)
