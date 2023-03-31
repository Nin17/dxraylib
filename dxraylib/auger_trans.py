"""
Auger rates and yields.
"""

from __future__ import annotations

from ._indexors import _index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_AR = _load("auger_rates")
_AY = _load("auger_yields")


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def AugerRate(Z: ArrayLike, auger_trans: ArrayLike) -> Array:
    """
    Non-radiative rate.

    Parameters
    ----------
    Z : array_like
        atomic number
    auger_trans : array_like
        Auger-type macro corresponding with the electrons involved

    Returns
    -------
    array
        non-radiative rate
    """
    return _index2d(_AR, Z - 6, auger_trans)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def AugerYield(Z: ArrayLike, shell: ArrayLike) -> Array:
    """
    Auger yield.

    Parameters
    ----------
    Z : array_like
        atomic number
    shell : array_like
        shell-type macro

    Returns
    -------
    array
        auger yield
    """
    return _index2d(_AY, Z - 3, shell)
