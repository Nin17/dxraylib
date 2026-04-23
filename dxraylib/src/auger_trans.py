"""Fractional non Radiative Rates and Auger Yields."""

from __future__ import annotations

from dxraylib._src._utils import array_namespace

__all__ = ["AugerRate", "AugerYield"]


from typing import TYPE_CHECKING

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

AUGERRATE_DATA = _load("auger_rates")
AUGERYIELD_DATA = _load("auger_yields")


def AugerRate(Z: NDArray[integer], auger_trans: NDArray[integer]) -> NDArray[floating]:
    """Non-radiative rate.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    auger_trans : NDArray[integer]
        Auger-type macro corresponding with the electrons involved

    Returns
    -------
    NDArray[floating]
        non-radiative rate

    """
    xp = array_namespace(Z, auger_trans)
    data = xp.asarray(AUGERRATE_DATA)
    return index2d(data, Z - 6, auger_trans, xp=xp)


def AugerYield(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Auger yield.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        auger yield

    """
    xp = array_namespace(Z, shell)
    data = xp.asarray(AUGERYIELD_DATA)
    return index2d(data, Z - 3, shell, xp=xp)
