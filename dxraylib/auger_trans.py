"""Fractional non Radiative Rates and Auger Yields."""

from __future__ import annotations

from dxraylib._src._utils import array_namespace

__all__: list[str] = ["AugerRate", "AugerYield"]


from typing import TYPE_CHECKING

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import float64, floating, integer
    from numpy.typing import NDArray

AUGERRATE_DATA: NDArray[float64] = _load("auger_rates")
AUGERYIELD_DATA: NDArray[float64] = _load("auger_yields")


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
    xp: ModuleType = array_namespace(Z, auger_trans)
    data: NDArray[floating] = xp.asarray(AUGERRATE_DATA)
    return index2d(data, Z - 6, auger_trans)


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
    xp: ModuleType = array_namespace(Z, shell)
    data: NDArray[floating] = xp.asarray(AUGERYIELD_DATA)
    return index2d(data, Z - 3, shell)
