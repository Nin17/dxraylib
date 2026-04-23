"""Fluorescence yield."""

from __future__ import annotations

__all__: list[str] = ["FluorYield"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from dxraylib._load import _load

from ._index import index2d

if TYPE_CHECKING:
    from numpy import float64, floating, integer
    from numpy.typing import NDArray

FLUORYIELD_DATA: NDArray[float64] = _load("fluor_yield")


def FluorYield(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Fluoresence yield.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        fluorescence yield

    """
    xp = array_namespace(Z, shell)
    data = xp.asarray(FLUORYIELD_DATA)
    return index2d(data, Z - 3, shell, xp=xp)
