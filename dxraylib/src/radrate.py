"""Fractional Radiative Rate."""

from __future__ import annotations

__all__: list[str] = ["RadRate"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from dxraylib._load import _load

from ._index import index2d

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import float64, floating, integer
    from numpy.typing import NDArray


RADRATE_DATA: NDArray[float64] = _load("rad_rate")


def RadRate(Z: NDArray[integer], line: NDArray[integer]) -> NDArray[floating]:
    # TODO(nin17): FIXME incorrect for negative lines
    """Radiative rate.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    line : NDArray[integer]
        line-type macro

    Returns
    -------
    NDArray[floating]
        radiative rate

    """
    xp: ModuleType = array_namespace(Z, line)
    data: NDArray[floating] = xp.asarray(RADRATE_DATA)
    return index2d(data, Z - 5, line)
