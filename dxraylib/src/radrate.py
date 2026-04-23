"""Fractional Radiative Rate."""

from __future__ import annotations

__all__ = ["RadRate"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray


RADRATE_DATA = _load("rad_rate")


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
    xp = array_namespace(Z, line)
    data = xp.asarray(RADRATE_DATA)
    return index2d(data, Z - 5, line, xp=xp)
