"""Atomic level widths."""

from __future__ import annotations

__all__ = ["AtomicLevelWidth"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

ATOMICLEVELWIDTH_DATA = _load("atomic_level_width")


def AtomicLevelWidth(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Atomic level width.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        atomic level width
        (keV)

    """
    xp = array_namespace(Z, shell)
    data = xp.asarray(ATOMICLEVELWIDTH_DATA)
    return index2d(data, Z - 10, shell, xp=xp)
