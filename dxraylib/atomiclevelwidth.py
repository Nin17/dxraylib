"""Atomic level widths."""

from __future__ import annotations

__all__: list[str] = ["AtomicLevelWidth"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import float64, floating, integer
    from numpy.typing import NDArray

ATOMICLEVELWIDTH_DATA: NDArray[float64] = _load("atomic_level_width")


def AtomicLevelWidth(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Atomic level width (keV).

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        atomic level width (keV)

    """
    xp: ModuleType = array_namespace(Z, shell)
    data: NDArray[floating] = xp.asarray(ATOMICLEVELWIDTH_DATA)
    return index2d(data, Z - 10, shell)
