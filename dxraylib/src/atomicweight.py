"""Standard Atomic Weight."""

from __future__ import annotations

__all__ = ["AtomicWeight"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from dxraylib._load import _load

from ._index import index1d

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

ATOMICWEIGHT_DATA = _load("atomic_weight")


def AtomicWeight(Z: NDArray[integer]) -> NDArray[floating]:
    """Standard atomic weight (g/mol).

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number

    Returns
    -------
    NDArray[floating]
        standard atomic weight (g/mol)

    """
    xp = array_namespace(Z)
    data = xp.asarray(ATOMICWEIGHT_DATA)
    return index1d(data, Z - 1, xp=xp)
