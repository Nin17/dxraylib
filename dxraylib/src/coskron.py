"""Coster-Kronig transition probabilities."""

from __future__ import annotations

__all__ = ["CosKronTransProb"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray


COSKRON_DATA = _load("coskron")


def CosKronTransProb(Z: NDArray[integer], trans: NDArray[integer]) -> NDArray[floating]:
    """Coster-Kronig transition probability.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    trans : NDArray[integer]
        Coster-Kronig transition macro

    Returns
    -------
    NDArray[floating]
        Coster-Kronig transition probability

    """
    xp = array_namespace(Z, trans)
    data = xp.asarray(COSKRON_DATA)
    return index2d(data, Z - 1, trans, xp=xp)
