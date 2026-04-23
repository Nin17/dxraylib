"""Jump factors."""

from __future__ import annotations

__all__ = ["JumpFactor"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray


JUMPFACTOR_DATA = _load("jump")


def JumpFactor(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Jump factor.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        jump factor

    """
    xp = array_namespace(Z, shell)
    data = xp.asarray(JUMPFACTOR_DATA)
    return index2d(data, Z - 1, shell, xp=xp)
