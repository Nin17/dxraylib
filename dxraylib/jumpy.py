"""Jump factors."""

from __future__ import annotations

__all__: list[str] = ["JumpFactor"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index2d
from ._load import _load

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import float64, floating, integer
    from numpy.typing import NDArray


JUMPFACTOR_DATA: NDArray[float64] = _load("jump")


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
    xp: ModuleType = array_namespace(Z, shell)
    data: NDArray[floating] = xp.asarray(JUMPFACTOR_DATA)
    return index2d(data, Z - 1, shell)
