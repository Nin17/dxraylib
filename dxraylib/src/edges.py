"""Absorption edge energies."""

from __future__ import annotations

__all__: list[str] = ["EdgeEnergy"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from dxraylib._load import _load

from ._index import index2d

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import floating, integer
    from numpy.typing import NDArray

EDGEENERGY_DATA: NDArray = _load("edges")


def EdgeEnergy(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Absorption edge energy (keV).

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        absorption edge energy (keV)

    """
    xp: ModuleType = array_namespace(Z, shell)
    data: NDArray[floating] = xp.asarray(EDGEENERGY_DATA)
    return index2d(data, Z - 1, shell)
