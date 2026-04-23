"""Absorption edge energies."""

from __future__ import annotations

__all__ = ["EdgeEnergy"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from dxraylib._load import _load

from ._index import index2d

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

EDGEENERGY_DATA = _load("edges")


def EdgeEnergy(Z: NDArray[integer], shell: NDArray[integer]) -> NDArray[floating]:
    """Absorption edge energy.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell-type macro

    Returns
    -------
    NDArray[floating]
        absorption edge energy
        (keV)

    """
    xp = array_namespace(Z, shell)
    data = xp.asarray(EDGEENERGY_DATA)
    return index2d(data, Z - 1, shell, xp=xp)
