"""Element density (g/cm³) at room temperature."""

from __future__ import annotations

__all__: list[str] = ["ElementDensity"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._index import index1d
from ._load import _load

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import float64, floating, integer
    from numpy.typing import NDArray


ELEMENTDENSITY_DATA: NDArray[float64] = _load("densities")


def ElementDensity(Z: NDArray[integer]) -> NDArray[floating]:
    """Element density (g/cm³) at room temperature.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number

    Returns
    -------
    NDArray[floating]
        element density (g/cm³)

    """
    xp: ModuleType = array_namespace(Z)
    data: NDArray[floating] = xp.asarray(ELEMENTDENSITY_DATA)
    return index1d(data, Z - 1)
