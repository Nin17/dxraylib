"""Anomalous Scattering Factor Δf''."""

from __future__ import annotations

__all__ = ["Fii"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._interpolate import interpolate1d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

FII_DATA = _load("fii")


def Fii(Z: NDArray[integer], E: NDArray[floating]) -> NDArray[floating]:
    """Anomalous scattering factor Δf''.

    Parameters
    ----------
    Z : NDArray[floating]
        atomic number
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        anomalous scattering factor Δf''

    """
    xp = array_namespace(Z, E)
    data = xp.asarray(FII_DATA)
    return interpolate1d(data, Z, E, E, xp=xp)
