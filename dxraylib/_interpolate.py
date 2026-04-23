"""Cubic spline interpolation of data using _splint.

Respects validity of atomic numbers and energies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._splint import _splint

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import floating, integer
    from numpy.typing import NDArray


def interpolate1d(
    data: NDArray[floating],
    Z: NDArray[integer],
    E: NDArray[floating],
    E2: NDArray[floating],
    /,
    *,
    xp: ModuleType,
) -> NDArray[floating]:
    """Apply _splint to a dataset at valid Z & E, NaN otherwise.

    Parameters
    ----------
    data : NDArray[floating]
        cubic spline dataset of shape (NZ, 3, NE) where NZ is the number of
        elements Z, NE is the number of energies (possibly scaled)
        x, y and y'' are stacked on the second axis
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)
    E2 : NDArray[floating]
        energy scaled to same units as the energy in y (data[:, 1, :])
    xp : ModuleType
        array namespace of data, Z, E & E2

    Returns
    -------
    NDArray[floating]
        interpolated value at valid atomic numbers and energies, NaN otherwise

    """  # !!! possibly returns inf as well
    dims = (*range(Z.ndim + E.ndim),)
    e = xp.expand_dims(E, dims[: Z.ndim])
    valid_z = (Z >= 1) & (Z <= data.shape[0])  # noqa: SIM300
    # TODO(nin17): pre-pad arrays to avoid - 1 issue
    output = _splint(data[xp.where(valid_z, Z - 1, 0)], E2, xp=xp)
    valid_z = xp.expand_dims(valid_z, dims[Z.ndim :])
    return xp.where(valid_z & (e >= 0), output, xp.nan)


def interpolate2d(  # noqa: PLR0913
    data: NDArray[floating],
    Z: NDArray[integer],
    shell: NDArray[integer],
    E: NDArray[floating],
    E2: NDArray[floating],
    /,
    *,
    xp: ModuleType,
) -> NDArray[floating]:
    """Apply _splint to dataset at valid Z, shell & E, NaN otherwise.

    Parameters
    ----------
    data : NDArray[floating]
        cubic spline dataset of shape (NZ, NSHELL, 3, NE) where NZ is the number of
        elements Z, NSHELL is the number of shells, NE is the number of energies
        (possibly scaled)
        x, y and y'' are stacked on the third axis
    Z : NDArray[integer]
        atomic number
    shell : NDArray[floating]
        shell number
    E : NDArray[floating]
        energy (keV)
    E2 : NDArray[floating]
        energy scaled to same units as the energy in y (data[:, :, 1, :])
    xp : ModuleType
        array namespace of data, Z, shell, E & E2

    Returns
    -------
    NDArray[floating]
        interpolated value at valid atomic numbers, shells and energies, NaN otherwise

    """  # !!! possibly returns inf as well
    dims = (*range(Z.ndim + shell.ndim + E.ndim),)

    z = xp.expand_dims(Z, dims[Z.ndim : -E.ndim])
    _shell = xp.expand_dims(shell, dims[: Z.ndim])

    valid_z = (z >= 1) & (z <= data.shape[0])
    valid_shell = (_shell >= 0) & (_shell < data.shape[1])
    # TODO(nin17): pre-pad array to avoid - 1
    valid_data = data[xp.where(valid_z, z - 1, 0), xp.where(valid_shell, _shell, 0)]
    output = _splint(valid_data, E2, xp=xp)
    valid_z = xp.expand_dims(valid_z, dims[-E.ndim :])
    valid_shell = xp.expand_dims(valid_shell, dims[-E.ndim :])
    valid_e = xp.expand_dims(E > 0, dims[: -E.ndim])
    return xp.where(valid_z & valid_shell & valid_e, output, xp.nan)
