"""
Cubic spline interpolation of data using _splint, respecting validity of
atomic numbers and energies.
"""

from ._splint import _splint
from ._utilities import wrapped_partial
from .config import Array, jit, jit_kwargs, xp


@wrapped_partial(jit, **jit_kwargs)
def _interpolate(data: Array, Z: Array, E: Array, E2: Array) -> Array:
    """
    Apply _splint to a dataset at valid atomic numbers and energies, NaN
    otherwise.

    Parameters
    ----------
    data : Array
        cubic spline dataset of shape (NZ, 3, NE) where NZ is the number of
        elements Z, NE is the number of energies (possibly scaled)
        x, y and y'' are stacked on the second axis
    Z : Array
        atomic number
    E : Array
        energy (keV)
    E2 : Array
        energy scaled to same units as the energy in y (data[:, 1, :])

    Returns
    -------
    Array
        interpolated value at valid atomic numbers and energies, NaN otherwise
    """
    _z = Z.reshape(Z.shape + (1,) * E.ndim)
    _e = E.reshape((1,) * Z.ndim + E.shape)

    output = _splint(
        data[xp.where((Z >= 1) & (Z <= data.shape[0]), Z - 1, 0)], E2
    )
    output = xp.where(
        (_z >= 1) & (_z <= data.shape[0]) & (_e >= 0), output, xp.nan
    )
    return output
