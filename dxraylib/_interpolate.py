"""
Cubic spline interpolation of data using _splint, respecting validity of
atomic numbers and energies.
"""

from __future__ import annotations

from . import _splint, config as cfg


def interpolate1d(
    data: cfg.Array, Z: cfg.Array, E: cfg.Array, E2: cfg.Array
) -> cfg.Array:
    """
    Apply _splint to a dataset at valid atomic numbers and energies, NaN
    otherwise.

    Parameters
    ----------
    data : array
        cubic spline dataset of shape (NZ, 3, NE) where NZ is the number of
        elements Z, NE is the number of energies (possibly scaled)
        x, y and y'' are stacked on the second axis
    Z : array
        atomic number
    E : array
        energy (keV)
    E2 : array
        energy scaled to same units as the energy in y (data[:, 1, :])

    Returns
    -------
    Array
        interpolated value at valid atomic numbers and energies, NaN otherwise
    """
    _z = Z.reshape(Z.shape + (1,) * E.ndim)
    _e = E.reshape((1,) * Z.ndim + E.shape)

    output = _splint.splint(
        data[cfg.xp.where((Z >= 1) & (Z <= data.shape[0]), Z - 1, 0)], E2
    )
    output = cfg.xp.where(
        (_z >= 1) & (_z <= data.shape[0]) & (_e >= 0), output, cfg.xp.nan
    )
    return output


def interpolate2d(
    data: cfg.Array,
    Z: cfg.Array,
    shell: cfg.Array,
    E: cfg.Array,
    E2: cfg.Array,
) -> cfg.Array:
    """_summary_

    Parameters
    ----------
    data : Array
        _description_
    Z : Array
        _description_
    shell : Array
        _description_
    E : Array
        _description_
    E2 : Array
        _description_

    Returns
    -------
    Array
        _description_
    """
    _z = Z.reshape(Z.shape + (1,) * (shell.ndim + E.ndim))
    _shell = shell.reshape((1,) * Z.ndim + shell.shape + (1,) * E.ndim)
    _e = E.reshape((1,) * (Z.ndim + shell.ndim) + E.shape)

    _z2 = Z.reshape(Z.shape + (1,) * shell.ndim)
    _shell2 = shell.reshape((1,) * Z.ndim + shell.shape)

    output = _splint.splint(
        data[
            cfg.xp.where((_z2 >= 1) & (_z2 <= data.shape[0]), _z2 - 1, 0),
            cfg.xp.where(
                (_shell2 >= 0) & (_shell2 < data.shape[1]), _shell2, 0
            ),
        ],
        E2,
    )

    output = cfg.xp.where(
        (_z >= 1)
        & (_z <= data.shape[0])
        & (_shell >= 0)
        & (_shell < data.shape[1])
        & (_e > 0),
        output,
        cfg.xp.nan,
    )

    return output
