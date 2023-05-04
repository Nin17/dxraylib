"""
helper functions to index 1D & 2D datasets
"""

from __future__ import annotations

from . import config as cfg


def index1d(data: cfg.Array, a: cfg.Array) -> cfg.Array:
    """
    Index valid integer indices (a) from a 1d data array (data).
    NaN for invalid indices: those larger than or equal to data.size or less
    than 0.

    Output is of shape a.shape.

    Parameters
    ----------
    data : Array
        array to index
    a : Array
        integer indices to select

    Returns
    -------
    Array
        indices (a) indexed from data for valid values, NaN otherwise
    """
    # TODO can probably do this in a uniform way between modules
    assert data.ndim == 1
    condition = (a >= 0) & (a < data.shape[0])
    # if cfg.xp.__name__ != "torch":
    #     condition = (a >= 0) & (a < data.size)
    # else:
    #     condition = (a >= 0) & (a < data.size(dim=0))
    output = data[cfg.xp.where(condition, a, 0)]
    return cfg.xp.where(condition, output, cfg.xp.nan)


def index2d(data: cfg.Array, a: cfg.Array, b: cfg.Array) -> cfg.Array:
    """
    Index valid integer indices (a & b) from a 2d data array (data).
    NaN for invalid indices: those larger than or equal to the shape of data
    along the respective axes (a -> 0, b -> 1) or less than 0.

    Output is of shape a.shape + b.shape.

    Parameters
    ----------
    data : Array
        array to index
    a : Array
        integer indices to select on 0th axis of data
    b : Array
        integer indices to select on 1st axis of data

    Returns
    -------
    Array
        indices (a & b) broadcast and indexed from data for valid values, NaN
        otherwise
    """
    assert data.ndim == 2
    _a = a.reshape(a.shape + (1,) * b.ndim)
    _b = b.reshape((1,) * a.ndim + b.shape)
    condition_a = (_a >= 0) & (_a < data.shape[0])
    condition_b = (_b >= 0) & (_b < data.shape[1])
    output = data[
        cfg.xp.where(condition_a, _a, 0), cfg.xp.where(condition_b, _b, 0)
    ]
    return cfg.xp.where(condition_a & condition_b, output, cfg.xp.nan)
