"""helper functions to index 1D & 2D datasets."""

from __future__ import annotations

__all__: list[str] = ["index1d", "index2d"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import integer
    from numpy.typing import NDArray


ONE_D = 1
TWO_D = 2


def index1d(
    data: NDArray,
    a: NDArray[integer],
    /,
    *,
    xp: ModuleType,
) -> NDArray:
    """Index valid integer indices (a) from a 1d data array (data).

    NaN for invalid indices: those larger than or equal to data.size or less
    than 0.

    Output is of shape a.shape.

    Parameters
    ----------
    data : NDArray
        array to index
    a : NDArray[integer]
        integer indices to select
    xp : ModuleType
        array namespace of data & a

    Returns
    -------
    NDArray
        indices (a) indexed from data for valid values, NaN otherwise

    """
    if data.ndim != ONE_D:
        msg = f"data is {data.ndim}D but must be 1D."
        raise ValueError(msg)
    cond = (a >= 0) & (a < data.shape[0])
    output = data[xp.where(cond, a, 0)]
    return xp.where(cond, output, xp.nan)


def index2d(
    data: NDArray,
    a: NDArray[integer],
    b: NDArray[integer],
    /,
    *,
    xp: ModuleType,
) -> NDArray:
    """Index valid integer indices (a & b) from a 2d data array (data).

    NaN for invalid indices: those larger than or equal to the shape of data
    along the respective axes (a -> 0, b -> 1) or less than 0.

    Output is of shape a.shape + b.shape.

    Parameters
    ----------
    data : NDArray
        array to index
    a : NDArray[integer]
        integer indices to select on 0th axis of data
    b : NDArray[integer]
        integer indices to select on 1st axis of data
    xp : ModuleType
        array namespace of data, a & b

    Returns
    -------
    NDArray
        indices (a & b) broadcast and indexed from data for valid values, NaN
        otherwise

    """
    if data.ndim != TWO_D:
        msg = f"data is {data.ndim}D but must be 2D."
        raise ValueError(msg)
    _a = xp.reshape(a, a.shape + (1,) * b.ndim)
    _b = xp.reshape(b, (1,) * a.ndim + b.shape)
    cond_a = (_a >= 0) & (_a < data.shape[0])
    cond_b = (_b >= 0) & (_b < data.shape[1])
    output = data[xp.where(cond_a, _a, 0), xp.where(cond_b, _b, 0)]
    return xp.where(cond_a & cond_b, output, xp.nan)
