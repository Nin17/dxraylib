"""_summary_
"""

from __future__ import annotations

from jax._src.typing import Array

from .config import jit, xp, NDArray


# TODO docstring
@jit
def _splint(data: NDArray, x: NDArray) -> NDArray:
    """
    Perform cubic spline interpolation to calculate an interpolated y value at
    a given x value.


    Parameters
    ----------
    data : xp.ndarray
        Array with the known x, y, and y'' stacked along
        the first axis.
    x : float
        Target x value at which to calculate the interpolated y value.

    Returns
    -------
    float
        Interpolated y value at the target x value
    """
    # klo = xp.apply_along_axis(lambda a: a.searchsorted(x), axis=-1, arr=data[..., 0, :]) - 1
    # TODO N-D splint function where multiple elements are queried at once
    # klo = xp.apply_along_axis(lambda a: a.searchsorted(x), axis = 1, arr = data[0]) - 1
    klo = xp.searchsorted(data[0], x, 'right') - 1
    # klo = data[0].searchsorted(x) - 1
    # print(data.shape)
    # klo = xp.apply_along_axis(lambda a: xp.searchsorted(a, x), axis=0, arr = data)
    # print(klo)
    # print(data[..., :, :].shape)
    # print(data[..., 0, :][klo])
    
    # output = xp.zeros_like(klo)
    # for index in xp.ndindex(klo.shape[:-x.ndim]):
    #     print(index)
    #     print(output[index].shape)
    #     print(data[(*index, 0)].shape)
    #     output[index] = data[(*index, 0)][klo[index]]
    # print(klo)
    # print(output)
    #     # print(index)
    # print(klo.shape)
    # print('hola', data[..., 0, klo].shape)
    # print(data[..., 0, :].shape)
    # print(data[0, 0][klo[0]])
    # print(data[1, 0][klo[1]])
    # print(xp.take(data[..., 0, :], klo, axis=-2 ))
    # raise Exception

    h = data[..., 0, :][klo + 1] - data[..., 0, :][klo]
    # print('hello', data[..., 0, :][klo])

    # TODO check if this is actually necessary
    # if h == 0.0:
    #     return (data[1][klo] + data[1][klo + 1]) / 2.0

    a = (data[..., 0, :][klo + 1] - x) / h
    b = (x - data[..., 0, :][klo]) / h
    return (
        a * data[..., 1, :][klo]
        + b * data[..., 1, :][klo + 1]
        + (
            (a * a * a - a) * data[..., 2, :][klo]
            + (b * b * b - b) * data[..., 2, :][klo + 1]
        )
        * (h * h)
        / 6.0
    )
