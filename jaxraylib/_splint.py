"""
Cubic spline interpolation from x, y & y'' data
"""
# TODO docstring
# TODO compatibility with numba
from __future__ import annotations

from ._utilities import wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp


@wrapped_partial(jit, **jit_kwargs)
def _splint(data: NDArray, x: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    data : NDArray
        _description_
    x : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    x = xp.asarray(x)
    klo = xp.apply_along_axis(
        lambda y: xp.searchsorted(y, x, side="right") - 1,
        axis=-1,
        arr=data[..., 0, :],
    )  # "right" - 1 # TODO try out different combos

    indices = xp.indices(klo.shape)[: data.ndim - 2]
    h = data[..., 0, :][(*indices, klo + 1)] - data[..., 0, :][(*indices, klo)]
    # ??? should i do this instead? avoid error warnings
    # h = xp.where(~xp.isnan(h), h, xp.nan)
    a = (data[..., 0, :][(*indices, klo + 1)] - x) / h
    b = (x - data[..., 0, :][(*indices, klo)]) / h

    d1_b = data[..., 1, :][(*indices, klo)]
    d1_a = data[..., 1, :][(*indices, klo + 1)]

    d2_b = data[..., 2, :][(*indices, klo)]
    d2_a = data[..., 2, :][(*indices, klo + 1)]
    output = (
        a * d1_b
        + b * d1_a
        + ((a * a * a - a) * d2_b + (b * b * b - b) * d2_a) * (h * h) / 6.0
    )
    # TODO where h == 0
    # return output
    # TODO check this works
    if xp.__name__ == "jax.numpy":
        return xp.where(h != 0, output, (d1_a + d1_b) / 2)
    output = xp.asarray(output)
    output[h == 0] = (d1_a + d1_b) / 2
    return output
