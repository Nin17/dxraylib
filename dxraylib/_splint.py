"""
Cubic spline interpolation from x, y & y'' data
"""
# TODO docstring
# TODO compatibility with numba
from __future__ import annotations

from ._utilities import wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp


@wrapped_partial(jit, **jit_kwargs)
def _splint(data: Array, x: ArrayLike) -> Array:
    """_summary_

    Parameters
    ----------
    data : array
        _description_
    x : array_like
        _description_

    Returns
    -------
    array
        _description_
    """
    x = xp.asarray(x)
    klo = xp.apply_along_axis(
        lambda y: xp.searchsorted(y, x, side="right") - 1,
        axis=-1,
        arr=data[..., 0, :],
    )  # "right" - 1 # TODO try out different combos

    indices = xp.indices(klo.shape)[: data.ndim - 2]

    # TODO check this doesn't change the results
    # Prevent out of bounds index error with numpy
    klo_0 = xp.where(klo < data.shape[-1] - 2, klo, 0)
    klo_1 = xp.where(klo < data.shape[-1] - 1, klo + 1, 0)

    h = data[..., 0, :][(*indices, klo_1)] - data[..., 0, :][(*indices, klo_0)]
    # ??? should i do this instead? avoid error warnings
    # h = xp.where(~xp.isnan(h), h, xp.nan)
    a = (data[..., 0, :][(*indices, klo_1)] - x) / h
    b = (x - data[..., 0, :][(*indices, klo_0)]) / h

    d1_b = data[..., 1, :][(*indices, klo_0)]
    d1_a = data[..., 1, :][(*indices, klo_1)]

    d2_b = data[..., 2, :][(*indices, klo_0)]
    d2_a = data[..., 2, :][(*indices, klo_1)]
    output = (
        a * d1_b
        + b * d1_a
        + ((a * a * a - a) * d2_b + (b * b * b - b) * d2_a) * (h * h) / 6.0
    )
    return xp.where(h != 0, output, (d1_a + d1_b) / 2)
