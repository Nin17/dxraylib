"""Cubic spline interpolation from x, y & y'' data."""

# TODO(nin17): vendoring of array_api_compat & array_api_extra
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from array_api_compat import is_jax_namespace, is_torch_namespace
from numpy import ndindex

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import floating, integer
    from numpy.typing import NDArray


def _searchsorted(
    data: NDArray[floating],
    x: NDArray[floating],
    /,
    *,
    xp: ModuleType,
) -> NDArray[integer]:
    """Apply xp.searchsorted(data, x, side="right") - 1 along axis=-1 of data.

    Parameters
    ----------
    data : NDArray[floating]
        array to search
    x : NDArray[floating]
        search values
    xp : ModuleType
        array namespace of data & x

    Returns
    -------
    NDArray[integer]
        found indices

    """

    def fcn(data: NDArray[floating], x: NDArray[floating]) -> NDArray[floating]:
        return xp.searchsorted(data, x, side="right") - 1

    if is_jax_namespace(xp):
        vmap = import_module("jax").vmap
    elif is_torch_namespace(xp):
        vmap = import_module("torch").vmap
    else:
        # apply_along_axis from numpy docs for axis=-1
        at = import_module("array_api_extra").at
        ni: tuple[int, ...] = data.shape[:-1]
        out: NDArray[integer] = xp.empty(ni + x.shape, dtype=int)
        for ii in ndindex(ni):
            out: NDArray[integer] = at(out)[ii].set(fcn(data[ii], x))
        return out
    _fcn = fcn
    for i in range(data.ndim - 1):
        _fcn = vmap(_fcn, (i, None), i)

    return _fcn(data, x)


def _splint(
    data: NDArray[floating],
    x: NDArray[floating],
    /,
    *,
    xp: ModuleType,
) -> NDArray[floating]:
    """Cubic spline interpolation.

    Parameters
    ----------
    data : NDArray[floating]
        data to interpolate
    x : NDArray[floating]
        values to interpolate data
    xp : ModuleType
        array namespace of data & x

    Returns
    -------
    NDArray[floating]
        interpolated values

    """
    klo: NDArray[integer] = _searchsorted(data[..., 0, :], x, xp=xp)

    indices: list[NDArray[integer]] = [
        xp.expand_dims(xp.arange(j), (*range(i), *range(i + 1, klo.ndim)))
        for i, j in enumerate(klo.shape[: data.ndim - 2])
    ]

    # TODO(nin17): check this doesn't change the results
    # Prevent out of bounds index error with numpy
    klo_0: NDArray[integer] = xp.where(klo < data.shape[-1] - 2, klo, 0)
    klo_1: NDArray[integer] = xp.where(klo < data.shape[-1] - 1, klo + 1, 0)

    dklo_0: NDArray[floating] = data[(*indices, slice(None), klo_0)]
    dklo_1: NDArray[floating] = data[(*indices, slice(None), klo_1)]

    dklo_0_0: NDArray[floating] = dklo_0[..., 0]
    dklo_1_0: NDArray[floating] = dklo_1[..., 0]

    h: NDArray[floating] = dklo_1_0 - dklo_0_0
    # ??? h = xp.where(~xp.isnan(h), h, xp.nan) should i do this instead? avoid warnings

    a: NDArray[floating] = (dklo_1_0 - x) / h
    b: NDArray[floating] = (x - dklo_0_0) / h

    d1_a: NDArray[floating] = dklo_1[..., 1]
    d1_b: NDArray[floating] = dklo_0[..., 1]
    d2_a: NDArray[floating] = dklo_1[..., 2]
    d2_b: NDArray[floating] = dklo_0[..., 2]

    output: NDArray[floating] = (
        a * d1_b + b * d1_a + ((a**3 - a) * d2_b + (b**3 - b) * d2_a) * (h**2) / 6.0
    )
    return xp.where(h != 0, output, (d1_a + d1_b) / 2)
