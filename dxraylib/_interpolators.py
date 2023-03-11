"""
Base function for cubic spline interpolation of data using _splint
"""
# TODO summary
# TODO docstring
# TODO type hinting

from ._splint import _splint
from ._utilities import wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp


@wrapped_partial(jit, **jit_kwargs)
def _args_splint(
    index: ArrayLike, value: ArrayLike, value_scaled: ArrayLike
) -> tuple[NDArray, ...]:
    _index = xp.asarray(index)
    _value = xp.asarray(value)
    _value_scaled = xp.asarray(value_scaled)
    __index = xp.expand_dims(_index, tuple(range(-1, -_value.ndim - 1, -1)))
    __value = xp.expand_dims(_value, tuple(range(_index.ndim)))
    return _index, _value_scaled, __index, __value


@wrapped_partial(jit, **jit_kwargs)
def _interpolate(
    data: NDArray, Z: ArrayLike, E: ArrayLike, E2: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    data : NDArray
        _description_
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_
    E2 : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    z, e, _z, _e = _args_splint(Z, E, E2)
    output = _splint(
        data[xp.where((z >= 1) & (z <= data.shape[0]), z - 1, 0)], e
    )
    output = xp.where(
        (_z >= 1) & (_z <= data.shape[0]) & (_e >= 0), output, xp.nan
    )
    return output


# TODO another implementation with inplace operations
@wrapped_partial(jit, **jit_kwargs)
def _interpolate2(data, Z, E, E2):
    z, e, _z, _e = _args_splint(Z, E, E2)
    ...
