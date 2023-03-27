"""
helper functions to index 1D & 2D datasets
"""
# TODO type hinting
# TODO docstring

from ._utilities import wrapped_partial
from .config import xp, jit, jit_kwargs


@wrapped_partial(jit, **jit_kwargs)
def _index1d(data, a):
    _a = xp.asarray(a)
    condition = (_a >= 0) & (_a < data.shape[0])
    output = data[xp.where(condition, _a, 0)]
    return xp.where(condition, output, xp.nan)


@wrapped_partial(jit, **jit_kwargs)
def _index2d(data, a, b):
    _a = xp.asarray(a).reshape((*a.shape, *(1,) * b.ndim))
    _b = xp.asarray(b).reshape((*(1,) * a.ndim, *b.shape))
    condition_a = (_a >= 0) & (_a < data.shape[0])
    condition_b = (_b >= 0) & (_b < data.shape[1])
    output = data[xp.where(condition_a, _a, 0), xp.where(condition_b, _b, 0)]
    return xp.where(condition_a & condition_b, output, xp.nan)
