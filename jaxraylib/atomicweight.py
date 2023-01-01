"""
Standard Atomic Weight
"""

from __future__ import annotations
import os
import functools

# TODO import this from config
from jax._src.typing import Array

from .config import jit, xp

DIRPATH = os.path.dirname(__file__)
AW_PATH = os.path.join(DIRPATH, "data/atomic_weight.npy")
data = xp.load(AW_PATH)
shape = data.shape[0]
VALUE_ERROR = f"Z out of range: 1 to {shape}"


@functools.partial(
    jit, **({"static_argnums": (0,)} if jit.__name__ == "jit" else {})
)
def _AtomicWeight(Z: int) -> float | Array:
    if Z < 1 or Z > shape:
        raise ValueError(VALUE_ERROR)
    return data[Z - 1]


def AtomicWeight(Z: int) -> float | Array:
    """
    Standard atomic weight

    Parameters
    ----------
    Z : int
        atomic number

    Returns
    -------
    float | Array
        standard atomic weight

    Raises
    ------
    ValueError
        if atomic number < 1 or > 103
    """
    return _AtomicWeight(Z)
