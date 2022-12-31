"""_summary_
"""

import os

from .config import jit, xp

DIRPATH = os.path.dirname(__file__)

FI_PATH = os.path.join(DIRPATH, "data/atomic_weight.npy")

data = xp.load(FI_PATH)

del DIRPATH, FI_PATH


@jit
def AtomicWeight(Z: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        atomic number

    Returns
    -------
    float
        standard atomic weight
    """
    # if Z < 1 or Z > 103:
    #     raise ValueError("Z out of range: 1 to 103")
    return data[Z - 1]
