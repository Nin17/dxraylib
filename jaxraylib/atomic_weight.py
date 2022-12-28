"""_summary_
"""

import os

import pandas as pd

df = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__), "xraylib", "data", "atomicweight.dat"
    ),
    delimiter="\t",
    header=None,
)

data = df[1].to_list()


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
    if Z < 1 or Z > 103:
        raise ValueError("Z out of range: 1 to 103")
    return data[Z - 1]


del df
