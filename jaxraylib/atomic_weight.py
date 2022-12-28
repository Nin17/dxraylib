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
df.set_index(0, inplace=True)
data = df.to_dict()[1]


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
    return data[Z]


del df
