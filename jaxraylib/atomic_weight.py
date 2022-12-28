"""_summary_
"""

import os

import pandas as pd

dirpath = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(
    os.path.join(dirpath, "xraylib", "data", "atomicweight.dat"),
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


del dirpath, df
