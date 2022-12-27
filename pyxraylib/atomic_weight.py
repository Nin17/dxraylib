"""_summary_
"""

import os

import pandas as pd

dirpath = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(
    os.path.join(dirpath, "data", "atomicweight.dat"),
    delimiter="\t",
    header=None,
)
df.set_index(0, inplace=True)
data = df.to_dict()[1]


def atomic_weight(atomic_number: int) -> float:
    """_summary_

    Parameters
    ----------
    z : int
        _description_

    Returns
    -------
    float
        _description_
    """
    return data[atomic_number]
