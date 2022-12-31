"""_summary_
"""

from __future__ import annotations
import csv

from .config import xp


# TODO type hint output
def _load(path: str, delimiter: str = " ", skiprows: int = 1):
    """_summary_

    Parameters
    ----------
    path : str
        the path to the file
    delimiter : str, optional
        the delimiter in the file, by default " "
    skiprows : int, optional
        the number of rows to skip, by default 1

    Returns
    -------
    xp.ndarray
        An xp.array containing the data padded with nans.
    """
    data = {}
    with open(path, mode="r") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar="|")
        for i in range(skiprows):
            next(reader)
        i = 1
        element: list[list[float]] = [[], [], []]
        for j in reader:
            row = [float(i) for i in j if i]
            if len(row) == 3:
                element = [[*x, y] for x, y in zip(element, row)]
            elif len(row) == 1:
                data[i] = element
                i += 1
                element = [[], [], []]
    data2 = {i: xp.array(j) for i, j in data.items()}
    maximum = max(i.shape[-1] for i in data2.values())
    return xp.array(
        [
            xp.pad(
                i, ((0, 0), (0, maximum - i.shape[-1])), constant_values=xp.nan
            )
            for i in data2.values()
        ]
    )
