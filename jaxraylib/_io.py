"""_summary_
"""

import csv


def _load(path: str) -> dict[int, list[list[float]]]:
    """_summary_

    Parameters
    ----------
    path : str
        _description_

    Returns
    -------
    dict[int, list[list[float]]]
        _description_
    """
    data = {}
    with open(path, mode="r") as f:
        reader = csv.reader(f, delimiter=" ", quotechar="|")
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
    return data
