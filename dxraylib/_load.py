"""
helper function to load the data
"""

# TODO centralize file loading
# TODO use .npz files for mygrad

# TODO document & type hint

import os

import numpy as np

from .config import xp

# _DIRPATH = os.path.dirname(__file__)


_PATH = os.path.join(os.path.dirname(__file__), "data/data.npz")
_DATA = np.load(_PATH)


def _load(file):
    if xp.__name__ == "mygrad":
        return _DATA[file]
    else:
        return xp.asarray(_DATA[file])
    # return (
    #     xp.asarray(_DATA[file])
    #     if xp.__name__ != "mygrad"
    #     else xp.tensor(_DATA[file])
    # )


# def _load(file):
#     # file += ".npy"
#     with np.load(_PATH) as f:
#         return (
#             xp.asarray(f[file])
#             if xp.__name__ != "mygrad"
#             else xp.tensor(f[file])
#         )


# def _load(file):
#     try:
#         path = os.path.join(_DIRPATH, "data", file + ".npy")
#         return xp.load(path)
#     # TODO find specific exceptions
#     except Exception as e:
#         path = os.path.join(_DIRPATH, "data", file + ".npz")
#         return xp.load(path)

# TODO save as a single .npz file
