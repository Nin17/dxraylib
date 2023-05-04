"""
helper function to load the data
"""

# TODO centralize file loading
# TODO use .npz files for mygrad

# TODO document & type hint

import os
import functools

import numpy as np

# from .config import xp
from . import config as cfg

# _DIRPATH = os.path.dirname(__file__)


_PATH = os.path.join(os.path.dirname(__file__), "data/data.npz")
_DATA = np.load(_PATH)


# TODO remove from here
_DATA = dict(_DATA)

_DATA["compton_profiles"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/compton_profiles.npy"
)

_DATA["compton_profiles_partial"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/compton_profiles_partial.npy"
)

_DATA["electron_config"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/electron_config.npy"
)

_DATA["kissel_pe"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/kissel_pe.npy"
)


# TODO remove to here


# Not functools.cache for compatibility with python 3.8
@functools.lru_cache(maxsize=None)
def _load(file, xp=cfg.xp):
    # TODO new functions here and then incoporate in data.npz

    if xp.__name__ == "mygrad":  # ???
        return _DATA[file]
    if xp.__name__ == "torch":
        # torch.searchsorted doesn't like nans
        # FIXME need to keep nans if it is indexing not interpolating
        # TODO think of condition, maybe axis[-2] of size 3
        return xp.nan_to_num(xp.tensor(_DATA[file]), nan=float("inf"))

    return xp.asarray(_DATA[file])
    # return (
    #     xp.asarray(_DATA[file])
    #     if xp.__name__ != "mygrad"
    #     else xp.tensor(_DATA[file])
    # )


def load(file):
    return _load(file, cfg.xp)


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
