"""helper function to load the data."""

# TODO centralize file loading

# TODO document & type hint
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray

_PATH = Path(__file__).parent.parent / "data/data.npz"
_DATA = np.load(_PATH)


# TODO remove from here
_DATA = dict(_DATA)

_DATA["compton_profiles"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/compton_profiles.npy",
)

_DATA["compton_profiles_partial"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/compton_profiles_partial.npy",
)

_DATA["electron_config"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/electron_config.npy",
)

_DATA["kissel_pe"] = np.load(
    "/Users/chris/Documents/PhD/Simulations/dxraylib69/dxraylib/data/kissel_pe.npy",
)


def _load(file: str) -> NDArray[float64]:
    data = _DATA[file]
    # TODO(nin17): redo files so that it's already padded with inf not nan
    data[np.isnan(data)] = np.inf
    return data
    # return _DATA[file]


# TODO save as a single .npz file
