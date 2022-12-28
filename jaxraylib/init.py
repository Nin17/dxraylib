"""_summary_
"""

import importlib
import logging
import os
import pkgutil
import sys


from . import config

log = logging.getLogger(__name__)


def init(backend: str = "numpy") -> None:
    """_summary_

    Parameters
    ----------
    backend : str, optional
        The backend to use for the calculations
        either "numpy", "cupy" or "jax", by default "numpy"

    Raises
    ------
    ValueError
        if backend is not "numpy", "cupy" or "jax"
    """
    if backend == "numpy":
        config.xp = importlib.import_module("numpy")
    elif backend == "cupy":
        config.xp = importlib.import_module("cupy.numpy")
    elif backend == "jax":
        config.xp = importlib.import_module("jax.numpy")
    else:
        raise ValueError('backend must be either "numpy" or "jax"!')
    pkgpath = os.path.dirname(__file__)
    mod_name = os.path.basename(pkgpath)
    for j in pkgutil.iter_modules([pkgpath]):
        if not j.ispkg and j.name != "config":
            try:
                importlib.reload(sys.modules[".".join([mod_name, j.name])])
            except KeyError as error:
                log.error(repr(error))
