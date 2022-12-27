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
    if backend == "numpy":
        config.xp = importlib.import_module("numpy")
    elif backend == "jax":
        config.xp = importlib.import_module("jax.numpy")
    else:
        raise ValueError('backend must be either "numpy" or "jax"!')
    pkgpath = os.path.dirname(__file__)
    for j in pkgutil.iter_modules([pkgpath]):
        if not j.ispkg and j.name != "config":
            # TODO should only need one of these
            try:
                importlib.reload(
                    sys.modules[".".join(["pyxraylib", "pyxraylib", j.name])]
                )
            except KeyError as error:
                log.error(repr(error))
            try:
                importlib.reload(sys.modules[".".join(["pyxraylib", j.name])])
            except KeyError as error:
                log.error(repr(error))
