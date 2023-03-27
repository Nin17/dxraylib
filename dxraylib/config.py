"""_summary_
"""
# TODO type hints as well when jax.typing is supported
# https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html?highlight=type%20hints#static-type-annotations

import importlib
import logging
import os
import sys
import typing

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

log = logging.getLogger(__name__)

PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PATH, "config.toml")

with open(CONFIG_PATH, "rb") as f:
    data = tomllib.load(f)

if data.get("double"):
    from jax.config import config

    config.update("jax_enable_x64", True)

xp = importlib.import_module(data.get("xp", "jax.numpy"))


if "jit" in data and data["jit"]:
    jit = getattr(
        importlib.import_module((split := data["jit"].rsplit(".", 1))[0]),
        split[1],
    )
else:

    def jit(func, *args, **kwargs):
        """_summary_

        Parameters
        ----------
        func : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return func


# TODO ArrayLike instead
if "Array" in data:
    Array = getattr(
        importlib.import_module((split := data["Array"].rsplit(".", 1))[0]),
        split[1],
    )
else:
    Array = typing.Any



ArrayLike = getattr(importlib.import_module("jax._src.typing"), "ArrayLike")

jit_kwargs = data.get("jit_kwargs", {})
