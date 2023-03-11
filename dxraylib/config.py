"""_summary_
"""
# TODO type hints as well when jax.typing is supported
# https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html?highlight=type%20hints#static-type-annotations

import importlib
import logging
import os
import pkgutil
import sys
import tomli_w
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
    
if data.get('double'):
    # TODO option to enable this
    from jax.config import config
    config.update("jax_enable_x64", True)

xp = importlib.import_module(data["xp"])


if "jit" in data and data["jit"]:
    jit = getattr(
        importlib.import_module((split := data["jit"].rsplit(".", 1))[0]),
        split[1],
    )
else:

    def jit(x):
        return x


# TODO ArrayLike instead
if "NDArray" in data:
    NDArray = getattr(
        importlib.import_module((split := data["NDArray"].rsplit(".", 1))[0]),
        split[1],
    )
else:
    NDArray = typing.Any


# # TODO ArrayLike instead
# if "NDArray" in data:
#     NDArray = getattr(
#         importlib.import_module((split := data["NDArray"].rsplit(".", 1))[0]),
#         split[1],
#     )
# else:
#     NDArray = typing.Any

ArrayLike = getattr(importlib.import_module('jax._src.typing'), 'ArrayLike')

jit_kwargs = data.get("jit_kwargs", {})

if "RAISE" in data or "xrl_np" in data:
    raise DeprecationWarning("Get rid of these, do decorators based on args!")
RAISE = data.get("RAISE", False)  # TODO deprecate this
xrl_np = data.get("xrl_np", True)  # TODO deprecate this


def _set_config(
    xp: str = "jax.numpy",
    jit: str = "",
    NDArray: str = "jax._src.typing.ArrayLike",
    **kwargs
):
    """_summary_

    Parameters
    ----------
    xp : str, optional
        _description_, by default "jax.numpy"
    jit : str, optional
        _description_, by default ""
    NDArray : str, optional
        _description_, by default "jax._src.typing.ArrayLike"
    """

    output = tomli_w.dumps({i: j for i, j in vars().items()})
    with open(CONFIG_PATH, "w") as f:
        f.write(output)


def init(
    xp: str = "jax.numpy",
    jit: str = "",
    NDArray: str = "jax._src.typing.ArrayLike",
    **kwargs
):
    """_summary_

    Parameters
    ----------
    xp : str, optional
        _description_, by default "jax.numpy"
    jit : str, optional
        _description_, by default ""
    NDArray : str, optional
        _description_, by default "jax._src.typing.ArrayLike"
    """
    _set_config(**{i: j for i, j in vars().items()})
    # ??? maybe problem del whilst iterating
    for i in pkgutil.walk_packages(
        [PATH], prefix="jaxraylib.", onerror=lambda x: None
    ):
        try:
            del sys.modules[i.name]
        except KeyError as error:
            log.error(repr(error))
            print(i.name)
    importlib.reload(sys.modules["jaxraylib"])
