"""
useful helper functions
"""

import functools

import jax

from .config import xp
from .xraylib_nist_compounds import GetCompoundDataNISTByName
from .xraylib_parser import CompoundParser


# TODO type hinting
# TODO no need to pass jit and **jit_kwargs
def wrapped_partial(func, *args, **kwargs):
    """
    functools.partial that updates __doc__ and __name__

    Parameters
    ----------
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # TODO use jax.tree_util.partial instead
    partial_func = functools.partial(func, *args, **kwargs)
    partial_func = functools.update_wrapper(partial_func, func)
    return partial_func


# TODO sensible function names
def asarray(argnums=(), argnames=()):
    def decorator(func):
        func = functools.update_wrapper(jax.tree_util.Partial(func), func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = [
                j if i in argnums else xp.asarray(j)
                for i, j in enumerate(args)
            ]
            kwargs = {
                k: v if k in argnames else xp.asarray(v)
                for k, v in kwargs.items()
            }
            # TODO
            # FIXME to
            # total = sum(i.ndim for i in args if hasattr(i, "ndim"))
            # total += sum(i.ndim for i in kwargs.values() if hasattr(i, "ndim"))
            output = func(*args, **kwargs)
            return output
            if output.shape == ():
                return output
            return xp.nan_to_num(output)

        return wrapper

    return decorator


def _compound_data(compound):
    try:
        compound_dict = CompoundParser(compound)
    except ValueError:
        try:
            compound_dict = GetCompoundDataNISTByName(compound)
        except ValueError as error:
            msg = """
            Compound is not a valid chemical formula and is not
             present in the NIST compound database
            """
            raise ValueError(
                msg.replace("\n", "").replace("  ", "")
            ) from error

    return compound_dict
