"""
useful helper functions
"""

import functools

import jax

from .config import xp


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
        func = jax.tree_util.Partial(func)

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
