"""
useful helper functions
"""

import functools
import itertools

import jax

from .config import xp, RAISE, jit, jit_kwargs, ArrayLike

XRL_NP = False


def xrl_np():
    ...
    # TODO 


def xrl_xrlnp(value_error: str = ""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            xrl = all(isinstance(i, (float, int, str)) for i in itertools.chain(args, kwargs.values()))
            xrl_np = all(isinstance(i, ArrayLike | str) for i in itertools.chain(args, kwargs.values()))
            if XRL_NP:
                # TODO some stuff that does argument broadcasting
                ...
            output, nan = func(*args, **kwargs)
            if nan and xrl:
                raise ValueError(value_error) # TODO from where it occurred
            if xrl:
                return xp.squeeze(output)
            if nan:
                return xp.nan_to_num(output)
            return output
        return wrapper
    return decorator

def error(output, value_error):
    raise ValueError(value_error)


def xrl_xrlnp_jax(value_error: str = ""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            xrl = all(isinstance(i, (float, int, str)) for i in itertools.chain(args, kwargs.values()))
            output, nan = func(*args, **kwargs)
            jax.lax.cond(xrl, )
            jax.lax.cond(nan and xrl, error, )
            return output
        return wrapper
    return decorator



# TODO another version of this that uses jax.lax.cond
# TODO type hints
def output_type(func):
    """
    Convert the output from Array to float if all the
    passed numeric args and kwargs are either int or float

    Parameters
    ----------
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if all(
            isinstance(i, (float, int, str))
            for i in itertools.chain(args, kwargs.values())
        ):
            return float(result)
        return result

    return wrapper


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
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


# TODO another version of this that uses jax.lax.cond
# TODO type hinting
def value_error(message: str = ""):
    """
    Decorator for functions of the type outlined below
    that raises ValueError(message) if config.xrl_np is False,
    or converts all NaNs in output to 0 if config.xrl_np is
    True

    def _func(*args, **kwargs) -> tuple[Array, bool]:
        ...
        return output, nan

    def func(*args, **kwargs):
        return _func(*args, **kwargs)

    Parameters
    ----------
    message : str, optional
        the error message to raise, by default ""
        Only raised if _func returns (..., True)
        and config is set to follow xraylib not xraylib_np.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            output, nan = function(*args, **kwargs)
            if nan:
                if RAISE:
                    # TODO proper traceback to the function
                    raise ValueError(message) from ValueError(
                        function.__name__
                    )
                return xp.nan_to_num(output)
            return output

        return wrapper

    return decorator


# def value_error2(message: str = ""):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             output, nan = func(*args, **kwargs)
#             if all(isinstance(i, (int, float, str)) for i in itertools.chain(args, kwargs.values())):

# jax.lax.cond()


def meh(error):
    raise ValueError("")


# @wrapped_partial(jit, **jit_kwargs)
# def chain(*iterables):
#     for it in iterables:
#         for element in it:
#             yield element
