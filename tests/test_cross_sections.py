"""
Tests for jaxraylib.CS_Compt, jaxraylib.CS_Energy, jaxraylib.CS_Photo,
jaxraylib.CS_Rayl and jaxraylib.CS_Total
"""

from typing import Callable

import numpy as np

from jaxraylib.cross_sections import CS_COMPT, CS_PHOTO, CS_RAYL, CS_ENERGY

from tests.utilities import CubicInterpolators


def scale(func: Callable, factor: float = 1):
    """
    Decorator which scales nanmin and nanmax properties of cls by:
        func(nanmin) / factor
        func(nanmax) / factor

    Parameters
    ----------
    func : Callable
        Function to apply to nanmin and nanmax
    factor : float, optional
        Division factor to apply, by default 1
    """

    def wrapper(cls):
        @property
        def nanmin(self):
            return func(super(cls, self).nanmin) / factor

        @property
        def nanmax(self):
            return func(super(cls, self).nanmax) / factor

        cls.nanmin = nanmin
        cls.nanmax = nanmax
        return cls

    return wrapper


@scale(np.exp, 1000)
class CS_Compt(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Compt

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = (CS_COMPT,)


class TestCS_Compt(CS_Compt):
    """
    Test class for jaxraylib.CS_Compt

    Parameters
    ----------
    CS_Compt : type
        Base test class for jaxraylib.CS_Compt
    """


@scale(np.exp)
class CS_Energy(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Energy

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = (CS_ENERGY,)


class TestCS_Energy(CS_Energy):
    """
    Test class for jaxraylib.CS_Energy

    Parameters
    ----------
    CS_Energy : type
        Base test class for jaxraylib.CS_Energy
    """


@scale(np.exp, 1000)
class CS_Photo(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Photo

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = (CS_PHOTO,)


class TestCS_Photo(CS_Photo):
    """
    Test class for jaxraylib.CS_Photo

    Parameters
    ----------
    CS_Photo : type
        Base test class for jaxraylib.CS_Photo
    """


@scale(np.exp, 1000)
class CS_Rayl(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Rayl

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = (CS_RAYL,)


class TestCS_Rayl(CS_Rayl):
    """
    Test class for jaxraylib.CS_Rayl

    Parameters
    ----------
    CS_Rayl : type
        Base test class for jaxraylib.CS_Rayl
    """


@scale(np.exp, 1000)
class CS_Total(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Total

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = CS_COMPT, CS_PHOTO, CS_RAYL


class TestCS_Total(CS_Total):
    """
    Test class for jaxraylib.CS_Total

    Parameters
    ----------
    CS_Total : type
        Base test class for jaxraylib.CS_Total
    """
