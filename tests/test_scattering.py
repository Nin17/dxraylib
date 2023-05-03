"""
Tests for scattering functions
"""

import functools
import typing

import numpy as np
import pytest

from dxraylib._load import load

from tests.config import N
from tests.utilities import Analytic, CubicInterpolators

rng = np.random.default_rng()


def interpolate_analytic(cls):
    @functools.cached_property
    def args(self):
        return super(cls, self).args + tuple(
            (i - j) * rng.random(N) + j for i, j in self.ab
        )

    @functools.cached_property
    def args0(self) -> tuple[typing.Any, ...]:
        """_summary_

        Returns
        -------
        tuple[typing.Any, ...]
            _description_
        """
        return tuple(
            i if isinstance(i, str) else rng.choice(i) for i in self.args
        )

    cls.args = args
    cls.args.__set_name__(cls, "args")
    cls.args0 = args0
    cls.args0.__set_name__(cls, "args0")
    return cls


class FF_Rayl(CubicInterpolators):
    """
    Base test class for dxraylib.FF_Rayl
    """

    data = (load("FF"),)
    scale = ((lambda x: x, 1),)


class TestFF_Rayl(FF_Rayl):
    """_summary_"""


class SF_Compt(CubicInterpolators):
    """
    Base test class for dxraylib.SF_Compt
    """

    data = (load("SF"),)
    scale = ((lambda x: x, 1),)


class TestSF_Compt(SF_Compt):
    """_summary_"""


class DCS_Thoms(Analytic):
    """_summary_"""

    ab = ((-2 * np.pi, 2 * np.pi),)


class TestDCS_Thoms(DCS_Thoms):
    """_summary_"""


class DCS_KN(Analytic):
    """_summary_"""

    ab = ((0, 1000), (-2 * np.pi, 2 * np.pi))


class TestDCS_KN(DCS_KN):
    """_summary_"""


# TODO another base class for amalgamation of interpolate and analytic
# @pytest.mark.skip("")
# TODO actually inherit from the correct things
@interpolate_analytic
class DCS_Rayl(SF_Compt, DCS_Thoms):
    """_summary_"""


class TestDCS_Rayl(DCS_Rayl):
    """_summary_"""


# TODO actually inherit from the correct things
# @interpolate_analytic
@pytest.mark.skip("")
class DCS_Compt(SF_Compt, DCS_KN):
    """_summary_"""

    # @functools.cached_property
    # def args(self):
    #     _args = super().args
    #     return

    # @functools.cached_property
    # def args0(self):
    #     _args0 = super().args0
    #     return


class TestDCS_Compt(DCS_Compt):
    """_summary_"""


class TestMomentTransf(Analytic):
    """_summary_"""

    ab = ((0, 1000), (-2 * np.pi, 2 * np.pi))


class TestCS_KN(Analytic):
    """_summary_"""

    ab = ((0, 1000),)


class TestComptonEnergy(Analytic):
    """_summary_"""

    ab = ((0, 1000), (-2 * np.pi, 2 * np.pi))
