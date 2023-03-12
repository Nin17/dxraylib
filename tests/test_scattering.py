"""
Tests for scattering functions
"""

import numpy as np
import pytest

from dxraylib.scattering import _FF_RAYL, _SF_COMPT

from tests.utilities import Analytic, CubicInterpolators


class FF_Rayl(CubicInterpolators):
    """
    Base test class for dxraylib.FF_Rayl
    """

    data = (_FF_RAYL,)
    scale = ((lambda x: x, 1),)


class TestFF_Rayl(FF_Rayl):
    """_summary_"""


class SF_Compt(CubicInterpolators):
    """
    Base test class for dxraylib.SF_Compt
    """

    data = (_SF_COMPT,)
    scale = ((lambda x: x, 1),)


class TestSF_Compt(SF_Compt):
    """_summary_"""


class TestDCS_Thoms(Analytic):
    """_summary_"""

    ab = ((-2 * np.pi, 2 * np.pi),)


class TestDCS_KN(Analytic):
    """_summary_"""

    ab = ((0, 1000), (-2 * np.pi, 2 * np.pi))


# TODO another base class for amalgamation of interpolate and analytic
@pytest.mark.skip("")
class DCS_Rayl(SF_Compt):
    """_summary_"""


class TestDCS_Rayl(DCS_Rayl):
    """_summary_"""


@pytest.mark.skip("")
class DCS_Compt(SF_Compt):
    """_summary_"""


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
