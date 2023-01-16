"""_summary_
"""
import numpy as np

import jaxraylib as jxrl
from jaxraylib.cross_sections import CS_COMPT, CS_PHOTO, CS_RAYL, CS_ENERGY

from tests.utilities import TestBaseInterpolators


class TestCS_Compt(TestBaseInterpolators):
    """_summary_

    Parameters
    ----------
    TestBaseInterpolators : _type_
        _description_
    """
    def test_spline_extrapolation(self):
        """_summary_
        """
        super().spline_extrapolation(
            CS_COMPT, jxrl.CS_Compt, lambda x: np.exp(x) / 1000
        )


class TestCS_Energy(TestBaseInterpolators):
    """_summary_

    Parameters
    ----------
    TestBaseInterpolators : _type_
        _description_
    """
    def test_spline_extrapolation(self):
        """_summary_
        """
        super().spline_extrapolation(
            CS_ENERGY, jxrl.CS_Energy, lambda x: np.exp(x)
        )


class TestCS_Photo(TestBaseInterpolators):
    """_summary_

    Parameters
    ----------
    TestBaseInterpolators : _type_
        _description_
    """
    def test_spline_extrapolation(self):
        """_summary_
        """
        super().spline_extrapolation(
            CS_PHOTO, jxrl.CS_Photo, lambda x: np.exp(x) / 1000
        )


class TestCS_Rayl(TestBaseInterpolators):
    """_summary_

    Parameters
    ----------
    TestBaseInterpolators : _type_
        _description_
    """
    def test_spline_extrapolation(self):
        """_summary_
        """
        super().spline_extrapolation(
            CS_RAYL, jxrl.CS_Rayl, lambda x: np.exp(x) / 1000
        )


class TestCS_Total(TestBaseInterpolators):
    """_summary_

    Parameters
    ----------
    TestBaseInterpolators : _type_
        _description_
    """
    def test_spline_extrapolation(self):
        """_summary_
        """
        super().spline_extrapolation(
            (CS_COMPT, CS_PHOTO, CS_RAYL),
            jxrl.CS_Total,
            lambda x: np.exp(x) / 1000,
        )
