"""_summary_
"""
# TODO docstrings

from dxraylib.auger_trans import _AR, _AY

from tests.utilities import Indexors2D


class TestAugerRate(Indexors2D):
    """ """

    shape = _AR.shape


class TestAugerYield(Indexors2D):
    """ """

    shape = _AY.shape
