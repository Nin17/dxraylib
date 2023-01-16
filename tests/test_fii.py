import jaxraylib as jxrl
from jaxraylib.fii import FII

from tests.utilities import TestBaseInterpolators


class TestFii(TestBaseInterpolators):
    def test_spline_extrapolation(self):
        return super().spline_extrapolation(FII, jxrl.Fii, lambda x: x)
