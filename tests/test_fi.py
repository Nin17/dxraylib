import jaxraylib as jxrl
from jaxraylib.fi import FI

from tests.utilities import TestBaseInterpolators


class TestFi(TestBaseInterpolators):
    def test_spline_extrapolation(self):
        return super().spline_extrapolation(FI, jxrl.Fi, lambda x: x)
