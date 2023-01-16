"""_summary_
"""


import numpy as np


import jaxraylib as jxrl
from jaxraylib.scattering import FF_RAYL, SF_COMPT

from tests.utilities import TestBaseInterpolators, TestBaseXrlXrlnp

rng = np.random.default_rng()


class TestFF_Rayl(TestBaseInterpolators):
    def test_spline_extrapolation(self):
        super().spline_extrapolation(FF_RAYL, jxrl.FF_Rayl, lambda q: q)


class TestSF_Compt(TestBaseInterpolators):
    def test_spline_extrapolation(self):
        super().spline_extrapolation(SF_COMPT, jxrl.SF_Compt, lambda q: q)


class TestDCS_Thoms(TestBaseXrlXrlnp):
    def test_xrl(self):
        theta = (rng.random() - 0.5) * 4 * np.pi
        super().xrl(theta)

    def test_xrlnp(self):
        theta = rng.random((100,))
        super().xrlnp(theta)


class TestDCS_KN(TestBaseXrlXrlnp):
    def test_value_error(self):
        super().raise_error(ValueError, 0.0, rng.random())

    def test_xrl(self):
        e = rng.random()
        theta = (rng.random() - 0.5) * 4 * np.pi
        super().xrl(e, theta)

    def test_xrlnp(self):
        e = rng.random((100,))
        theta = (rng.random((100,)) - 0.5) * 4 * np.pi
        super().xrlnp(e, theta, jxrl_args=(e[:, None], theta[None, :]))


class TestDCS_Rayl(TestBaseXrlXrlnp):
    def test_xrl(self):
        super().xrl(rng.integers(1, 98), 200*(rng.random()+1), rng.random())


class TestDCS_Compt(TestBaseXrlXrlnp):
    ...


class TestMomentTransf(TestBaseXrlXrlnp):
    def test_value_error(self):
        super().raise_error(ValueError, 0.0, rng.random())

    def test_xrl(self):
        super().xrl(rng.random(), rng.random())

    def test_xrlnp(self):
        e = rng.random((100,))
        theta = rng.random((100,))
        super().xrlnp(e, theta, jxrl_args=(e[:, None], theta[None, :]))


class TestCS_KN(TestBaseXrlXrlnp):
    def test_value_error(self):
        super().raise_error(ValueError, 0.0)

    def test_xrl(self):
        e = np.random.random((1,))[0] * np.random.randint(1, 500)
        super().xrl(e)

    def test_xrl_np(self):
        e = (100 + rng.random((100,))) * np.random.randint(1, 500)
        super().xrlnp(e)


class TestComptonEnergy(TestBaseXrlXrlnp):
    def test_value_error(self):
        super().raise_error(ValueError, 0.0, rng.random())
