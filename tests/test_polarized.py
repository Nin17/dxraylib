"""_summary_
"""

import unittest

import numpy as np

import xraylib as xrl
import xraylib_np as xrl_np

import jaxraylib as jxrl

from tests.config import ATOL, RTOL


rng = np.random.default_rng()


class TestDCSP_Rayl(unittest.TestCase):
    ...


class TestDCSP_Compt(unittest.TestCase):
    ...


class TestDCSP_KN(unittest.TestCase):
    def test_value_error(self):
        self.assertRaises(
            ValueError, jxrl.DCSP_KN, 0.0, rng.random(), rng.random()
        )


class TestDCSP_Thoms(unittest.TestCase):
    def test_xrl(self):
        theta, phi = rng.random((2,))
        np.testing.assert_allclose(
            jxrl.DCSP_Thoms(theta, phi),
            xrl.DCSP_Thoms(theta, phi),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_xrlnp(self):
        theta, phi = rng.random((2, 100))
        np.testing.assert_allclose(
            jxrl.DCSP_Thoms(theta[:, None], phi[None, :]),
            xrl_np.DCSP_Thoms(theta, phi),
            rtol=RTOL,
            atol=ATOL
        )
