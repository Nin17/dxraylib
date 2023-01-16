"""_summary_
"""

import numpy as np

from tests.utilities import TestBaseXrlXrlnp

rng = np.random.default_rng()
COMPOUND = "AuAg"


class TestRefractive_Index(TestBaseXrlXrlnp):
    def test_xrl(self):
        super().xrl(COMPOUND, *rng.random((2,)))

    def test_value_error_energy(self):
        super().raise_error(ValueError, COMPOUND, 0.0, rng.random())

    def test_value_error_density(self):
        super().raise_error(ValueError, COMPOUND, rng.random(), 0.0)


class TestRefractive_Index_Im(TestBaseXrlXrlnp):
    def test_xrl(self):
        super().xrl(COMPOUND, *rng.random((2,)))

    def test_value_error_energy(self):
        super().raise_error(ValueError, COMPOUND, 0.0, rng.random())

    def test_value_error_density(self):
        super().raise_error(ValueError, COMPOUND, rng.random(), 0.0)


class TestRefractive_Index_Re(TestBaseXrlXrlnp):
    def test_xrl(self):
        super().xrl(COMPOUND, *rng.random((2,)))

    def test_value_error_energy(self):
        super().raise_error(ValueError, COMPOUND, 0.0, rng.random())

    def test_value_error_density(self):
        super().raise_error(ValueError, COMPOUND, rng.random(), 0.0)
