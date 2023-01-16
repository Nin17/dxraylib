"""_summary_
"""

import numpy as np

from jaxraylib.atomicweight import AW

from tests.utilities import TestBaseXrlXrlnp

rng = np.random.default_rng()


class TestAtomicWeight(TestBaseXrlXrlnp):
    def test_xrl(self):
        super().xrl(rng.integers(1, AW.shape[0] + 1))

    def test_xrlnp(self):
        super().xrlnp(rng.integers(0, AW.shape[0] + 10, (100,)))

    def test_type_error(self):
        super().raise_error(
            TypeError,
            rng.integers(1, AW.shape[0] + 1).astype(float),
        )

    def test_value_error(self):
        super().raise_error(ValueError, -1)
