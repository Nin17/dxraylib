"""_summary_
"""

import numpy as np

from jaxraylib.densities import DEN

from tests.utilities import TestBaseXrlXrlnp

rng = np.random.default_rng()


class TestElementDensity(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """
    def test_xrl(self):
        """_summary_
        """
        super().xrl(rng.integers(1, DEN.shape[0] + 1))

    def test_xrlnp(self):
        """_summary_
        """
        super().xrlnp(rng.integers(-10, DEN.shape[0] + 10, (100,)))

    def test_type_error(self):
        """_summary_
        """
        super().raise_error(
            TypeError,
            rng.integers(1, DEN.shape[0] + 1).astype(float),
        )

    def test_value_error(self):
        """_summary_
        """
        super().raise_error(ValueError, -1)
