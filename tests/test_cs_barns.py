"""_summary_
"""
import numpy as np

from tests.utilities import TestBaseXrlXrlnp


rng = np.random.default_rng()


class TestCSb_Total(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(rng.integers(1, 98), (1 + rng.random()) * 100)


class TestCSb_Photo(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(rng.integers(1, 98), (1 + rng.random()) * 100)


class TestCSb_Rayl(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(rng.integers(1, 98), (1 + rng.random()) * 100)


class TestCSb_Compt(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(rng.integers(1, 98), (1 + rng.random()) * 100)


class TestDCSb_Rayl(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(
            rng.integers(1, 98),
            (1 + rng.random()) * 100,
            (rng.random() - 0.5) * 4 * np.pi,
        )


class TestDCSb_Compt(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(
            rng.integers(1, 98),
            (1 + rng.random()) * 100,
            (rng.random() - 0.5) * 4 * np.pi,
        )


class TestDCSPb_Rayl(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(
            rng.integers(1, 98),
            (1 + rng.random()) * 100,
            (rng.random() - 0.5) * 4 * np.pi,
            (rng.random() - 0.5) * 4 * np.pi,
        )


class TestDCSPb_Compt(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    def test_xrl(self):
        """_summary_"""
        super().xrl(
            rng.integers(1, 98),
            (1 + rng.random()) * 100,
            (rng.random() - 0.5) * 4 * np.pi,
            (rng.random() - 0.5) * 4 * np.pi,
        )
