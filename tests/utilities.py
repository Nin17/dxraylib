"""_summary_
"""
# TODO type hint functions
import unittest
import pytest

import numpy as np

import xraylib_np as xrl_np
import xraylib as xrl

import jaxraylib as jxrl

from tests.config import ATOL, RTOL

rng = np.random.default_rng()


class TestBaseXrlXrlnp:
    """_summary_

    Parameters
    ----------
    unittest : _type_
        _description_
    """

    @property
    def func(self) -> str:
        """_summary_

        Returns
        -------
        str
            _description_
        """
        return self.__class__.__name__.removeprefix("Test")

    def xrl(self, *args, **kwargs):
        """_summary_"""
        xrl_output = getattr(xrl, self.func)(*args, **kwargs)
        jxrl_output = getattr(jxrl, self.func)(*args, **kwargs)
        assert jxrl_output.shape == ()
        assert np.allclose(xrl_output, jxrl_output, atol=ATOL, rtol=RTOL)

    def xrlnp(self, *args, jxrl_args=None):
        """_summary_

        Parameters
        ----------
        jxrl_args : _type_, optional
            _description_, by default None
        """
        if jxrl_args is None:
            jxrl_args = args
        xrlnp_output = getattr(xrl_np, self.func)(*args)
        jxrl_output = getattr(jxrl, self.func)(*jxrl_args)
        assert xrlnp_output.shape == jxrl_output.shape
        assert np.allclose(xrlnp_output, jxrl_output, atol=ATOL, rtol=RTOL)

    def raise_error(self, error, *args, **kwargs):
        """_summary_

        Parameters
        ----------
        error : _type_
            _description_
        """
        try:
            getattr(xrl, self.func)(*args, **kwargs)
        except error:
            with pytest.raises(error):
                getattr(jxrl, self.func)(*args, **kwargs)


# TODO merge with TestBaseXrlXrlnp
# TODO change to pytest
class TestBaseInterpolators(unittest.TestCase):
    def spline_extrapolation(self, data, test_func, func):
        if isinstance(data, tuple):
            data_shape = min(i.shape[0] for i in data)
        else:
            data_shape = data.shape[0]
        for i in range(1, data_shape + 1):
            if isinstance(data, tuple):
                minimum = max(np.nanmin(j[i - 1, 0]) for j in data)
                maximum = min(np.nanmax(j[i - 1, 0]) for j in data)
            else:
                minimum = np.nanmin(data[i - 1, 0])
                maximum = np.nanmax(data[i - 1, 0])
            self.assertRaises(ValueError, test_func, i, func(minimum) / 2)
            self.assertRaises(ValueError, test_func, i, func(maximum) * 2)
