"""_summary_
"""


import pytest

import numpy as np

from dxraylib.cross_sections import _CS_COMPT, _CS_PHOTO, _CS_RAYL
from dxraylib.fi import _FI

from tests.config import N
from tests.utilities import (
    random_formula,
    CubicInterpolators,
)
from tests.test_atomicweight import AtomicWeight
from tests.test_cross_sections import CS_Total
from tests.test_fi import Fi


rng = np.random.default_rng()
COMPOUND = "AuAg"


def refractive_index(cls):
    """_summary_

    Parameters
    ----------
    og_class : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    @property
    def args0(self):
        compound = random_formula(3)
        e = (
            self.nanmax.min() - self.nanmin.max()
        ) * rng.random() + self.nanmin.max()
        d = rng.random()
        return compound, e, d

    @pytest.mark.skip("..._CP not implemented in xrl_np")
    def test_xrlnp(_):
        ...

    @pytest.mark.skip("Not in xraylib_np")
    def test_data_points(_):
        ...

    @pytest.mark.skip("")
    def test_extrapolate_value_error(_):
        ...

    @pytest.mark.skip("")
    def test_z_out_of_range(_):
        ...

    @pytest.mark.skip("")
    def test_negative(_):
        ...

    @pytest.mark.skip("")
    def test_z_1_2big(_):
        ...

    @pytest.mark.skip("")
    def test_z_1_2small(_):
        ...

    @pytest.mark.skip("")
    def test_extrapolate_2small(_):
        ...

    @pytest.mark.skip("")
    def test_extrapolate_2big(_):
        ...

    @pytest.mark.skip("")
    def test_0(_):
        ...

    cls.args0 = args0
    cls.test_xrlnp = test_xrlnp
    cls.test_data_points = test_data_points
    cls.test_negative = test_negative
    cls.test_z_1_2big = test_z_1_2big
    cls.test_z_1_2small = test_z_1_2small
    cls.test_extrapolate_2big = test_extrapolate_2big
    cls.test_extrapolate_2small = test_extrapolate_2small
    cls.test_0 = test_0
    return cls



@refractive_index
class TestRefractive_Index_Re(Fi, AtomicWeight):
    """_summary_"""


@refractive_index
class TestRefractive_Index_Im(CS_Total):
    """_summary_"""


@refractive_index
class TestRefractive_Index(CubicInterpolators, AtomicWeight):
    """_summary_"""

    data = _CS_RAYL, _CS_COMPT, _CS_PHOTO, _FI
    scale = (*((np.exp, 1000),) * 3, (lambda x: x, 1))
