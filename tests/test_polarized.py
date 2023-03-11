"""_summary_
"""

import numpy as np
import pytest

from tests.test_atomicweight import AtomicWeight
from tests.test_scattering import FF_Rayl, SF_Compt
from tests.utilities import Analytic


@pytest.mark.skip()
class DCSP_Rayl(FF_Rayl, AtomicWeight):
    """_summary_"""
    # TODO actually implement some tests


class TestDCSP_Rayl(DCSP_Rayl):
    """_summary_"""


@pytest.mark.skip()
class DCSP_Compt(SF_Compt, AtomicWeight):
    """_summary_"""
    # TODO actually implement some tests


class TestDCSP_Compt(DCSP_Compt):
    """_summary_"""


class TestDCSP_KN(Analytic):
    """_summary_"""

    ab = ((0, 1000), (-2 * np.pi, 2 * np.pi), (-np.pi, np.pi))


class TestDCSP_Thoms(Analytic):
    """_summary_"""

    ab = ((-2 * np.pi, 2 * np.pi), (-np.pi, np.pi))
