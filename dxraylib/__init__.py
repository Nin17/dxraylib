"""
dxraylib: differentiable X-ray-matter interactions
"""

__version__ = "0.0.1"
from .atomiclevelwidth import AtomicLevelWidth
from .atomicweight import AtomicWeight
from .auger_trans import AugerRate, AugerYield
from .constants import AVOGNUM, KEV2ANGST, MEC2, RE2
from .coskron import CosKronTransProb
from .cross_sections import (
    CS_Compt,
    CS_Energy,
    CS_Photo,
    CS_Rayl,
    CS_Total,
)
from .cs_barns import (
    CSb_Total,
    CSb_Compt,
    CSb_Rayl,
    CSb_Photo,
    DCSb_Rayl,
    DCSb_Compt,
    DCSPb_Rayl,
    DCSPb_Compt,
)
from .cs_cp import (
    CS_Total_CP,
    CS_Photo_CP,
    CS_Rayl_CP,
    CS_Compt_CP,
    CSb_Total_CP,
    CSb_Photo_CP,
    CSb_Rayl_CP,
    CSb_Compt_CP,
    CS_Energy_CP,
    DCS_Rayl_CP,
    DCS_Compt_CP,
    DCSb_Rayl_CP,
    DCSb_Compt_CP,
    DCSP_Rayl_CP,
    DCSP_Compt_CP,
    DCSPb_Rayl_CP,
    DCSPb_Compt_CP,
)
from .densities import ElementDensity
from .edges import EdgeEnergy
from .fi import Fi
from .fii import Fii
from .fluor_yield import FluorYield
from .jumpy import JumpFactor
from .polarized import DCSP_Compt, DCSP_KN, DCSP_Rayl, DCSP_Thoms
from .radrate import RadRate
from .refractive_indices import (
    Refractive_Index,
    Refractive_Index_Im,
    Refractive_Index_Re,
)
from .scattering import (
    ComptonEnergy,
    CS_KN,
    DCS_Compt,
    DCS_KN,
    DCS_Rayl,
    DCS_Thoms,
    FF_Rayl,
    MomentTransf,
    SF_Compt,
)
from .xraylib_nist_compounds import GetCompoundDataNISTByName
from .xraylib_parser import CompoundParser, AtomicNumberToSymbol
from .xraylib_radionuclides import (
    GetRadioNuclideDataByIndex,
    GetRadioNuclideDataByName,
    GetRadioNuclideDataList,
)
