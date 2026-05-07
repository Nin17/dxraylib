Scattering factors
==================

`Relevant section in the xraylib documentation <xraylib_>`_.

.. _xraylib: https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#scattering-factors


Included from the xraylib documentation:
In this section, we introduce the momentum transfer parameter q, which is used 
in several of the following functions. It should be noted that several 
definitions can be found of this parameter throughout the scientific 
literature, which vary mostly depending on the community where it is used. The 
crystallography and diffraction community for example, use the following 
definition:

.. math:: q = \frac{4\pi \times sin(\theta)}{\lambda}

with θ the angle between the incident X-ray and the crystal scattering planes 
according to Bragg's law, and λ the wavelength. xraylib (and dxraylib) uses 
however, a different definition, in which θ corresponds to the scattering angle
, which in case of Bragg scattering is equal to twice the angle from the 
previous definition. This new definition has the advantage of being useful 
when working with amorphous materials, as well as with incoherent scattering. 
Furthermore, our definition drops the 4π scale factor, in line with the 
definition by Hubbell et al in Atomic form factors, incoherent scattering 
functions, and photon scattering cross-sections, J. Phys. Chem. Ref. Data, 
Vol.4, No. 3, 1975:

.. math:: q = Ehc \times sin(\frac{\theta}{2}) \times 10^{8}

with E the energy of the photon, h Planck's constant and c the speed of light. 
The unit of the returned momentum transfer is then Å⁻¹.


|
.. autofunction:: dxraylib.FF_Rayl
|
.. autofunction:: dxraylib.SF_Compt
|
.. autofunction:: dxraylib.MomentTransf
|
.. autofunction:: dxraylib.Fi
|
.. autofunction:: dxraylib.Fii

