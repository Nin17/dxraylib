Compound parser
===============

`Relevant section in the xraylib documentation <xraylib_>`_.

.. _xraylib: https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#compound-parser

.. currentmodule:: dxraylib.xraylib_parser
.. autoclass:: dxraylib.xraylib_parser.compoundData
    :members:
    :show-inheritance:


CompoundParser
--------------

The CompoundParser function will parse a chemical formula compoundString and 
will allocate a compoundData structure with the results if successful, 
otherwise NULL is returned. Chemical formulas may contain (nested) brackets, 
followed by an integer or real number (with a dot) subscript. Examples of 
accepted formulas are: H2O, Ca5(PO4)3F, Ca5(PO4)F0.33Cl0.33(OH)0.33.

.. autofunction:: dxraylib.CompoundParser


AtomicNumberToSymbol
--------------------

The AtomicNumberToSymbol function returns a pointer to a string containing the 
element for atomic number Z. If an error occurred, the NULL string is returned. 

.. autofunction:: dxraylib.AtomicNumberToSymbol

SymbolToAtomicNumber
--------------------

The SymbolToAtomicNumber function returns the atomic number that corresponds
with element symbol. If the element does not exist, 0 is returned.

.. autofunction:: dxraylib.SymbolToAtomicNumber
