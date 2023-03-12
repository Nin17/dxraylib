<!--
header-includes:
 - \usepackage{fvextra}
 - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
-->

<!--  pandoc --pdf-engine=xelatex --highlight-style breezedark -V colorlinks README.md -o README.pdf -->

# $\partial xraylib$

A differentiable python reimplementation of [xraylib](https://github.com/tschoonj/xraylib) using [JAX](https://github.com/google/jax).

## Divergences from xraylib & xraylib_np

### Runtime errors

Currently, there is [no runtime error mechanism in XLA](https://github.com/google/jax/issues/4257#issuecomment-690844567) and by extension JAX.
Errors are instead indicated by ```NaNs```:

```python
import xraylib as xrl
import dxraylib as dxrl
Z = 69
E = -69.
try:
    print(xrl.Fi(Z, E))
except ValueError as e:
    print(e)
print(dxrl.Fi(Z, E))
```

Results in:

```text
Energy must be strictly positive
nan
```

In xraylib_np, errors aren't raised and are instead signalled by a value of ```0```.
Therefore, to obtain the same result with $\partial xraylib$, [jax.numpy.nan_to_num](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nan_to_num.html) or equivalent must be used:

```python
import numpy as np
import jax.numpy as jnp
import xraylib_np as xrl_np
Z = np.asarray([Z])
E = np.asarray([E])
print(dxrl.Fi(Z, E))
print(jnp.nan_to_num(dxrl.Fi(Z, E)))
print(xrl_np.Fi(Z, E))
```

Results in:

```python
[[nan]]
[[0.]]
[[0.]]
```

#### Argument dtypes

In xraylib_np, the arguments are required to have the correct dtype.
In $\partial xraylib$ this requirement is relaxed, and arguments can take any numerical type - except for ```Z``` which is still required to have an integer or boolean type:

```python
E = -E.astype(int)
try:
    print(xrl_np.Fi(Z, E))
except ValueError as e:
    print(e)
print(dxrl.Fi(Z, E))
Z = Z.astype(float)
try:
    print(xrl_np.Fi(Z, E))
except ValueError as e:
    print(e)
try:
    print(dxrl.Fi(Z, E))
except TypeError as e:
    print(e)
```

Results in:

```python
Buffer dtype mismatch, expected 'double' but got 'long'
[[-1.19547792]]
Buffer dtype mismatch, expected 'int_t' but got 'double'
Indexer must have integer or boolean type, got indexer with type float64 at position 0, indexer value Traced<ShapedArray(float64[1])>with<DynamicJaxprTrace(level=0/2)>
```

#### Argument shapes & dimensions

In xraylib_np arguments must be numpy arrays with ```ndim==1```, the output shape is the result of concatenating the input shapes.
In $\partial xraylib$, the output shape is still the result of concatenating the input shapes, but the requirement for 1d arguments is removed:

```python
Z = np.atleast_2d(Z).astype(int)
try:
    print(xrl_np.Fi(Z, E))
except ValueError as e:
    print(e)
print(dxrl.Fi(Z, E), dxrl.Fi(Z, E).shape, Z.shape, E.shape)
```

Results in:

```python
Buffer has wrong number of dimensions (expected 1, got 2)
[[[-2.65118007e-05]]] (1, 1, 1) (1, 1) (1,)
```

### Functions with string arguments

Functions in xraylib that require an argument as a string are not supported in xraylib_np. In $\partial xraylib$, all functions support arrays for all numerical arguments regardless of whether they also take string arguments.

```python
C = "Tm"
E = E.astype(float)
D = np.array([69])
try:
    print(xrl_np.Refractive_Index(C, E, D))
except AttributeError as e:
    print(e)
print(dxrl.Refractive_Index(C, E, D))
```

Results in:

```python
module 'xraylib_np' has no attribute 'Refractive_Index'
[[0.99999756+9.28136082e-08j]]
```

Functions that take a single string argument or return a string such as: ```AtomicNumberToSymbol```, ```CompoundParser``` and ```GetCompoundDataNISTByName``` are just wrappers around the functions of the same name in xraylib and therefore don't accept arrays as arguments.

## Currently supported functions

* AtomicLevelWidth
* AtomicNumberToSymbol
* AtomicWeight
* AugerRate
* AugerYield
* CS_Compt
* CS_Compt_CP
* CS_Energy
* CS_Energy_CP
* CS_KN
* CS_Photo
* CS_Photo_CP
* CS_Rayl
* CS_Rayl_CP
* CS_Total
* CS_Total_CP
* CSb_Compt
* CSb_Compt_CP
* CSb_Photo
* CSb_Photo_CP
* CSb_Rayl
* CSb_Rayl_CP
* CSb_Total
* CSb_Total_CP
* CompoundParser
* ComptonEnergy
* CosKronTransProb
* DCSP_Compt
* DCSP_Compt_CP
* DCSP_KN
* DCSP_Rayl
* DCSP_Rayl_CP
* DCSP_Thoms
* DCSPb_Compt
* DCSPb_Compt_CP
* DCSPb_Rayl
* DCSPb_Rayl_CP
* DCS_Compt
* DCS_Compt_CP
* DCS_KN
* DCS_Rayl
* DCS_Rayl_CP
* DCS_Thoms
* DCSb_Compt
* DCSb_Compt_CP
* DCSb_Rayl
* DCSb_Rayl_CP
* EdgeEnergy
* ElementDensity
* FF_Rayl
* Fi
* Fii
* FluorYield
* GetCompoundDataNISTByName
* JumpFactor
* MomentTransf
* Refractive_Index
* Refractive_Index_Im
* Refractive_Index_Re
* SF_Compt
