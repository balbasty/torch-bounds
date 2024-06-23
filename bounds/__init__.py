"""

Modules
-------
indexing
    Functions that wrap out-of-bound indices back in-bounds,
    according to some boundary condition.
padding
    Reimplements [`torch.nn.functional.pad`][] and [`torch.roll`][]
    with a larger set of boundary conditions.
realtransforms
    Implements the discrete sine and cosine transforms, variants I, II, III.
types
    Defines names and aliases for different boundary conditions,
    as well as tools to convert between different naming conventions.

Functions
---------
pad
    Pad a tensor
roll
    Roll a tensor
ensure_shape
    Pad/crop a tensor so that it has a given shape
to_enum
    Convert any boundary type to `BoundType`
to_int
    Convert any boundary type to `BoundType`-based integer values
to_fourier
    Convert any boundary type to discrete transforms
to_scipy
    Convert any boundary type to `scipy.ndimage` convention
to_torch
    Convert any boundary type to `torch.grid_sample` convention
dct
    One-dimensional Discrete Cosine Transform (DCT)
dst
    One-dimensional Discrete Sine Transform (DST)
idct
    One-dimensional Inverse Discrete Cosine Transform (DCT)
idst
    One-dimensional Inverse Discrete Sine Transform (DST)
dctn
    N-dimensional Discrete Cosine Transform (IDCT)
dstn
    N-dimensional Discrete Sine Transform (IDST)
idctn
    N-dimensional Inverse Discrete Cosine Transform (IDCT)
idstn
    N-dimensional Inverse Discrete Sine Transform (IDST)

Classes
-------
BoundType
    Enum type for bounds

Attributes
----------
BoundLike
    A type hint for any boundary type
SequenceOrScalar
    A type hint for values or sequences of values
"""
import torch as _torch

from .padding import *      # noqa: F401, F403
from .types import *        # noqa: F401, F403

from . import indexing      # noqa: F401
from . import padding       # noqa: F401
from . import types         # noqa: F401

__all__ = [
    'indexing',
    'padding',
    'types',
]
__all__ += padding.__all__
__all__ += types.__all__


if hasattr(getattr(_torch, 'fft', None), 'fft'):

    from .realtransforms import *        # noqa: F401, F403
    from . import realtransforms         # noqa: F401

    __all__ += ['realtransforms']
    __all__ += realtransforms.__all__


from . import _version  # noqa: E402
__version__ = _version.get_versions()['version']
