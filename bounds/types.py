"""
This module defines names and aliases for different boundary conditions,
as well as tools to convert between different naming conventions.

Classes
-------
BoundType
    Enum type for bounds

Functions
---------
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

"""  # noqa: E501
__all__ = [
    'BoundType', 'to_enum', 'to_int', 'to_fourier', 'to_scipy', 'to_torch',
]
from enum import Enum
from typing import Union


class BoundType(Enum):
    zero = zeros = constant = gridconstant = 0
    replicate = repeat = nearest = border = edge = 1
    dct1 = mirror = 2
    dct2 = reflect = reflection = gridmirror = neumann = 3
    dst1 = antimirror = 4
    dst2 = antireflect = dirichlet = 5
    dft = wrap = gridwrap = circular = circulant = 6
    nocheck = -1


BoundLike = Union[BoundType, str, int]


bounds_fourier = ('replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft')
bounds_scipy = ('nearest', 'constant', 'reflect', 'mirror', 'wrap')
bounds_torch = ('nearest', 'zeros', 'reflection')
bounds_torch_pad = ('border', 'constant', 'reflect', 'circular')
bounds_other = ('repeat', 'neumann', 'circular', 'circulant',
                'antireflect', 'dirichlet', 'antimirror')
enum_bounds = (BoundType.zero, BoundType.repeat, BoundType.dct1,
               BoundType.dct2, BoundType.dst1, BoundType.dst2, BoundType.dft)
int_bounds = tuple(range(7))


zero_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.zero]
rept_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.repeat]
dct1_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dct1]
dct2_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dct2]
dst1_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dst1]
dst2_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dst2]
dft_bounds = [b for b in BoundType.__members__.keys()
              if getattr(BoundType, b) == BoundType.dft]


def to_enum(bound) -> BoundType:
    """Convert boundary type to enum type.

    !!! note "See also"
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_torch`][bounds.types.to_torch]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] BoundType
        Boundary condition

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append(BoundType.zero)
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append(BoundType.replicate)
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append(BoundType.dct1)
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append(BoundType.dct2)
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            obound.append(BoundType.dst1)
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            obound.append(BoundType.dst2)
        elif b in (*dft_bounds, BoundType.dft, 6):
            obound.append(BoundType.dft)
        elif b in ('nocheck', BoundType.nocheck, -1):
            obound.append(BoundType.nocheck)
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_int(bound) -> int:
    """Convert boundary type to enum integer.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_torch`][bounds.types.to_torch]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] {0..6}
        Boundary condition

    """
    bound = to_enum(bound)
    if isinstance(bound, (list, tuple)):
        bound = type(bound)(map(lambda x: x.value, bound))
    else:
        bound = bound.value
    return bound


def to_fourier(bound):
    """Convert boundary type to discrete transforms.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_torch`][bounds.types.to_torch]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] {'replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft'}
        Boundary condition in terms of discrete transforms

    """  # noqa: E501
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append('zero')
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append('replicate')
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append('dct1')
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append('dct2')
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            obound.append('dst1')
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            obound.append('dst2')
        elif b in (*dft_bounds, BoundType.dft, 6):
            obound.append('dft')
        elif b in ('nocheck', BoundType.nocheck, -1):
            obound.append('nocheck')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_scipy(bound):
    """Convert boundary type to SciPy's convention.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_torch`][bounds.types.to_torch]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] {'border', 'constant', 'reflect', 'mirror', 'wrap'}
        Boundary condition in SciPy's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append('constant')
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append('border')
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append('mirror')
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append('reflect')
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        elif b in (*dft_bounds, BoundType.dft, 6):
            obound.append('wrap')
        elif b in ('nocheck', BoundType.nocheck, -1):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_torch(bound):
    """Convert boundary type to PyTorch's convention.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] ({'nearest', 'zero', 'reflection'}, bool)
        The first element is the boundary condition in PyTorchs's
        convention, and the second element is the value of `align_corners`.

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append(('zero', None))
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append(('nearest', None))
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append(('reflection', False))
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append(('reflection', True))
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in (*dft_bounds, BoundType.dft, 6):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in ('nocheck', BoundType.nocheck, -1):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound
