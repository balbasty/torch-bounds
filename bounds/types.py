"""
This module defines names and aliases for different boundary conditions,
as well as tools to convert between different naming conventions.

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
    'BoundType',
    'BoundLike',
    'SequenceOrScalar',
    'to_enum',
    'to_int',
    'to_fourier',
    'to_scipy',
    'to_torch',
]
from enum import Enum
from typing import Union, Sequence, TypeVar


class BoundType(Enum):
    """
    An Enum type that maps boundry modes of any convention to a
    unique set of values.

    ```python
    class BoundType(Enum):
        zero = zeros = constant = gridconstant = 0
        replicate = repeat = nearest = border = edge = 1
        dct1 = mirror = 2
        dct2 = reflect = reflection = gridmirror = neumann = 3
        dst1 = antimirror = 4
        dst2 = antireflect = dirichlet = 5
        dft = fft = wrap = gridwrap = circular = circulant = 6
        nocheck = -1
    ```
    """
    zero = zeros = constant = gridconstant = 0
    replicate = repeat = nearest = border = edge = 1
    dct1 = mirror = 2
    dct2 = reflect = reflection = gridmirror = neumann = 3
    dst1 = antimirror = 4
    dst2 = antireflect = dirichlet = 5
    dft = fft = wrap = gridwrap = circular = circulant = 6
    nocheck = -1


T = TypeVar('T')

SequenceOrScalar = Union[T, Sequence[T]]
"""Either an element or type `T`, or a sequence of elements of type `T`."""

BoundLike = Union[BoundType, str, int]
"""
A boundary mode.

Most conventions are handled (numpy, scipy, torch, see below). Can be one of:

0. `zero`, `zeros`, `constant` or `gridconstant`;
1. `replicate`, `nearest`, `border` or `edge`;
2. `dct1` or `mirror`;
3. `dct2`, `reflect`, `reflection`, `gridmirror` or `neumann`;
4. `dst1` or `antimirror`;
5. `dst2`, `antireflect` or `dirichlet`;
6. `dft`, `fft`, `wrap`, `gridwrap`, `circular` or `circulant`.

Each of these modes can be a [`BoundType`][bounds.types.BoundType] value
(e.g., `BoundType.mirror`), or its string representation (e.g., `"mirror"`).

The aliases `dft`, `dct1`, `dct2`, `dst1` and `dst2` exist because these
boundary modes correspond to the implicit boundary conditions of each of
these frequency transform:

- [Discrete Fourier Transform (DFT)](https://w.wiki/92by)
- [Discrete Cosine Transform I & II (DCT-I, DCT-II)](https://w.wiki/AQEt)
- [Discrete Sine Transform I & II (DST-I, DST-II)](https://w.wiki/ATnn)

The reason why so many aliases are supported is that there is no common
convention across python packages to name boundary conditions.
This table contains an extensive list of aliases:

+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
| Fourier | SciPy `ndimage` | Numpy `pad`    | PyTorch `pad` | PyTorch `grid_sample` | Other        | Description                                    |
+=========+=================+================+===============+=======================+==============+================================================+
|         | nearest         | edge           | border        | replicate             | repeat       | <code> a  a &#124; a b c d &#124;  d  d</code> |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
|         | constant,<br /> | constant       | constant      | zeros                 | zero         | <code> 0  0 &#124; a b c d &#124;  0  0</code> |
|         | grid-constant   |                |               |                       |              |                                                |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
| dct1    | mirror          | reflect        | reflect       | reflection<br />      |              | <code> c  b &#124; a b c d &#124;  c  b</code> |
|         |                 |                |               | (`False`)             |              |                                                |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
| dct2    | reflect,<br />  | symmetric      |               | reflection<br />      | neumann      | <code> b  a &#124; a b c d &#124;  d  c</code> |
|         | grid-mirror     |                |               | (`True`)              |              |                                                |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
| dst1    |                 |                |               |                       | antimirror   | <code>-a  0 &#124; a b c d &#124;  0 -d</code> |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
| dst2    |                 |                |               |                       | antireflect, | <code>-b -a &#124; a b c d &#124; -d -c</code> |
|         |                 |                |               |                       | dirichlet    |                                                |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
| dft     | grid-wrap       | wrap           | circular      |                       | circulant    | <code> c  d &#124; a b c d &#124;  a  b</code> |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
|         | wrap            |                |               |                       |              | <code> c  d &#124; a b c d &#124;  b  c</code> |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
|         |                 | linear_ramp    |               |                       |              |                                                |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+
|         |                 | minimum,<br /> |               |                       |              |                                                |
|         |                 | maximum,<br /> |               |                       |              |                                                |
|         |                 | mean,<br />    |               |                       |              |                                                |
|         |                 | median         |               |                       |              |                                                |
+---------+-----------------+----------------+---------------+-----------------------+--------------+------------------------------------------------+

Some of these conventions are inconsistant with each other. For example
`"wrap"` in `scipy.ndimage` is different from `"wrap"` in `numpy.pad`,
which corresponds to `"grid-wrap"` in `scipy.ndimage`. Also, `"reflect"`
in `numpy.pad` and `torch.pad` is different from `"reflect"` in `scipy.ndimage`,
which correspond to `"symmetric"` in `numpy.pad`.
"""  # noqa: E501


bounds_fourier = ('replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft')
bounds_scipy = ('nearest', 'constant', 'reflect', 'mirror', 'wrap')
bounds_torch = ('nearest', 'zeros', 'reflection')
bounds_torch_pad = ('border', 'constant', 'reflect', 'circular')
bounds_other = ('repeat', 'neumann', 'circular', 'circulant',
                'antireflect', 'dirichlet', 'antimirror', 'fft')
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


def to_enum(bound: SequenceOrScalar[BoundLike]) -> SequenceOrScalar[BoundType]:
    """Convert boundary type to enum type.

    !!! note "See also"
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_torch`][bounds.types.to_torch]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : SequenceOrScalar[BoundLike]
        Boundary condition in any convention

    Returns
    -------
    bound : SequenceOrScalar[BoundType]
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


def to_int(bound: SequenceOrScalar[BoundLike]) -> SequenceOrScalar[int]:
    """Convert boundary type to enum integer.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_torch`][bounds.types.to_torch]

    Parameters
    ----------
    bound : SequenceOrScalar[BoundLike]
        Boundary condition in any convention

    Returns
    -------
    bound : SequenceOrScalar[{0..6}]
        Boundary condition

    """
    bound = to_enum(bound)
    if isinstance(bound, (list, tuple)):
        bound = type(bound)(map(lambda x: x.value, bound))
    else:
        bound = bound.value
    return bound


def to_fourier(bound: SequenceOrScalar[BoundLike]) -> SequenceOrScalar[str]:
    """Convert boundary type to discrete transforms.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_torch`][bounds.types.to_torch]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : SequenceOrScalar[BoundLike]
        Boundary condition in any convention

    Returns
    -------
    bound : SequenceOrScalar[{'replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft'}]
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


def to_scipy(bound: SequenceOrScalar[BoundLike]) -> SequenceOrScalar[str]:
    """Convert boundary type to SciPy's convention.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_torch`][bounds.types.to_torch]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : SequenceOrScalar[BoundLike]
        Boundary condition in any convention

    Returns
    -------
    bound : SequenceOrScalar[{'border', 'constant', 'reflect', 'mirror', 'wrap'}]
        Boundary condition in SciPy's convention

    """  # noqa: E501
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


def to_torch(bound: SequenceOrScalar[BoundLike]) -> SequenceOrScalar[str]:
    """Convert boundary type to PyTorch's convention.

    !!! note "See also"
        * [`to_enum`][bounds.types.to_enum]
        * [`to_fourier`][bounds.types.to_fourier]
        * [`to_scipy`][bounds.types.to_scipy]
        * [`to_int`][bounds.types.to_int]

    Parameters
    ----------
    bound : SequenceOrScalar[BoundLike]
        Boundary condition in any convention

    Returns
    -------
    bound : SequenceOrScalar[({'nearest', 'zero', 'reflection'}, bool)]
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
