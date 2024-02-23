# torch-bounds

Boundary conditions (circulant, mirror, reflect) and real transforms (dct, dst) in PyTorch.

## Installation

### Dependency

- `torch >= 1.3`
- `torch >= 1.8` if real transforms are needed (dct, dst)

### Conda

```shell
conda install torch-bounds -c balbasty -c pytorch
```

### Pip

```shell
pip install torch-bounds
```

## Overview

There is no common convention across python packages to name boundary
conditions. This table contains an extensive list of aliases:

| Fourier   | SciPy `ndimage`          | Numpy `pad`   | PyTorch `pad` | PyTorch `grid_sample`            | Other                   | Description               |
| --------- | ------------------------ | ------------- | ------------- | -------------------------------- | ----------------------- | ------------------------- |
|           | nearest                  | edge          | border        | replicate                        | repeat                  | ` a  a | a b c d |  d  d` |
|           | constant, grid-constant  | constant      | constant      | zeros                            | zero                    | ` 0  0 | a b c d |  0  0` |
| dct1      | mirror                   | reflect       | reflect       | reflection (align_corners=False) |                         | ` c  b | a b c d |  c  b` |
| dct2      | reflect, grid-mirror     | symmetric     |               | reflection (align_corners=True)  | neumann                 | ` b  a | a b c d |  d  c` |
| dst1      |                          |               |               |                                  | antimirror              | `-a  0 | a b c d |  0 -d` |
| dst2      |                          |               |               |                                  | antireflect, dirichlet  | `-b -a | a b c d | -d -c` |
| dft       | grid-wrap                | wrap          | circular      |                                  | circulant               | ` c  d | a b c d |  a  b` |
|           | wrap                     |               |               |                                  |                         | ` c  d | a b c d |  b  c` |
|           |                          | linear_ramp   |
|           |                          | minimum, maximum, mean, median |

Some of these conventions are inconsistant with each other. For example
`"wrap"` in `scipy.ndimage` is different from `"wrap"` in `numpy.pad`,
which corresponds to `"grid-wrap"` in `scipy.ndimage`. Also, `"reflect"`
in `numpy.pad` and `torch.pad` is different from `"reflect"` in `scipy.ndimage`,
which correspond to `"symmetric"` in `numpy.pad`.

## Conversion between boundary names

We provide a series of functions to convert names between these
different conventions. In case of inconsistency, we assume that
- `"wrap"` means `"dft"`/`"grid-wrap"`
- `"reflect"` means `"dct2"`/`"grid-mirror"`

We also introduce an internal `Enum` type that maps of all these names
to a fixed set of integers:

```
class BoundType(Enum):
    zero = zeros = constant = gridconstant = 0
    replicate = repeat = nearest = border = edge = 1
    dct1 = mirror = 2
    dct2 = reflect = reflection = gridmirror = neumann = 3
    dst1 = antimirror = 4
    dst2 = antireflect = dirichlet = 5
    dft = wrap = gridwrap = circular = circulant = 6
    nocheck = -1
```

A series of functions allow any boundary name to be converted to any
convention:

```python
BoundLike = Union[BoundType, str, int]
ScalarOrList = Union[T, Sequence[T]]

def to_enum(bound: ScalarOrList[BoundLike]) -> ScalarOrList[BoundType]:
    """Convert boundary type to enum type.

    Parameters
    ----------
    bound : [sequence of] BoundLike
        Boundary condition in any convention

    Returns
    -------
    bound : [sequence of] BoundType
        Boundary condition
    """
    ...

def to_int(bound: ScalarOrList[BoundLike]) -> ScalarOrList[int]:
    """Convert boundary type to enum integer.

    Parameters
    ----------
    bound : [sequence of] BoundLike
        Boundary condition in any convention

    Returns
    -------
    bound : [sequence of] int
        Boundary condition
    """
    ...

def to_fourier(bound: ScalarOrList[BoundLike]) -> ScalarOrList[str]:
    """Convert boundary type to discrete transforms.

    Parameters
    ----------
    bound : [sequence of] BoundLike
        Boundary condition in any convention

    Returns
    -------
    bound : [sequence of] {'replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft'}
        Boundary condition
    """
    ...

def to_scipy(bound: ScalarOrList[BoundLike]) -> ScalarOrList[str]:
    """Convert boundary type to SciPy's convention.

    Parameters
    ----------
    bound : [sequence of] BoundLike
        Boundary condition in any convention

    Returns
    -------
    bound : [sequence of] {'border', 'constant', 'reflect', 'mirror', 'wrap'}
        Boundary condition
    """
    ...
```

## PyTorch limitations

It is clear from the PyTorch columns in this table that PyTorch does not
implement all possible boundary conditions. In particular, it does not
imeplement the boundary condition of a type II DCT (mirroring along the
edge of the first voxel). We reimplement `pad` and `roll` with this larger
set of boundary conditions.

```python
def pad(inp, padsize, mode='constant', value=0, side=None):
    """Pad a tensor.

    This function is a bit more generic than torch's native pad
    (`torch.nn.functional.pad`), but probably a bit slower:

    - works with any input type
    - works with arbitrarily large padding size
    - crops the tensor for negative padding values
    - implements additional padding modes

    When used with defaults parameters (`side=None`), it behaves
    exactly like `torch.nn.functional.pad`

    !!! info "Boundary modes"
        Like in PyTorch's `pad`, boundary modes include:

        - `'circular'`  (or `'dft'`)
        - `'mirror'`    (or `'dct1'`)
        - `'reflect'`   (or `'dct2'`)
        - `'replicate'` (or `'nearest'`)
        - `'constant'`  (or `'zero'`)

        as well as the following new modes:

        - `'antimirror'`    (or `'dst1'`)
        - `'antireflect'`   (or `'dst2'`)

    !!! info "Side modes"
        Side modes are `'pre'`, `'post'`, `'both'` or `None`.

        - If side is not `None`, `inp.dim()` values (or less) should be
          provided.
        - If side is `None`, twice as many values should be provided,
          indicating different padding sizes for the `'pre'` and `'post'`
          sides.
        - If the number of padding values is less than the dimension of the
          input tensor, zeros are prepended.

    Parameters
    ----------
    inp : tensor
        Input tensor
    padsize : [sequence of] int
        Amount of padding in each dimension.
    mode : [sequence of] BoundLike
        Padding mode
    value : scalar
        Value to pad with in mode 'constant'.
    side : "{'left', 'right', 'both', None}"
        Use padsize to pad on left side ('pre'), right side ('post') or
        both sides ('both'). If None, the padding side for the left and
        right sides should be provided in alternate order.

    Returns
    -------
    tensor
        Padded tensor.
    """
    ...

def roll(inp, shifts=1, dims=None, bound='circular'):
    r"""Like `torch.roll`, but with any boundary condition

    !!! warning
        When `dims` is `None`, we do not flatten but shift all dimensions.
        This differs from the behavior of `torch.roll` .

    Parameters
    ----------
    inp : tensor
        Input
    shifts : [sequence of] int
        Amount by which to roll.
        Positive shifts to the right, negative to the left.
    dims : [sequence of] int
        Dimensions to roll.
        By default, shifts apply to all dimensions if a scalar,
        or to the last N if a sequence.
    bound : "{'constant', 'replicate', 'reflect', 'mirror', 'circular'}"
        Boundary condition

    Returns
    -------
    out : tensor
        Rolled tensor
    """
    ...

def ensure_shape(inp, shape, mode='constant', value=0, side='post',
                 ceil=False):
    """Pad/crop a tensor so that it has a given shape

    Parameters
    ----------
    inp : tensor
        Input tensor
    shape : [sequence of] int
        Output shape
    mode : "{'constant', 'replicate', 'reflect', 'mirror', 'circular'}"
        Boundary mode
    value : scalar, default=0
        Value for mode 'constant'
    side : "{'pre', 'post', 'both'}"
        Side to crop/pad

    Returns
    -------
    out : tensor
        Padded tensor with shape `shape`
    """
    ...

def make_vector(input, n=None, crop=True, *args,
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    ...
```
