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

| Fourier   | SciPy `ndimage`          | Numpy `pad`   | PyTorch `pad` | PyTorch `grid_sample`| Other                   | Description               |
| --------- | ------------------------ | ------------- | ------------- | -------------------- | ----------------------- | ------------------------- |
|           | nearest                  | edge          | border        | replicate            | repeat                  | <code> a  a &#124; a b c d &#124;  d  d</code> |
|           | constant, <br />grid-constant  | constant      | constant      | zeros                | zero                    | <code> 0  0 &#124; a b c d &#124;  0  0</code> |
| dct1      | mirror                   | reflect       | reflect       | reflection  <br />(`False`) |                         | <code> c  b &#124; a b c d &#124;  c  b</code> |
| dct2      | reflect,  <br />grid-mirror     | symmetric     |               | reflection  <br />(`True`)  | neumann                 | <code> b  a &#124; a b c d &#124;  d  c</code> |
| dst1      |                          |               |               |                      | antimirror              | <code>-a  0 &#124; a b c d &#124;  0 -d</code> |
| dst2      |                          |               |               |                      | antireflect,  <br />dirichlet  | <code>-b -a &#124; a b c d &#124; -d -c</code> |
| dft       | grid-wrap                | wrap          | circular      |                      | circulant               | <code> c  d &#124; a b c d &#124;  a  b</code> |
|           | wrap                     |               |               |                      |                         | <code> c  d &#124; a b c d &#124;  b  c</code> |
|           |                          | linear_ramp   |
|           |                          | minimum,  <br />maximum,  <br />mean,  <br />median |

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

```python
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

## Real frequency transforms (DCT/DST)

PyTorch does not implement discrete sine and cosine transforms.

We follow the trick used in [`cupy`](https://cupy.dev) and implement these
transforms using the FFT applied to replicated/flipped inputs followed
by shuffling rescaling. These tricks are described in the following
references:

1.  J. Makhoul, **"A fast cosine transform in one and two dimensions,"** in
    _IEEE Transactions on Acoustics, Speech, and Signal Processing_, vol. 28,
    no. 1, pp. 27-34, February 1980.
2.  M.J. Narasimha and A.M. Peterson,
    **“On the computation of the discrete cosine  transform,”**
    _IEEE Trans. Commun._, vol. 26, no. 6, pp. 934–936, 1978.
3.  http://fourier.eng.hmc.edu/e161/lectures/dct/node2.html
4.  https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
5.  X. Shao, S. G. Johnson.
    **Type-II/III DCT/DST algorithms with reduced number of arithmetic operations**,
    _Signal Processing_, Volume 88, Issue 6, pp. 1553-1564, 2008.

We also implement the type 1 DCT/DST (whereas cupy only implements types
2 and 3). Type 4 is not implemented yet.

```python
def dct(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Discrete Cosine Transform

    !!! warning "Type IV not implemented"

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...


def idct(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Inverse Discrete Cosine Transform

    !!! warning
        Type IV not implemented

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dct(x, dim, norm, type)


def dst(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Discrete Sine Transform

    !!! warning "Type IV not implemented"

    !!! warning
        `dst(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy"}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...


def idst(
    x: Tensor,
    dim: int = -1,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return the Inverse Discrete Sine Transform

    !!! warning "Type IV not implemented"

    !!! warning
        `idst(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy"}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...


def dctn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Discrete Cosine Transform
    along the specified axes.

    !!! warning "Type IV not implemented"

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...


def idctn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Inverse Discrete Cosine Transform
    along the specified axes.

    !!! warning "Type IV not implemented"

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...


def dstn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Discrete Sine Transform
    along the specified axes.

    !!! warning "Type IV not implemented"

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy"}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...


def idstn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Inverse Discrete Sine Transform
    along the specified axes.

    !!! warning "Type IV not implemented"

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”, "ortho_scipy}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    ...

```

We further have the following aliases:

```python
dct1 = partial(dct, type=1)
dct2 = partial(dct, type=2)
dct3 = partial(dct, type=3)
dct4 = partial(dct, type=4)

idct1 = partial(idct, type=1)
idct2 = partial(idct, type=2)
idct3 = partial(idct, type=3)
idct4 = partial(idct, type=4)

dst1 = partial(dst, type=1)
dst2 = partial(dst, type=2)
dst3 = partial(dst, type=3)
dst4 = partial(dst, type=4)

idst1 = partial(idst, type=1)
idst2 = partial(idst, type=2)
idst3 = partial(idst, type=3)
idst4 = partial(idst, type=4)

dctn1 = partial(dctn, type=1)
dctn2 = partial(dctn, type=2)
dctn3 = partial(dctn, type=3)
dctn4 = partial(dctn, type=4)

idctn1 = partial(idctn, type=1)
idctn2 = partial(idctn, type=2)
idctn3 = partial(idctn, type=3)
idctn4 = partial(idctn, type=4)

dstn1 = partial(dstn, type=1)
dstn2 = partial(dstn, type=2)
dstn3 = partial(dstn, type=3)
dstn4 = partial(dstn, type=4)

idstn1 = partial(idstn, type=1)
idstn2 = partial(idstn, type=2)
idstn3 = partial(idstn, type=3)
idstn4 = partial(idstn, type=4)
```
