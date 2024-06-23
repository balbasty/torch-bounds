"""
This module implements discrete transforms for real signals:

- [Discrete Cosine Transform](https://w.wiki/AQEt)
- [Discrete Sine Transform](https://w.wiki/ATnn)

The implementations relies on the FFT under-the-hood, with memory-saving
tricks (borrowed from [`cupy`](https://github.com/cupy/cupy)).

The table belows lists all functions implemented in the module.
In addition to these, wrappers of the form
`#!python FUNCTYPE = partial(FUNC, type=TYPE)` are defined. For example:
```python
dct1 = partial(dct, type=1)
idct1 = partial(idct, type=1)
dctn1 = partial(dctn, type=1)
```

Functions
---------
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
```

"""
__all__ = [
    # 1D
    'dct', 'dst', 'idct', 'idst',
    # ND
    'dctn', 'dstn', 'idctn', 'idstn',
    # convenience wrappers
    'dct1', 'dct2', 'dct3', 'dct4',
    'dst1', 'dst2', 'dst3', 'dst4',
    'idct1', 'idct2', 'idct3', 'idct4',
    'idst1', 'idst2', 'idst3', 'idst4',
    'dctn1', 'dctn2', 'dctn3', 'dctn4',
    'dstn1', 'dstn2', 'dstn3', 'dstn4',
    'idctn1', 'idctn2', 'idctn3', 'idctn4',
    'idstn1', 'idstn2', 'idstn3', 'idstn4',
]
from torch import Tensor
from typing import Optional
from functools import partial
from ._realtransforms_autograd import DCTN, DSTN, flipnorm, fliptype
_IMPLEMENTED_TYPES = (1, 2, 3)


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
    norm : {"backward", "ortho", "forward"}
        Normalization mode. Default is "backward".
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    if type in _IMPLEMENTED_TYPES:
        return DCTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


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
    norm : {"backward", "ortho", "forward"}
        Normalization mode. Default is "backward".
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
    norm : {"backward", "ortho", "forward", "ortho_scipy"}
        Normalization mode. Default is "backward".
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = -1
    if type in _IMPLEMENTED_TYPES:
        return DSTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


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
    norm : {"backward", "ortho", "forward", "ortho_scipy"}
        Normalization mode. Default is "backward".
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
    return dst(x, dim, norm, type)


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
    norm : {"backward", "ortho", "forward"}
        Normalization mode. Default is "backward".
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    if type in _IMPLEMENTED_TYPES:
        return DCTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


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
    norm : {"backward", "ortho", "forward"}
        Normalization mode. Default is "backward".
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dctn(x, dim, norm, type)


def dstn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Discrete Sine Transform
    along the specified axes.

    !!! warning "Type IV not implemented"

    !!! warning
        `dstn(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {"backward", "ortho", "forward", "ortho_scipy"}
        Normalization mode. Default is "backward".
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    if type in _IMPLEMENTED_TYPES:
        return DSTN.apply(x, type, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


def idstn(
    x: Tensor,
    dim: Optional[int] = None,
    norm: str = 'backward',
    type: int = 2,
) -> Tensor:
    """Return multidimensional Inverse Discrete Sine Transform
    along the specified axes.

    !!! warning "Type IV not implemented"

    !!! warning
        `idstn(..., norm="ortho")` yields a different result than `scipy`
        and `cupy` for types 2 and 3. This is because their DST is not
        properly orthogonalized. Use `norm="ortho_scipy"` to get results
        matching their implementation.

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {"backward", "ortho", "forward", "ortho_scipy}
        Normalization mode. Default is "backward".
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    if dim is None:
        dim = list(range(x.ndim))
    norm = flipnorm[norm or "backward"]
    type = fliptype[type]
    return dstn(x, dim, norm, type)


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
