"""
This module reimplements [`torch.nn.functional.pad`][] and [`torch.roll`][]
with a larger set of boundary conditions.

Functions
---------
pad
    Pad a tensor
roll
    Roll a tensor
ensure_shape
    Pad/crop a tensor so that it has a given shape
"""
__all__ = ['pad', 'roll', 'ensure_shape']
import torch
import math
from torch import Tensor
from typing import Optional
from numbers import Number
from . import indexing
from .types import to_fourier, BoundLike, SequenceOrScalar
from ._utils import prod, ensure_list, meshgrid_list_ij, sub2ind_list


def pad(
    inp: Tensor,
    padsize: SequenceOrScalar[int],
    mode: SequenceOrScalar[BoundLike] = 'constant',
    value: Number = 0,
    side: Optional[str] = None
):
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
        Side modes are `'pre'` (or `'left'`), `'post'` (or `'right'`),
        `'both'` or `None`.

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
    padsize : SequenceOrScalar[int]
        Amount of padding in each dimension.
    mode : SequenceOrScalar[BoundLike]
        Padding mode
    value : scalar
        Value to pad with in mode `'constant'`.
    side : "{'pre', 'post', 'both', None}"
        Use padsize to pad on left side (`'pre'`), right side (`'post'`) or
        both sides (`'both'`). If `None`, the padding side for the left and
        right sides should be provided in alternate order.

    Returns
    -------
    out : tensor
        Padded tensor.

    """
    # Argument checking
    mode = to_fourier(mode)
    mode = ensure_list(mode, len(padsize) // (1 if side else 2))

    padsize = tuple(padsize)
    if not side:
        if len(padsize) % 2:
            raise ValueError('Padding length must be divisible by 2')
        padpre = padsize[::2]
        padpost = padsize[1::2]
    else:
        side = side.lower()
        if side == 'both':
            padpre = padsize
            padpost = padsize
        elif side in ('pre', 'left'):
            padpre = padsize
            padpost = (0,) * len(padpre)
        elif side in ('post', 'right'):
            padpost = padsize
            padpre = (0,) * len(padpost)
        else:
            raise ValueError(f'Unknown side `{side}`')
    padpre = (0,) * max(0, inp.ndim-len(padpre)) + padpre
    padpost = (0,) * max(0, inp.ndim-len(padpost)) + padpost
    if inp.dim() != len(padpre) or inp.dim() != len(padpost):
        raise ValueError('Padding length too large')

    # Pad
    mode = ['nocheck'] * max(0, inp.ndim-len(mode)) + mode
    if all(m in ('zero', 'nocheck') for m in mode):
        return _pad_constant(inp, padpre, padpost, value)
    else:
        bound = [getattr(indexing, m) for m in mode]
        return _pad_bound(inp, padpre, padpost, bound)


def _pad_constant(inp, padpre, padpost, value):
    new_shape = [s + pre + post
                 for s, pre, post in zip(inp.shape, padpre, padpost)]
    out = inp.new_full(new_shape, value)
    slicer = [slice(pre, pre + s) for pre, s in zip(padpre, inp.shape)]
    out[tuple(slicer)] = inp
    return out


def _pad_bound(inp, padpre, padpost, bound):
    begin = list(map(lambda x: -x, padpre))
    end = tuple(d+p for d, p in zip(inp.shape, padpost))

    grid = [
        torch.arange(b, e, device=inp.device) for (b, e) in zip(begin, end)
    ]
    mult = [None] * inp.dim()
    for d, n in enumerate(inp.shape):
        grid[d], mult[d] = bound[d](grid[d], n)
    grid = list(meshgrid_list_ij(grid))
    if any(map(torch.is_tensor, mult)):
        for d in range(len(mult)):
            if not torch.is_tensor(mult[d]):
                continue
            for _ in range(d+1, len(mult)):
                mult[d].unsqueeze_(-1)
    mult = prod(mult)
    grid = sub2ind_list(grid, inp.shape)

    out = inp.flatten()[grid]
    if torch.is_tensor(mult) or mult != 1:
        out *= mult
    return out


def ensure_shape(
    inp: Tensor,
    shape: SequenceOrScalar[Optional[int]],
    mode: SequenceOrScalar[BoundLike] = 'constant',
    value: Number = 0,
    side: str = 'post',
    ceil: bool = False
):
    """Pad/crop a tensor so that it has a given shape

    Parameters
    ----------
    inp : tensor
        Input tensor
    shape : SequenceOrScalar[int]
        Output shape
    mode : SequenceOrScalar[BoundLike]
        Boundary mode
    value : scalar, default=0
        Value for mode `'constant'`
    side : "{'pre', 'post', 'both'}"
        Side to crop/pad

    Returns
    -------
    out : tensor
        Padded tensor with shape `shape`

    """
    if isinstance(shape, int):
        shape = [shape]
    shape = list(shape)
    shape = [None] * max(0, inp.ndim - len(shape)) + shape
    if inp.ndim < len(shape):
        inp = inp.reshape((1,) * max(0, len(shape) - inp.ndim) + inp.shape)
    inshape = inp.shape
    shape = [inshape[d] if shape[d] is None else shape[d]
             for d in range(len(shape))]
    ndim = len(shape)

    half = (lambda x: int(math.ceil(x/2))) if ceil else (lambda x: x//2)

    # crop
    if side == 'both':
        crop = [max(0, inshape[d] - shape[d]) for d in range(ndim)]
        index = tuple(slice(half(c), (half(c) - c) or None) for c in crop)
    elif side == 'pre':
        crop = [max(0, inshape[d] - shape[d]) for d in range(ndim)]
        index = tuple(slice(-c or None) for c in crop)
    else:  # side == 'post'
        index = tuple(slice(min(shape[d], inshape[d])) for d in range(ndim))
    inp = inp[index]

    # pad
    pad_size = [max(0, shape[d] - inshape[d]) for d in range(ndim)]
    if side == 'both':
        pad_size = [[half(p), p-half(p)] for p in pad_size]
        pad_size = [q for p in pad_size for q in p]
        side = None
    inp = pad(inp, tuple(pad_size), mode=mode, value=value, side=side)

    return inp


def roll(
    inp: Tensor,
    shifts: SequenceOrScalar[int] = 1,
    dims: Optional[SequenceOrScalar[int]] = None,
    bound: SequenceOrScalar[BoundLike] = 'circular'
):
    r"""Like `torch.roll`, but with any boundary condition

    !!! warning
        When `dims` is `None`, we do not flatten but shift all dimensions.
        This differs from the behavior of `torch.roll` .

    Parameters
    ----------
    inp : tensor
        Input
    shifts : SequenceOrScalar[int]
        Amount by which to roll.
        Positive shifts to the right, negative to the left.
    dims : SequenceOrScalar[int]
        Dimensions to roll.
        By default, shifts apply to all dimensions if a scalar,
        or to the last N if a sequence.
    bound : SequenceOrScalar[BoundLike]
        Boundary condition

    Returns
    -------
    out : tensor
        Rolled tensor

    """
    if dims is None:
        if isinstance(shifts, int):
            dims = list(range(inp.dim()))
        else:
            shifts = ensure_list(shifts)
            dims = list(range(-len(shifts), 0))
    dims = ensure_list(dims)
    shifts = ensure_list(shifts, len(dims))
    bound = map(to_fourier, ensure_list(bound, len(dims)))
    bound = [getattr(indexing, b + '_') for b in bound]

    grid = [torch.arange(n, device=inp.device) for n in inp.shape]
    mult = [1] * inp.dim()
    for d, s, b in zip(dims, shifts, bound):
        grid[d] -= s
        grid[d], mult[d] = b(grid[d], inp.shape[d])
    grid = list(meshgrid_list_ij(grid))
    if any(map(torch.is_tensor, mult)):
        mult = meshgrid_list_ij(mult)
    mult = prod(mult)
    grid = sub2ind_list(grid, inp.shape)

    out = inp.flatten()[grid]
    out *= mult
    return out
