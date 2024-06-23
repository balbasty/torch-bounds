"""
This module contains functions that wrap out-of-bound indices back in-bounds,
according to some boundary condition.

Functions
---------
replicate
    Apply replicate boundary conditions to an index.
    **Aliases:**
    [`border`][bounds.indexing.border],
    [`nearest`][bounds.indexing.nearest],
    [`repeat`][bounds.indexing.repeat],
    [`edge`][bounds.indexing.edge].
dft
    Apply DFT boundary conditions to an index.
    **Aliases:**
    [`wrap`][bounds.indexing.wrap],
    [`gridwrap`][bounds.indexing.gridwrap],
    [`circular`][bounds.indexing.circular],
    [`circulant`][bounds.indexing.circulant].
dct1
    Apply DCT-I boundary conditions to an index.
    **Alias:** [`mirror`][bounds.indexing.mirror]
dct2
    Apply DCT-II boundary conditions to an index.
    **Aliases**:
    [`reflect`][bounds.indexing.reflect],
    [`gridmirror`][bounds.indexing.gridmirror],
    [`neumann`][bounds.indexing.neumann].
dst1
    Apply DST-I boundary conditions to an index.
    **Alias:** [`antimirror`][bounds.indexing.antimirror]
dst2
    Apply DST-II boundary conditions to an index.
    **Aliases**:
    [`antireflect`][bounds.indexing.antireflect],
    [`dirichlet`][bounds.indexing.dirichlet].
nocheck
    Do not wrap indices (assume they are inbounds)

"""
__all__ = [
    'nocheck',
    'replicate', 'repeat', 'nearest', 'border', 'edge',
    'dft', 'wrap', 'gridwrap', 'circular', 'circulant',
    'dct1', 'mirror',
    'dct2', 'reflect', 'reflection', 'gridmirror', 'neumann',
    'dst1', 'antimirror',
    'dst2', 'antireflect', 'dirichlet',
]
import torch
from torch import Tensor
from typing import Tuple
from ._utils import floor_div_int


def nocheck(i, n):
    """Assume all indices are inbounds"""
    return i, 1


def replicate(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index

    !!! info "Aliases"
        [`border`][bounds.indexing.border],
        [`nearest`][bounds.indexing.nearest],
        [`repeat`][bounds.indexing.repeat]

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for replicate)

    """
    return (replicate_script(i, n) if torch.is_tensor(i) else
            replicate_int(i, n))


def replicate_int(i, n):
    return min(max(i, 0), n-1), 1


@torch.jit.script
def replicate_script(i, n: int) -> Tuple[Tensor, int]:
    return i.clamp(min=0, max=n-1), 1


def dft(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index

    !!! info "Aliases"
        [`wrap`][bounds.indexing.wrap],
        [`circular`][bounds.indexing.circular]

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dft)

    """
    return dft_script(i, n) if torch.is_tensor(i) else dft_int(i, n)


def dft_int(i, n):
    return i % n, 1


@torch.jit.script
def dft_script(i, n: int) -> Tuple[Tensor, int]:
    return i.remainder(n), 1


def dct2(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index

    !!! info "Aliases"
        [`reflect`][bounds.indexing.reflect],
        [`neumann`][bounds.indexing.neumann]

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dct2)

    """
    return dct2_script(i, n) if torch.is_tensor(i) else dct2_int(i, n)


def dct2_int(i: int, n: int) -> Tuple[int, int]:
    n2 = n * 2
    i = (n2 - 1) - i if i < 0 else i
    i = i % n2
    i = (n2 - 1) - i if i >= n else i
    return i, 1


@torch.jit.script
def dct2_script(i, n: int) -> Tuple[Tensor, int]:
    n2 = n * 2
    i = torch.where(i < 0, (n2 - 1) - i, i)
    i = i.remainder(n2)
    i = torch.where(i >= n, (n2 - 1) - i, i)
    return i, 1


def dct1(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    !!! info "Aliases"
        [`mirror`][bounds.indexing.mirror]

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dct1)

    """
    return dct1_script(i, n) if torch.is_tensor(i) else dct1_int(i, n)


def dct1_int(i: int, n: int) -> Tuple[int, int]:
    if n == 1:
        return 0, 1
    n2 = (n - 1) * 2
    i = abs(i) % n2
    i = n2 - i if i >= n else i
    return i, 1


@torch.jit.script
def dct1_script(i, n: int) -> Tuple[Tensor, int]:
    if n == 1:
        return torch.zeros_like(i), 1
    n2 = (n - 1) * 2
    i = i.abs().remainder(n2)
    i = torch.where(i >= n, n2 - i, i)
    return i, 1


def dst1(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index

    !!! info "Aliases"
        [`antimirror`][bounds.indexing.antimirror]

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, 0, -1}
        Sign of the transformation

    """
    return dst1_script(i, n) if torch.is_tensor(i) else dst1_int(i, n)


def dst1_int(i: int, n: int) -> Tuple[int, int]:
    n2 = 2 * (n + 1)

    # sign
    ii = (2*n - i) if i < 0 else i
    ii = (ii % n2) % (n + 1)
    x = 0 if ii == n else 1
    x = -x if (i / (n + 1)) % 2 >= 1 else x

    # index
    i = -i - 2 if i < 0 else i
    i = i % n2
    i = (n2 - 2) - i if i > n else i
    i = min(max(i, 0), n-1)
    return i, x


@torch.jit.script
def dst1_script(i, n: int) -> Tuple[Tensor, Tensor]:
    n2 = 2 * (n + 1)

    # sign
    #   zeros
    ii = torch.where(i < 0, 2*n - i, i).remainder(n2).remainder(n + 1)
    x = (ii != n).to(torch.int8)
    #   +/- ones
    ii = torch.where(i < 0, n - 1 - i, i)
    x = torch.where(floor_div_int(ii, n + 1).remainder(2) >= 1, -x, x)

    # index
    i = torch.where(i < 0, -2 - i, i)
    i = i.remainder(n2)
    i = torch.where(i > n, (n2 - 2) - i, i)
    i = i.clamp(0, n-1)
    return i, x


def dst2(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index

    !!! info "Aliases"
        [`antireflect`][bounds.indexing.antireflect],
        [`dirichlet`][bounds.indexing.dirichlet]

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, 0, -1}
        Sign of the transformation (always 1 for dct1)

    """
    return dst2_script(i, n) if torch.is_tensor(i) else dst2_int(i, n)


def dst2_int(i: int, n: int) -> Tuple[int, int]:
    x = -1 if (i/n) % 2 >= 1 else 1
    return dct2_int(i, n)[0], x


@torch.jit.script
def dst2_script(i, n: int) -> Tuple[Tensor, Tensor]:
    x = torch.ones([1], dtype=torch.int8, device=i.device)
    x = torch.where(floor_div_int(i, n).remainder(2) >= 1, -x, x)
    return dct2_script(i, n)[0], x


nearest = border = repeat = edge = replicate
"""Alias for [`replicate`][bounds.indexing.replicate]"""

reflect = reflection = gridmirror = neumann = dct2
"""Alias for [`dct2`][bounds.indexing.dct2]"""

mirror = dct1
"""Alias for [`dct1`][bounds.indexing.dct1]"""

antireflect = dirichlet = dst2
"""Alias for [`dst2`][bounds.indexing.dst2]"""

antimirror = dst1
"""Alias for [`dst1`][bounds.indexing.dst1]"""

wrap = gridwrap = circular = circulant = circulant = dft
"""Alias for [`dft`][bounds.indexing.dft]"""
