import os
import torch
from torch import Tensor
from types import GeneratorType as generator
from typing import List, Any, Optional


def ensure_list(x: Any, length: Optional[int] = None, crop: bool = True,
                **kwargs) -> List:
    """
    Ensure that an object is a list

    The output list is of length at least `length`.
    When `crop` is `True`, its length is also at most `length`.
    If needed, the last value is replicated, unless `default` is provided.

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, range, or generator, it is converted to a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if length and len(x) < length:
        default = [kwargs.get('default', x[-1] if x else None)]
        x += default * (length - len(x))
    if length and crop:
        x = x[:length]
    return x


def prod(sequence, inplace=False):
    """Perform the product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    inplace : bool, default=False
        Perform the product inplace (using `__imul__` instead of `__mul__`).

    Returns
    -------
    product :
        Product of the elements in the sequence.

    """
    accumulate = None
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        elif inplace:
            accumulate *= elem
        else:
            accumulate = accumulate * elem
    return accumulate


def _compare_versions(version1, mode, version2):
    for v1, v2 in zip(version1, version2):
        if mode in ('gt', '>'):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('ge', '>='):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('lt', '<'):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        elif mode in ('le', '<='):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
    if mode in ('gt', 'lt', '>', '<'):
        return False
    else:
        return True


def torch_version(mode, version):
    """Check torch version

    Parameters
    ----------
    mode : {'<', '<=', '>', '>='}
    version : tuple[int]

    Returns
    -------
    True if "torch.version <mode> version"

    """
    current_version, *cuda_variant = torch.__version__.split('+')
    major, minor, patch, *_ = current_version.split('.')
    # strip alpha tags
    for x in 'abcdefghijklmnopqrstuvwxy':
        if x in patch:
            patch = patch[:patch.index(x)]
    current_version = (int(major), int(minor), int(patch))
    version = ensure_list(version)
    return _compare_versions(current_version, mode, version)


# floor_divide returns wrong results for negative values, because it truncates
# instead of performing a proper floor. In recent version of pytorch, it is
# advised to use div(..., rounding_mode='trunc'|'floor') instead.
# Here, we only use floor_divide on positive values so we do not care.
if torch_version('>=', (1, 8)):

    @torch.jit.script
    def floor_div(x, y) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='floor')

    @torch.jit.script
    def floor_div_int(x, y: int) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='floor')

    @torch.jit.script
    def trunc_div(x, y) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='trunc')

    @torch.jit.script
    def trunc_div_int(x, y: int) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='trunc')

else:

    @torch.jit.script
    def floor_div(x, y) -> torch.Tensor:
        return (x / y).floor().to(x.dtype)

    @torch.jit.script
    def floor_div_int(x, y: int) -> torch.Tensor:
        return (x / y).floor().to(x.dtype)

    @torch.jit.script
    def trunc_div(x, y) -> torch.Tensor:
        int_dtype = torch.long if x.is_floating_point() else x.dtype
        return (x / y).to(int_dtype).to(x.dtype)

    @torch.jit.script
    def trunc_div_int(x, y: int) -> torch.Tensor:
        int_dtype = torch.long if x.is_floating_point() else x.dtype
        return (x / y).to(int_dtype).to(x.dtype)


if torch_version('>=', (1, 10)):
    # torch >= 1.10
    # -> use `indexing` keyword

    if not int(os.environ.get('PYTORCH_JIT', '1')):
        # JIT deactivated -> torch.meshgrid takes an unpacked list of tensors

        @torch.jit.script
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(*tensors, indexing='ij'))

        @torch.jit.script
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(*tensors, indexing='xy'))

    else:
        # JIT activated -> torch.meshgrid takes a packed list of tensors

        @torch.jit.script
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors, indexing='ij'))

        @torch.jit.script
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors, indexing='xy'))

else:
    # torch < 1.10
    # -> implement "xy" mode manually

    if not int(os.environ.get('PYTORCH_JIT', '1')):
        # JIT deactivated -> torch.meshgrid takes an unpacked list of tensors

        @torch.jit.script
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors))

        @torch.jit.script
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            grid = list(torch.meshgrid(*tensors))
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid

    else:
        # JIT activated -> torch.meshgrid takes a packed list of tensors

        @torch.jit.script
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors))

        @torch.jit.script
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            grid = list(torch.meshgrid(tensors))
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid


@torch.jit.script
def reverse_list_int(x: List[int]) -> List[int]:
    """TorchScript equivalent to `x[::-1]`"""
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]


@torch.jit.script
def cumprod_list_int(x: List[int], reverse: bool = False,
                     exclusive: bool = False) -> List[int]:
    """Cumulative product of elements in the list

    Parameters
    ----------
    x : list[int]
        List of integers
    reverse : bool
        Cumulative product from right to left.
        Else, cumulative product from left to right (default).
    exclusive : bool
        Start series from 1.
        Else start series from first element (default).

    Returns
    -------
    y : list[int]
        Cumulative product

    """
    if len(x) == 0:
        lx: List[int] = []
        return lx
    if reverse:
        x = reverse_list_int(x)

    x0 = 1 if exclusive else x[0]
    lx = [x0]
    all_x = x[:-1] if exclusive else x[1:]
    for x1 in all_x:
        x0 = x0 * x1
        lx.append(x0)
    if reverse:
        lx = reverse_list_int(lx)
    return lx


@torch.jit.script
def sub2ind_list(subs: List[Tensor], shape: List[int]):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D,) list[tensor]
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) list[int]
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    ind = subs[-1]
    subs = subs[:-1]
    ind = ind.clone()
    stride = cumprod_list_int(shape[1:], reverse=True, exclusive=False)
    for i, s in zip(subs, stride):
        ind += i * s
    return ind
