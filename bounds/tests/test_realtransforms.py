import torch
import scipy
from bounds import dct, dst, idct, idst
from scipy.fft import (
    dct as scipy_dct, idct as scipy_idct,
    dst as scipy_dst, idst as scipy_idst,
)
import pytest

dtype = torch.double        # data type (double advised to check gradients)
types = (1, 2, 3)
norms = ("backward", "ortho", "forward")
sizes = (4, 5)


scipy_version = tuple(map(int, scipy.__version__.split('.')))
if scipy_version >= (1, 12, 0):
    def scipy_norm(norm):
        return norm
else:
    def scipy_norm(norm):
        return "ortho_scipy" if norm == "ortho" else norm


@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_dct(type, norm, size):
    print(f'dct({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        dct(dat, norm=norm, type=type),
        torch.as_tensor(scipy_dct(dat, norm=scipy_norm(norm), type=type))
    )


@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_idct(type, norm, size):
    print(f'idct({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        idct(dat, norm=norm, type=type),
        torch.as_tensor(scipy_idct(dat, norm=scipy_norm(norm), type=type))
    )


@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_dst(type, norm, size):
    print(f'dst({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        dst(dat, norm=norm, type=type),
        torch.as_tensor(scipy_dst(dat, norm=scipy_norm(norm), type=type))
    )


@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_idst(type, norm, size):
    print(f'idst({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        idst(dat, norm=norm, type=type),
        torch.as_tensor(scipy_idst(dat, norm=scipy_norm(norm), type=type))
    )
