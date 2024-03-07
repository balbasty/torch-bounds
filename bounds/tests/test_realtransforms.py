import torch
import scipy
import pytest
from ..utils import torch_version
torch_has_fft = torch_version('>=', (1, 8))

if torch_has_fft:
    from bounds import dct, dst, idct, idst
    from scipy.fft import (
        dct as scipy_dct, idct as scipy_idct,
        dst as scipy_dst, idst as scipy_idst,
    )


dtype = torch.double        # data type (double advised to check gradients)
types = (1, 2, 3)
norms = ("backward", "ortho", "forward")
sizes = (4, 5)


scipy_version = tuple(map(int, scipy.__version__.split('.')))
if scipy_version >= (1, 12, 0):
    def my_norm(norm):
        return norm
else:
    def my_norm(norm):
        return "ortho_scipy" if norm == "ortho" else norm


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_dct(type, norm, size):
    print(f'dct({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        dct(dat, norm=my_norm(norm), type=type),
        torch.as_tensor(
            scipy_dct(dat.numpy(), norm=norm, type=type)
        )
    )


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_idct(type, norm, size):
    print(f'idct({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        idct(dat, norm=my_norm(norm), type=type),
        torch.as_tensor(
            scipy_idct(dat.numpy(), norm=norm, type=type)
        )
    )


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_dst(type, norm, size):
    print(f'dst({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        dst(dat, norm=my_norm(norm), type=type),
        torch.as_tensor(
            scipy_dst(dat.numpy(), norm=norm, type=type)
        )
    )


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_idst(type, norm, size):
    print(f'idst({type}, {norm})[{size}]')
    dat = torch.randn([size], dtype=dtype)
    assert torch.allclose(
        idst(dat, norm=my_norm(norm), type=type),
        torch.as_tensor(
            scipy_idst(dat.numpy(), norm=norm, type=type)
        )
    )
