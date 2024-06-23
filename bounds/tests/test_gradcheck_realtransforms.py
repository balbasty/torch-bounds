import torch
from torch.autograd import gradcheck
import pytest
import inspect
from .._utils import torch_version
torch_has_fft = torch_version('>=', (1, 8))

if torch_has_fft:
    from bounds import dct, dst, idct, idst

# global parameters
dtype = torch.double        # data type (double advised to check gradients)

if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(True)
kwargs = dict(rtol=1., raise_exception=True)
if 'check_undefined_grad' in inspect.signature(gradcheck).parameters:
    kwargs['check_undefined_grad'] = False
if 'nondet_tol' in inspect.signature(gradcheck).parameters:
    kwargs['nondet_tol'] = 1e-3

# parameters
devices = [('cpu', 1)]
if torch.backends.openmp.is_available() or torch.backends.mkl.is_available():
    print('parallel backend available')
    devices.append(('cpu', 10))
if torch.cuda.is_available():
    print('cuda backend available')
    devices.append('cuda')

types = (1, 2, 3)
norms = ("backward", "ortho", "forward")
sizes = (4, 5)


def init_device(device):
    if isinstance(device, (list, tuple)):
        device, param = device
    else:
        param = 1 if device == 'cpu' else 0
    if device == 'cuda':
        torch.cuda.set_device(param)
        torch.cuda.init()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        device = '{}:{}'.format(device, param)
    else:
        assert device == 'cpu'
        torch.set_num_threads(param)
    return torch.device(device)


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_dct(device, type, norm, size):
    print(f'dct({type}, {norm})[{size}] on {device}')
    device = init_device(device)
    dat = torch.randn([size], dtype=dtype, device=device)
    dat.requires_grad = True
    assert gradcheck(dct, (dat, -1, norm, type), **kwargs)


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_idct(device, type, norm, size):
    print(f'idct({type}, {norm})[{size}] on {device}')
    device = init_device(device)
    dat = torch.randn([size], dtype=dtype, device=device)
    dat.requires_grad = True
    assert gradcheck(idct, (dat, -1, norm, type), **kwargs)


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_dst(device, type, norm, size):
    print(f'dst({type}, {norm})[{size}] on {device}')
    device = init_device(device)
    dat = torch.randn([size], dtype=dtype, device=device)
    dat.requires_grad = True
    assert gradcheck(dst, (dat, -1, norm, type), **kwargs)


@pytest.mark.skipif(not torch_has_fft, reason="requires pytorch 1.8")
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("size", sizes)
def test_gradcheck_idst(device, type, norm, size):
    print(f'idst({type}, {norm})[{size}] on {device}')
    device = init_device(device)
    dat = torch.randn([size], dtype=dtype, device=device)
    dat.requires_grad = True
    assert gradcheck(idst, (dat, -1, norm, type), **kwargs)
