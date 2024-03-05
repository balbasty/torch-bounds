import torch
from . import _realtransforms_from_fft as F


flipnorm = {
    'forward': 'backward',
    'backward': 'forward',
    'ortho': 'ortho',
    'ortho_scipy': 'ortho_scipy',
}
fliptype = {1: 1, 2: 3, 3: 2, 4: 4}


class DCTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        norm = norm or 'backward'
        if type not in (2, 3) and norm == 'ortho_scipy':
            norm = 'ortho'
        ctx.dim = dim
        ctx.norm = norm
        ctx.type = type
        return F.dctn(x, type=type, dim=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        if ctx.norm == 'ortho':
            # orthogonal: inverse == transpose
            x = F.idctn(x, type=ctx.type, dim=ctx.dim, norm=ctx.norm)
            return x, None, None, None

        scipy = 'scipy' in ctx.norm
        norm = "backward" if scipy else ctx.norm
        type = fliptype[ctx.type]
        prm = dict(dim=ctx.dim, type=type, square=True)
        spprm = dict(dim=ctx.dim, type=type, scipy=True, inverse=True)
        if type in (1, 3):
            x = x.clone()
        if scipy:
            x = F.ortho_prescale_(x, **spprm)
        x = F.ortho_prescale_(x, **prm)
        x = F.dctn(x, type=type, dim=ctx.dim, norm=norm)
        x = F.ortho_postscale_(x, **prm)
        if scipy:
            x = F.ortho_postscale_(x, **spprm)
            x = F.ortho_scale_(x, ctx.dim, type=type)
        return x, None, None, None


class DSTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        norm = norm or 'backward'
        if type not in (2, 3) and norm == 'ortho_scipy':
            norm = 'ortho'
        ctx.dim = dim
        ctx.norm = norm
        ctx.type = type
        return F.dstn(x, type=type, dim=dim, norm=norm)

    @staticmethod
    def backward(ctx, x):
        if ctx.norm == 'ortho':
            # orthogonal: inverse == transpose
            x = F.idstn(x, type=ctx.type, dim=ctx.dim, norm=ctx.norm)
            return x, None, None, None

        scipy = 'scipy' in ctx.norm
        norm = "backward" if scipy else ctx.norm
        type = fliptype[ctx.type]
        prm = dict(
            dim=ctx.dim, type=type, square=True, dst=True,
        )
        spprm = dict(
            dim=ctx.dim, type=type, scipy=True, inverse=True, dst=True,
        )
        if type in (1, 3):
            x = x.clone()
        if scipy:
            x = F.ortho_prescale_(x, **spprm)
        x = F.ortho_prescale_(x, **prm)
        x = F.dstn(x, type=type, dim=ctx.dim, norm=norm)
        x = F.ortho_postscale_(x, **prm)
        if scipy:
            x = F.ortho_postscale_(x, **spprm)
            x = F.ortho_scale_(x, ctx.dim, type=type)
        return x, None, None, None
