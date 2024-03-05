# copied and adapted from
#   https://github.com/cupy/cupy/blob/v12.0.0/cupyx/scipy/fft/_realtransforms.py
#   https://github.com/cupy/cupy/blob/v12.0.0/LICENSE
#
# NOTE (YB 2023/05/12)
#   Scipy's "orthogonal" DST is not orthogonal
#   I've modified it to make it properly orthogonal, but it means my
#   version is not exactly equivalent to scipy/cupy anymore.
#   Scipy's buggy behaviour can be recovered with the option
#   `norm="ortho_scipy"`.
#
# NOTE (YB 2024/02/23)
#   Reported in https://github.com/scipy/scipy/issues/18452
#   Fixed by https://github.com/scipy/scipy/pull/18613      (scipy)
#   Fixed by https://github.com/cupy/cupy/pull/8058         (cupy)
"""Real-to-real transforms

cuFFT does not implement real-to-real FFTs. This module implements forward
and inverse DCT-II and DCT-III transforms using FFTs.

A length N DCT can be computed using a length N FFT and some additional
multiplications and reordering of entries.

The approach taken here is based on the work in [1]_, [2]_ and is discussed in
the freely-available online resources [3]_, [4]_.

The implementation here follows that approach with only minor modification to
match the normalization conventions in SciPy.

The modifications to turn a type II or III DCT to a DST were implemented as
described in [5]_.

.. [1] J. Makhoul, "A fast cosine transform in one and two dimensions," in
    IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28,
    no. 1, pp. 27-34, February 1980.

.. [2] M.J. Narasimha and A.M. Peterson, “On the computation of the discrete
    cosine  transform,” IEEE Trans. Commun., vol. 26, no. 6, pp. 934–936, 1978.

.. [3] http://fourier.eng.hmc.edu/e161/lectures/dct/node2.html

.. [4] https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft  # noqa

.. [5] X. Shao, S. G. Johnson. Type-II/III DCT/DST algorithms with reduced
    number of arithmetic operations, Signal Processing, Volume 88, Issue 6,
    pp. 1553-1564, 2008.
"""
import math
import numbers
import operator
import torch
from torch import fft as _fft


__all__ = ['dct', 'dctn', 'dst', 'dstn', 'idct', 'idctn', 'idst', 'idstn']


def dct(x, type=2, n=None, dim=-1, norm=None, overwrite_x=False):
    """Return the Discrete Cosine Transform of an array, x.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2. Currently CuPy only
        supports types 2 and 3.
    n : int, optional:
        Length of the transform.  If ``n < x.shape[dim]``, `x` is
        truncated. If ``n > x.shape[dim]``, `x` is zero-padded.
        The default results in ``n = x.shape[dim]``.
    dim : int, optional
        Axis along which the dct is computed; the default is over the
        last dim (i.e., ``dim=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : torch.Tensor
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.dct`

    Notes
    -----
    For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal
    to MATLAB ``dct(x)``.

    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 1, 2 and 3 means the transform definition is
    modified to give orthogonality of the DCT matrix (see below).

    For ``norm="backward"``, there is no scaling on `dct` and the `idct` is
    scaled by ``1/N`` where ``N`` is the "logical" size of the DCT. For
    ``norm="forward"`` the ``1/N`` normalization is applied to the forward
    `dct` instead and the `idct` is unnormalized.

    CuPy currently only supports DCT types 2 and 3. 'The' DCT generally
    refers to DCT type 2, and 'the' Inverse DCT generally refers to DCT
    type 3 [1]_. See the :func:`scipy.fft.dct` documentation for a full
    description of each type.

    References
    ----------
    .. [1] Wikipedia, "Discrete cosine transform",
           https://en.wikipedia.org/wiki/Discrete_cosine_transform

    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = dct(x.real, type, n, dim, norm, overwrite_x)
        out = out + 1j * dct(x.imag, type, n, dim, norm, overwrite_x)
        return out

    x = _promote_dtype(x)

    kwargs = dict(n=n, dim=dim, norm=norm, forward=True)
    if type == 2:
        return _dct_or_dst_type2(x, dst=False, **kwargs)
    elif type == 3:
        return _dct_or_dst_type3(x, dst=False, **kwargs)
    elif type == 1:
        return _dct_type1(x, **kwargs)
    elif type == 4:
        raise NotImplementedError(
            'Only DCT I, II and III have been implemented.'
        )
    else:
        raise ValueError('invalid DCT type')


def dst(x, type=2, n=None, dim=-1, norm=None, overwrite_x=False):
    """Return the Discrete Sine Transform of an array, x.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[dim]``, `x` is
        truncated.  If ``n > x.shape[dim]``, `x` is zero-padded. The
        default results in ``n = x.shape[dim]``.
    dim : int, optional
        Axis along which the dst is computed; the default is over the
        last dim (i.e., ``dim=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    dst : torch.Tensor
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.dst`

    Notes
    -----

    For ``norm="ortho"`` both the `dst` and `idst` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 2 and 3 means the transform definition is
    modified to give orthogonality of the DST matrix (see below).

    For ``norm="backward"``, there is no scaling on the `dst` and the `idst` is
    scaled by ``1/N`` where ``N`` is the "logical" size of the DST.

    See the :func:`scipy.fft.dst` documentation for a full description of each
    type. CuPy currently only supports DST types 2 and 3.
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = dst(x.real, type, n, dim, norm, overwrite_x)
        out = out + 1j * dst(x.imag, type, n, dim, norm, overwrite_x)
        return out

    x = _promote_dtype(x)

    kwargs = dict(n=n, dim=dim, norm=norm, forward=True)
    if type == 2:
        return _dct_or_dst_type2(x, dst=True, **kwargs)
    elif type == 3:
        return _dct_or_dst_type3(x, dst=True, **kwargs)
    elif type == 1:
        return _dst_type1(x, **kwargs)
    elif type == 4:
        raise NotImplementedError(
            'Only DST I, II and III have been implemented.'
        )
    else:
        raise ValueError('invalid DST type')


def idct(x, type=2, n=None, dim=-1, norm=None, overwrite_x=False):
    """Return the Inverse Discrete Cosine Transform of an array, x.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[dim]``, `x` is
        truncated.  If ``n > x.shape[dim]``, `x` is zero-padded. The
        default results in ``n = x.shape[dim]``.
    dim : int, optional
        Axis along which the idct is computed; the default is over the
        last dim (i.e., ``dim=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    idct : torch.Tensor
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.idct`

    Notes
    -----
    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
    MATLAB ``idct(x)``.

    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 1, 2 and 3 means the transform definition is
    modified to give orthogonality of the IDCT matrix (see `dct` for the full
    definitions).

    'The' IDCT is the IDCT-II, which is the same as the normalized DCT-III
    [1]_. See the :func:`scipy.fft.dct` documentation for a full description of
    each type. CuPy currently only supports DCT types 2 and 3.

    References
    ----------
    .. [1] Wikipedia, "Discrete sine transform",
           https://en.wikipedia.org/wiki/Discrete_sine_transform
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = idct(x.real, type, n, dim, norm, overwrite_x)
        out = out + 1j * idct(x.imag, type, n, dim, norm, overwrite_x)
        return out

    x = _promote_dtype(x)

    kwargs = dict(n=n, dim=dim, norm=norm, forward=False)
    if type == 2:
        # DCT-III is the inverse of DCT-II
        return _dct_or_dst_type3(x, dst=False, **kwargs)
    elif type == 3:
        # DCT-II is the inverse of DCT-III
        return _dct_or_dst_type2(x, dst=False, **kwargs)
    elif type == 1:
        # DCT-I is the inverse of DCT-I
        return _dct_type1(x, **kwargs)
    elif type == 4:
        raise NotImplementedError(
            'Only DCT I, II and III have been implemented.'
        )
    else:
        raise ValueError('invalid DCT type')


def idst(x, type=2, n=None, dim=-1, norm=None, overwrite_x=False):
    """Return the Inverse Discrete Sine Transform of an array, x.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[dim]``, `x` is
        truncated.  If ``n > x.shape[dim]``, `x` is zero-padded. The
        default results in ``n = x.shape[dim]``.
    dim : int, optional
        Axis along which the idst is computed; the default is over the
        last dim (i.e., ``dim=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    idst : torch.Tensor of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.idst`

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see :func:`scipy.fft.dst`.
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = idst(x.real, type, n, dim, norm, overwrite_x)
        out = out + 1j * idst(x.imag, type, n, dim, norm, overwrite_x)
        return out

    x = _promote_dtype(x)

    kwargs = dict(n=n, dim=dim, norm=norm, forward=False)
    if type == 2:
        # DST-III is the inverse of DST-II
        return _dct_or_dst_type3(x, dst=True, **kwargs)
    elif type == 3:
        # DST-II is the inverse of DST-III
        return _dct_or_dst_type2(x, dst=True, **kwargs)
    elif type == 1:
        # DST-I is the inverse of DST-I
        return _dst_type1(x, **kwargs)
    elif type == 4:
        raise NotImplementedError(
            'Only DST I, II and III have been implemented.'
        )
    else:
        raise ValueError('invalid DST type')


def dctn(x, type=2, s=None, dim=None, norm=None, overwrite_x=False):
    """Compute a multidimensional Discrete Cosine Transform.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `dim` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `dim` is not None, then `s` is
        ``numpy.take(x.shape, dim, dim=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    dim : int or array_like of ints or None, optional
        Axes over which the DCT is computed. If not given, the last ``len(s)``
        dimensions are used, or all dimensions if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : torch.Tensor of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.dctn`

    Notes
    -----
    For full details of the DCT types and normalization modes, as well as
    references, see `dct`.
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = dctn(x.real, type, s, dim, norm, overwrite_x)
        out = torch.complex(out, dctn(x.imag, type, s, dim, norm, overwrite_x))
        return out

    shape, dim = _init_nd_shape_and_dims(x, s, dim)
    x = _promote_dtype(x)

    if len(dim) == 0:
        return x

    kwargs = dict(type=type, norm=norm, overwrite_x=overwrite_x)
    for n, dim1 in zip(shape, dim):
        x = dct(x, n=n, dim=dim1, **kwargs)
    return x


def idctn(x, type=2, s=None, dim=None, norm=None, overwrite_x=False):
    """Compute a multidimensional Discrete Cosine Transform.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `dim` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `dim` is not None, then `s` is
        ``numpy.take(x.shape, dim, dim=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    dim : int or array_like of ints or None, optional
        Axes over which the IDCT is computed. If not given, the last ``len(s)``
        dimensions are used, or all dimensions if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : torch.Tensor of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.idctn`

    Notes
    -----
    For full details of the IDCT types and normalization modes, as well as
    references, see :func:`scipy.fft.idct`.
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = idctn(x.real, type, s, dim, norm, overwrite_x)
        out = out + 1j * idctn(x.imag, type, s, dim, norm, overwrite_x)
        return out

    shape, dim = _init_nd_shape_and_dims(x, s, dim)
    x = _promote_dtype(x)

    if len(dim) == 0:
        return x

    for n, dim1 in zip(shape, dim):
        x = idct(
            x, type=type, n=n, dim=dim1, norm=norm, overwrite_x=overwrite_x
        )
    return x


def dstn(x, type=2, s=None, dim=None, norm=None, overwrite_x=False):
    """Compute a multidimensional Discrete Sine Transform.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `dim` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `dim` is not None, then `s` is
        ``numpy.take(x.shape, dim, dim=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    dim : int or array_like of ints or None, optional
        Axes over which the DST is computed. If not given, the last ``len(s)``
        dimensions are used, or all dimensions if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : torch.Tensor of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.dstn`

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see :func:`scipy.fft.dst`.
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = dstn(x.real, type, s, dim, norm, overwrite_x)
        out = out + 1j * dstn(x.imag, type, s, dim, norm, overwrite_x)
        return out

    shape, dim = _init_nd_shape_and_dims(x, s, dim)
    x = _promote_dtype(x)

    if len(dim) == 0:
        return x

    for n, dim1 in zip(shape, dim):
        x = dst(
            x, type=type, n=n, dim=dim1, norm=norm, overwrite_x=overwrite_x
        )
    return x


def idstn(x, type=2, s=None, dim=None, norm=None, overwrite_x=False):
    """Compute a multidimensional Discrete Sine Transform.

    Parameters
    ----------
    x : torch.Tensor
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `dim` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `dim` is not None, then `s` is
        ``numpy.take(x.shape, dim, dim=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    dim : int or array_like of ints or None, optional
        Axes over which the IDST is computed. If not given, the last ``len(s)``
        dimensions are used, or all dimensions if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : torch.Tensor of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.idstn`

    Notes
    -----
    For full details of the IDST types and normalization modes, as well as
    references, see :func:`scipy.fft.idst`.
    """
    if x.dtype.is_complex:
        # separable application on real and imaginary parts
        out = idstn(x.real, type, s, dim, norm, overwrite_x)
        out = out + 1j * idstn(x.imag, type, s, dim, norm, overwrite_x)
        return out

    shape, dim = _init_nd_shape_and_dims(x, s, dim)
    x = _promote_dtype(x)

    if len(dim) == 0:
        return x

    for n, dim1 in zip(shape, dim):
        x = idst(
            x, type=type, n=n, dim=dim1, norm=norm, overwrite_x=overwrite_x
        )
    return x


def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int."""
    if isinstance(x, numbers.Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or 'value'
        raise ValueError(
            f'{name} must be a scalar or iterable of integers'
        ) from e

    return x


def _init_nd_shape_and_dims(x, shape, dims):
    """Handles shape and dims arguments for nd transforms."""
    noshape = shape is None
    nodims = dims is None

    if not nodims:
        dims = _iterable_of_int(dims, 'dim')
        dims = [a + x.ndim if a < 0 else a for a in dims]

        if any(a >= x.ndim or a < 0 for a in dims):
            raise ValueError('dims exceeds dimensionality of input')
        if len(set(dims)) != len(dims):
            raise ValueError('all dims must be unique')

    if not noshape:
        shape = _iterable_of_int(shape, 'shape')
        nshape = len(shape)
        if dims and len(dims) != nshape:
            raise ValueError(
                'when given, dims and shape arguments'
                ' have to be of the same length'
            )
        if nodims:
            if nshape > x.ndim:
                raise ValueError('shape requires more dims than are present')
            dims = range(x.ndim - len(shape), x.ndim)

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, dims)]
    elif nodims:
        shape = list(x.shape)
        dims = range(x.ndim)
    else:
        shape = [x.shape[a] for a in dims]

    if any(s < 1 for s in shape):
        raise ValueError(
            f'invalid number of data points ({shape}) specified'
        )

    return shape, dims


sqrt2 = math.sqrt(2)


def ortho_prescale_(
        x, dim, type, dst=False, inverse=False, square=False, scipy=False
):
    if not isinstance(dim, numbers.Number):
        for d in dim:
            x = ortho_prescale_(x, d, type, dst, inverse, square, scipy)
        return x

    f = 0.5 * sqrt2 if inverse else sqrt2
    if square:
        f *= f
    x = x.movedim(dim, 0)
    if type == 3:
        index = -1 if (dst and not scipy) else 0
        x[index] *= f
    elif type == 1 and not dst:
        x[0] *= f
        x[-1] *= f
    x = x.movedim(0, dim)
    return x


def ortho_postscale_(
        x, dim, type, dst=False, inverse=False, square=False, scipy=False
):
    if not isinstance(dim, numbers.Number):
        for d in dim:
            x = ortho_postscale_(x, d, type, dst, inverse, square, scipy)
        return x

    f = sqrt2 if inverse else 0.5 * sqrt2
    if square:
        f *= f
    x = x.movedim(dim, 0)
    if type == 2:
        index = -1 if (dst and not scipy) else 0
        x[index] *= f
    elif type == 1 and not dst:
        x[0] *= f
        x[-1] *= f
    x = x.movedim(0, dim)
    return x


def get_ortho_scale(x, dim, type):
    if not isinstance(dim, numbers.Number):
        f = 1
        for d in dim:
            f *= get_ortho_scale(x, d, type)
        return f
    n = x.shape[dim]
    delta = -1 if type == 1 else 0
    f = 1 / math.sqrt(2 * (n + delta))
    return f


def ortho_scale_(x, dim, type, inverse=False, square=False):
    f = get_ortho_scale(x, dim, type)
    if square:
        f *= f
    if inverse:
        f = 1 / f
    x *= f
    return x


def _cook_shape(a, s, dims, value_type):
    if s is None or s == a.shape:
        return a
    if (value_type == 'C2R') and (s[-1] is not None):
        s = list(s)
        s[-1] = s[-1] // 2 + 1
    for sz, dim in zip(s, dims):
        if (sz is not None) and (sz != a.shape[dim]):
            shape = list(a.shape)
            if shape[dim] > sz:
                index = [slice(None)] * a.ndim
                index[dim] = slice(0, sz)
                a = a[tuple(index)]
            else:
                index = [slice(None)] * a.ndim
                index[dim] = slice(0, shape[dim])
                shape[dim] = sz
                z = a.new_zeros(shape)  # C layout by default
                z[tuple(index)] = a
                a = z
    return a


def _promote_dtype(x):
    dtype = x.dtype
    if not dtype.is_floating_point:
        # use float64 instead of promote_types to match SciPy's behavior
        float_dtype = torch.float64
    elif dtype is torch.float16:
        float_dtype = torch.float32
    else:
        float_dtype = dtype
    assert float_dtype in (torch.float32, torch.float64)
    return x.to(float_dtype, copy=False)


def _get_dct_norm_factor(n, inorm, dct_type=2):
    """Normalization factors for DCT/DST I-IV.

    Parameters
    ----------
    n : int
        Data size.
    inorm : {'none', 'sqrt', 'full'}
        When `inorm` is 'none', the scaling factor is 1.0 (unnormalized). When
        `inorm` is 'sqrt', scaling by ``1/sqrt(d)`` as needed for an orthogonal
        transform is used. When `inorm` is 'full', normalization by ``1/d`` is
        applied. The value of ``d`` depends on both `n` and the `dct_type`.
    dct_type : {1, 2, 3, 4}
        Which type of DCT or DST is being normalized?.

    Returns
    -------
    fct : float
        The normalization factor.
    """
    if inorm == 'none':
        return 1
    delta = -1 if dct_type == 1 else 0
    d = 2 * (n + delta)
    if inorm == 'full':
        fct = 1 / d
    elif inorm == 'sqrt':
        fct = 1 / math.sqrt(d)
    else:
        raise ValueError('expected inorm = "none", "sqrt" or "full"')
    return fct


def _reshuffle_dct2(x, dim, dst=False):
    """Reorder entries to allow computation of DCT/DST-II via FFT."""
    sl_even = [slice(None)] * x.ndim
    sl_even[dim] = slice(0, None, 2)
    sl_even = tuple(sl_even)
    sl_odd = [slice(None)] * x.ndim
    sl_odd[dim] = slice(1, None, 2)
    sl_odd = tuple(sl_odd)
    if dst:
        x = torch.cat((x[sl_even], -x[sl_odd].flip(dim)), dim=dim)
    else:
        x = torch.cat((x[sl_even], x[sl_odd].flip(dim)), dim=dim)
    return x


def _mult_factor_dct2(
        n, n_truncate, norm_factor, dtype=torch.float32, device=None
):
    real = torch.zeros(n_truncate, dtype=dtype, device=device)
    imag = torch.arange(n_truncate, dtype=dtype, device=device)
    imag *= -math.pi / (2 * n)
    out = torch.complex(real, imag).exp_().mul_(2 * norm_factor)
    return out


def _exp_factor_dct2(x, n, dim, norm_factor, n_truncate=None):
    """Twiddle & scaling factors for computation of DCT/DST-II via FFT."""
    if n_truncate is None:
        n_truncate = n
    tmp = _mult_factor_dct2(n, n_truncate, norm_factor, x.dtype, x.device)

    if x.ndim == 1:
        return tmp
    tmp_shape = [1] * x.ndim
    tmp_shape[dim] = n_truncate
    tmp_shape = tuple(tmp_shape)
    return tmp.reshape(tmp_shape)


def _dct_or_dst_type2(
    x, n=None, dim=-1, forward=True, norm=None, dst=False,
):
    """Forward DCT/DST-II (or inverse DCT/DST-III) along a single dim

    Parameters
    ----------
    x : torch.Tensor
        The data to transform.
    n : int
        The size of the transform. If None, ``x.shape[dim]`` is used.
    dim : int
        Axis along which the transform is applied.
    forward : bool
        Set true to indicate that this is a forward DCT-II as opposed to an
        inverse DCT-III (The difference between the two is only in the
        normalization factor).
    norm : {None, 'ortho', 'forward', 'backward', 'ortho_scipy'}
        The normalization convention to use.
        If `'ortho_scipy'`, use scipy's incorrect `'ortho`` normalization.
    dst : bool
        If True, a discrete sine transform is computed rather than the discrete
        cosine transform.
    overwrite_x : bool
        Indicates that it is okay to overwrite x. In practice, the current
        implementation never performs the transform in-place.

    Returns
    -------
    y: torch.Tensor
        The transformed array.
    """
    if dim < -x.ndim or dim >= x.ndim:
        raise IndexError('dim out of range')
    if dim < 0:
        dim += x.ndim
    if n is not None and n < 1:
        raise ValueError(
            f'invalid number of data points ({n}) specified'
        )

    x = _cook_shape(x, (n,), (dim,), 'R2R')
    n = x.shape[dim]

    x = _reshuffle_dct2(x, dim, dst)

    ortho_scipy = norm == 'ortho_scipy'
    if ortho_scipy:
        norm = 'ortho'

    if norm == 'ortho':
        inorm = 'sqrt'
    elif norm == 'forward':
        inorm = 'full' if forward else 'none'
    else:
        inorm = 'none' if forward else 'full'
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=2)

    x = _fft.fft(x, n=n, dim=dim)
    tmp = _exp_factor_dct2(x.real, n, dim, norm_factor)
    x *= tmp  # broadcasting
    x = torch.real(x)

    if dst and ortho_scipy:
        x = x.flip(dim)

    if norm == 'ortho':
        x = x.movedim(dim, 0)
        x[0] *= sqrt2 * 0.5
        x = x.movedim(0, dim)

    if dst and not ortho_scipy:
        x = x.flip(dim)

    return x


def _reshuffle_dct3(y, n, dim, dst):
    """Reorder entries to allow computation of DCT/DST-II via FFT."""
    x = torch.empty_like(y)
    n_half = (n + 1) // 2

    # Store first half of y in the even entries of the output
    sl_even = [slice(None)] * y.ndim
    sl_even[dim] = slice(0, None, 2)
    sl_even = tuple(sl_even)

    sl_half = [slice(None)] * y.ndim
    sl_half[dim] = slice(0, n_half)
    x[sl_even] = y[tuple(sl_half)]

    # Store the second half of y in the odd entries of the output
    sl_odd = [slice(None)] * y.ndim
    sl_odd[dim] = slice(1, None, 2)
    sl_odd = tuple(sl_odd)

    # sl_half[dim] = slice(-1, n_half - 1, -1)
    sl_half[dim] = slice(n_half, None)
    if dst:
        x[sl_odd] = -y[tuple(sl_half)].flip(dim)
    else:
        x[sl_odd] = y[tuple(sl_half)].flip(dim)
    return x


def _mult_factor_dct3(n, norm_factor, dtype=torch.float32, device=None):
    real = torch.zeros(n, dtype=dtype, device=device)
    imag = torch.arange(n, dtype=dtype, device=device)
    imag *= math.pi / (2 * n)
    out = torch.complex(real, imag).exp_().mul_(2 * norm_factor * n)
    return out


# _mult_factor_dct3 = ElementwiseKernel(
#     in_params='R xr, int32 N, R norm_factor',
#     out_params='C y',
#     operation="""
#     C j(0., 1.);
#     y = (R)(2 * N * norm_factor) * exp(j * (R)(i * M_PI / (2 * N)));""",
# )


def _exp_factor_dct3(x, n, dim, norm_factor):
    """Twiddle & scaling factors for computation of DCT/DST-III via FFT."""
    tmp = _mult_factor_dct3(n, norm_factor, x.dtype, x.device)
    if x.ndim == 1:
        return tmp
    # prepare shape for broadcasting along non-transformed axes
    tmp_shape = [1] * x.ndim
    tmp_shape[dim] = n
    tmp_shape = tuple(tmp_shape)
    return tmp.reshape(tmp_shape)


def _dct_or_dst_type3(
    x, n=None, dim=-1, norm=None, forward=True, dst=False,
):
    """Forward DCT/DST-III (or inverse DCT/DST-II) along a single dim.

    Parameters
    ----------
    x : torch.Tensor
        The data to transform.
    n : int
        The size of the transform. If None, ``x.shape[dim]`` is used.
    dim : int
        Axis along which the transform is applied.
    forward : bool
        Set true to indicate that this is a forward DCT-III as opposed to an
        inverse DCT-II (The difference between the two is only in the
        normalization factor).
    norm : {None, 'ortho', 'forward', 'backward', 'ortho_scipy'}
        The normalization convention to use.
        If `'ortho_scipy'`, use scipy's incorrect `'ortho`` normalization.
    dst : bool
        If True, a discrete sine transform is computed rather than the discrete
        cosine transform.
    overwrite_x : bool
        Indicates that it is okay to overwrite x. In practice, the current
        implementation never performs the transform in-place.

    Returns
    -------
    y: torch.Tensor
        The transformed array.

    """
    if dim < -x.ndim or dim >= x.ndim:
        raise IndexError('dim out of range')
    if dim < 0:
        dim += x.ndim
    if n is not None and n < 1:
        raise ValueError(
            f'invalid number of data points ({n}) specified'
        )

    x = _cook_shape(x, (n,), (dim,), 'R2R')
    n = x.shape[dim]

    ortho_scipy = norm == 'ortho_scipy'
    if ortho_scipy:
        norm = 'ortho'

    # determine normalization factor
    if norm == 'ortho':
        sl0_scale = 0.5 * sqrt2
        inorm = 'sqrt'
    elif norm == 'forward':
        sl0_scale = 0.5
        inorm = 'full' if forward else 'none'
    elif norm == 'backward' or norm is None:
        sl0_scale = 0.5
        inorm = 'none' if forward else 'full'
    else:
        raise ValueError(f'Invalid norm value "{norm}", should be "backward", '
                         '"ortho" or "forward"')
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=3)

    if dst:
        x = x.flip(dim)
        if ortho_scipy:
            sl0 = [slice(None)] * x.ndim
            sl0[dim] = slice(-1, None)
            x[tuple(sl0)] *= sqrt2
            sl0_scale = 0.5

    # scale by exponentials and normalization factor
    tmp = _exp_factor_dct3(x, n, dim, norm_factor)
    x = x * tmp  # broadcasting

    sl0 = [slice(None)] * x.ndim
    sl0[dim] = slice(1)
    x[tuple(sl0)] *= sl0_scale

    # inverse fft
    x = _fft.ifft(x, n=n, dim=dim)
    x = torch.real(x)

    # reorder entries
    return _reshuffle_dct3(x, n, dim, dst)


def _dct_type1(
    x, n=None, dim=-1, norm=None, forward=True,
):
    if dim < -x.ndim or dim >= x.ndim:
        raise IndexError('dim out of range')
    if dim < 0:
        dim += x.ndim
    if n is not None and n < 1:
        raise ValueError(
            f'invalid number of data points ({n}) specified'
        )

    x = _cook_shape(x, (n,), (dim,), 'R2R')
    n = x.shape[dim]

    # replicate signal
    slicer_post = [slice(None)] * x.ndim
    slicer_post[dim] = slice(1, -1)
    x_post = x[tuple(slicer_post)]
    x = torch.cat([x, x_post.flip(dim)], dim=dim)

    # orthogonalization step (pre-transform)
    if norm.startswith('ortho'):
        x = x.movedim(dim, 0)
        x[0] *= sqrt2
        x[n-1] *= sqrt2
        x = x.movedim(0, dim)

    # determine normalization factor
    if norm.startswith('ortho'):
        inorm = 'sqrt'
    elif norm == 'forward':
        inorm = 'full' if forward else 'none'
    elif norm == 'backward' or norm is None:
        inorm = 'none' if forward else 'full'
    else:
        raise ValueError(f'Invalid norm value "{norm}", should be "backward", '
                         '"ortho" or "forward"')
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=1)

    # fft
    x = _fft.fft(x, n=2*(n-1), dim=dim)
    slicer_pre = [slice(None)] * x.ndim
    slicer_pre[dim] = slice(n)
    x = torch.real(x[tuple(slicer_pre)])
    x *= norm_factor

    # orthogonalization step (post-transform)
    if norm.startswith('ortho'):
        x = x.movedim(dim, 0)
        x[0] /= sqrt2
        x[-1] /= sqrt2
        x = x.movedim(0, dim)

    return x


def _dst_type1(
        x, n=None, dim=-1, norm=None, forward=True,
):
    if dim < -x.ndim or dim >= x.ndim:
        raise IndexError('dim out of range')
    if dim < 0:
        dim += x.ndim
    if n is not None and n < 1:
        raise ValueError(
            f'invalid number of data points ({n}) specified'
        )

    x = _cook_shape(x, (n,), (dim,), 'R2R')
    n = x.shape[dim]

    # replicate signal
    slicer_pre = [slice(None)] * x.ndim
    slicer_pre[dim] = slice(1, n+1)
    slicer_post = [slice(None)] * x.ndim
    slicer_post[dim] = slice(n+2, None)
    bigshape = list(x.shape)
    bigshape[dim] = 2*(n+1)
    tmp = x
    x = x.new_zeros(bigshape)
    x[tuple(slicer_pre)] = tmp
    x[tuple(slicer_post)] = tmp.flip(dim)
    x[tuple(slicer_post)] *= -1
    del tmp

    # determine normalization factor
    if norm.startswith('ortho'):
        inorm = 'sqrt'
    elif norm == 'forward':
        inorm = 'full' if forward else 'none'
    elif norm == 'backward' or norm is None:
        inorm = 'none' if forward else 'full'
    else:
        raise ValueError(f'Invalid norm value "{norm}", should be "backward", '
                         '"ortho" or "forward"')
    norm_factor = _get_dct_norm_factor(n+2, inorm=inorm, dct_type=1)
    # TODO: I am not exactly sure why I need the `+2`

    # fft
    x = _fft.fft(x, n=2*(n+1), dim=dim)
    slicer_pre = [slice(None)] * x.ndim
    slicer_pre[dim] = slice(1, n+1)
    x = torch.imag(x[tuple(slicer_pre)])
    x *= -norm_factor

    return x
