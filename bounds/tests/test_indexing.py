# import pytest
import torch
from .. import indexing


def _assert_eq(x, y):
    assert x == y, f"{x} != {y}"


def test_dft():
    # int
    _assert_eq(indexing.dft(0, 10)[0], 0)
    _assert_eq(indexing.dft(9, 10)[0], 9)
    _assert_eq(indexing.dft(10, 10)[0], 0)
    _assert_eq(indexing.dft(-1, 10)[0], 9)
    _assert_eq(indexing.dft(39, 10)[0], 9)
    _assert_eq(indexing.dft(40, 10)[0], 0)
    # tensor
    _assert_eq(indexing.dft(torch.as_tensor(0), 10)[0].item(), 0)
    _assert_eq(indexing.dft(torch.as_tensor(9), 10)[0].item(), 9)
    _assert_eq(indexing.dft(torch.as_tensor(10), 10)[0].item(), 0)
    _assert_eq(indexing.dft(torch.as_tensor(-1), 10)[0].item(), 9)
    _assert_eq(indexing.dft(torch.as_tensor(39), 10)[0].item(), 9)
    _assert_eq(indexing.dft(torch.as_tensor(40), 10)[0].item(), 0)


def test_dct2():
    # int
    _assert_eq(indexing.dct2(0, 10)[0], 0)
    _assert_eq(indexing.dct2(9, 10)[0], 9)
    _assert_eq(indexing.dct2(10, 10)[0], 9)
    _assert_eq(indexing.dct2(-1, 10)[0], 0)
    _assert_eq(indexing.dct2(19, 10)[0], 0)
    _assert_eq(indexing.dct2(20, 10)[0], 0)
    _assert_eq(indexing.dct2(29, 10)[0], 9)
    _assert_eq(indexing.dct2(30, 10)[0], 9)
    _assert_eq(indexing.dct2(39, 10)[0], 0)
    _assert_eq(indexing.dct2(40, 10)[0], 0)
    _assert_eq(indexing.dct2(-10, 10)[0], 9)
    _assert_eq(indexing.dct2(-11, 10)[0], 9)
    _assert_eq(indexing.dct2(-20, 10)[0], 0)
    _assert_eq(indexing.dct2(-21, 10)[0], 0)
    _assert_eq(indexing.dct2(-30, 10)[0], 9)
    _assert_eq(indexing.dct2(-31, 10)[0], 9)
    # tensor
    _assert_eq(indexing.dct2(torch.as_tensor(0), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(9), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(10), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(-1), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(19), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(20), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(29), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(30), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(39), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(40), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(-10), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(-11), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(-20), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(-21), 10)[0].item(), 0)
    _assert_eq(indexing.dct2(torch.as_tensor(-30), 10)[0].item(), 9)
    _assert_eq(indexing.dct2(torch.as_tensor(-31), 10)[0].item(), 9)


def test_dst2():
    # int
    _assert_eq(indexing.dst2(0, 10)[0], 0)
    _assert_eq(indexing.dst2(9, 10)[0], 9)
    _assert_eq(indexing.dst2(10, 10)[0], 9)
    _assert_eq(indexing.dst2(-1, 10)[0], 0)
    _assert_eq(indexing.dst2(19, 10)[0], 0)
    _assert_eq(indexing.dst2(20, 10)[0], 0)
    _assert_eq(indexing.dst2(29, 10)[0], 9)
    _assert_eq(indexing.dst2(30, 10)[0], 9)
    _assert_eq(indexing.dst2(39, 10)[0], 0)
    _assert_eq(indexing.dst2(40, 10)[0], 0)
    _assert_eq(indexing.dst2(-10, 10)[0], 9)
    _assert_eq(indexing.dst2(-11, 10)[0], 9)
    _assert_eq(indexing.dst2(-20, 10)[0], 0)
    _assert_eq(indexing.dst2(-21, 10)[0], 0)
    _assert_eq(indexing.dst2(-30, 10)[0], 9)
    _assert_eq(indexing.dst2(-31, 10)[0], 9)
    # tensor
    _assert_eq(indexing.dst2(torch.as_tensor(0), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(9), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(10), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(-1), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(19), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(20), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(29), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(30), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(39), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(40), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(-10), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(-11), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(-20), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(-21), 10)[0].item(), 0)
    _assert_eq(indexing.dst2(torch.as_tensor(-30), 10)[0].item(), 9)
    _assert_eq(indexing.dst2(torch.as_tensor(-31), 10)[0].item(), 9)
    # sign
    _assert_eq(indexing.dst2(0, 10)[1], 1)
    _assert_eq(indexing.dst2(9, 10)[1], 1)
    _assert_eq(indexing.dst2(10, 10)[1], -1)
    _assert_eq(indexing.dst2(-1, 10)[1], -1)
    _assert_eq(indexing.dst2(19, 10)[1], -1)
    _assert_eq(indexing.dst2(20, 10)[1], 1)
    _assert_eq(indexing.dst2(29, 10)[1], 1)
    _assert_eq(indexing.dst2(30, 10)[1], -1)
    _assert_eq(indexing.dst2(39, 10)[1], -1)
    _assert_eq(indexing.dst2(40, 10)[1], 1)
    _assert_eq(indexing.dst2(-10, 10)[1], -1)
    _assert_eq(indexing.dst2(-11, 10)[1], 1)
    _assert_eq(indexing.dst2(-20, 10)[1], 1)
    _assert_eq(indexing.dst2(-21, 10)[1], -1)
    _assert_eq(indexing.dst2(-30, 10)[1], -1)
    _assert_eq(indexing.dst2(-31, 10)[1], 1)


def test_dct1():
    # int
    _assert_eq(indexing.dct1(-5, 5)[0], 3)
    _assert_eq(indexing.dct1(-4, 5)[0], 4)
    _assert_eq(indexing.dct1(-1, 5)[0], 1)
    _assert_eq(indexing.dct1(0, 5)[0], 0)
    _assert_eq(indexing.dct1(4, 5)[0], 4)
    _assert_eq(indexing.dct1(5, 5)[0], 3)
    _assert_eq(indexing.dct1(8, 5)[0], 0)
    _assert_eq(indexing.dct1(9, 5)[0], 1)
    _assert_eq(indexing.dct1(12, 5)[0], 4)
    _assert_eq(indexing.dct1(13, 5)[0], 3)
    # tensor
    _assert_eq(indexing.dct1(-5, 5)[0], 3)
    _assert_eq(indexing.dct1(-4, 5)[0], 4)
    _assert_eq(indexing.dct1(-1, 5)[0], 1)
    _assert_eq(indexing.dct1(0, 5)[0], 0)
    _assert_eq(indexing.dct1(4, 5)[0], 4)
    _assert_eq(indexing.dct1(5, 5)[0], 3)
    _assert_eq(indexing.dct1(8, 5)[0], 0)
    _assert_eq(indexing.dct1(9, 5)[0], 1)
    _assert_eq(indexing.dct1(12, 5)[0], 4)
    _assert_eq(indexing.dct1(13, 5)[0], 3)


def test_dst1():
    n = 3
    inputs = list(range(-9, 12))
    inputs = [
        -9, -8, -7, -6, -5, -4, -3, -2, -1, +0,
        +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11,
    ]
    outputs = [
        +0, +0, +1, +2, +2, +2, +1, +0, +0, +0,
        +1, +2, +2, +2, +1, +0, +0, +0, +1, +2, +2,
    ]
    signs = [
        +0, +1, +1, +1, +0, -1, -1, -1, +0, +1,
        +1, +1, +0, -1, -1, -1, +0, +1, +1, +1, +0,
    ]

    def assert_eq(i, o, s):
        po, ps = indexing.dst1(i, n)
        assert ps == s, f"{i}: {ps} != {s}"
        if s:
            assert po == o, f"{i}: {po} != {o}"
        po, ps = indexing.dst1(torch.as_tensor(i), n)
        po, ps = po.item(), ps.item()
        assert ps == s, f"{i}: {ps} != {s}"
        if s:
            assert po == o, f"{i}: {po} != {o}"

    for i, o, s in zip(inputs, outputs, signs):
        assert_eq(i, o, s)
