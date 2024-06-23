# torch-bounds

Boundary conditions (circulant, mirror, reflect) and real transforms (DCT, DST) in PyTorch.

## Overview

This small package implements a wide range of boundary conditions used
to extrapolate a given discrete signal outside of its native bounds.

Based on these additional boundary conditions, it implements:

- [`pad`](https://torch-bounds.readthedocs.io/en/latest/api/padding/#bounds.padding.pad): an extension of `torch.nn.functional.pad`
- [`roll`](https://torch-bounds.readthedocs.io/en/latest/api/padding/#bounds.padding.roll): an extension of `torch.roll`

It also implements discrete
[sine and cosine transforms](https://torch-bounds.readthedocs.io/en/latest/api/realtransforms)
(variants 1, 2 and 3), using a trick similar to `cupy`.

Finally, it implements additional utilities:

- [`ensure_shape`](https://torch-bounds.readthedocs.io/en/latest/api/padding/#bounds.padding.ensure_shape)
  crops or pads a tensor (with any boundary condition) so that it matches a give shape.
- [`indexing`](https://torch-bounds.readthedocs.io/en/latest/api/indexing)
  is a module that implements functions to tranforms out-of-bounds
  coordinates into in-bounds coordinates according to any boundary condition.
- [`types`](https://torch-bounds.readthedocs.io/en/latest/api/types)
  is a module that defines names and aliases for different boundary conditions,
  as well as tools to convert between different naming conventions.

## Documentation

See our [**documentation**](https://torch-bounds.readthedocs.io) and
[**notebooks**](docs/notebooks/).

## Installation

### Dependency

- `torch >= 1.3`
- `torch >= 1.8` if real transforms are needed (dct, dst)

### Conda

```shell
conda install torch-bounds -c balbasty -c pytorch
```

### Pip

```shell
pip install torch-bounds
```



## Related packages

- [`torch-interpol`](https://github.com/balbasty/torch-interpol):
  B-spline interpolation with the same boundary conditions as those
  implemented here.
- [`torch-distmap`](https://github.com/balbasty/torch-distmap):
  Euclidean distance transform.
