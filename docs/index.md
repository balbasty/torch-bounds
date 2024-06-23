# torch-bounds

This small package implements a wide range of boundary conditions used
to extrapolate a given discrete signal outside of its native bounds.

Based on these additional boundary conditions, it implements
- [`pad`](api/padding.md#pad): an extension of `torch.nn.functional.pad`
- [`roll`](api/padding.md#roll): an extension of `torch.roll`

It also implements discrete [sine and cosine transforms](api/realtransforms.md)
(variants 1, 2 and 3), using a trick similar to `cupy`.

Finally, it implements additional utilities:
- [`ensure_shape`](api/padding.md#ensure_shape) crops or pads a tensor
  (with any boundary condition) so that it matches a give shape.
- [`indexing`](api/indexing.md) is a module that implements functions to
  tranforms out-of-bounds coordinates into in-bounds coordinates according
  to any boundary condition.
- [`types`](api/indexing.md) is a module that defines names and aliases
  for different boundary conditions, as well as tools to convert between
  different naming conventions.


## Related packages
- [`torch-interpol`](https://github.com/balbasty/torch-interpol):
  B-spline interpolation with the same boundary conditions as those
  implemented here.
- [`torch-distmap`](https://github.com/balbasty/torch-distmap):
  Euclidean distance transform.
