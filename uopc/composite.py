"""Week 4: composite movement ops, built only from RESHAPE/EXPAND/PERMUTE/SHRINK/STACK.

PAD and FLIP are not primitives in this course. Building them out of primitives is
a good exercise in thinking about index maps, and rangeify + symbolic turn the
result back into tight code (PAD becomes a bounds-checked WHERE).
"""
from __future__ import annotations
from uopc.uop import UOp, Ops, prod


def zeros_like(x: UOp) -> UOp:
  z = UOp.const(0, x.dtype)
  return z.reshape((1,) * len(x.shape)).expand(x.shape) if x.shape else z


def move_axis(x: UOp, src: int, dst: int) -> UOp:
  order = [i for i in range(len(x.shape)) if i != src]
  order.insert(dst, src)
  return x.permute(order) if order != list(range(len(x.shape))) else x


def concat2(a: UOp, b: UOp, axis: int) -> UOp:
  """Concatenate two same-shaped tensors along `axis`: STACK, move the new axis next to `axis`, merge."""
  assert a.shape == b.shape
  s = UOp(Ops.STACK, (a, b))                       # (2, *shape)
  s = move_axis(s, 0, axis)                        # (..., 2, n, ...)
  shape = list(a.shape)
  shape[axis] *= 2
  return s.reshape(tuple(shape))


def _shrink_axis(x: UOp, axis: int, b: int, e: int) -> UOp:
  arg = [(0, s) for s in x.shape]
  arg[axis] = (b, e)
  return x.shrink(arg) if (b, e) != (0, x.shape[axis]) else x


def pad_axis(x: UOp, axis: int, before: int, after: int) -> UOp:
  """Zero-pad one axis by doubling with zeros and shrinking back."""
  while after > 0:
    n = x.shape[axis]
    x = _shrink_axis(concat2(x, zeros_like(x), axis), axis, 0, n + min(after, n))
    after -= min(after, n)
  while before > 0:
    n = x.shape[axis]
    x = _shrink_axis(concat2(zeros_like(x), x, axis), axis, n - min(before, n), 2 * n)
    before -= min(before, n)
  return x


def pad(x: UOp, arg: tuple[tuple[int, int], ...]) -> UOp:
  for axis, (b, a) in enumerate(arg): x = pad_axis(x, axis, b, a)
  return x


def anti_eye(n: int, dtype) -> UOp:
  """M[i, j] = (i + j == n - 1), from one-hot + the expand/reshape/shrink "diagonal trick".

  u is one-hot at n-1, length 2n-1. Expand u to (n+1, 2n-1), flatten, keep n*2n
  elements, view as (n, 2n): element (i, j) is u[(2n*i + j) mod (2n-1)] = u[i + j].
  """
  if n == 1: return UOp.const(1, dtype).reshape((1, 1))
  one = UOp.const(1, dtype).reshape((1,))
  u = pad_axis(one, 0, n - 1, n - 1)                              # (2n-1,)
  m = u.reshape((1, 2 * n - 1)).expand((n + 1, 2 * n - 1)).reshape(((n + 1) * (2 * n - 1),))
  m = m.shrink(((0, n * 2 * n),)).reshape((n, 2 * n))
  return m.shrink(((0, n), (0, n)))


def flip_axis(x: UOp, axis: int) -> UOp:
  """Reverse one axis: y[i] = sum_j x[j] * [i + j == n-1]. O(n^2), but only primitives."""
  n = x.shape[axis]
  if n == 1: return x
  xl = move_axis(x, axis, len(x.shape) - 1)                      # (..., n)
  lead = xl.shape[:-1]
  xe = xl.reshape(lead + (1, n)).expand(lead + (n, n))
  m = anti_eye(n, x.dtype).reshape((1,) * len(lead) + (n, n)).expand(lead + (n, n))
  y = (xe * m).reduce(Ops.ADD, 1)                                 # (..., n)
  return move_axis(y, len(x.shape) - 1, axis)


def sum_to(g: UOp, shape: tuple[int, ...]) -> UOp:
  """Reverse of EXPAND: sum g over the axes where `shape` has 1 and g doesn't."""
  axes = [i for i, (s, t) in enumerate(zip(g.shape, shape)) if s != t]
  if not axes: return g
  keep = [i for i in range(len(shape)) if i not in axes]
  return g.permute(keep + axes).reduce(Ops.ADD, len(axes)).reshape(shape)


def is_contiguous_buffer(u: UOp) -> bool:
  while u.op is Ops.RESHAPE: u = u.src[0]
  return u.op is Ops.BUFFER and prod(u.shape) == u.arg.shape[0]
