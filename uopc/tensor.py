"""Week 5: a lazy Tensor frontend.

A Tensor is a thin wrapper around a shaped UOp graph. Nothing runs until you call
.realize() / .numpy(): then the graph is kernelized, rangeified, rendered to C,
compiled, and executed. Every op below is a few lines of graph building on top of
the course's primitives -- there is no other math library underneath.
"""
from __future__ import annotations
import math, os
import numpy as np
from uopc.uop import UOp, Ops, prod
from uopc.dtype import DType, dtypes
from uopc.autodiff import gradient
from uopc.composite import pad as _pad, flip_axis
from uopc.kernelize import new_buffer, buffer_data, base_buffer, realize as _realize
from uopc.runtime import np_dtype

LOG2E, LN2 = 1 / math.log(2), math.log(2)


def _broadcast_shape(*shapes):
  n = max(len(s) for s in shapes)
  shapes = [(1,) * (n - len(s)) + tuple(s) for s in shapes]
  out = []
  for dims in zip(*shapes):
    d = max(dims)
    if any(x not in (1, d) for x in dims): raise ValueError(f"can't broadcast {shapes}")
    out.append(d)
  return tuple(out)


class Tensor:
  rng = np.random.default_rng(0)
  training = False
  fuse_reduce = False
  opt = None   # week 6: kernel optimizer (set below)

  def __init__(self, data, dtype: DType | None = None, requires_grad: bool = False):
    if isinstance(data, UOp): self.uop = data
    else:
      arr = np.asarray(data)
      if dtype is None: dtype = dtypes.f32 if arr.dtype.kind == "f" else dtypes.bool if arr.dtype.kind == "b" else dtypes.i32
      arr = arr.astype(np_dtype(dtype))
      buf = new_buffer(dtype, arr.size, arr)
      self.uop = buf.reshape(arr.shape) if arr.shape != buf.shape else buf
    self.requires_grad = requires_grad
    self.grad: Tensor | None = None

  # ---- basics -------------------------------------------------------------
  @property
  def shape(self) -> tuple[int, ...]: return self.uop.shape
  @property
  def dtype(self) -> DType: return self.uop.dtype
  @property
  def ndim(self) -> int: return len(self.shape)
  def numel(self) -> int: return prod(self.shape)
  def __repr__(self): return f"<Tensor {self.shape} {self.dtype.name}{' grad' if self.requires_grad else ''}>"

  def realize(self) -> Tensor:
    Tensor.realize_all(self)
    return self

  @staticmethod
  def realize_all(*ts: Tensor):
    todo = [t for t in ts if t.uop.op is not Ops.BUFFER and not (t.uop.op is Ops.RESHAPE and base_buffer(t.uop).op is Ops.BUFFER and prod(t.shape) == base_buffer(t.uop).arg.shape[0])]
    if not todo: return
    bufs = _realize([t.uop for t in todo], Tensor.fuse_reduce, Tensor.opt)
    for t, b in zip(todo, bufs):
      t._graph = t.uop   # keep the graph: loss.item() before loss.backward() must still have gradients
      t.uop = b.reshape(t.shape) if b.shape != t.shape else b

  def numpy(self) -> np.ndarray:
    self.realize()
    return buffer_data(base_buffer(self.uop)).reshape(self.shape).copy()

  def item(self): return self.numpy().item()
  def detach(self) -> Tensor:
    """Same value, but gradients stop here. (Kernelize strips the marker, so it costs nothing to run.)"""
    return Tensor(UOp(Ops.STAGE, (self.uop,), "detach"))

  def assign(self, x: Tensor) -> Tensor:
    """Make this tensor (e.g. a parameter) hold x's value. Functional: rebinds to a new buffer."""
    assert x.shape == self.shape and x.dtype == self.dtype, f"assign {x.shape} to {self.shape}"
    self.uop = x.uop
    return self

  # ---- creation -----------------------------------------------------------
  @staticmethod
  def full(shape, v, dtype: DType = dtypes.f32) -> Tensor:
    c = UOp.const(v, dtype)
    return Tensor(c.reshape((1,) * len(shape)).expand(tuple(shape)) if shape else c)
  @staticmethod
  def zeros(*shape, dtype=dtypes.f32) -> Tensor: return Tensor.full(shape, 0, dtype)
  @staticmethod
  def ones(*shape, dtype=dtypes.f32) -> Tensor: return Tensor.full(shape, 1, dtype)
  @staticmethod
  def randn(*shape, requires_grad=False) -> Tensor:
    return Tensor(Tensor.rng.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)
  @staticmethod
  def uniform(*shape, low=-1.0, high=1.0, requires_grad=False) -> Tensor:
    return Tensor(Tensor.rng.uniform(low, high, shape).astype(np.float32), requires_grad=requires_grad)
  @staticmethod
  def kaiming(*shape, requires_grad=True) -> Tensor:
    bound = 1 / math.sqrt(prod(shape[1:]) if len(shape) > 1 else shape[0])
    return Tensor.uniform(*shape, low=-bound, high=bound, requires_grad=requires_grad)
  @staticmethod
  def arange(n: int) -> Tensor: return Tensor(np.arange(n, dtype=np.int32))
  @staticmethod
  def manual_seed(s: int): Tensor.rng = np.random.default_rng(s)

  # ---- elementwise ----------------------------------------------------------
  def _const(self, x) -> Tensor: return x if isinstance(x, Tensor) else Tensor.full((), x, self.dtype)

  def broadcast_to(self, shape) -> Tensor:
    shape = tuple(shape)
    if self.shape == shape: return self
    u = self.uop
    if len(u.shape) < len(shape): u = u.reshape((1,) * (len(shape) - len(u.shape)) + u.shape)
    return Tensor(u.expand(shape) if u.shape != shape else u)

  def _binop(self, op: Ops, x, reverse=False) -> Tensor:
    # numpy-style promotion for python scalars: int tensor (op) 0.5 is a float computation
    if isinstance(x, float) and not self.dtype.is_float: return self.float()._binop(op, x, reverse)
    x = self._const(x)
    a, b = (x, self) if reverse else (self, x)
    shape = _broadcast_shape(a.shape, b.shape)
    return Tensor(UOp(op, (a.broadcast_to(shape).uop, b.broadcast_to(shape).uop)))

  def __add__(self, x): return self._binop(Ops.ADD, x)
  def __radd__(self, x): return self._binop(Ops.ADD, x, True)
  def __mul__(self, x): return self._binop(Ops.MUL, x)
  def __rmul__(self, x): return self._binop(Ops.MUL, x, True)
  def __neg__(self): return self * -1
  def __sub__(self, x): return self + -self._const(x)
  def __rsub__(self, x): return self._const(x) + -self
  def __truediv__(self, x): return self * self._const(x).reciprocal()
  def __rtruediv__(self, x): return self._const(x) * self.reciprocal()
  def __lt__(self, x): return self._binop(Ops.CMPLT, x)
  def __gt__(self, x): return self._binop(Ops.CMPLT, x, True)
  def __ge__(self, x): return (self < x).logical_not()
  def __le__(self, x): return (self > x).logical_not()
  def eq(self, x): return self._binop(Ops.CMPNE, x).logical_not()
  def ne(self, x): return self._binop(Ops.CMPNE, x)
  def logical_not(self): return self._binop(Ops.CMPNE, True)
  def maximum(self, x): return self._binop(Ops.MAX, x)
  def minimum(self, x): return -((-self).maximum(-self._const(x)))
  def where(self, a, b) -> Tensor:
    a = a if isinstance(a, Tensor) else Tensor.full((), a)
    b = b if isinstance(b, Tensor) else Tensor.full((), b, a.dtype)
    shape = _broadcast_shape(self.shape, a.shape, b.shape)
    return Tensor(UOp(Ops.WHERE, (self.broadcast_to(shape).uop, a.broadcast_to(shape).uop, b.broadcast_to(shape).uop)))
  def cast(self, d: DType) -> Tensor: return Tensor(self.uop.cast(d))
  def float(self) -> Tensor: return self.cast(dtypes.f32)

  def reciprocal(self): return Tensor(self.uop.recip())
  def exp2(self): return Tensor(UOp(Ops.EXP2, (self.uop,)))
  def log2(self): return Tensor(UOp(Ops.LOG2, (self.uop,)))
  def exp(self): return (self * LOG2E).exp2()
  def log(self): return self.log2() * LN2
  def sqrt(self): return Tensor(UOp(Ops.SQRT, (self.uop,)))
  def rsqrt(self): return self.sqrt().reciprocal()
  def sin(self): return Tensor(UOp(Ops.SIN, (self.uop,)))
  def cos(self): return (self + math.pi / 2).sin()
  def square(self): return self * self
  def abs(self): return self.maximum(-self)
  def relu(self): return (self > 0).where(self, 0.0)
  def sigmoid(self): return ((-self).exp() + 1).reciprocal()
  def tanh(self): return (self * 2).sigmoid() * 2 - 1
  def gelu(self): return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self * self * self)).tanh())
  def silu(self): return self * self.sigmoid()

  # ---- movement -------------------------------------------------------------
  def reshape(self, *shape) -> Tensor:
    shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    if -1 in shape:
      known = prod(s for s in shape if s != -1)
      shape = tuple(self.numel() // known if s == -1 else s for s in shape)
    return self if shape == self.shape else Tensor(self.uop.reshape(shape))
  def permute(self, *order) -> Tensor:
    order = tuple(order[0]) if len(order) == 1 and isinstance(order[0], (tuple, list)) else order
    order = tuple(o % self.ndim for o in order)
    return self if order == tuple(range(self.ndim)) else Tensor(self.uop.permute(order))
  def transpose(self, a=-1, b=-2) -> Tensor:
    order = list(range(self.ndim))
    order[a], order[b] = order[b], order[a]
    return self.permute(order)
  @property
  def T(self) -> Tensor: return self.transpose()
  def expand(self, *shape) -> Tensor:
    shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return self.broadcast_to(tuple(s if s != -1 else o for s, o in zip(shape, self.shape)))
  def flatten(self, start=0) -> Tensor: return self.reshape(self.shape[:start] + (-1,))
  def unsqueeze(self, d) -> Tensor:
    d = d % (self.ndim + 1)
    return self.reshape(self.shape[:d] + (1,) + self.shape[d:])
  def shrink(self, arg) -> Tensor:
    arg = tuple((0, s) if a is None else a for a, s in zip(arg, self.shape))
    return self if all(a == (0, s) for a, s in zip(arg, self.shape)) else Tensor(self.uop.shrink(arg))
  def pad(self, arg) -> Tensor: return Tensor(_pad(self.uop, tuple(arg)))
  def flip(self, axis: int) -> Tensor: return Tensor(flip_axis(self.uop, axis % self.ndim))
  def __getitem__(self, idx) -> Tensor:
    """Ints and step-1 slices only."""
    idx = idx if isinstance(idx, tuple) else (idx,)
    arg, squeeze = [], []
    for d, i in enumerate(idx):
      n = self.shape[d]
      if isinstance(i, int):
        i = i % n
        arg.append((i, i + 1)); squeeze.append(d)
      else:
        b, e, step = i.indices(n)
        assert step == 1, "only step-1 slices"
        arg.append((b, max(b, e)))
    arg += [(0, s) for s in self.shape[len(arg):]]
    t = self.shrink(arg)
    return t.reshape(tuple(s for d, s in enumerate(t.shape) if d not in squeeze))
  @staticmethod
  def cat(ts: list[Tensor], dim: int = 0) -> Tensor:
    """Concatenate via pad + add: each piece is padded into place, then summed."""
    dim = dim % ts[0].ndim
    total = sum(t.shape[dim] for t in ts)
    out, off = None, 0
    for t in ts:
      p = [(0, 0)] * t.ndim
      p[dim] = (off, total - off - t.shape[dim])
      piece = t.pad(p)
      out = piece if out is None else out + piece
      off += t.shape[dim]
    return out
  def stack_with(self, *others) -> Tensor: return Tensor(UOp(Ops.STACK, (self.uop, *(o.uop for o in others))))

  # ---- reductions -----------------------------------------------------------
  def _reduce(self, op: Ops, axis=None, keepdim=False) -> Tensor:
    axes = list(range(self.ndim)) if axis is None else [a % self.ndim for a in (axis if isinstance(axis, (tuple, list)) else [axis])]
    keep = [i for i in range(self.ndim) if i not in axes]
    t = self.permute(keep + axes)
    r = Tensor(t.uop.reduce(op, len(axes)))
    if keepdim: r = r.reshape(tuple(1 if i in axes else s for i, s in enumerate(self.shape)))
    return r
  def sum(self, axis=None, keepdim=False): return self._reduce(Ops.ADD, axis, keepdim)
  def max(self, axis=None, keepdim=False): return self._reduce(Ops.MAX, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis, keepdim))
  def mean(self, axis=None, keepdim=False):
    s = (self if self.dtype.is_float else self.float()).sum(axis, keepdim)
    return s * (s.numel() / self.numel())
  def var(self, axis=None, keepdim=False):
    return (self - self.mean(axis, keepdim=True)).square().mean(axis, keepdim)
  def softmax(self, axis=-1):
    e = (self - self.max(axis, keepdim=True)).exp()
    return e / e.sum(axis, keepdim=True)
  def log_softmax(self, axis=-1):
    m = self - self.max(axis, keepdim=True)
    return m - m.exp().sum(axis, keepdim=True).log()
  def logsumexp(self, axis=-1):
    m = self.max(axis, keepdim=True)
    return (self - m).exp().sum(axis, keepdim=True).log() + m
  def argmax(self, axis=-1):
    """Index of the first maximum, as i32."""
    axis = axis % self.ndim
    n = self.shape[axis]
    idx = Tensor.arange(n).reshape((n,) + (1,) * (self.ndim - axis - 1))
    is_max = self.eq(self.max(axis, keepdim=True))
    return is_max.where(n - idx, 0).max(axis) * -1 + n

  # ---- linear algebra -------------------------------------------------------
  def matmul(self, w: Tensor) -> Tensor:
    """(..., M, K) @ (..., K, N): broadcast to (..., M, N, K), multiply, reduce K."""
    x = self if self.ndim > 1 else self.reshape(1, -1)
    wt = w.reshape(-1, 1) if w.ndim == 1 else w
    wt = wt.transpose()                                     # (..., N, K)
    x = x.reshape(x.shape[:-1] + (1, x.shape[-1]))          # (..., M, 1, K)
    wt = wt.reshape(wt.shape[:-2] + (1,) + wt.shape[-2:])   # (..., 1, N, K)
    out = (x * wt).sum(-1)
    if self.ndim == 1: out = out.reshape(out.shape[:-2] + out.shape[-1:])
    if w.ndim == 1: out = out.reshape(out.shape[:-1])
    return out
  __matmul__ = matmul
  def linear(self, w: Tensor, b: Tensor | None = None) -> Tensor:
    out = self @ w
    return out + b if b is not None else out

  def conv2d(self, w: Tensor, b: Tensor | None = None, padding: int = 0, stride: int = 1) -> Tensor:
    """x (N, C, H, W), w (O, C, KH, KW). im2col from shrinks, then a matmul."""
    x = self.pad(((0, 0), (0, 0), (padding, padding), (padding, padding))) if padding else self
    N, C, H, W = x.shape
    O, _, KH, KW = w.shape
    OH, OW = (H - KH) // stride + 1, (W - KW) // stride + 1
    cols = []
    for ky in range(KH):
      for kx in range(KW):
        win = x.shrink(((0, N), (0, C), (ky, ky + stride * (OH - 1) + 1), (kx, kx + stride * (OW - 1) + 1)))
        if stride > 1: win = win._stride2d(stride, OH, OW)
        cols.append(win)                                     # (N, C, OH, OW) each
    patches = cols[0].stack_with(*cols[1:]) if len(cols) > 1 else cols[0].unsqueeze(0)   # (KH*KW, N, C, OH, OW)
    patches = patches.permute(1, 3, 4, 2, 0).reshape(N, OH, OW, C * KH * KW)
    out = patches @ w.reshape(O, C * KH * KW).T              # (N, OH, OW, O)
    out = out.permute(0, 3, 1, 2)
    return out + b.reshape(1, O, 1, 1) if b is not None else out

  def _stride2d(self, s: int, OH: int, OW: int) -> Tensor:
    N, C, H, W = self.shape
    t = self.pad(((0, 0), (0, 0), (0, OH * s - H), (0, OW * s - W)))
    return t.reshape(N, C, OH, s, OW, s)[:, :, :, 0, :, 0]

  def max_pool2d(self, k: int = 2) -> Tensor:
    N, C, H, W = self.shape
    return self[:, :, :H // k * k, :W // k * k].reshape(N, C, H // k, k, W // k, k).max((3, 5))

  # ---- losses ---------------------------------------------------------------
  def one_hot(self, n: int) -> Tensor:
    return self.unsqueeze(-1).eq(Tensor.arange(n)).float()
  def cross_entropy(self, labels: Tensor) -> Tensor:
    """self: (B, C) logits, labels: (B,) i32."""
    return -(self.log_softmax(-1) * labels.one_hot(self.shape[-1])).sum(-1).mean()

  # ---- autodiff ------------------------------------------------------------
  def backward(self, params: list[Tensor]) -> Tensor:
    """Set p.grad = d(self)/d(p) for each p. Gradients are lazy Tensors like everything else."""
    assert self.shape == (), "backward needs a scalar"
    root = getattr(self, "_graph", self.uop)
    for p, g in zip(params, gradient(root, [p.uop for p in params])):
      p.grad = Tensor(g) if p.grad is None else p.grad + Tensor(g)
    return self

  # ---- staging -------------------------------------------------------------
  def contiguous(self) -> Tensor:
    """Ask for a materialization point: the kernelizer will write this into a temp buffer."""
    return Tensor(UOp(Ops.STAGE, (self.uop,)))


if os.environ.get("OPT", "1") != "0":
  from uopc.opt import HandOpt, SearchOpt
  Tensor.opt = SearchOpt() if os.environ.get("OPT") == "2" else HandOpt()
