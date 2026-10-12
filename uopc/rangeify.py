"""Week 4: rangeify. From shapes to loops.

A shaped graph says *what* each element is. Rangeify says *how to compute one
element*: give every output axis a RANGE, then ask each node "what is your value
at index (i0, i1, ...)?" and push the index down the graph:

    BUFFER    value[i]               -> LOAD(INDEX(buf, i))
    ALU       f(a, b)[idx]           -> f(a[idx], b[idx])
    RESHAPE   x.reshape(s)[idx]      -> x[unravel(ravel(idx, s), x.shape)]
    EXPAND    x.expand(s)[idx]       -> x[0 where x had size 1]
    PERMUTE   x.permute(p)[idx]      -> x[idx permuted back]
    SHRINK    x.shrink(b)[idx]       -> x[idx + begin]
    STACK     stack(a, b)[i, *rest]  -> where(i < 1, a[rest], b[rest])
    REDUCE    sum over the last n    -> new RANGEs, an accumulator, a loop

After rangeify no op has a shape: everything is a scalar in some loop. Symbolic
simplification then cleans up the index arithmetic (ravel . unravel == id).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from uopc.uop import UOp, Ops, ParamArg, AxisType, ALU, MOVEMENT, sink, range_
from uopc.dtype import dtypes, DType
from uopc.symbolic import simplify


def ravel(idx: tuple[UOp, ...], shape: tuple[int, ...]) -> UOp:
  flat = UOp.const(0)
  for i, s in zip(idx, shape): flat = flat * s + i
  return flat

def unravel(flat: UOp, shape: tuple[int, ...]) -> tuple[UOp, ...]:
  out, stride = [], 1
  for s in reversed(shape):
    out.append((flat // stride) % s if s != 1 else UOp.const(0))
    stride *= s
  return tuple(reversed(out))

def identity(op: Ops, d: DType):
  if op is Ops.ADD: return 0.0 if d.is_float else 0
  return -float("inf") if d.is_float else dtypes.min(d)


@dataclass
class Opts:
  """Week 6: how to map a kernel onto the machine. Chosen by uopc.opt, applied during rangeify."""
  upcast: dict[int, int] = field(default_factory=dict)   # output axis -> unroll factor (register tile)
  thread: int | None = None                              # output axis run in parallel (OpenMP)
  unroll: int = 1                                        # unroll factor of the innermost reduce axis
  gpu: bool = False                                      # weeks 7-8: output axes become GLOBAL (blockIdx) ranges
  local: dict[int, int] = field(default_factory=dict)    # output axis -> threads per block on that axis (LOCAL)


@dataclass
class Ctx:
  next_range: int = 0
  next_alloc: int = 0
  memo: dict = field(default_factory=dict)
  reduce_ranges: dict = field(default_factory=dict)
  opts: Opts = field(default_factory=Opts)

  def rng(self, size: int, kind: AxisType = AxisType.LOOP) -> UOp:
    self.next_range += 1
    return range_(size, self.next_range - 1, kind)

  def alloc(self, d: DType, n: int = 1) -> UOp:
    self.next_alloc += 1
    return UOp(Ops.ALLOC, (), ParamArg(self.next_alloc - 1, d, (n,)))


def ranges_in(*us: UOp) -> list[UOp]:
  return sorted({r for u in us for r in u.toposort() if r.op is Ops.RANGE}, key=lambda r: r.arg)


def rangeify(u: UOp, idx: tuple[UOp, ...], ctx: Ctx) -> UOp:
  """The scalar value of u at index idx."""
  key = (u, idx)
  if (r := ctx.memo.get(key)) is not None: return r
  assert len(idx) == len(u.shape), f"{u.op}: {len(idx)} indices for shape {u.shape}"
  op = u.op
  if op is Ops.CONST: r = u
  elif op is Ops.BUFFER: r = u.index(idx[0]).load()
  elif op is Ops.STAGE: r = rangeify(u.src[0], idx, ctx)      # no kernelizer: compute it inline
  elif op in ALU: r = u.replace(src=tuple(rangeify(s, idx, ctx) for s in u.src))
  elif op is Ops.INDEX:
    iv = tuple(rangeify(i, (), ctx) for i in u.src[1:])
    r = rangeify(u.src[0], iv + idx, ctx)
  elif op is Ops.STACK:
    i, rest = idx[0], idx[1:]
    r = rangeify(u.src[-1], rest, ctx)
    for k in range(len(u.src) - 2, -1, -1):
      r = (i < k + 1).where(rangeify(u.src[k], rest, ctx), r)
  elif op is Ops.RESHAPE:
    # simplify now: ((i*N + j)//N) % N mentions j but *is* i. Dependence analysis
    # (which loops a reduce must sit in) and the memo both need the truth.
    r = rangeify(u.src[0], tuple(simplify(i) for i in unravel(ravel(idx, u.shape), u.src[0].shape)), ctx)
  elif op is Ops.EXPAND:
    r = rangeify(u.src[0], tuple(UOp.const(0) if s == 1 and n != 1 else i for i, s, n in zip(idx, u.src[0].shape, u.shape)), ctx)
  elif op is Ops.PERMUTE:
    inv = [0] * len(idx)
    for k, p in enumerate(u.arg): inv[p] = idx[k]
    r = rangeify(u.src[0], tuple(inv), ctx)
  elif op is Ops.SHRINK: r = rangeify(u.src[0], tuple(i + b if b else i for i, (b, _) in zip(idx, u.arg)), ctx)
  elif op is Ops.REDUCE: r = _reduce(u, idx, ctx)
  else: raise NotImplementedError(f"rangeify: {op}")
  ctx.memo[key] = r
  return r


def _axis_parts(ctx: Ctx, size: int, factor: int, kind: AxisType) -> tuple[UOp, int]:
  """An axis of `size` as (outer RANGE or CONST 0, unroll factor): index = outer * factor + c."""
  if factor > 1 and size % factor == 0:
    return (ctx.rng(size // factor, kind) if size > factor else UOp.const(0)), factor
  return (ctx.rng(size, kind) if size > 1 else UOp.const(0)), 1

def _combos(parts: list[tuple[UOp, int]]) -> list[tuple[UOp, ...]]:
  """Every index tuple obtained by fixing the unrolled part of each axis to a constant.
  parts[k] = (base, f): the index of axis k is base * f + c for c in 0..f-1."""
  out = [()]
  for b, f in parts:
    known0 = b.op is Ops.CONST and b.arg == 0
    out = [c + (UOp.const(k) if known0 else (b * f + k if f > 1 else b),) for c in out for k in range(f)]
  return out


def _reduce(u: UOp, idx: tuple[UOp, ...], ctx: Ctx) -> UOp:
  rop, n = u.arg
  x = u.src[0]
  rshape = x.shape[len(x.shape) - n:]
  # unrolled copies of the same reduce (they differ only by constants in idx) share one loop
  rkey = (u, frozenset(ranges_in(*idx)))
  if (parts := ctx.reduce_ranges.get(rkey)) is None:
    parts = [_axis_parts(ctx, s, ctx.opts.unroll if k == n - 1 else 1, AxisType.REDUCE) for k, s in enumerate(rshape)]
    ctx.reduce_ranges[rkey] = parts
  real = [r for r, _ in parts if r.op is Ops.RANGE]
  vals = [rangeify(x, idx + c, ctx) for c in _combos(parts)]
  while len(vals) > 1:                                    # unrolled reduce axis: a balanced tree
    vals = [UOp(rop, (vals[i], vals[i + 1])) if i + 1 < len(vals) else vals[i] for i in range(0, len(vals), 2)]
  if not real: return vals[0]
  acc = ctx.alloc(u.dtype)
  # the accumulator lives inside the loops the output index uses...
  acc0 = acc.after(*ranges_in(*idx))
  init = acc0.index(0).store(UOp.const(identity(rop, u.dtype), u.dtype))
  # ...is re-read every iteration of the reduce loops...
  accl = acc0.after(init, *real)
  upd = accl.index(0).store(UOp(rop, (accl.index(0).load(), vals[0])))
  # ...and is read once more after the loops end
  return acc0.after(upd.end(*real)).index(0).load()


def merge_ends(root: UOp) -> UOp:
  """Unrolled copies each END the same shared loop. Merge them into one END over a SINK."""
  while True:
    groups: dict[tuple, list[UOp]] = {}
    for u in root.toposort():
      if u.op is Ops.END: groups.setdefault(u.src[1:], []).append(u)
    dup = next((g for g in groups.values() if len(g) > 1), None)
    if dup is None: return root
    merged = UOp(Ops.END, (UOp(Ops.SINK, tuple(e.src[0] for e in dup)), *dup[0].src[1:]))
    root = root.substitute({e: merged for e in dup})


def _gpu_axis(ctx: Ctx, size: int, local: int, factor: int) -> tuple[UOp, int]:
  """size = blocks * local * factor: index = (gidx * local + lidx) * factor + c."""
  assert size % (local * factor) == 0, f"axis {size} not divisible by local {local} * upcast {factor}"
  g = ctx.rng(size // (local * factor), AxisType.GLOBAL) if size // (local * factor) > 1 else UOp.const(0)
  if local == 1: return g, factor
  return g * local + ctx.rng(local, AxisType.LOCAL), factor


def lower(root: UOp, out: UOp, ctx: Ctx | None = None) -> UOp:
  """Lower one shaped value into a SINK that stores it, element by element, into BUFFER `out`."""
  ctx = ctx or Ctx()
  o = ctx.opts
  if o.gpu:
    # each output axis gets a GLOBAL range and (optionally) a LOCAL one; neither becomes a loop on a GPU
    gl = [_gpu_axis(ctx, s, o.local.get(k, 1), o.upcast.get(k, 1)) for k, s in enumerate(root.shape)]
    plist = gl
  else:
    # the THREAD axis gets the lowest range index, so it is the outermost loop
    order = ([o.thread] if o.thread is not None else []) + [k for k in range(len(root.shape)) if k != o.thread]
    parts: dict[int, tuple[UOp, int]] = {}
    for k in order:
      parts[k] = _axis_parts(ctx, root.shape[k], o.upcast.get(k, 1), AxisType.THREAD if k == o.thread else AxisType.LOOP)
    plist = [parts[k] for k in range(len(root.shape))]
  stores = []
  for idx in _combos(plist):
    val = rangeify(root, idx, ctx)
    stores.append(out.index(ravel(idx, root.shape) if root.shape else UOp.const(0)).store(val))
  loops = ranges_in(*(b for b, _ in plist))
  body = stores[0] if len(stores) == 1 else UOp(Ops.SINK, tuple(stores))
  return simplify(merge_ends(sink(body.end(*loops))))
