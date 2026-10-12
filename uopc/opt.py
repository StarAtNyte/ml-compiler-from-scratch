"""Week 6: making kernels fast on a CPU.

Three knobs, all applied while rangeifying (see rangeify.Opts):

  UPCAST  unroll output axes into a register tile. A 4x8 tile of a GEMM keeps 32
          accumulators in registers and reuses each loaded A value 8 times and each
          B value 4 times: arithmetic intensity goes from 1/2 to ~2.7 FLOP/load.
  UNROLL  unroll the innermost reduce axis; the loop body becomes a balanced tree.
  THREAD  run the outermost output loop on every core with OpenMP.
  TILE    split output axes into cache blocks (Opts.tile). Only search_opt tries it: with the layout
          pass making the reduce axis contiguous, it measured no faster on the reference machine.

`hand_opt` picks them with simple rules. `search_opt` times candidates and keeps
the fastest (a tiny BEAM search).
"""
from __future__ import annotations
import itertools, os, time
import numpy as np
from uopc.uop import UOp, Ops, ParamArg, ALU, MOVEMENT, prod
from uopc.rangeify import lower, rangeify, Ctx, Opts
from uopc.symbolic import simplify
from uopc.rewrite import UPat, PatternMatcher, graph_rewrite

NTHREADS = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# layout: make the innermost reduce axis contiguous in memory

def _views(x: UOp):
  """The movement-op chains feeding x through ALU ops: (top of chain, materialized base)."""
  seen, stack = set(), [x]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u)
    if u.op in ALU: stack.extend(u.src)
    elif u.op in MOVEMENT:
      base = u
      while base.op in MOVEMENT: base = base.src[0]
      if base.op in (Ops.BUFFER, Ops.REDUCE, Ops.STAGE): yield u, base

def _last_axis_stride(view: UOp, base: UOp) -> int | None:
  """Distance in memory between view[..., k] and view[..., k+1], or None if not affine-looking."""
  if view.shape[-1] == 1: return 0
  fake = UOp(Ops.BUFFER, (), ParamArg(-999, base.dtype, (prod(base.shape),)))
  v = view.substitute({base: fake.reshape(base.shape) if base.shape != fake.shape else fake})
  ctx = Ctx()
  rngs = tuple(ctx.rng(s) for s in v.shape)
  val = rangeify(v, rngs, ctx)
  if val.op is not Ops.LOAD or val.src[0].op is not Ops.INDEX: return None
  e = val.src[0].src[1]
  at = lambda k: simplify(e.substitute({r: UOp.const(k if i == len(rngs) - 1 else 0) for i, r in enumerate(rngs)}))
  e0, e1 = at(0), at(1)
  return e1.arg - e0.arg if e0.op is Ops.CONST and e1.op is Ops.CONST else None

# movement ops commute with elementwise ops: permute(a * b) == permute(a) * permute(b).
# Pushing them down to the leaves makes every view chain end at real data.
push_views = PatternMatcher([
  (UPat(MOVEMENT, src=(UPat(ALU, name="a"),), name="m"),
   lambda m, a: a.replace(src=tuple(m.replace(src=(s,)) for s in a.src))),
])

def layout_pass(roots: list[UOp]) -> list[UOp]:
  """Before broadcasting, copy operands whose reduce axis is strided into a contiguous temp (a STAGE).

  The copy costs O(size of the operand); the reduce it feeds is usually much bigger.
  This is what turns `A @ B` (B read down its columns) into a dot product of rows.
  """
  roots = [graph_rewrite(r, push_views) for r in roots]
  todo: set[UOp] = set()
  for r in UOp(Ops.SINK, tuple(roots)).toposort():
    if r.op is not Ops.REDUCE: continue
    for view, base in _views(r.src[0]):
      s = _last_axis_stride(view, base)
      if s in (0, 1, None): continue
      c = view
      while c.op is Ops.EXPAND: c = c.src[0]                    # stage before the broadcast
      # only stage operands that get broadcast into the reduce: copying them is cheap
      if prod(c.shape) > prod(base.shape) or prod(c.shape) >= prod(r.src[0].shape) or c in todo: continue
      staged = UOp(Ops.STAGE, (c,))
      if _last_axis_stride(view.substitute({c: staged}), staged) == 1: todo.add(c)
  return UOp.wrap(roots, todo, lambda n: UOp(Ops.STAGE, (n,))) if todo else roots


def _reduce_size(body: UOp) -> int:
  return max((prod(u.src[0].shape[len(u.src[0].shape) - u.arg[1]:]) for u in body.toposort() if u.op is Ops.REDUCE), default=1)


def hand_opt_choice(body: UOp) -> Opts:
  shape, rsize = body.shape, _reduce_size(body)
  o = Opts()
  if rsize > 1:
    # register tile: 4x4 on the last two output axes. With AVX2's 16 vector registers, 16 accumulators plus
    # the loaded operands don't quite fit (gcc spills a few), but measured it beats 2x8, 4x8 and 8x8 here
    budget = 16
    for axis, want in [(len(shape) - 1, 4), (len(shape) - 2, 4)]:
      if axis < 0: continue
      for f in (want, 4, 2):
        if f <= budget and shape[axis] % f == 0 and shape[axis] >= f and f > 1:
          o.upcast[axis] = f; budget //= f; break
    if not o.upcast and rsize % 4 == 0 and rsize < 64: o.unroll = 4
  # threads: only when there is enough work to amortize waking them up
  work = prod(shape or (1,)) * rsize
  if work >= 1 << 16:
    for axis, s in enumerate(shape):
      if s // o.upcast.get(axis, 1) >= 2:
        o.thread = axis; break
  return o


def hand_opt(body: UOp, out: UOp) -> UOp:
  return lower(body, out, Ctx(opts=hand_opt_choice(body)))


def candidate_opts(body: UOp) -> list[Opts]:
  shape, rsize = body.shape, _reduce_size(body)
  outs = [a for a, s in enumerate(shape) if s > 1]
  cands = [Opts(), hand_opt_choice(body)]
  for a, b in itertools.product(outs[-2:], repeat=2):
    for fa, fb in [(4, 4), (8, 4), (4, 8), (8, 8), (16, 2), (2, 16)]:
      up = {a: fa} if a == b else {a: fa, b: fb}
      if all(shape[k] % f == 0 for k, f in up.items()):
        for u in (1, 4) if rsize % 4 == 0 else (1,):
          cands.append(Opts(dict(up), hand_opt_choice(body).thread, u))
  # cache blocking of the last two output axes (helps when operand rows don't fit in cache)
  base = hand_opt_choice(body)
  for t in (32, 64, 128):
    tile = {a: t for a in outs[-2:] if shape[a] % t == 0}
    if tile: cands.append(Opts(dict(base.upcast), base.thread, base.unroll, tile=tile))
  return cands


def search_opt(body: UOp, out: UOp, inputs: list[np.ndarray] | None = None, reps: int = 3) -> UOp:
  """Time every candidate on random data, keep the fastest."""
  from uopc.render_c import render_c
  from uopc.runtime import Program, np_dtype
  bufs = sorted({u for u in body.toposort() if u.op is Ops.BUFFER}, key=lambda u: u.arg.slot)
  data = inputs or [np.random.randn(b.arg.shape[0]).astype(np_dtype(b.dtype)) for b in bufs]
  res = np.zeros(out.arg.shape[0], np_dtype(out.dtype))
  best, best_t = None, float("inf")
  seen = set()
  for o in candidate_opts(body):
    key = (tuple(sorted(o.upcast.items())), o.thread, o.unroll, tuple(sorted(o.tile.items())))
    if key in seen: continue
    seen.add(key)
    ast = lower(body, out, Ctx(opts=o))
    prg = Program(render_c(ast, "k"), "k")
    ts = []
    for _ in range(reps):
      t0 = time.perf_counter(); prg(*data, res); ts.append(time.perf_counter() - t0)
    if min(ts) < best_t: best, best_t = ast, min(ts)
  return best


class HandOpt:
  """The default optimizer kernelize uses: layout pass + hand-coded opts."""
  layout = staticmethod(layout_pass)
  lower = staticmethod(hand_opt)

class SearchOpt(HandOpt):
  """Like HandOpt, but times candidate opts for every new kernel (slow to compile, fast to run)."""
  lower = staticmethod(search_opt)
