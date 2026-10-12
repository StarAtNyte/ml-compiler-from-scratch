"""Week 5: kernel fission and fusion.

Rangeify can lower *any* shape graph into one giant loop nest, but that is often
terrible: a REDUCE consumed through an EXPAND would be recomputed for every
element it is broadcast to ((A@B)@C would recompute A@B for every output). So we cut the graph into
kernels at *materialization points*:

  * every root (the tensors the user asked for)
  * every STAGE (the user, or a rule, asked for a temp buffer)
  * every REDUCE, unless fuse_reduce is on (then reductions nest, see flash attention)

Everything between materialization points is fused into one kernel. A kernel
reads BUFFERs and writes exactly one. The schedule is the list of kernels in
dependency order. Whole schedules are cached on the *structure* of the graph,
so a training step only pays for Python-side compilation once.
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
import numpy as np
from uopc.uop import UOp, Ops, ParamArg, prod, _argkey
from uopc.dtype import DType
from uopc.rangeify import lower
from uopc.composite import is_contiguous_buffer
from uopc.runtime import Program, np_dtype
from uopc.rewrite import UPat, PatternMatcher, graph_rewrite

DEBUG = int(os.environ.get("DEBUG", "0"))


# ---------------------------------------------------------------------------
# buffers: a BUFFER UOp owns a numpy array. When the UOp dies, so does the memory.

_next_slot = 0

def new_buffer(dtype: DType, size: int, data: np.ndarray | None = None) -> UOp:
  global _next_slot
  _next_slot += 1
  u = UOp(Ops.BUFFER, (), ParamArg(_next_slot, dtype, (max(size, 1),)))
  if data is not None: u.__dict__["np"] = np.ascontiguousarray(data.reshape(-1), dtype=np_dtype(dtype))
  return u

def buffer_data(u: UOp) -> np.ndarray:
  if "np" not in u.__dict__: u.__dict__["np"] = np.zeros(u.arg.shape[0], np_dtype(u.dtype))
  return u.__dict__["np"]

def base_buffer(u: UOp) -> UOp:
  while u.op is Ops.RESHAPE: u = u.src[0]
  return u


# ---------------------------------------------------------------------------
# the schedule

@dataclass
class Kernel:
  name: str
  src: str
  refs: list[tuple[str, int]]      # one per local slot: ("in", i) | ("tmp", j) | ("out", r)
  program: Program | None = None
  flops: int = 0

@dataclass
class Plan:
  kernels: list[Kernel]
  temps: list[tuple[DType, int]]
  outs: list[tuple[str, int] | tuple[DType, int]]   # per root: ("in", i) alias, or (dtype, size) fresh
  temp_arrays: list | None = None   # allocated on first run, then reused: no page faults on every step


def _materialize_points(roots: list[UOp], fuse_reduce: bool) -> list[UOp]:
  topo = UOp(Ops.SINK, tuple(roots)).toposort()
  m = []
  for u in topo:
    if u.op is Ops.SINK: continue
    if u.op is Ops.STAGE or (u.op is Ops.REDUCE and not fuse_reduce) or (u in roots and not is_contiguous_buffer(u)):
      m.append(u)
  return m


# Tensor.detach() wraps a node in STAGE(arg="detach"): autodiff stops there, execution ignores it
strip_detach = PatternMatcher([(UPat(Ops.STAGE, arg="detach", src=(UPat.var("x"),)), lambda x: x)])


def stage_broadcast_compute(roots: list[UOp], factor: int = 4) -> list[UOp]:
  """Fusing is only free if nothing gets recomputed. A computed tensor that is EXPANDed by
  `factor` or more (e.g. a LayerNorm output feeding a matmul) would be recomputed for every
  copy, so we materialize it first. This is the fusion/recompute trade-off, as one rule."""
  from uopc.uop import MOVEMENT, ALU
  todo: set[UOp] = set()
  for u in UOp(Ops.SINK, tuple(roots)).toposort():
    if u.op is not Ops.EXPAND or prod(u.shape) < factor * prod(u.src[0].shape): continue
    base = u.src[0]
    while base.op in MOVEMENT: base = base.src[0]
    if base.op in ALU and base.op is not Ops.CAST and base.shape and _fused_work(base) >= 4096: todo.add(base)
  return UOp.wrap(roots, todo, lambda n: UOp(Ops.STAGE, (n,))) if todo else roots


def _fused_work(u: UOp) -> int:
  """ALU ops * elements that recomputing u would cost (stopping at materialized nodes)."""
  from uopc.uop import ALU
  n, seen, stack = 0, set(), [u]
  while stack:
    x = stack.pop()
    if x in seen or x.op in (Ops.BUFFER, Ops.REDUCE, Ops.STAGE): continue
    seen.add(x)
    if x.op in ALU: n += 1
    stack.extend(x.src)
  return n * prod(u.shape)


def _kernel_name(root: UOp) -> str:
  red = any(u.op is Ops.REDUCE for u in root.toposort())
  return ("r_" if red else "e_") + "_".join(map(str, root.shape or (1,)))


def make_plan(roots: list[UOp], fuse_reduce: bool = False, opt=None) -> tuple[Plan, list[UOp]]:
  """Kernelize. Returns the plan and the canonical list of input BUFFERs it refers to as ("in", i)."""
  inputs = [u for u in UOp(Ops.SINK, tuple(roots)).toposort() if u.op is Ops.BUFFER]
  in_idx = {u: i for i, u in enumerate(inputs)}
  roots = [graph_rewrite(r, strip_detach) for r in roots]
  # fuse_reduce asks for "never write the big intermediate", so don't stage broadcasts either
  if not fuse_reduce: roots = stage_broadcast_compute(roots)
  if opt is not None: roots = opt.layout(roots)
  points = _materialize_points(roots, fuse_reduce)
  # every materialization point gets a placeholder buffer; consumers read it through a reshape
  ref: dict[UOp, tuple[str, int]] = {}
  temps: list[tuple[DType, int]] = []
  outs: list = []
  place: dict[UOp, UOp] = {}
  for k, u in enumerate(points):
    if u in roots: continue
    ref[u] = ("tmp", len(temps))
    temps.append((u.dtype, max(1, prod(u.shape))))
  for r, u in enumerate(roots):
    if is_contiguous_buffer(u): outs.append(("in", in_idx[base_buffer(u)]))
    else:
      ref[u] = ("out", r)
      outs.append((u.dtype, max(1, prod(u.shape))))
  for u in points:
    ph = UOp(Ops.BUFFER, (), ParamArg(-1 - len(place), u.dtype, (max(1, prod(u.shape)),)))   # negative slots: placeholders
    place[u] = ph.reshape(u.shape) if u.shape != ph.shape else ph
  ph_ref = {base_buffer(place[u]): ref[u] for u in points}

  kernels = []
  for u in points:
    body = u.src[0] if u.op is Ops.STAGE else u
    body = body.substitute({p: v for p, v in place.items() if p is not u})
    # renumber buffers 0..n-1 locally (output last) so identical kernels render identical C
    bufs = [b for b in body.toposort() if b.op is Ops.BUFFER]
    local = {b: UOp(Ops.BUFFER, (), ParamArg(i, b.dtype, b.arg.shape)) for i, b in enumerate(bufs)}
    out = UOp(Ops.BUFFER, (), ParamArg(len(bufs), u.dtype, (max(1, prod(u.shape)),)))
    body = body.substitute(local)
    kname = _kernel_name(body)
    if opt is not None: ast = opt.lower(body, out)
    else: ast = lower(body, out)
    from uopc.render_c import render_c
    src = render_c(ast, kname)
    refs = [ph_ref[b] if b.arg.slot < 0 else ("in", in_idx[b]) for b in bufs] + [ref[u]]
    # simplification can delete a read (x*0 -> 0): the C signature then has fewer parameters,
    # so bind exactly the slots that survived, or every later argument shifts by one
    alive = sorted({b.arg.slot for b in ast.toposort() if b.op is Ops.BUFFER})
    refs = [refs[s] for s in alive]
    kernels.append(Kernel(kname, src, refs, flops=_flops(body)))
  return Plan(kernels, temps, outs), inputs


def _flops(u: UOp) -> int:
  """Rough FLOP count of a shaped kernel body: one per ALU element computed."""
  from uopc.uop import ALU
  return sum(prod(x.shape or (1,)) for x in u.toposort() if x.op in ALU) + \
         sum(prod(x.src[0].shape) for x in u.toposort() if x.op is Ops.REDUCE)


# ---------------------------------------------------------------------------
# structural cache ("JIT"): same graph shape -> same plan, whatever the buffers hold

_plans: dict[tuple, Plan] = {}

def _structure_key(roots: list[UOp]) -> tuple:
  ids: dict[UOp, int] = {}
  key = []
  for u in UOp(Ops.SINK, tuple(roots)).toposort():
    ids[u] = len(ids)
    arg = ("buf", u.dtype.name, u.arg.shape) if u.op is Ops.BUFFER else _argkey(u.arg)
    key.append((u.op, arg, tuple(ids[s] for s in u.src)))
  return tuple(key)


def realize(roots: list[UOp], fuse_reduce: bool = False, opt=None) -> list[UOp]:
  """Compute each root. Returns, per root, a BUFFER UOp holding the result."""
  uniq = list(dict.fromkeys(roots))
  t0 = time.perf_counter()
  key = (_structure_key(uniq), fuse_reduce, opt)
  inputs = [u for u in UOp(Ops.SINK, tuple(uniq)).toposort() if u.op is Ops.BUFFER]
  if (plan := _plans.get(key)) is None:
    plan, _ = make_plan(uniq, fuse_reduce, opt)
    for k in plan.kernels:
      k.program = Program(k.src, k.name)
      if DEBUG >= 4: print(k.src)
    _plans[key] = plan
  t1 = time.perf_counter()
  if plan.temp_arrays is None: plan.temp_arrays = [np.empty(n, np_dtype(d)) for d, n in plan.temps]
  temps = plan.temp_arrays
  outs = [inputs[o[1]] if o[0] == "in" else new_buffer(o[0], o[1]) for o in plan.outs]
  def get(rf):
    kind, i = rf
    return buffer_data(inputs[i]) if kind == "in" else temps[i] if kind == "tmp" else buffer_data(outs[i])
  for k in plan.kernels:
    ts = time.perf_counter()
    k.program(*(get(r) for r in k.refs))
    if DEBUG >= 2:
      dt = time.perf_counter() - ts
      extra = ""
      if DEBUG >= 3:   # run again: the second run has warm caches and no first-touch page faults
        ts = time.perf_counter(); k.program(*(get(r) for r in k.refs)); extra = f"  (again: {(time.perf_counter()-ts)*1e3:.3f} ms)"
      print(f"  {k.name:30s} {dt*1e3:8.3f} ms  {k.flops/max(dt, 1e-9)/1e9:7.2f} GFLOP/s{extra}")
  if DEBUG >= 1: print(f"realize: {len(plan.kernels)} kernels, schedule {(t1-t0)*1e3:.1f} ms, run {(time.perf_counter()-t1)*1e3:.1f} ms")
  by_root = dict(zip(uniq, outs))
  return [by_root[r] for r in roots]
