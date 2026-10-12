"""Week 1: the UOp.

    UOp(op, src: tuple[UOp, ...], arg)

Every program in this course is a DAG of UOps. UOps are *hash consed*:
constructing a UOp that is structurally identical to a live one returns the
very same Python object. That gives us O(1) structural equality (`is`),
free common-subexpression elimination, and cheap memoization keyed on nodes.

dtype, shape and addrspace are *recursive properties*: they are computed
from the op, the arg and the sources, never stored by the user.
"""
from __future__ import annotations
import math, weakref
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from uopc.dtype import DType, dtypes, AddrSpace


class Ops(Enum):
  # ---- week 1: values ---------------------------------------------------
  PARAM = auto()      # arg: ParamArg. An input to the enclosing CALL.
  CALL = auto()       # src[0] = body, src[1:] = arguments. arg: CallArg
  CONST = auto()      # arg: python value (bool/int/float). shape ()
  STACK = auto()      # new leading axis; all srcs share one shape
  INDEX = auto()      # src[0] indexed by src[1:]; pops axes off the left
  CAST = auto()       # arg: DType
  RECIP = auto()      # 1/x
  ADD = auto(); MUL = auto(); CMPLT = auto()
  FLOORDIV = auto(); FLOORMOD = auto()
  WHERE = auto()      # (cond, a, b)
  # extra ALU, introduced when we need them (masks, max-reduce, softmax)
  CMPNE = auto(); MAX = auto()
  EXP2 = auto(); LOG2 = auto(); SQRT = auto(); SIN = auto()
  # ---- week 2: memory ---------------------------------------------------
  BUFFER = auto()     # arg: ParamArg. global persistent memory (never in a CALL body)
  ALLOC = auto()      # arg: ParamArg. call-scope storage `float a[10];`
  LOAD = auto()       # (ptr,)
  STORE = auto()      # (ptr, value)
  AFTER = auto()      # (mem, *deps): `mem`, but only once deps have happened
  SINK = auto()       # (*effects): groups side effects, the root of a program
  # ---- week 3: loops ----------------------------------------------------
  RANGE = auto()      # src[0] = end; arg: RangeArg. value is the loop counter
  END = auto()        # (passthrough, *ranges): closes the ranges
  # ---- week 4: movement + reduction -------------------------------------
  RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto(); SHRINK = auto()
  REDUCE = auto()     # arg: (Ops.ADD | Ops.MAX, n) -- reduces the last n axes
  # ---- week 5: kernels ----------------------------------------------------
  STAGE = auto()      # materialize src[0] into a temp buffer you can index into
  # ---- weeks 7-8: GPUs ----------------------------------------------------
  BARRIER = auto()    # (*deps): __syncthreads()
  WMMA = auto()       # reserved for tensor cores (week 8 lowers GEMMs to WMMA by pattern instead)

  def __repr__(self): return f"Ops.{self.name}"
  __hash__ = object.__hash__   # members are singletons: identity hashing is correct and much faster


UNARY = {Ops.RECIP, Ops.CAST, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN}
BINARY = {Ops.ADD, Ops.MUL, Ops.CMPLT, Ops.CMPNE, Ops.FLOORDIV, Ops.FLOORMOD, Ops.MAX}
ALU = UNARY | BINARY | {Ops.WHERE}
COMMUTATIVE = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE}
MOVEMENT = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.SHRINK}


class AxisType(Enum):
  LOOP = auto()      # a plain serial for-loop
  REDUCE = auto()    # a for-loop that a reduction accumulates over
  UPCAST = auto()    # fully unrolled (week 6). uopc unrolls *during* rangeify (Opts.upcast), so no RANGE of
                     # this kind is ever built; the name stays for the course's vocabulary
  THREAD = auto()    # parallel across CPU threads (week 6)
  GLOBAL = auto()    # GPU blockIdx (week 7)
  LOCAL = auto()     # GPU threadIdx (week 7)
  def __repr__(self): return f"AxisType.{self.name}"
  __hash__ = object.__hash__


@dataclass(frozen=True)
class ParamArg:
  slot: int
  dtype: DType
  shape: tuple[int, ...] = ()
  addrspace: AddrSpace | None = None   # only used by ALLOC (LOCAL or SHARED)


@dataclass(frozen=True)
class CallArg:
  name: str = "f"


@dataclass(frozen=True, order=True)
class RangeArg:
  idx: int                         # loops nest in increasing idx order
  kind: AxisType = AxisType.LOOP


def prod(xs) -> int: return math.prod(xs)


def _argkey(arg):
  # 1 == 1.0 == True and 0.0 == -0.0 in python; the cache must not merge them.
  if isinstance(arg, float): return (float, repr(arg))
  if isinstance(arg, tuple): return (tuple, tuple(_argkey(a) for a in arg))
  return (type(arg), arg)


class UOp:
  __slots__ = ("op", "src", "arg", "dtype", "shape", "addrspace", "__weakref__", "__dict__")
  _cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

  def __new__(cls, op: Ops, src: tuple[UOp, ...] = (), arg: Any = None):
    key = (op, src, _argkey(arg))
    if (ret := UOp._cache.get(key)) is not None: return ret
    ret = super().__new__(cls)
    ret.op, ret.src, ret.arg = op, src, arg
    ret.dtype, ret.shape, ret.addrspace = _infer(op, src, arg)
    UOp._cache[key] = ret
    return ret

  # hash consing means identity *is* structural equality, so we keep object's default
  # __eq__/__hash__ (identity, implemented in C). Note __lt__ etc. build UOps instead.
  def __reduce__(self): return (UOp, (self.op, self.src, self.arg))
  def __repr__(self):
    s = f"UOp({self.op}, arg={self.arg!r}" if self.arg is not None else f"UOp({self.op}"
    return s + (f", src=({', '.join(repr(x) for x in self.src)},))" if self.src else ")")

  def replace(self, **kw) -> UOp:
    return UOp(kw.get("op", self.op), kw.get("src", self.src), kw.get("arg", self.arg))

  # ---- graph helpers ----------------------------------------------------
  def toposort(self) -> list[UOp]:
    """Every node reachable from self, children before parents. Iterative (graphs get deep)."""
    out, seen, stack = [], set(), [(self, False)]
    while stack:
      u, done = stack.pop()
      if done: out.append(u); continue
      if u in seen: continue
      seen.add(u)
      stack.append((u, True))
      stack.extend((s, False) for s in reversed(u.src) if s not in seen)
    return out

  def substitute(self, mapping: dict[UOp, UOp]) -> UOp:
    """Replace nodes (by identity) everywhere in the graph. Rebuilds only what changed."""
    new: dict[UOp, UOp] = {}
    for u in self.toposort():
      if u in mapping: new[u] = mapping[u]; continue
      src = tuple(new[s] for s in u.src)
      new[u] = u if src == u.src else u.replace(src=src)
    return new[self]

  @staticmethod
  def wrap(roots: list[UOp], targets: set[UOp], fn) -> list[UOp]:
    """Rebuild the graph bottom-up, replacing each node t in targets by fn(t'), where t' is t
    with *its* sources already rebuilt. (substitute() would wrap the original, unrewritten t.)"""
    new: dict[UOp, UOp] = {}
    for u in UOp(Ops.SINK, tuple(roots)).toposort():
      src = tuple(new[s] for s in u.src)
      n = u if src == u.src else u.replace(src=src)
      new[u] = fn(n) if u in targets else n
    return [new[r] for r in roots]

  def ranges(self) -> list[UOp]:
    return sorted((u for u in self.toposort() if u.op is Ops.RANGE), key=lambda u: u.arg)

  # ---- constructors -----------------------------------------------------
  @staticmethod
  def const(v, dtype: DType | None = None) -> UOp:
    if dtype == dtypes.f16: return UOp(Ops.CONST, (), float(v)).cast(dtypes.f16)   # python has no f16 values
    if dtype is not None and dtype != dtypes.from_py(v):
      v = {"bool": bool, "i32": int}.get(dtype.name, float)(v)
    return UOp(Ops.CONST, (), v)

  def const_like(self, v) -> UOp:
    """A constant with self's dtype *and shape* (broadcast with RESHAPE + EXPAND)."""
    c = UOp.const(v, self.dtype)
    return c.reshape((1,) * len(self.shape)).expand(self.shape) if self.shape else c
  def _c(self, x) -> UOp: return x if isinstance(x, UOp) else self.const_like(x)
  def alu(self, op: Ops, *src) -> UOp: return UOp(op, (self, *(self._c(s) for s in src)))

  def __add__(self, x): return self.alu(Ops.ADD, x)
  def __radd__(self, x): return self._c(x).alu(Ops.ADD, self)
  def __mul__(self, x): return self.alu(Ops.MUL, x)
  def __rmul__(self, x): return self._c(x).alu(Ops.MUL, self)
  def __neg__(self): return self.alu(Ops.MUL, -1 if self.dtype == dtypes.i32 else -1.0)
  def __sub__(self, x): return self + -self._c(x)
  def __rsub__(self, x): return self._c(x) + -self
  def __floordiv__(self, x): return self.alu(Ops.FLOORDIV, x)
  def __mod__(self, x): return self.alu(Ops.FLOORMOD, x)
  def __lt__(self, x): return self.alu(Ops.CMPLT, x)
  def __gt__(self, x): return self._c(x).alu(Ops.CMPLT, self)
  def ne(self, x): return self.alu(Ops.CMPNE, x)
  def eq(self, x): return self.ne(x).ne(True)
  def maximum(self, x): return self.alu(Ops.MAX, x)
  def recip(self): return UOp(Ops.RECIP, (self,))
  def cast(self, d: DType): return self if d == self.dtype else UOp(Ops.CAST, (self,), d)
  def where(self, a, b):
    a = a if isinstance(a, UOp) else (b.const_like(a) if isinstance(b, UOp) else UOp.const(a))
    b = b if isinstance(b, UOp) else a.const_like(b)
    return UOp(Ops.WHERE, (self, a, b))

  def index(self, *idx) -> UOp:
    return UOp(Ops.INDEX, (self, *(i if isinstance(i, UOp) else UOp.const(int(i)) for i in idx)))
  def load(self) -> UOp: return UOp(Ops.LOAD, (self,))
  def store(self, v) -> UOp: return UOp(Ops.STORE, (self, v))
  def after(self, *deps) -> UOp: return UOp(Ops.AFTER, (self, *deps)) if deps else self
  def end(self, *rngs) -> UOp: return UOp(Ops.END, (self, *rngs)) if rngs else self

  def reshape(self, shape): return UOp(Ops.RESHAPE, (self,), tuple(shape))
  def expand(self, shape): return UOp(Ops.EXPAND, (self,), tuple(shape))
  def permute(self, perm): return UOp(Ops.PERMUTE, (self,), tuple(perm))
  def shrink(self, arg): return UOp(Ops.SHRINK, (self,), tuple(tuple(x) for x in arg))
  def reduce(self, op: Ops, n: int = 1): return UOp(Ops.REDUCE, (self,), (op, n))


def sink(*srcs: UOp) -> UOp: return UOp(Ops.SINK, tuple(srcs))
def range_(end: int | UOp, idx: int, kind: AxisType = AxisType.LOOP) -> UOp:
  return UOp(Ops.RANGE, (end if isinstance(end, UOp) else UOp.const(int(end)),), RangeArg(idx, kind))


# ---------------------------------------------------------------------------
# shape / dtype / addrspace inference. This is the type checker of the IR:
# every malformed UOp dies here, at construction time, with a useful message.

class UOpError(Exception): pass

def _check(cond, msg):
  if not cond: raise UOpError(msg)

def _infer(op: Ops, src: tuple[UOp, ...], arg) -> tuple[DType, tuple[int, ...], AddrSpace]:
  A = AddrSpace.ALU
  if op is Ops.CONST:
    _check(not src, "CONST has no sources")
    return dtypes.from_py(arg), (), A
  if op is Ops.PARAM: return arg.dtype, arg.shape, A
  if op is Ops.BUFFER:
    _check(len(arg.shape) == 1, "BUFFER is flat")
    return arg.dtype, arg.shape, AddrSpace.GLOBAL
  if op is Ops.ALLOC:
    _check(len(arg.shape) == 1, "ALLOC is flat")
    return arg.dtype, arg.shape, arg.addrspace or AddrSpace.LOCAL
  if op is Ops.CALL:
    params = {u.arg.slot: u for u in call_params(src[0])}
    _check(sorted(params) == list(range(len(src) - 1)), f"CALL needs params 0..{len(src)-2}, body has {sorted(params)}")
    for i, a in enumerate(src[1:]):
      p = params[i]
      _check(a.shape == p.shape and a.dtype == p.dtype, f"CALL arg {i}: {a.dtype}{a.shape} != param {p.dtype}{p.shape}")
    return src[0].dtype, src[0].shape, src[0].addrspace
  if op is Ops.STACK:
    _check(len(src) > 0, "STACK of nothing")
    _check(all(s.shape == src[0].shape and s.dtype == src[0].dtype for s in src), "STACK srcs must agree in shape and dtype")
    return src[0].dtype, (len(src),) + src[0].shape, A
  if op is Ops.INDEX:
    x, idx = src[0], src[1:]
    _check(len(idx) <= len(x.shape), f"INDEX: {len(idx)} indices into shape {x.shape}")
    _check(all(i.dtype == dtypes.i32 and i.shape == () for i in idx), "INDEX indices are i32 scalars")
    return x.dtype, x.shape[len(idx):], x.addrspace
  if op in ALU:
    if op is Ops.WHERE:
      c, a, b = src
      _check(c.dtype == dtypes.bool, "WHERE condition must be bool")
      _check(a.dtype == b.dtype, f"WHERE branches disagree: {a.dtype} vs {b.dtype}")
      _check(c.shape == a.shape == b.shape, f"WHERE shape mismatch {c.shape} {a.shape} {b.shape}")
      return a.dtype, a.shape, A
    if op in UNARY:
      _check(len(src) == 1, f"{op} is unary")
      if op is Ops.CAST: return arg, src[0].shape, A
      _check(src[0].dtype.is_float, f"{op} needs a float, got {src[0].dtype}")
      return src[0].dtype, src[0].shape, A
    a, b = src
    _check(a.shape == b.shape, f"{op}: shape mismatch {a.shape} vs {b.shape} (broadcast with EXPAND)")
    _check(a.dtype == b.dtype, f"{op}: dtype mismatch {a.dtype} vs {b.dtype} (use CAST)")
    if op in (Ops.FLOORDIV, Ops.FLOORMOD): _check(a.dtype == dtypes.i32, f"{op} is integer only")
    return (dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else a.dtype), a.shape, A
  if op is Ops.LOAD:
    _check(src[0].addrspace is not A, "LOAD needs memory")
    return src[0].dtype, src[0].shape, A
  if op is Ops.STORE:
    _check(src[0].addrspace is not A, "STORE needs memory")
    _check(src[0].dtype == src[1].dtype, f"STORE of {src[1].dtype} into {src[0].dtype}")
    return dtypes.void, (), A
  if op is Ops.AFTER: return src[0].dtype, src[0].shape, src[0].addrspace
  if op in (Ops.SINK, Ops.END, Ops.BARRIER): return dtypes.void, (), A
  if op is Ops.RANGE:
    _check(src[0].dtype == dtypes.i32, "RANGE end is i32")
    return dtypes.i32, (), A
  if op is Ops.RESHAPE:
    _check(prod(arg) == prod(src[0].shape), f"RESHAPE {src[0].shape} -> {arg}")
    return src[0].dtype, arg, src[0].addrspace
  if op is Ops.EXPAND:
    s = src[0].shape
    _check(len(arg) == len(s) and all(a == b or a == 1 for a, b in zip(s, arg)), f"EXPAND {s} -> {arg}")
    return src[0].dtype, arg, src[0].addrspace
  if op is Ops.PERMUTE:
    _check(sorted(arg) == list(range(len(src[0].shape))), f"PERMUTE {arg} of {src[0].shape}")
    return src[0].dtype, tuple(src[0].shape[i] for i in arg), src[0].addrspace
  if op is Ops.SHRINK:
    s = src[0].shape
    _check(len(arg) == len(s) and all(0 <= b <= e <= n for (b, e), n in zip(arg, s)), f"SHRINK {arg} of {s}")
    return src[0].dtype, tuple(e - b for b, e in arg), src[0].addrspace
  if op is Ops.REDUCE:
    rop, n = arg
    _check(rop in (Ops.ADD, Ops.MAX) and 0 <= n <= len(src[0].shape), f"REDUCE {arg} of {src[0].shape}")
    return src[0].dtype, src[0].shape[:len(src[0].shape) - n], A
  if op is Ops.STAGE: return src[0].dtype, src[0].shape, A
  if op is Ops.WMMA: return src[2].dtype, src[2].shape, A
  raise UOpError(f"no inference rule for {op}")


def call_params(body: UOp) -> list[UOp]:
  """PARAMs that belong to this body. A nested CALL's body is its own scope, so we don't look inside it."""
  out, seen, stack = [], set(), [body]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u)
    if u.op is Ops.PARAM: out.append(u)
    stack.extend(u.src[1:] if u.op is Ops.CALL else u.src)
  return out

