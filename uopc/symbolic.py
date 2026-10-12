"""Week 1: constant folding, symbolic simplification, and "running" programs by rewriting.

Three rule sets:
  * fold      -- evaluate ALU ops whose inputs are all constants
  * symbolic  -- algebra: identities, reassociation, div/mod, bounds (vmin/vmax)
  * evaluate  -- push ALU through STACK, resolve INDEX of STACK, inline CALLs

graph_rewrite(program, fold + symbolic + evaluate) on a closed program (no PARAMs,
no memory) reduces it all the way to CONST/STACK: rewriting *is* the interpreter.
"""
from __future__ import annotations
import math
from uopc.uop import UOp, Ops, ALU, call_params, prod
from uopc.dtype import dtypes, truncate
from uopc.rewrite import UPat, PatternMatcher, graph_rewrite


# ---------------------------------------------------------------------------
# 1. the reference semantics of every ALU op. The C renderer must agree with this.

def _safe(f, *a):
  try: return f(*a)
  except (OverflowError, ValueError, ZeroDivisionError): return None

def exec_alu(op: Ops, dtype, vals: list):
  # inputs first: an f32 CONST may hold a double (16777217.0) that C rounds when it parses the literal
  if op is not Ops.CAST and dtype.is_float: vals = [truncate(dtype, v) if isinstance(v, float) else v for v in vals]
  if op is Ops.WHERE and dtype.is_float: vals = [vals[0]] + [truncate(dtype, v) for v in vals[1:]]
  match op:
    case Ops.ADD: r = vals[0] + vals[1]
    case Ops.MUL: r = vals[0] * vals[1] if not (dtype.is_float and any(math.isinf(v) for v in vals) and 0 in vals) else math.nan
    case Ops.MAX: r = max(vals[0], vals[1])
    case Ops.CMPLT: return vals[0] < vals[1]
    case Ops.CMPNE: return vals[0] != vals[1]
    case Ops.FLOORDIV:
      if vals[1] == 0 or (dtype == dtypes.i32 and vals[0] == -2**31 and vals[1] == -1): return None   # C traps here
      r = vals[0] // vals[1]
    case Ops.FLOORMOD: r = vals[0] % vals[1] if vals[1] != 0 else None
    case Ops.WHERE: r = vals[1] if vals[0] else vals[2]
    case Ops.RECIP: r = 1.0 / vals[0] if vals[0] != 0 else math.copysign(math.inf, vals[0])
    case Ops.EXP2: r = _safe(math.pow, 2.0, vals[0]) if vals[0] < 1024 else math.inf
    case Ops.LOG2: r = math.log2(vals[0]) if vals[0] > 0 else (-math.inf if vals[0] == 0 else math.nan)
    case Ops.SQRT: r = math.sqrt(vals[0]) if vals[0] >= 0 else math.nan
    case Ops.SIN: r = math.sin(vals[0]) if math.isfinite(vals[0]) else math.nan
    case Ops.CAST:
      v = vals[0]
      if dtype == dtypes.i32:
        if not math.isfinite(v): return None          # undefined behavior in C: refuse to fold
        r = int(v)                                    # C truncates toward zero
      elif dtype == dtypes.bool: return bool(v)
      else: r = float(v)
    case _: return None
  return None if r is None else truncate(dtype, r)


# ---------------------------------------------------------------------------
# 2. bounds analysis. vmin/vmax are conservative: the true value is always inside.

def bounds(u: UOp) -> tuple:
  if (b := u.__dict__.get("_bounds")) is not None: return b
  b = _bounds(u)
  if u.dtype == dtypes.i32 and (b[0] < dtypes.min(u.dtype) or b[1] > dtypes.max(u.dtype)):
    b = (dtypes.min(u.dtype), dtypes.max(u.dtype))   # might wrap around: give up
  u.__dict__["_bounds"] = b
  return b

def vmin(u: UOp): return bounds(u)[0]
def vmax(u: UOp): return bounds(u)[1]

def _bounds(u: UOp) -> tuple:
  d = u.dtype
  full = (dtypes.min(d), dtypes.max(d))
  if u.op is Ops.CONST: return (u.arg, u.arg)
  if u.op is Ops.RANGE: return (0, max(0, vmax(u.src[0]) - 1))
  if d.is_float or u.shape != (): return full
  if u.op is Ops.CAST and u.src[0].dtype == dtypes.bool: return (int(vmin(u.src[0])), int(vmax(u.src[0])))
  if u.op is Ops.WHERE:
    (c0, c1), (a0, a1), (b0, b1) = bounds(u.src[0]), bounds(u.src[1]), bounds(u.src[2])
    if c0: return (a0, a1)
    if not c1: return (b0, b1)
    return (min(a0, b0), max(a1, b1))
  if u.op not in ALU or len(u.src) != 2: return full
  (a0, a1), (b0, b1) = bounds(u.src[0]), bounds(u.src[1])
  match u.op:
    case Ops.ADD: return (a0 + b0, a1 + b1)
    case Ops.MUL:
      if any(math.isinf(x) for x in (a0, a1, b0, b1)): return full
      ps = [a0 * b0, a0 * b1, a1 * b0, a1 * b1]
      return (min(ps), max(ps))
    case Ops.MAX: return (max(a0, b0), max(a1, b1))
    case Ops.FLOORDIV if b0 == b1 and b0 > 0: return (a0 // b0, a1 // b0)
    case Ops.FLOORMOD if b0 == b1 and b0 > 0:
      return (a0, a1) if 0 <= a0 and a1 < b0 else (0, b0 - 1)   # floor mod by c>0 is always in [0, c)
    case Ops.CMPLT: return (True, True) if a1 < b0 else (False, False) if a0 >= b1 else (False, True)
    case Ops.CMPNE: return (True, True) if a1 < b0 or b1 < a0 else (False, False) if a0 == a1 == b0 == b1 else (False, True)
  return full


# ---------------------------------------------------------------------------
# 3. rules

def _fold(x: UOp) -> UOp | None:
  if x.shape != () or not all(s.op is Ops.CONST for s in x.src): return None
  r = exec_alu(x.op, x.dtype, [s.arg for s in x.src])
  return None if r is None else UOp.const(r, x.dtype)

fold = PatternMatcher([(UPat(ALU, name="x"), _fold)])


def _divides(u: UOp, c: int) -> UOp | None:
  """u / c if every term of u is a multiple of c, else None."""
  if u.op is Ops.CONST: return u.const_like(u.arg // c) if u.arg % c == 0 else None
  if u.op is Ops.MUL and u.src[1].op is Ops.CONST:
    return u.src[0] * (u.src[1].arg // c) if u.src[1].arg % c == 0 else None
  if u.op is Ops.ADD:
    if (a := _divides(u.src[0], c)) is not None and (b := _divides(u.src[1], c)) is not None: return a + b
  return None

def _split_terms(u: UOp) -> list[UOp]:
  return _split_terms(u.src[0]) + _split_terms(u.src[1]) if u.op is Ops.ADD else [u]

def _div_sum(x: UOp, c: UOp, mod: bool) -> UOp | None:
  """(a*c + b) // c == a + b//c  and  (a*c + b) % c == b % c. Always true for *floor* division."""
  if c.arg <= 0: return None
  terms = _split_terms(x)
  quot, rest = [], []
  for t in terms: (quot if _divides(t, c.arg) is not None else rest).append(t)
  if not quot: return None
  r = sum(rest[1:], rest[0]) if rest else x.const_like(0)
  if mod: return r % c
  return sum((_divides(t, c.arg) for t in quot[1:]), _divides(quot[0], c.arg)) + r // c

def _const_bounds(x: UOp) -> UOp | None:
  # an int/bool expression whose bounds pin it to one value *is* that value
  if x.shape == () and not x.dtype.is_float and x.op in ALU and (b := bounds(x))[0] == b[1]:
    return UOp.const(b[0], x.dtype)
  return None

def _lt_mul(x: UOp, c1: UOp, c2: UOp) -> UOp | None:
  # x*c1 < c2  <=>  x < ceil(c2/c1)  for c1 > 0
  return x < -(-c2.arg // c1.arg) if c1.arg > 0 else None

def _i32(v: int) -> int: return truncate(dtypes.i32, v)   # int32 arithmetic wraps, so folded constants must too

x, y, z = UPat.var("x"), UPat.var("y"), UPat.var("z")
c1, c2 = UPat.cvar("c1"), UPat.cvar("c2")
ints = (dtypes.i32, dtypes.bool)
xi = UPat.var("x", dtype=dtypes.i32)

symbolic = fold + PatternMatcher([
  # canonical form: constants on the right of commutative ops
  (UPat((Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE), src=(UPat(Ops.CONST, name="c"), UPat(name="y")), name="u"),
   lambda u, c, y: u.replace(src=(y, c)) if y.op is not Ops.CONST else None),
  # identities
  (x + 0, lambda x: x), (x + 0.0, lambda x: x),   # x+0.0 is not exact for x=-0.0; we accept that
  (x * 1, lambda x: x), (x * 1.0, lambda x: x),
  (UPat.var("x", dtype=ints) * 0, lambda x: x.const_like(0)),
  (x // 1, lambda x: x), (x % 1, lambda x: x.const_like(0)),
  (UPat(Ops.MAX, src=(x, x)), lambda x: x),
  (UPat(Ops.WHERE, src=(UPat(), x, x)), lambda x: x),
  (UPat(Ops.WHERE, src=(UPat.cvar("c"), x, y)), lambda c, x, y: x if c.arg else y),
  (UPat(Ops.WHERE, src=(z, UPat(Ops.WHERE, src=(z, x, UPat())), y)), lambda z, x, y: z.where(x, y)),
  (UPat(Ops.WHERE, src=(z, x, UPat(Ops.WHERE, src=(z, UPat(), y)))), lambda z, x, y: z.where(x, y)),
  (UPat(Ops.CAST, src=(x,), name="c"), lambda c, x: x if c.dtype == x.dtype else None),
  (UPat(Ops.CMPNE, src=(UPat.var("x", dtype=dtypes.bool), UPat(Ops.CONST, arg=False))), lambda x: x),
  (UPat((Ops.CMPLT, Ops.CMPNE), src=(xi, xi)), lambda x: UOp.const(False)),
  (xi + xi, lambda x: x * 2),
  (xi + xi * -1, lambda x: x.const_like(0)),
  # reassociate constants outward, so that they can meet and fold
  ((xi + c1) + c2, lambda x, c1, c2: x + _i32(c1.arg + c2.arg)),   # ints only: float addition isn't associative
  ((xi * c1) * c2, lambda x, c1, c2: x * _i32(c1.arg * c2.arg)),
  ((xi + c1) * c2, lambda x, c1, c2: x * c2 + _i32(c1.arg * c2.arg)),
  # distribute constants over sums so div/mod can see every term: (a + b) * c -> a*c + b*c
  (UPat(Ops.MUL, src=(UPat(Ops.ADD, src=(xi, UPat.var("y")), name="s"), UPat.cvar("c"))), lambda x, y, s, c: x * c + y * c),
  (UPat(Ops.ADD, src=[xi + c1, UPat.var("y")]), lambda x, c1, y: (x + y) + c1 if y.op is not Ops.CONST else None),
  (xi * c1 + xi * c2, lambda x, c1, c2: x * _i32(c1.arg + c2.arg)),
  (xi * c1 + xi, lambda x, c1: x * _i32(c1.arg + 1)),
  # division and modulo (floor semantics make these unconditional)
  ((xi // c1) // c2, lambda x, c1, c2: x // (c1.arg * c2.arg) if c1.arg > 0 and c2.arg > 0 and c1.arg * c2.arg < 2**31 else None),
  ((xi % c1) % c2, lambda x, c1, c2: x % c2 if c2.arg > 0 and c1.arg % c2.arg == 0 else None),
  ((xi % c1) // c2, lambda x, c1, c2: (x // c2) % (c1.arg // c2.arg) if c2.arg > 0 and c1.arg % c2.arg == 0 else None),
  (UPat(Ops.FLOORDIV, src=(xi, UPat.cvar("c"))), lambda x, c: _div_sum(x, c, mod=False)),
  (UPat(Ops.FLOORMOD, src=(xi, UPat.cvar("c"))), lambda x, c: _div_sum(x, c, mod=True)),
  (UPat(Ops.FLOORDIV, src=(xi, UPat.cvar("c"))), lambda x, c: x.const_like(0) if 0 <= vmin(x) and vmax(x) < c.arg else None),
  (UPat(Ops.FLOORMOD, src=(xi, UPat.cvar("c"))), lambda x, c: x if 0 <= vmin(x) and vmax(x) < c.arg else None),
  # recombine: (x//c)*c + x%c == x
  (UPat(Ops.ADD, src=[(xi // c1) * c1, xi % c1]), lambda x, c1: x),
  (UPat(Ops.ADD, src=[(xi // c1) * c1 + y, xi % c1]), lambda x, c1, y: x + y),
  # comparisons
  ((xi + c1) < c2, lambda x, c1, c2: x < (c2.arg - c1.arg) if -2**31 <= c2.arg - c1.arg < 2**31 else None),
  ((xi * c1) < c2, _lt_mul),
  # bounds say this int expression is a constant
  (UPat(ALU, name="x"), _const_bounds),
])


# ---------------------------------------------------------------------------
# 4. evaluation: STACK, INDEX and CALL

def _elem(s: UOp, i: int) -> UOp: return s.src[i] if s.op is Ops.STACK else s.index(i)

def _push_stack(x: UOp) -> UOp | None:
  # f(STACK(a0, a1), STACK(b0, b1)) -> STACK(f(a0, b0), f(a1, b1))
  if x.shape == () or not any(s.op is Ops.STACK for s in x.src): return None
  return UOp(Ops.STACK, tuple(x.replace(src=tuple(_elem(s, i) for s in x.src)) for i in range(x.shape[0])))

def _index_stack(s: UOp, i: UOp, rest: tuple[UOp, ...]) -> UOp | None:
  if not 0 <= i.arg < len(s.src): raise IndexError(f"index {i.arg} out of bounds for STACK of {len(s.src)}")
  return s.src[i.arg].index(*rest) if rest else s.src[i.arg]

def inline_call(call: UOp) -> UOp:
  """Beta reduction: replace each PARAM of the body with its argument.
  Scoped: a nested CALL's body (its src[0]) has its own PARAMs, so we never rewrite inside it."""
  args = {p: call.src[1 + p.arg.slot] for p in call_params(call.src[0])}
  order, seen, stack = [], set(), [(call.src[0], False)]
  while stack:
    u, done = stack.pop()
    if done: order.append(u); continue
    if u in seen: continue
    seen.add(u)
    stack.append((u, True))
    stack.extend((s, False) for s in (u.src[1:] if u.op is Ops.CALL else u.src))
  new: dict[UOp, UOp] = {}
  for u in order:
    if u in args: new[u] = args[u]; continue
    src = (u.src[0], *(new[s] for s in u.src[1:])) if u.op is Ops.CALL else tuple(new[s] for s in u.src)
    new[u] = u if src == u.src else u.replace(src=src)
  return new[call.src[0]]

def _literal(u: UOp):
  """A CONST/STACK tree as (flat list of python values, shape), or None if anything is symbolic."""
  if u.op is Ops.CONST: return [u.arg], ()
  if u.op is not Ops.STACK: return None
  parts = [_literal(s) for s in u.src]
  if any(p is None for p in parts): return None
  return [v for p in parts for v in p[0]], u.shape

def _from_flat(flat: list, shape: tuple, dtype) -> UOp:
  if not shape: return UOp.const(flat[0], dtype)
  n = len(flat) // shape[0]
  return UOp(Ops.STACK, tuple(_from_flat(flat[i*n:(i+1)*n], shape[1:], dtype) for i in range(shape[0])))

def _unravel(i: int, shape) -> list[int]:
  out = []
  for s in reversed(shape): out.append(i % s); i //= s
  return out[::-1]

def _ravel(idx, shape) -> int:
  i = 0
  for x, s in zip(idx, shape): i = i * s + x
  return i

def _eval_movement(u: UOp) -> UOp | None:
  """Movement ops and REDUCE on fully known values: just compute them."""
  if (lit := _literal(u.src[0])) is None: return None
  flat, ishape = lit
  oshape, n_out = u.shape, prod(u.shape)
  match u.op:
    case Ops.RESHAPE: out = flat
    case Ops.EXPAND: out = [flat[_ravel([0 if s == 1 else x for x, s in zip(_unravel(i, oshape), ishape)], ishape)] for i in range(n_out)]
    case Ops.PERMUTE:
      inv = [u.arg.index(k) for k in range(len(ishape))]
      out = [flat[_ravel([_unravel(i, oshape)[inv[k]] for k in range(len(ishape))], ishape)] for i in range(n_out)]
    case Ops.SHRINK: out = [flat[_ravel([x + b for x, (b, _) in zip(_unravel(i, oshape), u.arg)], ishape)] for i in range(n_out)]
    case Ops.REDUCE:
      rop, k = u.arg
      inner = prod(ishape[len(ishape) - k:])
      out = []
      for i in range(n_out):
        acc = flat[i * inner]
        for v in flat[i * inner + 1:(i + 1) * inner]: acc = exec_alu(rop, u.dtype, [acc, v])
        out.append(acc)
  return _from_flat(out, oshape, u.dtype)

evaluate = PatternMatcher([
  (UPat({Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.SHRINK, Ops.REDUCE}, name="u"), _eval_movement),
  (UPat(Ops.INDEX, src=(x,)), lambda x: x),
  (UPat(Ops.INDEX, src=(UPat(Ops.INDEX, name="a"),), allow_any_len=True, name="b"),
   lambda a, b: a.src[0].index(*a.src[1:], *b.src[1:])),
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name="s"), UPat.cvar("i")), allow_any_len=True, name="u"),
   lambda s, i, u: _index_stack(s, i, u.src[2:])),
  (UPat(ALU, name="x"), _push_stack),
  (UPat(Ops.CALL, name="call"), inline_call),
])


_simplified: dict[UOp, UOp] = {}

def simplify(u: UOp) -> UOp:
  """graph_rewrite(u, symbolic), memoized: rewriting is a pure function of a (hash-consed) node."""
  if (r := _simplified.get(u)) is not None: return r
  if len(_simplified) > 200_000: _simplified.clear()
  r = _simplified[u] = graph_rewrite(u, symbolic)
  _simplified[r] = r
  return r
def run(u: UOp) -> UOp: return graph_rewrite(u, symbolic + evaluate)

def to_python(u: UOp):
  """A fully evaluated program as nested python lists."""
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.STACK: return [to_python(s) for s in u.src]
  raise ValueError(f"program did not evaluate to a value, stuck at {u.op}")
