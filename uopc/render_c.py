"""Weeks 2-3: rendering a linearized program as a C function.

Each UOp becomes at most one line of C. CONST, INDEX, AFTER and pure ALU ops
used once are rendered inline as expressions; everything else gets a named
variable. BUFFERs become function parameters, sorted by slot.
"""
from __future__ import annotations
import math
from uopc.uop import UOp, Ops, AxisType, ALU
from uopc.dtype import dtypes, DType, AddrSpace
from uopc.linearize import linearize
from uopc.symbolic import vmin

PRELUDE = """#include <math.h>
#include <stdbool.h>
static inline int floordiv_i32(int a, int b) { int q = a / b; return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q; }
static inline int floormod_i32(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
"""

def cconst(v, d: DType) -> str:
  if d == dtypes.bool: return "1" if v else "0"
  if d == dtypes.i32: return str(v) if v != -2**31 else "(-2147483647-1)"
  if math.isnan(v): return "NAN"
  if math.isinf(v): return "INFINITY" if v > 0 else "-INFINITY"
  s = repr(float(v))
  return s + "f" if d == dtypes.f32 else f"(({d.ctype}){s}f)"

def ctype(d: DType) -> str: return d.ctype

def _binop(op: Ops, a: str, b: str, ua: UOp, ub: UOp, d: DType) -> str:
  match op:
    case Ops.ADD: return f"({a}+{b})"
    case Ops.MUL: return f"({a}*{b})"
    case Ops.CMPLT: return f"({a}<{b})"
    case Ops.CMPNE: return f"({a}!={b})"
    case Ops.MAX: return f"(({a}>{b})?{a}:{b})"
    case Ops.FLOORDIV:
      # C division truncates; it equals floor division when both sides are non-negative
      if vmin(ua) >= 0 and ub.op is Ops.CONST and ub.arg > 0: return f"({a}/{b})"
      return f"floordiv_i32({a},{b})"
    case Ops.FLOORMOD:
      if vmin(ua) >= 0 and ub.op is Ops.CONST and ub.arg > 0: return f"({a}%{b})"
      return f"floormod_i32({a},{b})"
  raise NotImplementedError(op)

UNARY_FN = {Ops.EXP2: "exp2", Ops.LOG2: "log2", Ops.SQRT: "sqrt", Ops.SIN: "sin"}

def _unop(op: Ops, a: str, d: DType, arg) -> str:
  if op is Ops.CAST: return f"(({arg.ctype})({a}))"
  if op is Ops.RECIP: return f"(1.0f/{a})" if d == dtypes.f32 else f"(({d.ctype})1.0f/{a})"
  return f"{UNARY_FN[op]}{'f' if d == dtypes.f32 else ''}({a})"


class CRenderer:
  """Render a SINK of stores into one C function. Subclassed for CUDA in week 7."""
  kernel_prefix = "void"
  buffer_prefix = ""
  shared_prefix = ""
  barrier = None
  restrict = "restrict"

  def render_range(self, u: UOp, name: str, end: str) -> list[str]:
    pragma = ["#pragma omp parallel for"] if u.arg.kind is AxisType.THREAD else []
    return pragma + [f"for (int {name} = 0; {name} < {end}; {name}++) {{"]

  def render(self, sink: UOp, name: str = "kernel") -> str:
    lin = linearize(sink)
    uses: dict[UOp, int] = {}
    for u in lin:
      for s in u.src: uses[s] = uses.get(s, 0) + 1
    r: dict[UOp, str] = {}
    lines: list[str] = []
    depth = 1
    bufs = sorted({u for u in lin if u.op is Ops.BUFFER}, key=lambda u: u.arg.slot)
    counter = 0
    def var(prefix: str) -> str:
      nonlocal counter
      counter += 1
      return f"{prefix}{counter}"
    def line(s: str): lines.append("  " * depth + s)

    for u in lin:
      op = u.op
      if op is Ops.CONST: r[u] = cconst(u.arg, u.dtype)
      elif op is Ops.BUFFER: r[u] = f"data{u.arg.slot}"
      elif op is Ops.ALLOC:
        r[u] = f"{'smem' if u.addrspace is AddrSpace.SHARED else 'acc'}{u.arg.slot}"
        pre = self.shared_prefix if u.addrspace is AddrSpace.SHARED else ""
        if self._scalar(u): line(f"{ctype(u.dtype)} {r[u]};")     # a size-1 accumulator is just a variable
        else: line(f"{pre}{ctype(u.dtype)} {r[u]}[{u.arg.shape[0]}];")
      elif op is Ops.AFTER: r[u] = r[u.src[0]]
      elif op is Ops.INDEX:
        base = u.src[0]
        while base.op is Ops.AFTER: base = base.src[0]
        r[u] = r[u.src[0]] if base.op is Ops.ALLOC and self._scalar(base) else f"{r[u.src[0]]}[{r[u.src[1]]}]"
      elif op is Ops.LOAD:
        r[u] = var("val")
        line(f"{ctype(u.dtype)} {r[u]} = {r[u.src[0]]};")
      elif op is Ops.STORE: line(f"{r[u.src[0]]} = {r[u.src[1]]};")
      elif op is Ops.RANGE:
        r[u] = self.range_name(u)
        for l in self.render_range(u, r[u], r[u.src[0]]): line(l)
        if self.is_loop(u): depth += 1
      elif op is Ops.END:
        for rng in u.src[1:]:
          if self.is_loop(rng):
            depth -= 1
            line("}")
      elif op is Ops.BARRIER: line(self.barrier + ";")
      elif op is Ops.SINK: pass
      elif op is Ops.WHERE:
        expr = f"({r[u.src[0]]}?{r[u.src[1]]}:{r[u.src[2]]})"
        self._assign(u, expr, uses, r, line, var)
      elif op in ALU:
        if len(u.src) == 1: expr = _unop(op, r[u.src[0]], u.dtype, u.arg)
        else: expr = _binop(op, r[u.src[0]], r[u.src[1]], u.src[0], u.src[1], u.dtype)
        self._assign(u, expr, uses, r, line, var)
      elif op is Ops.WMMA: self.render_wmma(u, r, line)
      else: raise NotImplementedError(f"C renderer can't handle {op}")

    params = ", ".join(f"{self.buffer_prefix}{ctype(b.dtype)}* {self.restrict} data{b.arg.slot}" for b in bufs)
    return self.header(sink, lin) + f"{self.kernel_prefix} {name}({params}) {{\n" + "\n".join(lines) + "\n}\n"

  def header(self, sink: UOp, lin: list[UOp]) -> str: return PRELUDE

  def _scalar(self, alloc: UOp) -> bool: return alloc.arg.shape == (1,) and alloc.addrspace is AddrSpace.LOCAL

  def range_name(self, u: UOp) -> str: return f"ridx{u.arg.idx}"

  def is_loop(self, u: UOp) -> bool:
    """On a CPU every range is a for loop -- GPU ranges too, which makes this an emulator for GPU kernels."""
    return True

  def _assign(self, u, expr, uses, r, line, var):
    if uses.get(u, 0) <= 1: r[u] = expr   # used once: inline it
    else:
      r[u] = var("alu")
      line(f"{ctype(u.dtype)} {r[u]} = {expr};")

  def render_wmma(self, u, r, line): raise NotImplementedError("WMMA needs the CUDA renderer")


def render_c(sink: UOp, name: str = "kernel") -> str: return CRenderer().render(sink, name)
