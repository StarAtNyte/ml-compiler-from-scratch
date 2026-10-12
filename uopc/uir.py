"""Week 1: the .uir interchange format.

    # gemm_256x256.uir
    %0 = buffer 65536 : dtype=f32 slot=0
    %1 = buffer 65536 : dtype=f32 slot=1
    %2 = reshape %0, (256, 1, 256)
    %3 = reshape %1, (1, 256, 256)
    %4 = mul %2, %3
    %5 = reduce %4 : op=add pop=1

Grammar (one statement per line, `#` starts a comment):

    line     := '%' NAME '=' opname operands? (':' attr*)?
    operands := operand (',' operand)*
    operand  := '%' NAME | literal          literal := int | float | true | false | tuple
    attr     := key '=' value

Integer/float literals in a *source* position become CONSTs (`add %x, 1.0`, `range 10`).
Tuple literals are the arg of movement ops. The last statement is the program.

Two pieces of sugar make hand-written programs readable:
  * binary ops broadcast: size-1 axes are EXPANDed, missing leading axes RESHAPEd in
  * a bare literal adopts the dtype of the other operand (`mul %f32, 2` means 2.0)
"""
from __future__ import annotations
import ast, math, re
from uopc.uop import UOp, Ops, ParamArg, CallArg, RangeArg, AxisType, BINARY, UOpError
from uopc.dtype import dtypes, AddrSpace

POSITIONAL_ARG = {Ops.BUFFER, Ops.ALLOC, Ops.CONST, Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.SHRINK}
LITERAL_WORDS = {"true": True, "false": False, "inf": math.inf, "-inf": -math.inf, "nan": math.nan}


class ParseError(Exception): pass


def _split_top(s: str) -> list[str]:
  """Split on commas that are not inside parentheses."""
  out, depth, cur = [], 0, ""
  for ch in s:
    if ch == "," and depth == 0: out.append(cur.strip()); cur = ""; continue
    depth += (ch == "(") - (ch == ")")
    cur += ch
  if cur.strip(): out.append(cur.strip())
  return out


def _literal(tok: str):
  if tok in LITERAL_WORDS: return LITERAL_WORDS[tok]
  try: return ast.literal_eval(tok)
  except (ValueError, SyntaxError): raise ParseError(f"bad literal {tok!r}") from None


def _attr_value(key: str, v: str):
  if key == "dtype": return dtypes.from_name(v)
  if key == "op": return Ops[v.upper()]
  if key == "kind": return AxisType[v.upper()]
  if key == "space": return AddrSpace[v.upper()]
  if key == "name": return v
  return _literal(v)


def _broadcast(a: UOp, b: UOp) -> tuple[UOp, UOp]:
  if a.shape == b.shape: return a, b
  n = max(len(a.shape), len(b.shape))
  a = a.reshape((1,) * (n - len(a.shape)) + a.shape) if len(a.shape) < n else a
  b = b.reshape((1,) * (n - len(b.shape)) + b.shape) if len(b.shape) < n else b
  out = []
  for x, y in zip(a.shape, b.shape):
    if x != y and 1 not in (x, y): raise ParseError(f"can't broadcast {a.shape} with {b.shape}")
    out.append(max(x, y))
  return (a if a.shape == tuple(out) else a.expand(out)), (b if b.shape == tuple(out) else b.expand(out))


def parse(text: str) -> UOp:
  env: dict[str, UOp] = {}
  last = None
  for lineno, raw in enumerate(text.splitlines(), 1):
    line = raw.split("#", 1)[0].strip()
    if not line: continue
    try: last = _parse_line(line, env)
    except (ParseError, UOpError, KeyError, ValueError, TypeError) as e:
      raise ParseError(f"line {lineno}: {raw.strip()!r}: {e}") from None
  if last is None: raise ParseError("empty program")
  return last


def _parse_line(line: str, env: dict[str, UOp]) -> UOp:
  lhs, rhs = (p.strip() for p in line.split("=", 1))
  if not lhs.startswith("%"): raise ParseError("statement must start with %name =")
  body, _, attrs_s = rhs.partition(":")
  opname, _, ops_s = body.strip().partition(" ")
  op = Ops[opname.upper()]
  attrs = {}
  for k, v in re.findall(r"(\w+)=(\([^=]*\)|\S+)", attrs_s):
    attrs[k] = _attr_value(k, v)
  toks = _split_top(ops_s)
  lit = None
  if op in POSITIONAL_ARG:
    if op in (Ops.BUFFER, Ops.ALLOC, Ops.CONST): lit, toks = _literal(toks[0]), toks[1:]
    else: lit, toks = _literal(toks[-1]), toks[:-1]
  src = []
  for t in toks:
    if t.startswith("%"):
      if t[1:] not in env: raise ParseError(f"undefined {t}")
      src.append(env[t[1:]])
    else: src.append(_literal(t))
  env[lhs[1:]] = u = _build(op, src, lit, attrs)
  return u


def _build(op: Ops, src: list, lit, attrs: dict) -> UOp:
  # bare literals in source position become CONSTs, typed like their neighbours
  # (a WHERE's condition is bool: its branches decide the literal's type)
  like = next((s for s in (src[1:] if op is Ops.WHERE else src) if isinstance(s, UOp)), None)
  def const(v):
    if op in (Ops.INDEX, Ops.RANGE) or like is None or isinstance(v, bool): return UOp.const(v)
    return UOp.const(v, like.dtype)
  src = [s if isinstance(s, UOp) else const(s) for s in src]
  match op:
    case Ops.BUFFER | Ops.ALLOC:
      return UOp(op, (), ParamArg(attrs.get("slot", 0), attrs.get("dtype", dtypes.f32), (lit,), attrs.get("space")))
    case Ops.PARAM: return UOp(op, (), ParamArg(attrs["slot"], attrs.get("dtype", dtypes.f32), tuple(attrs.get("shape", ()))))
    case Ops.CONST: return UOp.const(lit, attrs.get("dtype"))
    case Ops.CALL: return UOp(op, tuple(src), CallArg(attrs.get("name", "f")))
    case Ops.CAST: return src[0].cast(attrs["dtype"])
    case Ops.RANGE: return UOp(op, tuple(src), RangeArg(attrs.get("idx", 0), attrs.get("kind", AxisType.LOOP)))
    case Ops.REDUCE: return src[0].reduce(attrs.get("op", Ops.ADD), attrs.get("pop", 1))
    case Ops.RESHAPE | Ops.EXPAND | Ops.PERMUTE: return UOp(op, (src[0],), tuple(lit))
    case Ops.SHRINK: return src[0].shrink(lit)
  if op in BINARY:
    a, b = _broadcast(*src)
    return UOp(op, (a, b))
  if op is Ops.WHERE:
    c, a, b = src
    a, b = _broadcast(a, b); c, a = _broadcast(c, a); c, b = _broadcast(c, b)
    return UOp(op, (c, a, b))
  return UOp(op, tuple(src))


# ---------------------------------------------------------------------------
# printing

def _fmt_lit(v) -> str:
  if isinstance(v, bool): return "true" if v else "false"
  if isinstance(v, float):
    if math.isnan(v): return "nan"
    if math.isinf(v): return "inf" if v > 0 else "-inf"
    return repr(v)
  if isinstance(v, tuple): return "(" + ", ".join(_fmt_lit(x) for x in v) + ("," if len(v) == 1 else "") + ")"
  return str(v)


def to_uir(root: UOp) -> str:
  names: dict[UOp, str] = {}
  lines = []
  for u in root.toposort():
    if u.op is Ops.CONST and u is not root: continue   # constants are printed inline
    names[u] = f"%{len(names)}"
    ops = [_fmt_lit(s.arg) if s.op is Ops.CONST else names[s] for s in u.src]
    attrs = []
    a = u.arg
    match u.op:
      case Ops.BUFFER | Ops.ALLOC:
        ops = [str(a.shape[0])]
        attrs = [f"dtype={a.dtype.name}", f"slot={a.slot}"] + ([f"space={a.addrspace.name.lower()}"] if a.addrspace else [])
      case Ops.PARAM: attrs = [f"slot={a.slot}", f"dtype={a.dtype.name}"] + ([f"shape={_fmt_lit(a.shape)}"] if a.shape else [])
      case Ops.CONST: ops = [_fmt_lit(a)]
      case Ops.CALL: attrs = [f"name={a.name}"]
      case Ops.CAST: attrs = [f"dtype={a.name}"]
      case Ops.RANGE: attrs = [f"idx={a.idx}", f"kind={a.kind.name.lower()}"]
      case Ops.REDUCE: attrs = [f"op={a[0].name.lower()}", f"pop={a[1]}"]
      case Ops.RESHAPE | Ops.EXPAND | Ops.PERMUTE | Ops.SHRINK: ops.append(_fmt_lit(a))
    line = f"{names[u]} = {u.op.name.lower()}"
    if ops: line += " " + ", ".join(ops)
    if attrs: line += " : " + " ".join(attrs)
    lines.append(line)
  return "\n".join(lines) + "\n"
