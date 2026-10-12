"""Week 1: pattern matching and graph rewriting.

A *rewrite rule* is (pattern, function). The pattern is a UPat tree that matches
UOps structurally and binds names; the function receives the bindings as keyword
arguments and returns a replacement UOp, or None for "rule does not apply".

graph_rewrite applies a PatternMatcher bottom-up and greedily: children are
rewritten to a fixed point before their parent is looked at, and whenever a rule
fires the result is itself rewritten again. Because UOps are hash consed, the
memo table is just a dict keyed by node.
"""
from __future__ import annotations
import itertools
from typing import Any, Callable
from uopc.uop import UOp, Ops
from uopc.dtype import DType


class UPat:
  __slots__ = ("op", "src", "name", "arg", "dtype", "allow_any_len")

  def __init__(self, op: Ops | tuple[Ops, ...] | set[Ops] | None = None, src=None, name: str | None = None,
               arg: Any = None, dtype: DType | tuple[DType, ...] | None = None, allow_any_len: bool = False):
    self.op = None if op is None else (frozenset([op]) if isinstance(op, Ops) else frozenset(op))
    self.dtype = None if dtype is None else (frozenset([dtype]) if isinstance(dtype, DType) else frozenset(dtype))
    # src: tuple -> match in order. list -> match in any order (for commutative ops).
    self.src, self.name, self.arg, self.allow_any_len = src, name, arg, allow_any_len

  @staticmethod
  def var(name: str | None = None, dtype=None) -> UPat: return UPat(name=name, dtype=dtype)
  @staticmethod
  def cvar(name: str | None = None, dtype=None) -> UPat: return UPat(Ops.CONST, name=name, dtype=dtype)

  def __repr__(self): return f"UPat({self.op and set(self.op)}, name={self.name})"

  def match(self, u: UOp, store: dict[str, UOp]) -> list[dict[str, UOp]]:
    """All ways this pattern matches u, each a dict of bindings."""
    if self.op is not None and u.op not in self.op: return []
    if self.dtype is not None and u.dtype not in self.dtype: return []
    if self.arg is not None and u.arg != self.arg: return []
    if self.name is not None:
      if self.name in store and store[self.name] is not u: return []   # same name must bind same node
      store = {**store, self.name: u}
    if self.src is None: return [store]
    orders = itertools.permutations(self.src) if isinstance(self.src, list) else [self.src]
    out = []
    for pats in orders:
      if len(pats) != len(u.src) and not (self.allow_any_len and len(pats) <= len(u.src)): continue
      stores = [store]
      for p, s in zip(pats, u.src):
        stores = [m for st in stores for m in p.match(s, st)]
        if not stores: break
      out.extend(stores)
    return out

  # sugar so rules read like math: UPat.var("x") + UPat.cvar("c")
  def _bin(self, op, other, swap=False):
    other = other if isinstance(other, UPat) else UPat(Ops.CONST, arg=other)
    src = [other, self] if swap else [self, other]
    return UPat(op, src if op in (Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE) else tuple(src))
  def __add__(self, o): return self._bin(Ops.ADD, o)
  def __mul__(self, o): return self._bin(Ops.MUL, o)
  def __floordiv__(self, o): return self._bin(Ops.FLOORDIV, o)
  def __mod__(self, o): return self._bin(Ops.FLOORMOD, o)
  def __lt__(self, o): return self._bin(Ops.CMPLT, o)


Rule = tuple[UPat, Callable[..., UOp | None]]


class PatternMatcher:
  def __init__(self, rules: list[Rule]):
    self.rules = rules
    # index by root op so we only try rules that could possibly match
    self.by_op: dict[Ops, list[Rule]] = {}
    for pat, fn in rules:
      for op in (pat.op if pat.op is not None else Ops):
        self.by_op.setdefault(op, []).append((pat, fn))

  def __add__(self, other: PatternMatcher) -> PatternMatcher: return PatternMatcher(self.rules + other.rules)

  def rewrite(self, u: UOp, ctx=None) -> UOp | None:
    for pat, fn in self.by_op.get(u.op, ()):
      for binding in pat.match(u, {}):
        kw = {**binding, "ctx": ctx} if ctx is not None else binding
        if (ret := fn(**kw)) is not None and ret is not u: return ret
    return None


def graph_rewrite(root: UOp, pm: PatternMatcher, ctx=None, max_steps: int = 10_000_000) -> UOp:
  """Bottom-up greedy rewrite to a fixed point. Iterative, so deep graphs don't blow the stack.

  Stage 0: schedule children.   Stage 1: children done -> rebuild, try rules.
  Stage 2: a rule fired and we rewrote its result -> record that as our answer.
  """
  done: dict[UOp, UOp] = {}
  stack: list[tuple[UOp, int, UOp | None]] = [(root, 0, None)]
  steps = 0
  while stack:
    u, stage, aux = stack.pop()
    if u in done and stage != 2: continue
    if stage == 0:
      stack.append((u, 1, None))
      stack.extend((s, 0, None) for s in reversed(u.src) if s not in done)
    elif stage == 1:
      src = tuple(done[s] for s in u.src)
      n = u if src == u.src else u.replace(src=src)
      steps += 1
      if steps > max_steps: raise RuntimeError("graph_rewrite did not converge (a rule loops?)")
      if (m := pm.rewrite(n, ctx)) is None: done[u] = n
      elif m in done: done[u] = done[m]
      else: stack.extend([(u, 2, m), (m, 0, None)])
    else:
      done[u] = done[aux]
  return done[root]
