"""Weeks 2-3: turning a graph into a straight-line list of instructions with loops.

A graph has no order; C does. Linearizing means picking a topological order
that also respects *scope*: the set of loops a node must sit inside.

    scope(RANGE r)       = scope(end) | {r}
    scope(END x, *rs)    = scope(x) - rs
    scope(anything else) = union of scope(srcs)

Loops nest in increasing RangeArg.idx order. If two ranges a < b ever appear in
one scope, b is nested inside a, so *every* node that uses b must also sit inside a.
Closing scopes under "encloses" turns each scope into a path in the loop tree:
a sorted tuple that is exactly the stack of open loops when the node runs.

Then we list-schedule:
  1. emit a ready node whose scope is exactly the open loops
  2. else open a loop that pending nodes need next (sibling loops whose END
     another sibling's nodes read must go first)
  3. else close the innermost loop
ALLOCs are declared in the deepest scope shared by all their users (so each CPU
thread gets its own private accumulator).
"""
from __future__ import annotations
import heapq
from uopc.uop import UOp, Ops
from uopc.dtype import AddrSpace


class LinearizeError(Exception): pass


def scopes(sink: UOp) -> dict[UOp, frozenset[UOp]]:
  sc: dict[UOp, frozenset[UOp]] = {}
  for u in sink.toposort():
    s = frozenset().union(*(sc[x] for x in u.src))
    if u.op is Ops.RANGE: s = s | {u}
    if u.op is Ops.END: s = s - set(u.src[1:])
    sc[u] = s
  return sc


def _key(s) -> tuple[UOp, ...]: return tuple(sorted(s, key=lambda r: r.arg))


def loop_targets(topo: list[UOp], sink: UOp) -> dict[UOp, tuple[UOp, ...]]:
  sc = scopes(sink)
  # enclosing loops of each range: every smaller range it ever shares a scope with
  enc: dict[UOp, set[UOp]] = {}
  for s in set(sc.values()):
    t = _key(s)
    for j, r in enumerate(t): enc.setdefault(r, set()).update(t[:j])
  def close(s):
    out, todo = set(s), list(s)
    while todo:
      for a in enc.get(todo.pop(), ()):
        if a not in out: out.add(a); todo.append(a)
    return _key(out)
  target = {u: close(sc[u]) for u in topo}
  for u in topo:
    if u.op is Ops.RANGE: target[u] = close(sc[u] - {u} | enc.get(u, set()))
  return target


def linearize(sink: UOp) -> list[UOp]:
  topo = sink.toposort()
  order = {u: i for i, u in enumerate(topo)}
  target = loop_targets(topo, sink)
  users: dict[UOp, list[UOp]] = {u: [] for u in topo}
  for u in topo:
    for s in set(u.src): users[s].append(u)
  for u in topo:
    if u.op is Ops.ALLOC and u.addrspace is AddrSpace.SHARED: target[u] = ()   # shared memory: whole block, top of kernel
    elif u.op is Ops.ALLOC and users[u]:
      ts = [target[x] for x in users[u]]
      n = 0
      while all(len(t) > n and t[n] is ts[0][n] for t in ts): n += 1
      target[u] = ts[0][:n]

  waiting = {u: len(set(u.src)) for u in topo}
  ready: set[UOp] = set()                          # ready RANGE / END nodes
  heaps: dict[tuple, list[tuple[int, UOp]]] = {}   # ready everything-else, bucketed by scope
  pending, stack, out = set(topo), [], []

  def make_ready(u: UOp):
    if u.op in (Ops.RANGE, Ops.END): ready.add(u)
    else: heapq.heappush(heaps.setdefault(target[u], []), (order[u], u))

  def emit(u: UOp):
    out.append(u); pending.discard(u); ready.discard(u)
    for x in users[u]:
      waiting[x] -= 1
      if waiting[x] == 0: make_ready(x)

  for u in topo:
    if waiting[u] == 0: make_ready(u)
  while pending:
    cur = tuple(stack)
    # 1. anything that belongs exactly here
    if heaps.get(cur):
      emit(heapq.heappop(heaps[cur])[1]); continue
    # 2. open a loop that pending nodes need next
    need = {u: target[u][len(cur)] for u in pending
            if u.op is not Ops.RANGE and len(target[u]) > len(cur) and target[u][:len(cur)] == cur}
    cands = {r for r in need.values() if r in ready and target[r] == cur}
    if cands:
      # if nodes needing loop A read (through an END) the results of sibling loop B, open B first
      ended: dict[UOp, set[UOp]] = {}
      blocked: set[UOp] = set()
      for u in topo:
        if u not in pending: continue
        e = set(u.src[1:]) & cands if u.op is Ops.END else set()
        for x in u.src:
          if x in pending: e |= ended.get(x, set())
        ended[u] = e
        if u in need and e - {need[u]}: blocked.add(need[u])
      r = min(cands - blocked or cands, key=lambda r: r.arg)
      emit(r); stack.append(r); continue
    # 3. close the innermost loop(s)
    ends = [u for u in ready if u.op is Ops.END and target[u] == cur[:len(cur) - (len(u.src) - 1)]
            and set(u.src[1:]) == set(cur[len(cur) - (len(u.src) - 1):])]
    if ends:
      e = min(ends, key=order.__getitem__)
      emit(e); del stack[len(stack) - (len(e.src) - 1):]; continue
    stuck = sorted(pending, key=order.__getitem__)[:5]
    raise LinearizeError(f"can't schedule; open loops {[r.arg.idx for r in cur]}, stuck at "
                         f"{[(u.op, [r.arg.idx for r in target[u]]) for u in stuck]}")
  if stack: raise LinearizeError(f"RANGE without END: loops {[r.arg.idx for r in stack]} are never closed")
  return out
