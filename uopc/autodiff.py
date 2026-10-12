"""Week 5: reverse-mode automatic differentiation on shape graphs.

Autodiff is just another graph transformation: given the graph for `loss` and
some target nodes, build the graph for d(loss)/d(target). It works on pure
shape graphs only -- no STORE, CALL, AFTER, RANGE or END -- which is why we do
it *before* rangeify.

Each rule maps (node, grad of node) -> grad of each source (None = no gradient).
"""
from __future__ import annotations
import math
from uopc.uop import UOp, Ops
from uopc.composite import sum_to, pad, zeros_like

LN2 = math.log(2)


def _full(like: UOp, v: float) -> UOp:
  c = UOp.const(v, like.dtype)
  return c.reshape((1,) * len(like.shape)).expand(like.shape) if like.shape else c

def _rule(u: UOp, g: UOp) -> tuple[UOp | None, ...]:
  s = u.src
  match u.op:
    case Ops.ADD: return (g, g)
    case Ops.MUL: return (g * s[1], g * s[0])
    case Ops.MAX:
      a, b = s
      half = g * _full(g, 0.5)
      zero = zeros_like(g)
      return ((b < a).where(g, (a < b).where(zero, half)), (a < b).where(g, (b < a).where(zero, half)))
    case Ops.RECIP: return (-(g * u * u),)
    case Ops.EXP2: return (g * u * _full(u, LN2),)
    case Ops.LOG2: return (g * s[0].recip() * _full(u, 1 / LN2),)
    case Ops.SQRT: return (g * u.recip() * _full(u, 0.5),)
    case Ops.SIN: return (g * UOp(Ops.SIN, (s[0] + _full(u, math.pi / 2),)),)
    case Ops.CAST: return (g.cast(s[0].dtype) if s[0].dtype.is_float else None,)
    case Ops.WHERE: return (None, s[0].where(g, zeros_like(g)), s[0].where(zeros_like(g), g))
    case Ops.CMPLT | Ops.CMPNE | Ops.FLOORDIV | Ops.FLOORMOD: return (None,) * len(s)
    case Ops.RESHAPE: return (g.reshape(s[0].shape),)
    case Ops.EXPAND: return (sum_to(g, s[0].shape),)
    case Ops.PERMUTE:
      inv = [0] * len(u.arg)
      for i, p in enumerate(u.arg): inv[p] = i
      return (g.permute(inv),)
    case Ops.SHRINK: return (pad(g, tuple((b, n - e) for (b, e), n in zip(u.arg, s[0].shape))),)
    case Ops.STACK: return tuple(g.index(i) for i in range(len(s)))
    case Ops.STAGE: return (None,) if u.arg == "detach" else (g,)
    case Ops.REDUCE:
      rop, n = u.arg
      x = s[0]
      ge = g.reshape(u.shape + (1,) * n).expand(x.shape)
      if rop is Ops.ADD: return (ge,)
      mx = u.reshape(u.shape + (1,) * n).expand(x.shape)
      mask = x.eq(mx).cast(x.dtype)                     # ties share the gradient
      cnt = mask.reduce(Ops.ADD, n).reshape(u.shape + (1,) * n).expand(x.shape)
      return (ge * mask * cnt.recip(),)
    case Ops.INDEX:
      x, idx = s[0], s[1:]
      if not all(i.op is Ops.CONST for i in idx): raise NotImplementedError("gradient of a data-dependent INDEX (use a one-hot matmul)")
      gx = g
      for i, n in zip(reversed(idx), reversed(x.shape[:len(idx)])):
        z = zeros_like(gx)
        gx = UOp(Ops.STACK, tuple(gx if k == i.arg else z for k in range(n)))
      return (gx,) + (None,) * len(idx)
    case Ops.CONST | Ops.BUFFER | Ops.PARAM: return ()
  raise NotImplementedError(f"no gradient rule for {u.op} (autodiff only works on pure shape graphs)")


def gradient(root: UOp, targets: list[UOp]) -> list[UOp]:
  topo = root.toposort()
  # only walk nodes that lie on a path to some target
  on_path = set(targets)
  for u in topo:
    if any(x in on_path for x in u.src): on_path.add(u)
  grads: dict[UOp, UOp] = {root: _full(root, 1.0)}
  for u in reversed(topo):
    if u not in grads or u not in on_path or u in targets and not u.src: continue
    for x, gx in zip(u.src, _rule(u, grads[u])):
      if gx is None or x not in on_path: continue
      grads[x] = grads[x] + gx if x in grads else gx
  return [grads.get(t, zeros_like(t)) for t in targets]
