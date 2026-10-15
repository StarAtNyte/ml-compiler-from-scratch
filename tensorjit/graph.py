"""Immutable tensor expressions with bounded symbolic dimensions.

Supported operations are float32 and pure: add/mul/ReLU, transpose, reduction,
and rank-two matrix multiplication. Symbols identify equal logical dimensions.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Dim:
  name: str
  lower: int = 1
  upper: int = 4096

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name.isidentifier() or type(self.lower) is not int or type(self.upper) is not int or self.lower < 1 or self.upper < self.lower or self.upper >= 2**31:
      raise ValueError("invalid dimension name or bounds")


@dataclass(frozen=True, eq=False)
class Expr:
  op: str
  shape: tuple[int | Dim, ...]
  sources: tuple[Expr, ...] = ()
  arg: object = None

  def __add__(self, other): return binary("add", self, other)
  def __mul__(self, other): return binary("mul", self, other)
  def __neg__(self): return binary("mul", -1, self)
  def __sub__(self, other): return binary("add", self, -(other if isinstance(other, Expr) else constant(other)))
  def __rsub__(self, other): return binary("add", other, -self)
  def __radd__(self, other): return binary("add", other, self)
  def __rmul__(self, other): return binary("mul", other, self)
  def relu(self): return Expr("relu", self.shape, (self,))

  def expand_dims(self, axis):
    if not -len(self.shape)-1 <= axis <= len(self.shape): raise ValueError("invalid inserted axis")
    axis %= len(self.shape)+1
    return Expr("expand_dims", self.shape[:axis]+(1,)+self.shape[axis:], (self,), axis)

  def broadcast_to(self, shape):
    shape = tuple(shape)
    if any(not isinstance(d, Dim) and (type(d) is not int or d < 1 or d >= 2**31) for d in shape):
      raise ValueError("invalid target shape")
    if broadcast(self.shape, shape) != shape: raise ValueError("cannot broadcast to target shape")
    return Expr("broadcast", shape, (self,))

  def transpose(self):
    if len(self.shape) != 2: raise ValueError("transpose requires a matrix")
    return Expr("transpose", self.shape[::-1], (self,))

  def sum(self, axis=-1):
    if not self.shape or not -len(self.shape) <= axis < len(self.shape): raise ValueError("invalid reduction axis")
    axis %= len(self.shape)
    return Expr("sum", self.shape[:axis] + self.shape[axis+1:], (self,), axis)

  def __matmul__(self, other):
    if not isinstance(other, Expr) or len(self.shape) != 2 or len(other.shape) != 2 or self.shape[1] != other.shape[0]:
      raise ValueError("matmul requires rank-two operands with equal reduction dimensions")
    return Expr("matmul", (self.shape[0], other.shape[1]), (self, other))


def input(name, shape):
  shape = tuple(shape)
  if not name.isidentifier() or any(not isinstance(d, Dim) and (type(d) is not int or d < 1 or d >= 2**31) for d in shape):
    raise ValueError("invalid input name or shape")
  return Expr("input", shape, arg=name)


def constant(value):
  if type(value) not in (int, float) or not math.isfinite(value): raise ValueError("constant must be finite")
  if abs(value) > 3.4028234663852886e38: raise ValueError("constant outside finite float32 range")
  return Expr("constant", (), arg=float(value))


def broadcast(a, b):
  output = []
  for i in range(1, max(len(a), len(b))+1):
    left, right = a[-i] if i <= len(a) else 1, b[-i] if i <= len(b) else 1
    if left == right: output.append(left)
    elif left == 1: output.append(right)
    elif right == 1: output.append(left)
    else: raise ValueError(f"incompatible broadcast dimensions: {left}, {right}")
  return tuple(reversed(output))


def binary(op, left, right):
  left = left if isinstance(left, Expr) else constant(left)
  right = right if isinstance(right, Expr) else constant(right)
  return Expr(op, broadcast(left.shape, right.shape), (left, right))


def toposort(outputs):
  seen, order = set(), []
  pending = [(node, False) for node in reversed(outputs)]
  while pending:
    node, expanded = pending.pop()
    if node in seen: continue
    if expanded:
      seen.add(node); order.append(node)
    else:
      pending.append((node, True))
      pending.extend((source, False) for source in reversed(node.sources))
  return order
