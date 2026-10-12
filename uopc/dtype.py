"""Week 1: data types and address spaces.

A DType is a tiny frozen record. We keep the set deliberately small:
void, bool, i32, f32 (weeks 1-6) and f16 (week 8, tensor cores).
"""
from __future__ import annotations
import math, struct
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True, eq=True)
class DType:
  name: str
  itemsize: int
  ctype: str
  fmt: str | None  # struct / numpy format character

  def __repr__(self): return f"dtypes.{self.name}"
  @property
  def is_float(self) -> bool: return self.name in ("f32", "f16")
  @property
  def is_int(self) -> bool: return self.name == "i32"


class dtypes:
  void = DType("void", 0, "void", None)
  bool = DType("bool", 1, "_Bool", "?")
  i32 = DType("i32", 4, "int", "i")
  f32 = DType("f32", 4, "float", "f")
  f16 = DType("f16", 2, "_Float16", "e")
  all = (void, bool, i32, f32, f16)

  @staticmethod
  def from_name(name: str) -> DType:
    for d in dtypes.all:
      if d.name == name: return d
    raise ValueError(f"unknown dtype {name!r}")

  @staticmethod
  def from_py(x) -> DType:
    if isinstance(x, bool): return dtypes.bool
    if isinstance(x, int): return dtypes.i32
    if isinstance(x, float): return dtypes.f32
    raise TypeError(f"no dtype for python value {x!r}")

  @staticmethod
  def min(d: DType):
    return {"bool": False, "i32": -2**31}.get(d.name, -math.inf)

  @staticmethod
  def max(d: DType):
    return {"bool": True, "i32": 2**31 - 1}.get(d.name, math.inf)


def truncate(d: DType, x):
  """Bring a python value into the exact value set of dtype d (used by const folding)."""
  if d == dtypes.bool: return bool(x)
  if d == dtypes.i32: return ((int(x) + 2**31) % 2**32) - 2**31   # two's complement wraparound
  if d == dtypes.f32: return struct.unpack("f", struct.pack("f", float(x)))[0] if math.isfinite(x) else float(x)
  if d == dtypes.f16: return struct.unpack("e", struct.pack("e", float(x)))[0] if math.isfinite(x) and abs(x) < 65520 else float(x)
  raise TypeError(f"can't truncate to {d}")


class AddrSpace(Enum):
  """Where a UOp lives. ALU = a plain value in a register. The rest are memory."""
  ALU = auto()      # a value (week 1)
  GLOBAL = auto()   # device memory, a function argument (week 2)
  LOCAL = auto()    # call-scope storage: `float acc[4];` (week 2)
  SHARED = auto()   # GPU workgroup shared memory (week 7)

  def __repr__(self): return f"AddrSpace.{self.name}"
