"""Small control-flow IR with explicit effects and signed, wrapping i32 arithmetic."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

Operand = str | int


class Kind(Enum):
  I32 = "i32"
  BUFFER = "buffer"


@dataclass(frozen=True)
class Parameter:
  name: str
  kind: Kind = Kind.I32


@dataclass
class Instruction:
  op: str
  dest: str | None
  args: tuple[Operand, ...]


@dataclass
class Phi:
  base: str
  dest: str
  incoming: dict[str, Operand] = field(default_factory=dict)


@dataclass
class Terminator:
  op: str
  args: tuple[Operand, ...] = ()
  targets: tuple[str, ...] = ()


@dataclass
class Block:
  instructions: list[Instruction] = field(default_factory=list)
  terminator: Terminator | None = None
  phis: list[Phi] = field(default_factory=list)


@dataclass
class Function:
  name: str
  parameters: tuple[Parameter, ...]
  blocks: dict[str, Block]
  entry: str = "entry"
  ssa: bool = False


PURE = {"copy": 1, "add": 2, "sub": 2, "mul": 2, "lt": 2, "le": 2, "eq": 2, "ne": 2}
EFFECTS = {"load": 2, "store": 3}  # Even an unused load can raise a bounds error.


def i32(value: int) -> int:
  value = int(value) & 0xffffffff
  return value - 0x100000000 if value & 0x80000000 else value


def alu(op: str, args) -> int:
  if op == "copy": return i32(args[0])
  a, b = args
  if op == "add": return i32(a + b)
  if op == "sub": return i32(a - b)
  if op == "mul": return i32(a * b)
  if op == "lt": return int(a < b)
  if op == "le": return int(a <= b)
  if op == "eq": return int(a == b)
  if op == "ne": return int(a != b)
  raise ValueError(f"unknown arithmetic operation: {op}")


def successors(function, name):
  term = function.blocks[name].terminator
  if term is None: raise ValueError(f"unterminated block: {name}")
  return term.targets


def graph(function):
  if function.entry not in function.blocks: raise ValueError("missing entry")
  predecessors = {name: set() for name in function.blocks}
  for name in function.blocks:
    for target in successors(function, name):
      if target not in function.blocks: raise ValueError(f"unknown block: {target}")
      predecessors[target].add(name)
  reachable, todo = set(), [function.entry]
  while todo:
    name = todo.pop()
    if name in reachable: continue
    reachable.add(name)
    todo.extend(successors(function, name))
  return predecessors, reachable


def prune(function):
  _, reachable = graph(function)
  function.blocks = {name: block for name, block in function.blocks.items() if name in reachable}
  predecessors, _ = graph(function)
  for name, block in function.blocks.items():
    for phi in block.phis:
      phi.incoming = {p: value for p, value in phi.incoming.items() if p in predecessors[name]}
  return function


def operands(args): return {arg for arg in args if isinstance(arg, str)}


def format_ir(function):
  params = ", ".join(f"{p.name}: {p.kind.value}" for p in function.parameters)
  lines = [f"{'ssa ' if function.ssa else ''}func {function.name}({params}):"]
  for name, block in function.blocks.items():
    lines.append(f"{name}:")
    for phi in block.phis:
      incoming = ", ".join(f"{p}: {v}" for p, v in sorted(phi.incoming.items()))
      lines.append(f"  {phi.dest} = phi [{incoming}]")
    for instruction in block.instructions:
      lhs = f"{instruction.dest} = " if instruction.dest else ""
      lines.append(f"  {lhs}{instruction.op} {', '.join(map(str, instruction.args))}")
    term = block.terminator
    if term is not None: lines.append(f"  {term.op} {', '.join(map(str, (*term.args, *term.targets)))}")
  return "\n".join(lines)
