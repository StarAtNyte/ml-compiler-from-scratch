"""Phi destruction, graph coloring, and a bounded-register interpreter with spills.

This teaching machine permits spill-slot operands directly. It counts their traffic,
not host CPU cycles. It is not a physical ISA encoder or a hardware cost model.
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from minicc.analysis import liveness, verify
from minicc.interpreter import bind, memory
from minicc.ir import Block, Instruction, Kind, Terminator, alu, operands


def destroy_phis(original):
  if not original.ssa: raise ValueError("phi destruction requires SSA")
  verify(original)
  function = deepcopy(original)
  counter = 0
  for parent, block in list(function.blocks.items()):
    targets = list(block.terminator.targets)
    for index, target in enumerate(targets):
      phis = function.blocks[target].phis
      if not phis: continue
      counter += 1
      edge = f"$edge{counter}"
      reads, writes = [], []
      # Save every source before writing any destination: parallel-copy semantics.
      for i, phi in enumerate(phis):
        temporary = f"$edge{counter}_copy{i}"
        reads.append(Instruction("copy", temporary, (phi.incoming[parent],)))
        writes.append(Instruction("copy", phi.dest, (temporary,)))
      function.blocks[edge] = Block(reads + writes, Terminator("jump", targets=(target,)))
      targets[index] = edge
    block.terminator.targets = tuple(targets)
  for block in function.blocks.values(): block.phis.clear()
  function.ssa = False
  verify(function)
  return function


def interference(function):
  if function.ssa or any(b.phis for b in function.blocks.values()): raise ValueError("destroy phis first")
  verify(function)
  _, outgoing = liveness(function)
  buffers = {p.name for p in function.parameters if p.kind == Kind.BUFFER}
  neighbors = {}
  def ensure(values):
    for value in values - buffers: neighbors.setdefault(value, set())
  def connect(a, b):
    if a == b or a in buffers or b in buffers: return
    ensure({a, b})
    neighbors[a].add(b); neighbors[b].add(a)
  parameters = {p.name for p in function.parameters if p.kind == Kind.I32}
  ensure(parameters)
  # Arguments are installed together before entry. Protect even unused parameters
  # from clobbering another parameter during that initial binding.
  for a, b in combinations(sorted(parameters), 2): connect(a, b)
  for name, block in function.blocks.items():
    live = (outgoing[name] | operands(block.terminator.args)) - buffers
    ensure(live)
    for instruction in reversed(block.instructions):
      if instruction.dest is not None:
        ensure({instruction.dest})
        live.discard(instruction.dest)
        for value in live: connect(instruction.dest, value)
      live |= operands(instruction.args) - buffers
      ensure(live)
  return neighbors


@dataclass(frozen=True)
class Location:
  kind: str
  index: int


@dataclass
class Allocation:
  function: object
  locations: dict[str, Location]
  registers: int
  slots: int
  neighbors: dict[str, set[str]]


def allocate(function, registers=4):
  if registers < 1: raise ValueError("at least one register is required")
  neighbors = interference(function)
  locations, spills = {}, 0
  # Deterministic largest-degree-first greedy coloring; not optimal coloring.
  for value in sorted(neighbors, key=lambda v: (-len(neighbors[v]), v)):
    unavailable = {locations[n].index for n in neighbors[value] if n in locations and locations[n].kind == "register"}
    available = next((r for r in range(registers) if r not in unavailable), None)
    if available is None:
      locations[value] = Location("slot", spills)
      spills += 1
    else: locations[value] = Location("register", available)
  allocation = Allocation(function, locations, registers, spills, neighbors)
  for value, adjacent in neighbors.items():
    for other in adjacent:
      if locations[value] == locations[other]: raise ValueError("interfering values share storage")
  return allocation


def execute_allocated(allocation, *arguments, max_steps=100000):
  function = allocation.function
  env = bind(function, arguments)
  buffers = {p.name: env[p.name] for p in function.parameters if p.kind == Kind.BUFFER}
  storage = {"register": [0]*allocation.registers, "slot": [0]*allocation.slots}
  counts = {"spill_reads": 0, "spill_writes": 0, "instructions": 0}
  def write(name, value):
    location = allocation.locations[name]
    storage[location.kind][location.index] = value
    if location.kind == "slot": counts["spill_writes"] += 1
  def read(value):
    if isinstance(value, int): return value
    if value in buffers: return buffers[value]
    location = allocation.locations[value]
    if location.kind == "slot": counts["spill_reads"] += 1
    return storage[location.kind][location.index]
  for parameter in function.parameters:
    if parameter.kind == Kind.I32: write(parameter.name, env[parameter.name])
  current = function.entry
  for _ in range(max_steps):
    block = function.blocks[current]
    for instruction in block.instructions:
      counts["instructions"] += 1
      args = [read(value) for value in instruction.args]
      result = memory(instruction.op, args) if instruction.op in ("load", "store") else alu(instruction.op, args)
      if instruction.dest is not None: write(instruction.dest, result)
    term = block.terminator
    if term.op == "return": return read(term.args[0]), counts
    current = term.targets[0] if term.op == "jump" else term.targets[0 if read(term.args[0]) else 1]
  raise RuntimeError("execution step limit exceeded")
