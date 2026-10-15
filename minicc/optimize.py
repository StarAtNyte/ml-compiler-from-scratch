"""Sparse conditional constant propagation and effect-preserving dead code removal.

The worklist is expressed as a fixed-point scan to keep the lattice mechanics visible.
Values move only UNDEFINED -> constant -> OVERDEFINED; executable edges only grow.
"""
from copy import deepcopy
from enum import Enum
from minicc.analysis import verify
from minicc.ir import EFFECTS, PURE, Terminator, alu, operands, prune


class State(Enum):
  UNDEFINED = 0
  OVERDEFINED = 1


def join(a, b):
  if a == State.UNDEFINED: return b
  if b == State.UNDEFINED or a == b: return a
  return State.OVERDEFINED


def sccp(original):
  if not original.ssa: raise ValueError("SCCP requires SSA")
  verify(original)
  function = deepcopy(original)
  values = {p.name: State.OVERDEFINED for p in function.parameters}
  executable, edges = {function.entry}, set()

  def value(arg): return arg if isinstance(arg, int) else values.get(arg, State.UNDEFINED)

  def update(dest, result):
    nonlocal changed
    combined = join(values.get(dest, State.UNDEFINED), result)
    if combined != values.get(dest, State.UNDEFINED):
      values[dest], changed = combined, True

  while True:
    changed = False
    for name, block in function.blocks.items():
      if name not in executable: continue
      for phi in block.phis:
        result = State.UNDEFINED
        for parent, arg in phi.incoming.items():
          if (parent, name) in edges: result = join(result, value(arg))
        update(phi.dest, result)
      for instruction in block.instructions:
        if instruction.dest is None: continue
        args = [value(arg) for arg in instruction.args]
        if instruction.op in EFFECTS or State.OVERDEFINED in args: result = State.OVERDEFINED
        elif State.UNDEFINED in args: result = State.UNDEFINED
        else: result = alu(instruction.op, args)
        update(instruction.dest, result)
      term = block.terminator
      if term.op == "jump": targets = term.targets
      elif term.op == "return": targets = ()
      else:
        condition = value(term.args[0])
        if condition == State.UNDEFINED: targets = ()
        elif condition == State.OVERDEFINED: targets = term.targets
        else: targets = (term.targets[0 if condition else 1],)
      for target in targets:
        if (name, target) not in edges: edges.add((name, target)); changed = True
        if target not in executable: executable.add(target); changed = True
    if not changed: break

  def replace(arg):
    candidate = value(arg)
    return candidate if isinstance(candidate, int) else arg

  for name, block in function.blocks.items():
    for phi in block.phis: phi.incoming = {p: replace(arg) for p, arg in phi.incoming.items()}
    for instruction in block.instructions:
      instruction.args = tuple(replace(arg) for arg in instruction.args)
      if instruction.op in PURE and isinstance(value(instruction.dest), int):
        instruction.op, instruction.args = "copy", (value(instruction.dest),)
    term = block.terminator
    term.args = tuple(replace(arg) for arg in term.args)
    if term.op == "branch" and isinstance(term.args[0], int):
      block.terminator = Terminator("jump", targets=(term.targets[0 if term.args[0] else 1],))
  prune(function)
  verify(function)
  return function


def eliminate_dead_code(original):
  if not original.ssa: raise ValueError("dead-code removal requires SSA")
  function = deepcopy(original)
  definitions = {}
  roots = set()
  for block in function.blocks.values():
    roots |= operands(block.terminator.args)
    for phi in block.phis: definitions[phi.dest] = operands(phi.incoming.values())
    for instruction in block.instructions:
      if instruction.dest is not None: definitions[instruction.dest] = operands(instruction.args)
      if instruction.op in EFFECTS: roots |= operands(instruction.args)
  live, pending = set(), list(roots)
  while pending:
    value = pending.pop()
    if value in live: continue
    live.add(value)
    pending.extend(definitions.get(value, ()))
  for block in function.blocks.values():
    block.phis = [phi for phi in block.phis if phi.dest in live]
    block.instructions = [i for i in block.instructions if i.op in EFFECTS or i.dest in live]
  verify(function)
  return function


def optimize(function): return eliminate_dead_code(sccp(function))
