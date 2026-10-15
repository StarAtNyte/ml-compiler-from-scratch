"""Pruned iterated-dominance-frontier phi placement and dominator-tree renaming."""
from collections import defaultdict
from copy import deepcopy
from minicc.analysis import dominance_frontiers, dominance_tree, liveness, verify
from minicc.ir import Phi


def to_ssa(original):
  if original.ssa: raise ValueError("already in SSA")
  function = deepcopy(original)
  verify(function)
  frontiers = dominance_frontiers(function)
  _, children = dominance_tree(function)
  live_in, _ = liveness(function)
  defsites = defaultdict(set)
  for parameter in function.parameters: defsites[parameter.name].add(function.entry)
  for name, block in function.blocks.items():
    for instruction in block.instructions:
      if instruction.dest is not None: defsites[instruction.dest].add(name)
  for variable, sites in sorted(defsites.items()):
    pending, placed = sorted(sites), set()
    while pending:
      origin = pending.pop()
      for target in sorted(frontiers[origin]):
        if target in placed or variable not in live_in[target]: continue
        function.blocks[target].phis.append(Phi(variable, variable))
        placed.add(target)
        if target not in sites: pending.append(target)
  stacks = defaultdict(list)
  counters = defaultdict(int)
  for parameter in function.parameters: stacks[parameter.name].append(parameter.name)

  def read(operand):
    if isinstance(operand, int): return operand
    if not stacks[operand]: raise ValueError(f"uninitialized during rename: {operand}")
    return stacks[operand][-1]

  def define(variable, pushed):
    counters[variable] += 1
    name = f"{variable}.{counters[variable]}"
    stacks[variable].append(name)
    pushed.append(variable)
    return name

  def rename(name):
    block, pushed = function.blocks[name], []
    for phi in block.phis: phi.dest = define(phi.base, pushed)
    for instruction in block.instructions:
      instruction.args = tuple(read(arg) for arg in instruction.args)
      if instruction.dest is not None: instruction.dest = define(instruction.dest, pushed)
    block.terminator.args = tuple(read(arg) for arg in block.terminator.args)
    for target in block.terminator.targets:
      for phi in function.blocks[target].phis: phi.incoming[name] = read(phi.base)
    for child in children[name]: rename(child)
    for variable in reversed(pushed): stacks[variable].pop()

  rename(function.entry)
  function.ssa = True
  verify(function)
  return function
