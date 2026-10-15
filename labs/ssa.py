"""Reachability, dominators, liveness, and SSA verification on a branching CFG.

Run: python -m labs.ssa
Phi operands are uses on predecessor edges, not ordinary uses in the merge block.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instruction:
  result: str | None
  args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Block:
  instructions: tuple[Instruction, ...]
  successors: tuple[str, ...] = ()
  # (result, ((predecessor, value), ...))
  phis: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()


def graph(blocks: dict[str, Block], entry: str):
  if entry not in blocks: raise ValueError("missing entry")
  predecessors = {b: set() for b in blocks}
  for name, block in blocks.items():
    for successor in block.successors:
      if successor not in blocks: raise ValueError(f"unknown successor: {successor}")
      predecessors[successor].add(name)
  reachable, todo = set(), [entry]
  while todo:
    name = todo.pop()
    if name in reachable: continue
    reachable.add(name)
    todo.extend(blocks[name].successors)
  return predecessors, reachable


def dominators(blocks: dict[str, Block], entry: str):
  predecessors, reachable = graph(blocks, entry)
  dom = {b: ({b} if b == entry else set(reachable)) for b in reachable}
  changed = True
  while changed:
    changed = False
    for b in sorted(reachable - {entry}):
      incoming = [dom[p] for p in predecessors[b] & reachable]
      value = {b} | set.intersection(*incoming)
      if value != dom[b]: dom[b], changed = value, True
  return dom


def verify(blocks: dict[str, Block], entry: str, parameters: tuple[str, ...] = ()):
  predecessors, reachable = graph(blocks, entry)
  dom = dominators(blocks, entry)
  definitions = {}
  for b in reachable:
    for name, incoming in blocks[b].phis:
      if len(incoming) != len(dict(incoming)) or set(dict(incoming)) != predecessors[b]:
        raise ValueError(f"phi {name}: incoming edges do not match predecessors")
      if name in parameters or name in definitions: raise ValueError(f"duplicate definition: {name}")
      definitions[name] = (b, -1)
    for i, instruction in enumerate(blocks[b].instructions):
      if (name := instruction.result) is not None:
        if name in parameters or name in definitions: raise ValueError(f"duplicate definition: {name}")
        definitions[name] = (b, i)

  def use(value, block, position):
    if value in parameters: return
    if value not in definitions: raise ValueError(f"undefined value: {value}")
    definition_block, definition_position = definitions[value]
    if definition_block not in dom[block] or (definition_block == block and definition_position >= position):
      raise ValueError(f"definition of {value} does not dominate its use in {block}")

  for b in reachable:
    for _, incoming in blocks[b].phis:
      for predecessor, value in incoming:
        if predecessor in reachable: use(value, predecessor, len(blocks[predecessor].instructions))
    for i, instruction in enumerate(blocks[b].instructions):
      for value in instruction.args: use(value, b, i)


def liveness(blocks: dict[str, Block], entry: str):
  _, reachable = graph(blocks, entry)
  uses, definitions = {}, {}
  for b in reachable:
    defined = {name for name, _ in blocks[b].phis}
    used = set()
    for instruction in blocks[b].instructions:
      used.update(set(instruction.args) - defined)
      if instruction.result is not None: defined.add(instruction.result)
    uses[b], definitions[b] = used, defined
  live_in, live_out = {b: set() for b in reachable}, {b: set() for b in reachable}
  changed = True
  while changed:
    changed = False
    for b in sorted(reachable, reverse=True):
      outgoing = set()
      for successor in blocks[b].successors:
        phi_defs = {name for name, _ in blocks[successor].phis}
        edge_uses = {dict(incoming)[b] for _, incoming in blocks[successor].phis}
        outgoing |= (live_in[successor] - phi_defs) | edge_uses
      incoming = uses[b] | (outgoing - definitions[b])
      if incoming != live_in[b] or outgoing != live_out[b]:
        live_in[b], live_out[b], changed = incoming, outgoing, True
  return live_in, live_out


def diamond():
  return {
    "entry": Block((Instruction(None, ("p",)),), ("left", "right")),
    "left": Block((Instruction("a", ("x",)),), ("merge",)),
    "right": Block((Instruction("b", ("x",)),), ("merge",)),
    "merge": Block((Instruction("y", ("z",)), Instruction(None, ("y",))),
                   phis=(("z", (("left", "a"), ("right", "b"))),)),
  }


def main():
  blocks = diamond()
  verify(blocks, "entry", ("p", "x"))
  for name, values in sorted(dominators(blocks, "entry").items()):
    print(f"Dom({name}) = {','.join(sorted(values))}")
  incoming, outgoing = liveness(blocks, "entry")
  for name in sorted(blocks): print(f"{name}: IN={sorted(incoming[name])}, OUT={sorted(outgoing[name])}")


if __name__ == "__main__": main()
