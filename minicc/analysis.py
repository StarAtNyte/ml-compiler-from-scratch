"""Fixed-point dominance/liveness, definite assignment, and IR/SSA verification."""
from minicc.ir import EFFECTS, PURE, Kind, graph, operands, successors


def dominators(function):
  predecessors, reachable = graph(function)
  entry = function.entry
  result = {b: ({b} if b == entry else set(reachable)) for b in reachable}
  while True:
    changed = False
    for b in sorted(reachable - {entry}):
      value = {b} | set.intersection(*(result[p] for p in predecessors[b] & reachable))
      if value != result[b]: result[b], changed = value, True
    if not changed: return result


def dominance_tree(function):
  dom = dominators(function)
  immediate = {b: max(dom[b] - {b}, key=lambda d: len(dom[d])) for b in dom if b != function.entry}
  children = {b: [] for b in dom}
  for b, parent in immediate.items(): children[parent].append(b)
  for values in children.values(): values.sort()
  return immediate, children


def dominance_frontiers(function):
  predecessors, reachable = graph(function)
  immediate, _ = dominance_tree(function)
  result = {b: set() for b in reachable}
  for b in reachable:
    if len(predecessors[b] & reachable) < 2: continue
    for parent in predecessors[b] & reachable:
      runner = parent
      while runner != immediate.get(b):
        result[runner].add(b)
        if runner not in immediate: break
        runner = immediate[runner]
  return result


def liveness(function):
  _, reachable = graph(function)
  uses, defs = {}, {}
  for name in reachable:
    block = function.blocks[name]
    defined, used = {p.dest for p in block.phis}, set()
    for instruction in block.instructions:
      used |= operands(instruction.args) - defined
      if instruction.dest is not None: defined.add(instruction.dest)
    used |= operands(block.terminator.args) - defined
    uses[name], defs[name] = used, defined
  incoming, outgoing = {b: set() for b in reachable}, {b: set() for b in reachable}
  while True:
    changed = False
    for b in sorted(reachable, reverse=True):
      live = set()
      for target in successors(function, b):
        block = function.blocks[target]
        live |= (incoming[target] - {p.dest for p in block.phis}) | operands(p.incoming[b] for p in block.phis)
      value = uses[b] | (live - defs[b])
      if value != incoming[b] or live != outgoing[b]: incoming[b], outgoing[b], changed = value, live, True
    if not changed: return incoming, outgoing


def definite_assignment(function):
  predecessors, reachable = graph(function)
  parameters = {p.name for p in function.parameters}
  definitions = {b: {i.dest for i in function.blocks[b].instructions if i.dest is not None} for b in reachable}
  universe = parameters | set.union(set(), *definitions.values())
  incoming = {b: (set(parameters) if b == function.entry else set(universe)) for b in reachable}
  outgoing = {b: incoming[b] | definitions[b] for b in reachable}
  while True:
    changed = False
    for b in sorted(reachable):
      value = set(parameters) if b == function.entry else set.intersection(*(outgoing[p] for p in predecessors[b] & reachable))
      after = value | definitions[b]
      if value != incoming[b] or after != outgoing[b]: incoming[b], outgoing[b], changed = value, after, True
    if not changed: break
  for name in reachable:
    defined = set(incoming[name])
    block = function.blocks[name]
    for instruction in block.instructions:
      missing = operands(instruction.args) - defined
      if missing: raise ValueError(f"possibly uninitialized in {name}: {sorted(missing)}")
      if instruction.dest is not None: defined.add(instruction.dest)
    missing = operands(block.terminator.args) - defined
    if missing: raise ValueError(f"possibly uninitialized in {name}: {sorted(missing)}")


def verify(function):
  predecessors, reachable = graph(function)
  if reachable != set(function.blocks): raise ValueError("prune unreachable blocks before verification")
  if predecessors[function.entry]: raise ValueError("entry must not have predecessors")
  parameters = {p.name: p.kind for p in function.parameters}
  if len(parameters) != len(function.parameters): raise ValueError("duplicate parameter")
  definitions = {p.name: (function.entry, -2) for p in function.parameters}
  buffers = {name for name, kind in parameters.items() if kind == Kind.BUFFER}
  for name, block in function.blocks.items():
    if block.phis and not function.ssa: raise ValueError("phis require SSA form")
    for phi in block.phis:
      if set(phi.incoming) != predecessors[name]: raise ValueError(f"phi edge mismatch in {name}")
    for position, instruction in enumerate(block.instructions):
      if any(type(arg) not in (int, str) for arg in instruction.args): raise ValueError("operands must be i32 literals or value names")
      if any(isinstance(arg, int) and not -(2**31) <= arg < 2**31 for arg in instruction.args): raise ValueError("literal outside i32 domain")
      arity = (PURE | EFFECTS).get(instruction.op)
      if arity is None or len(instruction.args) != arity: raise ValueError(f"invalid instruction: {instruction}")
      if (instruction.dest is None) != (instruction.op == "store"): raise ValueError("invalid instruction result")
      if instruction.dest in buffers: raise ValueError("cannot redefine buffer parameter")
      if instruction.op in EFFECTS:
        if instruction.args[0] not in buffers: raise ValueError("memory operation requires a buffer")
        scalar_args = instruction.args[1:]
      else: scalar_args = instruction.args
      if operands(scalar_args) & buffers: raise ValueError("buffer used as scalar")
    term = block.terminator
    if term.op not in ("return", "branch", "jump"): raise ValueError("invalid terminator")
    if len(term.args) != (0 if term.op == "jump" else 1) or len(term.targets) != {"return": 0, "branch": 2, "jump": 1}[term.op]:
      raise ValueError("invalid terminator arity")
    if any(type(arg) not in (int, str) for arg in term.args): raise ValueError("invalid terminator operand")
    if any(isinstance(arg, int) and not -(2**31) <= arg < 2**31 for arg in term.args): raise ValueError("literal outside i32 domain")
    if operands(term.args) & buffers: raise ValueError("buffer used as scalar condition or return")
    if function.ssa:
      for position, dest in [(-1, p.dest) for p in block.phis] + [(i, instruction.dest) for i, instruction in enumerate(block.instructions) if instruction.dest is not None]:
        if dest in definitions: raise ValueError(f"duplicate SSA definition: {dest}")
        definitions[dest] = (name, position)
  if not function.ssa:
    definite_assignment(function)
    return
  dom = dominators(function)
  def check(args, block, position):
    for arg in args:
      if isinstance(arg, int):
        if not -(2**31) <= arg < 2**31: raise ValueError("literal outside i32 domain")
        continue
      if arg not in definitions: raise ValueError(f"undefined SSA value: {arg}")
      origin, index = definitions[arg]
      if origin not in dom[block] or (origin == block and index >= position):
        raise ValueError(f"{arg} does not dominate use in {block}")
  for name, block in function.blocks.items():
    for phi in block.phis:
      if operands(phi.incoming.values()) & buffers: raise ValueError("buffer phi unsupported")
      for parent, value in phi.incoming.items(): check((value,), parent, len(function.blocks[parent].instructions))
    for position, instruction in enumerate(block.instructions): check(instruction.args, name, position)
    check(block.terminator.args, name, len(block.instructions))
