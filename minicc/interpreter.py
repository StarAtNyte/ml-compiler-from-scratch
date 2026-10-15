"""Execute mutable or SSA IR with ordered memory effects and simultaneous phi updates."""
from minicc.analysis import verify
from minicc.ir import Kind, alu, i32


def bind(function, arguments):
  if len(arguments) != len(function.parameters): raise ValueError("wrong argument count")
  result = {}
  for parameter, value in zip(function.parameters, arguments):
    if parameter.kind == Kind.I32:
      if not isinstance(value, int): raise ValueError("scalar parameters require Python integers")
      result[parameter.name] = i32(value)
    else:
      if not hasattr(value, "__len__") or not hasattr(value, "__getitem__") or not hasattr(value, "__setitem__"):
        raise ValueError("buffer parameter requires mutable indexed storage")
      result[parameter.name] = value
  return result


def memory(op, args):
  buffer, index = args[:2]
  if not 0 <= index < len(buffer): raise IndexError(f"buffer index {index} outside [0,{len(buffer)})")
  if op == "load": return i32(buffer[index])
  buffer[index] = i32(args[2])
  return None


def execute(function, *arguments, max_steps=100000):
  verify(function)
  env = bind(function, arguments)
  current, previous = function.entry, None
  def read(value): return value if isinstance(value, int) else env[value]
  for _ in range(max_steps):
    block = function.blocks[current]
    incoming = {phi.dest: read(phi.incoming[previous]) for phi in block.phis}
    env.update(incoming)
    for instruction in block.instructions:
      args = [read(arg) for arg in instruction.args]
      value = memory(instruction.op, args) if instruction.op in ("load", "store") else alu(instruction.op, args)
      if instruction.dest is not None: env[instruction.dest] = value
    term = block.terminator
    if term.op == "return": return read(term.args[0])
    target = term.targets[0] if term.op == "jump" else term.targets[0 if read(term.args[0]) else 1]
    previous, current = current, target
  raise RuntimeError("execution step limit exceeded")
