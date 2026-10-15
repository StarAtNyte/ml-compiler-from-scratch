"""Render verified SSA as bounds-checked C and execute it through a declared ABI."""
import ctypes
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from minicc.analysis import verify
from minicc.ir import Kind


def render_c(function):
  if not function.ssa: raise ValueError("C backend expects SSA")
  verify(function)
  names = {}
  def symbol(name):
    if name not in names: names[name] = f"v{len(names)}"
    return names[name]
  def operand(value):
    if isinstance(value, int): return f"INT32_C({value})" if value != -(2**31) else "(-INT32_C(2147483647)-1)"
    return symbol(value)
  arguments = []
  parameters = {p.name for p in function.parameters}
  for parameter in function.parameters:
    name = symbol(parameter.name)
    arguments.append(f"int32_t *{name}" if parameter.kind == Kind.BUFFER else f"int32_t {name}")
    if parameter.kind == Kind.BUFFER: arguments.append(f"int32_t {name}_length")
  arguments.append("int32_t *error")
  labels = {name: f"block_{i}" for i, name in enumerate(function.blocks)}
  for block in function.blocks.values():
    for phi in block.phis: symbol(phi.dest)
    for instruction in block.instructions:
      if instruction.dest is not None: symbol(instruction.dest)
  lines = ["#include <stdint.h>", "#include <string.h>",
           # Bit-preserving conversion avoids signed-overflow UB and avoids
           # relying on an out-of-range unsigned-to-signed C cast.
           "static int32_t signed_bits(uint32_t x) { int32_t y; memcpy(&y, &x, sizeof(y)); return y; }",
           f"int32_t compiled({', '.join(arguments)}) {{", "  *error = 0;"]
  for name, c_name in names.items():
    if name not in parameters: lines.append(f"  int32_t {c_name};")
  lines.append(f"  goto {labels[function.entry]};")
  def edge(parent, target, indent):
    lines.append(indent + "{")
    phis = function.blocks[target].phis
    for i, phi in enumerate(phis): lines.append(indent + f"  int32_t edge_{i} = {operand(phi.incoming[parent])};")
    for i, phi in enumerate(phis): lines.append(indent + f"  {symbol(phi.dest)} = edge_{i};")
    lines.append(indent + f"  goto {labels[target]};")
    lines.append(indent + "}")
  comparisons = {"lt": "<", "le": "<=", "eq": "==", "ne": "!="}
  arithmetic = {"add": "+", "sub": "-", "mul": "*"}
  for name, block in function.blocks.items():
    lines.append(labels[name] + ": ;")
    for instruction in block.instructions:
      args = [operand(arg) for arg in instruction.args]
      if instruction.op in ("load", "store"):
        base, index = args[:2]
        lines.append(f"  if ({index} < 0 || {index} >= {base}_length) {{ *error = 1; return 0; }}")
        statement = f"{base}[{index}] = {args[2]}" if instruction.op == "store" else f"{symbol(instruction.dest)} = {base}[{index}]"
      else:
        if instruction.op == "copy": expression = args[0]
        elif instruction.op in comparisons: expression = f"({args[0]} {comparisons[instruction.op]} {args[1]})"
        else: expression = f"signed_bits((uint32_t){args[0]} {arithmetic[instruction.op]} (uint32_t){args[1]})"
        statement = f"{symbol(instruction.dest)} = {expression}"
      lines.append(f"  {statement};")
    term = block.terminator
    if term.op == "return": lines.append(f"  return {operand(term.args[0])};")
    elif term.op == "jump": edge(name, term.targets[0], "  ")
    else:
      lines.append(f"  if ({operand(term.args[0])})")
      edge(name, term.targets[0], "  ")
      lines.append("  else")
      edge(name, term.targets[1], "  ")
  lines.append("}")
  return "\n".join(lines)


class NativeFunction:
  def __init__(self, function):
    compiler = shutil.which("gcc")
    if compiler is None: raise RuntimeError("gcc is required for native execution")
    self.function, self.source = function, render_c(function)
    self.directory = TemporaryDirectory(prefix="minicc-")
    source, library = Path(self.directory.name)/"kernel.c", Path(self.directory.name)/"kernel.so"
    source.write_text(self.source)
    subprocess.run([compiler, "-std=c11", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    self.library = ctypes.CDLL(str(library))
    self.call = self.library.compiled
    types = []
    for parameter in function.parameters:
      types.extend([ctypes.POINTER(ctypes.c_int32), ctypes.c_int32] if parameter.kind == Kind.BUFFER else [ctypes.c_int32])
    self.call.argtypes, self.call.restype = [*types, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int32

  def __call__(self, *arguments):
    if len(arguments) != len(self.function.parameters): raise ValueError("wrong argument count")
    native = []
    for parameter, value in zip(self.function.parameters, arguments):
      if parameter.kind == Kind.I32:
        if not isinstance(value, int): raise ValueError("scalar parameters require Python integers")
        native.append(ctypes.c_int32(value))
      else:
        if not isinstance(value, np.ndarray) or value.dtype != np.int32 or value.ndim != 1 or not value.flags.c_contiguous or not value.flags.aligned or not value.flags.writeable:
          raise ValueError("native buffers require aligned, writable, contiguous rank-one int32 arrays")
        if len(value) >= 2**31: raise ValueError("buffer exceeds i32 length domain")
        native.extend([value.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32(len(value))])
    error = ctypes.c_int32()
    result = self.call(*native, ctypes.byref(error))
    if error.value: raise IndexError("compiled buffer access out of bounds")
    return result
