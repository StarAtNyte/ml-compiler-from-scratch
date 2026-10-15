"""Guarded dynamic-shape runtime, aligned memory planning, and native execution."""
import ctypes
import math
import shutil
import subprocess
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from labs.memory import Temporary, plan_memory, verify_plan
from tensorjit.compiler import compile_plan
from tensorjit.graph import Dim


class CompiledGraph:
  def __init__(self, outputs, *, fuse=True, max_bytes=256*1024*1024):
    if type(max_bytes) is not int or max_bytes < 1: raise ValueError("max_bytes must be a positive integer")
    self.plan = compile_plan(outputs, fuse=fuse)
    self.max_bytes = max_bytes
    self.directory = TemporaryDirectory(prefix="tensorjit-")
    self.source = self.plan.source
    path = Path(self.directory.name)
    (path/"kernels.c").write_text(self.source)
    compiler = shutil.which("gcc")
    if compiler is None: raise RuntimeError("gcc is required")
    subprocess.run([compiler, "-std=c11", "-shared", "-fPIC", "-O2", "-ffp-contract=off", str(path/"kernels.c"), "-lm", "-o", str(path/"kernels.so")],
                   check=True, capture_output=True, text=True)
    self.library = ctypes.CDLL(str(path/"kernels.so"))
    self.functions = []
    pointer = ctypes.POINTER(ctypes.c_float)
    for i, kernel in enumerate(self.plan.kernels):
      function = getattr(self.library, f"kernel_{i}")
      function.argtypes = [pointer] * (len(kernel.dependencies)+1) + [ctypes.c_int32]*len(self.plan.symbols)
      function.restype = None
      self.functions.append(function)
    self.compilations, self.calls = 1, 0
    self.last_memory = {}
    # Native calls release the GIL. Serialize calls so observable counters describe
    # one completed invocation; the actual arenas are already per-invocation.
    self.lock = threading.Lock()

  def bind(self, arguments, explicit=None):
    expected = {node.arg for node in self.plan.inputs}
    if set(arguments) != expected: raise ValueError(f"expected inputs {sorted(expected)}, received {sorted(arguments)}")
    bindings = dict(explicit or {})
    symbol_table = {dim.name: dim for dim in self.plan.symbols}
    if set(bindings) - set(symbol_table): raise ValueError("unknown explicit dimension")
    for name, value in bindings.items():
      dim = symbol_table[name]
      if type(value) is not int or not dim.lower <= value <= dim.upper:
        raise ValueError(f"{name}: invalid explicit dimension binding")
    total_input_bytes = 0
    for node in self.plan.inputs:
      array = arguments[node.arg]
      if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.ndim != len(node.shape) or not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{node.arg}: expected aligned contiguous float32 with rank {len(node.shape)}")
      total_input_bytes += array.nbytes
      if total_input_bytes > self.max_bytes: raise ValueError("input bytes exceed configured budget")
      for expected_dim, actual in zip(node.shape, array.shape):
        if isinstance(expected_dim, Dim):
          if not expected_dim.lower <= actual <= expected_dim.upper: raise ValueError(f"{expected_dim.name}: dimension outside bounds")
          if expected_dim.name in bindings and bindings[expected_dim.name] != actual: raise ValueError(f"{expected_dim.name}: inconsistent dimension binding")
          bindings[expected_dim.name] = actual
        elif expected_dim != actual: raise ValueError("static dimension mismatch")
    # Output symbols must all be grounded in input metadata, never guessed.
    if set(bindings) != {dim.name for dim in self.plan.symbols}: raise ValueError("unbound output dimension")
    return bindings

  def __call__(self, **arguments):
    with self.lock: return self._run(arguments)

  def run_with_bindings(self, bindings, **arguments):
    """Supply metadata for symbols absent from all data inputs (e.g. constant gradients)."""
    with self.lock: return self._run(arguments, bindings)

  def _run(self, arguments, explicit=None):
    bindings = self.bind(arguments, explicit)
    def shape(node): return tuple(bindings[d.name] if isinstance(d, Dim) else d for d in node.shape)
    def nbytes(node):
      size = 4*math.prod(shape(node))
      if size > min(self.max_bytes, 2**63-1): raise ValueError("tensor allocation exceeds configured budget")
      return size
    # Derive liveness from the actual fused schedule, not original graph edges.
    index = {kernel.node: i for i, kernel in enumerate(self.plan.kernels)}
    retained = set(self.plan.outputs)
    temporaries = []
    for kernel in self.plan.kernels:
      node = kernel.node
      if node in retained: continue
      last = max([index[node]] + [i for i, consumer in enumerate(self.plan.kernels) if node in consumer.dependencies])
      temporaries.append(Temporary(self.plan.names[node], nbytes(node), index[node], last))
    allocations, peak = plan_memory(temporaries)
    verify_plan(temporaries, allocations)
    output_bytes = sum(nbytes(node) for node in self.plan.outputs)
    input_bytes = sum(array.nbytes for array in arguments.values())
    if input_bytes+peak+output_bytes > self.max_bytes: raise ValueError("inputs, workspace, and outputs exceed configured budget")
    arena = np.empty(peak+63, dtype=np.uint8)
    shift = (-arena.ctypes.data) % 64
    arena = arena[shift:shift+peak]
    arrays = {node: arguments[node.arg] for node in self.plan.inputs}
    outputs = {node: np.empty(shape(node), dtype=np.float32) for node in self.plan.outputs}
    for node, array in outputs.items():
      if node.op != "input": arrays[node] = array
    for kernel in self.plan.kernels:
      if kernel.node not in retained:
        allocation = allocations[self.plan.names[kernel.node]]
        arrays[kernel.node] = np.ndarray(shape(kernel.node), dtype=np.float32, buffer=arena, offset=allocation.offset)
    ptr = ctypes.POINTER(ctypes.c_float)
    dimensions = [bindings[d.name] for d in self.plan.symbols]
    for kernel, function in zip(self.plan.kernels, self.functions):
      # An input requested as output is copied, preserving caller ownership.
      destination = outputs[kernel.node] if kernel.node in retained else arrays[kernel.node]
      function(*(arrays[node].ctypes.data_as(ptr) for node in kernel.dependencies), destination.ctypes.data_as(ptr), *dimensions)
    self.calls += 1
    self.last_memory = {"temporary_bytes": peak, "unreused_temporary_bytes": sum(t.nbytes for t in temporaries),
                        "output_bytes": output_bytes, "input_bytes": input_bytes, "allocations": allocations, "bindings": bindings,
                        "kernel_count": len(self.plan.kernels)}
    return tuple(outputs[node] for node in self.plan.outputs)
