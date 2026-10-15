"""Executable scalar compiler: source -> CFG -> SSA -> optimization -> C / register VM."""
from minicc.frontend import lower
from minicc.ssa import to_ssa
from minicc.optimize import optimize
from minicc.interpreter import execute
from minicc.c_backend import NativeFunction


def compile_ir(source, optimized=True):
  function = to_ssa(lower(source))
  return optimize(function) if optimized else function


def compile_native(source, optimized=True): return NativeFunction(compile_ir(source, optimized))
