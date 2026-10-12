"""uopc: a machine learning compiler built from scratch, one week at a time.

  week 1   uop, rewrite, symbolic, uir      the IR, rewriting, algebra, the text format
  week 2   render_c, runtime, linearize     memory, C code, compiling and running it
  week 3   linearize                        loops and scopes
  week 4   rangeify, composite              shapes -> loops, PAD/FLIP from primitives
  week 5   kernelize, autodiff, tensor, nn  fusion/fission, gradients, a Tensor library
  week 6   opt                              upcast, unroll, threads, layout
  week 7-8 gpu, cuda                        GPU mapping, CUDA, shared memory, tensor cores
"""
from uopc.uop import UOp, Ops, AxisType, ParamArg, CallArg, RangeArg, sink, range_
from uopc.dtype import dtypes, DType, AddrSpace
from uopc.tensor import Tensor
