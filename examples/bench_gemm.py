"""Week 6: GEMM GFLOP/s, step by step, vs numpy (which calls an optimized BLAS).

  python examples/bench_gemm.py 1024

  B strided   C[i,j] = sum_k A[i,k] B[k,j]: the reduce walks down a column of B (stride N)
  B^T         same math, B stored transposed: both operands are contiguous along k
"""
import sys, time
import numpy as np
from uopc.uop import UOp, Ops, ParamArg
from uopc.dtype import dtypes
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.runtime import Program
from uopc.tensor import Tensor


def gemm_graph(N, b_transposed=False):
  A = UOp(Ops.BUFFER, (), ParamArg(0, dtypes.f32, (N * N,)))
  B = UOp(Ops.BUFFER, (), ParamArg(1, dtypes.f32, (N * N,)))
  a = A.reshape((N, 1, N)).expand((N, N, N))
  bt = B.reshape((N, N)) if b_transposed else B.reshape((N, N)).permute((1, 0))
  return (a * bt.reshape((1, N, N)).expand((N, N, N))).reduce(Ops.ADD, 1)


def bench(fn, reps=5):
  fn(); best = float("inf")
  for _ in range(reps):
    t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
  return best


if __name__ == "__main__":
  N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
  a, b = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
  bt = np.ascontiguousarray(b.T)
  c = np.zeros(N * N, np.float32)
  out = UOp(Ops.BUFFER, (), ParamArg(2, dtypes.f32, (N * N,)))
  flop = 2 * N ** 3
  print(f"N={N}  ({flop / 1e9:.2f} GFLOP per GEMM)")
  rows = [("B strided", False, "naive", Opts()),
          ("B strided", False, "threads", Opts(thread=0)),
          ("B strided", False, "4x4 tile + threads", Opts(upcast={0: 4, 1: 4}, thread=0)),
          ("B^T", True, "naive", Opts()),
          ("B^T", True, "threads", Opts(thread=0)),
          ("B^T", True, "4x4 tile", Opts(upcast={0: 4, 1: 4})),
          ("B^T", True, "4x4 tile + threads", Opts(upcast={0: 4, 1: 4}, thread=0))]
  for layout, transposed, name, o in rows:
    prg = Program(render_c(lower(gemm_graph(N, transposed), out, Ctx(opts=o)), "gemm"), "gemm")
    rhs = bt if transposed else b
    t = bench(lambda: prg(a, rhs, c))
    err = np.abs(c.reshape(N, N) - a @ b).max()
    print(f"  {layout:10s} {name:20s} {flop / t / 1e9:8.1f} GFLOP/s   max err {err:.1e}")
  A, B = Tensor(a), Tensor(b)
  t = bench(lambda: (A @ B).realize())
  print(f"  {'Tensor A@B':10s} {'(layout pass+opt)':20s} {flop / t / 1e9:8.1f} GFLOP/s   (end to end, includes the transpose copy)")
  t = bench(lambda: a @ b)
  print(f"  {'numpy':31s} {flop / t / 1e9:8.1f} GFLOP/s")
