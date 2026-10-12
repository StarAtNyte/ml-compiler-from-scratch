"""Week 6: three ways to compute attention, out = softmax(Q K^T / sqrt(D)) V.

  1. unfused     -- the Tensor code, kernelized normally: S = QK^T is materialized (N*N floats)
  2. fused       -- the same Tensor code with fuse_reduce=True: reduces nest inside each other,
                    nothing N*N ever touches memory; QK^T is recomputed by every pass (max, sum,
                    output) and, in the output pass, once per 16-wide tile of the head dimension
  3. online      -- FlashAttention's online softmax, written by hand in the week-3 loop IR:
                    one pass over keys, running max m, running sum l, rescaled accumulator o[D]

  python examples/flash_attention.py 1024 64
"""
import math, sys, time
import numpy as np
from uopc.uop import UOp, Ops, ParamArg, AxisType, sink, range_
from uopc.dtype import dtypes
from uopc.render_c import render_c
from uopc.runtime import Program
from uopc.rangeify import lower, Ctx, Opts
from uopc.tensor import Tensor

LOG2E = 1 / math.log(2)


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  return (q @ k.T * (1 / math.sqrt(q.shape[-1]))).softmax(-1) @ v


def online_attention_ir(N: int, D: int) -> UOp:
  """FlashAttention-style online softmax as a UOp loop program. One query row per thread."""
  f32 = dtypes.f32
  Q, K, V, OUT = (UOp(Ops.BUFFER, (), ParamArg(s, f32, (N * D,))) for s in range(4))
  alloc = lambda slot, n: UOp(Ops.ALLOC, (), ParamArg(slot, f32, (n,)))
  M, L, O, S = alloc(0, 1), alloc(1, 1), alloc(2, D), alloc(3, 1)
  scale = (1 / math.sqrt(D)) * LOG2E                     # work in base 2: exp(x) = exp2(x * log2 e)

  i = range_(N, 0, AxisType.THREAD)
  Mi, Li, Oi = M.after(i), L.after(i), O.after(i)
  init_m = Mi.index(0).store(UOp.const(-math.inf))
  init_l = Li.index(0).store(UOp.const(0.0))
  d0 = range_(D, 1)
  init_o = Oi.index(d0).store(UOp.const(0.0)).end(d0)

  j = range_(N, 2)
  # s = (q_i . k_j) * scale
  Sj = S.after(i, j)
  init_s = Sj.index(0).store(UOp.const(0.0))
  e = range_(D, 3)
  Se = Sj.after(init_s, e)
  dot = Se.index(0).store(Se.index(0).load() + Q.index(i * D + e).load() * K.index(j * D + e).load()).end(e)
  s = Sj.after(dot).index(0).load() * scale
  # online softmax update
  Mj, Lj = Mi.after(init_m, j), Li.after(init_l, j)
  m_old = Mj.index(0).load()
  m_new = m_old.maximum(s)
  corr = UOp(Ops.EXP2, (m_old - m_new,))                # rescale what we accumulated so far
  p = UOp(Ops.EXP2, (s - m_new,))
  st_l = Lj.index(0).store(Lj.index(0).load() * corr + p)
  d = range_(D, 4)
  Od = Oi.after(init_o, j)
  st_o = Od.index(d).store(Od.index(d).load() * corr + p * V.index(j * D + d).load()).end(d)
  st_m = Mj.index(0).store(m_new)
  loop_j = UOp(Ops.SINK, (st_l, st_o, st_m)).end(j)
  # normalize
  d2 = range_(D, 5)
  out = OUT.index(i * D + d2).store(Oi.after(loop_j).index(d2).load() * Li.after(loop_j).index(0).load().recip())
  return sink(out.end(d2, i))


class FlashOpt:
  """fuse_reduce kernels: unroll the head dim (up to 16) so each QK^T value is shared by 16 accumulators."""
  from uopc.opt import layout_pass as layout
  @staticmethod
  def lower(body, out):
    return lower(body, out, Ctx(opts=Opts(upcast={len(body.shape) - 1: min(body.shape[-1], 16)}, thread=0)))


def bench(fn, reps=3):
  fn(); best = math.inf
  for _ in range(reps):
    t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
  return best


if __name__ == "__main__":
  N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
  D = int(sys.argv[2]) if len(sys.argv) > 2 else 64
  rng = np.random.default_rng(0)
  q, k, v = (rng.standard_normal((N, D)).astype(np.float32) for _ in range(3))
  s = q @ k.T / math.sqrt(D)
  p = np.exp(s - s.max(-1, keepdims=True))
  ref = (p / p.sum(-1, keepdims=True)) @ v

  def run_tensor():
    return attention(Tensor(q), Tensor(k), Tensor(v)).numpy()
  t = bench(run_tensor)
  print(f"unfused  {t*1e3:8.2f} ms  err {np.abs(run_tensor() - ref).max():.1e}   (materializes {N*N*4/1e6:.1f} MB of scores)")

  Tensor.fuse_reduce, old = True, Tensor.opt
  Tensor.opt = FlashOpt
  t = bench(run_tensor)
  print(f"fused    {t*1e3:8.2f} ms  err {np.abs(run_tensor() - ref).max():.1e}   (no N*N buffer, QK^T recomputed)")
  Tensor.fuse_reduce, Tensor.opt = False, old

  prg = Program(render_c(online_attention_ir(N, D), "flash"), "flash")
  out = np.zeros(N * D, np.float32)
  t = bench(lambda: prg(q, k, v, out))
  print(f"online   {t*1e3:8.2f} ms  err {np.abs(out.reshape(N, D) - ref).max():.1e}   (one pass, running max/sum)")
