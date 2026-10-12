"""Week 6: upcast, unroll, threads, layout. Optimized kernels must compute the same thing."""
import numpy as np, pytest
from uopc.uop import Ops
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.runtime import Program
from uopc.opt import hand_opt_choice, layout_pass
from helpers import buf

def gemm_graph(M, N, K, bt=False):
  A, B = buf(0, M * K), buf(1, K * N)
  b = B.reshape((N, K)) if bt else B.reshape((K, N)).permute((1, 0))
  return (A.reshape((M, 1, K)).expand((M, N, K)) * b.reshape((1, N, K)).expand((M, N, K))).reduce(Ops.ADD, 1)

@pytest.mark.parametrize("opts", [
  Opts(), Opts(thread=0), Opts(upcast={1: 4}), Opts(upcast={0: 4, 1: 4}, thread=0),
  Opts(unroll=4), Opts(upcast={0: 2, 1: 8}, unroll=4, thread=1), Opts(upcast={0: 32})])
def test_gemm_opts_correct(opts):
  M, N, K = 32, 48, 64
  a, b = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32)
  c = np.zeros(M * N, np.float32)
  src = render_c(lower(gemm_graph(M, N, K), buf(2, M * N), Ctx(opts=opts)), "k")
  Program(src, "k")(a, b, c)
  np.testing.assert_allclose(c.reshape(M, N), a @ b, rtol=1e-4, atol=1e-4)
  if opts.thread is not None: assert "#pragma omp parallel for" in src

def test_upcast_shares_reduce_loop():
  src = render_c(lower(gemm_graph(8, 8, 8), buf(2, 64), Ctx(opts=Opts(upcast={0: 2, 1: 2}))), "k")
  assert src.count("for (") == 3                        # 2 output loops + ONE shared reduce loop
  assert src.count("acc") >= 4

def test_reduce_unroll_has_no_inner_loop():
  x = buf(0, 16)
  src = render_c(lower(x.reduce(Ops.ADD, 1), buf(1, 1), Ctx(opts=Opts(unroll=16))), "k")
  assert "for (" not in src

def test_layout_pass_stages_transposed_operand():
  g = gemm_graph(64, 64, 64)
  (g2,) = layout_pass([g])
  assert any(u.op is Ops.STAGE for u in g2.toposort())
  (g3,) = layout_pass([gemm_graph(64, 64, 64, bt=True)])     # already contiguous: nothing to do
  assert not any(u.op is Ops.STAGE for u in g3.toposort())

def test_hand_opt_choice():
  o = hand_opt_choice(gemm_graph(64, 64, 64))
  assert o.upcast and o.thread is not None

def max_loop_depth(src: str) -> int:
  return max((len(l) - len(l.lstrip())) // 2 for l in src.splitlines() if l.lstrip().startswith("for ("))

def test_fused_attention_reduces_are_not_nested():                 # NOTES 12
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
  from flash_attention import attention, FlashOpt
  from uopc.tensor import Tensor
  from uopc.kernelize import make_plan
  N, D = 64, 16
  q, k, v = (Tensor(np.random.randn(N, D).astype(np.float32)) for _ in range(3))
  plan, _ = make_plan([attention(q, k, v).uop], fuse_reduce=True, opt=FlashOpt)
  # loops: i (thread), then a reduce over keys j, inside it a dot product over D: depth 3, never 4
  assert max(max_loop_depth(k.src) for k in plan.kernels) <= 3

@pytest.mark.parametrize("tile", [{0: 8, 1: 16}, {1: 32}, {0: 16}])
def test_cache_blocking_correct(tile):
  M, N, K = 32, 64, 48
  a, b = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32)
  c = np.zeros(M * N, np.float32)
  src = render_c(lower(gemm_graph(M, N, K), buf(2, M * N), Ctx(opts=Opts(upcast={0: 4, 1: 4}, thread=0, tile=tile))), "k")
  Program(src, "k")(a, b, c)
  np.testing.assert_allclose(c.reshape(M, N), a @ b, rtol=1e-4, atol=1e-4)
  assert src.count("for (") == 3 + len(tile)                # one extra block loop per tiled axis
