"""Weeks 7-8: GPU mapping. Every kernel is checked on the CPU through the emulator;
the CUDA test only runs when a GPU and NVRTC are available."""
import numpy as np, pytest
from uopc.uop import UOp, Ops, AxisType
from uopc.dtype import dtypes
from uopc.runtime import Program
from uopc.rangeify import Opts
from uopc.gpu import gpu_lower, gpu_opt_choice, emulate, render_cuda, launch_dims, smem_gemm_ir, wmma_gemm, match_gemm
from helpers import buf

def gemm_graph(M, N, K):
  A, B = buf(0, M * K), buf(1, K * N)
  return (A.reshape((M, 1, K)).expand((M, N, K)) * B.reshape((K, N)).permute((1, 0)).reshape((1, N, K)).expand((M, N, K))).reduce(Ops.ADD, 1)

@pytest.mark.parametrize("opts", [Opts(gpu=True), Opts(gpu=True, local={0: 4, 1: 8}),
                                  Opts(gpu=True, local={1: 16}, upcast={0: 2, 1: 2})])
def test_gpu_gemm_emulated(opts):
  M, N, K = 32, 64, 48
  a, b = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32)
  c = np.zeros(M * N, np.float32)
  ast = gpu_lower(gemm_graph(M, N, K), buf(2, M * N), opts)
  Program(emulate(ast), "kernel")(a, b, c)
  np.testing.assert_allclose(c.reshape(M, N), a @ b, atol=1e-4)
  src = render_cuda(ast)
  assert "__global__" in src and ("threadIdx" in src) == bool(opts.local)

def test_gpu_softmax_like_emulated():
  x = np.random.randn(16, 32).astype(np.float32)
  X = buf(0, 16 * 32).reshape((16, 32))
  m = X.reduce(Ops.MAX, 1).reshape((16, 1)).expand((16, 32))
  body = UOp(Ops.EXP2, (X + -m,))
  out = np.zeros(16 * 32, np.float32)
  ast = gpu_lower(body, buf(1, 16 * 32))
  Program(emulate(ast), "kernel")(x, out)
  np.testing.assert_allclose(out.reshape(16, 32), np.exp2(x - x.max(1, keepdims=True)), rtol=1e-5)

def test_launch_dims_and_auto_choice():
  o = gpu_opt_choice(gemm_graph(64, 64, 64))
  ast = gpu_lower(gemm_graph(64, 64, 64), buf(2, 64 * 64), o)
  grid, block = launch_dims(ast)
  assert np.prod(grid) * np.prod(block) * np.prod(list(o.upcast.values()) or [1]) == 64 * 64

def test_smem_gemm_renders():
  src = render_cuda(smem_gemm_ir(64, 16), "smem_gemm")
  assert src.count("__syncthreads();") == 2 and "__shared__ float smem0[256];" in src
  assert launch_dims(smem_gemm_ir(64, 16)) == ((4, 4, 1), (16, 16, 1))

def test_wmma():
  assert match_gemm(gemm_graph(32, 32, 32)) == (32, 32, 32)
  assert "mma_sync" in wmma_gemm(32, 32, 32)

def _has_cuda():
  try:
    from uopc.cuda import CUDA
    CUDA(); return True
  except Exception: return False

@pytest.mark.skipif(not _has_cuda(), reason="no CUDA device / NVRTC")
def test_cuda_smem_gemm_on_gpu():
  from uopc.cuda import run_cuda
  N = 64
  a, b, c = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32), np.zeros(N * N, np.float32)
  ast = smem_gemm_ir(N, 16)
  grid, block = launch_dims(ast)
  run_cuda(render_cuda(ast, "smem_gemm"), "smem_gemm", grid, block, a, b, c)
  np.testing.assert_allclose(c.reshape(N, N), a @ b, atol=1e-3)
