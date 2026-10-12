"""Week 3: RANGE/END. Hand-written gemm and conv in the loop language."""
import numpy as np
from uopc.uop import UOp, Ops, ParamArg, sink, range_
from uopc.dtype import dtypes
from uopc.runtime import run_c
from uopc.linearize import linearize
from helpers import buf

def gemm(N, M, K):
  A, B, C = buf(0, N * K), buf(1, K * M), buf(2, N * M)
  i, j, k = range_(N, 0), range_(M, 1), range_(K, 2)
  acc = UOp(Ops.ALLOC, (), ParamArg(0, dtypes.f32, (1,))).after(i, j)
  init = acc.index(0).store(UOp.const(0.0))
  acck = acc.after(init, k)
  upd = acck.index(0).store(acck.index(0).load() + A.index(i * K + k).load() * B.index(k * M + j).load())
  return sink(C.index(i * M + j).store(acc.after(upd.end(k)).index(0).load()).end(i, j))

def test_gemm_loops():
  N, M, K = 17, 9, 33
  a, b = np.random.randn(N, K).astype(np.float32), np.random.randn(K, M).astype(np.float32)
  c = np.zeros((N, M), np.float32)
  src = run_c(gemm(N, M, K), {0: a, 1: b, 2: c})
  np.testing.assert_allclose(c, a @ b, atol=1e-4)
  assert src.count("for (") == 3

def test_conv1d_loops():
  # out[o] = sum_k x[o + k] * w[k]
  n, kw = 20, 5
  X, W, O = buf(0, n), buf(1, kw), buf(2, n - kw + 1)
  o, k = range_(n - kw + 1, 0), range_(kw, 1)
  acc = UOp(Ops.ALLOC, (), ParamArg(0, dtypes.f32, (1,))).after(o)
  init = acc.index(0).store(UOp.const(0.0))
  acck = acc.after(init, k)
  upd = acck.index(0).store(acck.index(0).load() + X.index(o + k).load() * W.index(k).load())
  s = sink(O.index(o).store(acc.after(upd.end(k)).index(0).load()).end(o))
  x, w, out = np.random.randn(n).astype(np.float32), np.random.randn(kw).astype(np.float32), np.zeros(n - kw + 1, np.float32)
  run_c(s, {0: x, 1: w, 2: out})
  np.testing.assert_allclose(out, np.correlate(x, w, "valid"), atol=1e-5)

def test_loop_invariant_code_is_hoisted():
  N = 8
  A, C = buf(0, N * N), buf(1, N * N)
  i, j = range_(N, 0), range_(N, 1)
  row = A.index(i * N).load()            # depends only on i
  s = sink(C.index(i * N + j).store(row + A.index(i * N + j).load()).end(i, j))
  lin = linearize(s)
  pos = {u: n for n, u in enumerate(lin)}
  assert pos[row] < pos[j]                # emitted before the j loop opens

def test_missing_end_is_an_error():
  import pytest
  from uopc.linearize import LinearizeError
  out, r = buf(0, 4), range_(4, 0)
  with pytest.raises(LinearizeError, match="never closed"): linearize(sink(out.index(r).store(UOp.const(1.0))))
