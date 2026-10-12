"""Week 2: rendering loopless C, the runtime, memory (BUFFER/ALLOC/LOAD/STORE/AFTER)."""
import numpy as np
from uopc.uop import UOp, Ops, ParamArg, sink
from uopc.dtype import dtypes
from uopc.render_c import render_c
from uopc.runtime import run_c
from helpers import buf

def test_elementwise_loopless():
  a, b, c = buf(0, 4), buf(1, 4), buf(2, 4)
  s = sink(*[c.index(i).store(a.index(i).load() * b.index(i).load() + 1.0) for i in range(4)])
  A, B, C = np.arange(4, dtype=np.float32), np.full(4, 2, np.float32), np.zeros(4, np.float32)
  run_c(s, {0: A, 1: B, 2: C})
  np.testing.assert_allclose(C, A * B + 1)

def test_alloc_and_after():
  # tmp[0] = a[0] + a[1]; out[0] = tmp[0] * tmp[0]  -- AFTER orders the read after the write
  a, out = buf(0, 2), buf(1, 1)
  tmp = UOp(Ops.ALLOC, (), ParamArg(0, dtypes.f32, (1,)))
  w = tmp.index(0).store(a.index(0).load() + a.index(1).load())
  v = tmp.after(w).index(0).load()
  O = np.zeros(1, np.float32)
  run_c(sink(out.index(0).store(v * v)), {0: np.array([1, 2], np.float32), 1: O})
  assert O[0] == 9

def test_floordiv_rendering_negative():
  a, out = buf(0, 2, dtypes.i32), buf(1, 2, dtypes.i32)
  x = a.index(0).load()
  s = sink(out.index(0).store(x // a.index(1).load()), out.index(1).store(x % a.index(1).load()))
  O = np.zeros(2, np.int32)
  run_c(s, {0: np.array([-7, 2], np.int32), 1: O})
  assert list(O) == [-4, 1]
  assert "floordiv_i32" in render_c(s)

def test_casts_and_where():
  a, out = buf(0, 3), buf(1, 3, dtypes.i32)
  s = sink(*[out.index(i).store((a.index(i).load() < 0.0).where(UOp.const(-1), a.index(i).load().cast(dtypes.i32))) for i in range(3)])
  O = np.zeros(3, np.int32)
  run_c(s, {0: np.array([-2.5, 3.7, 0.2], np.float32), 1: O})
  assert list(O) == [-1, 3, 0]

def test_run_c_checks_buffer_dtypes():
  import pytest
  out = buf(0, 4)
  with pytest.raises(TypeError): run_c(sink(out.index(0).store(UOp.const(1.0))), {0: np.ones(4)})   # float64
