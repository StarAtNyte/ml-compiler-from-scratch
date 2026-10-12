"""Week 4: movement ops, broadcasting, reduction, rangeify, composite PAD/FLIP."""
import numpy as np, pytest
from uopc.uop import UOp, Ops
from uopc.uir import parse
from uopc.composite import pad, flip_axis, anti_eye
from uopc.symbolic import run, to_python
from helpers import buf, run_shaped

def test_gemm_from_movement_ops():
  N = 16
  a, b = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
  g = parse(f"""
  %0 = buffer {N*N} : dtype=f32 slot=0
  %1 = buffer {N*N} : dtype=f32 slot=1
  %2 = reshape %0, ({N}, 1, {N})
  %3 = reshape %1, ({N}, {N})
  %4 = permute %3, (1, 0)
  %5 = reshape %4, (1, {N}, {N})
  %6 = mul %2, %5
  %7 = reduce %6 : op=add pop=1""")
  np.testing.assert_allclose(run_shaped(g, [a.ravel(), b.ravel()]), a @ b, atol=1e-4)

def test_syllabus_gemm_is_not_a_gemm():
  # the example in the syllabus computes A * rowsum(B): a good reminder to test everything
  N = 8
  a, b = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
  g = parse(f"""
  %0 = buffer {N*N} : dtype=f32 slot=0
  %1 = buffer {N*N} : dtype=f32 slot=1
  %2 = reshape %0, ({N}, {N}, 1)
  %3 = reshape %1, ({N}, 1, {N})
  %4 = mul %2, %3
  %5 = reduce %4 : op=add pop=1""")
  np.testing.assert_allclose(run_shaped(g, [a.ravel(), b.ravel()]), a * b.sum(1, keepdims=True), atol=1e-4)

@pytest.mark.parametrize("seed", range(5))
def test_random_movement_chains(seed):
  rng = np.random.default_rng(seed)
  x = rng.standard_normal(24).astype(np.float32)
  u, ref = buf(0, 24).reshape((2, 3, 4)), x.reshape(2, 3, 4)
  for _ in range(6):
    k = rng.integers(4)
    if k == 0:
      p = tuple(rng.permutation(len(ref.shape))); u, ref = u.permute(p), ref.transpose(p)
    elif k == 1:
      arg = tuple((int(b), int(rng.integers(b + 1, s + 1))) for s in ref.shape for b in [rng.integers(0, s)])
      u, ref = u.shrink(arg), ref[tuple(slice(b, e) for b, e in arg)]
    elif k == 2:
      u, ref = u.reshape((1,) + u.shape), ref.reshape((1,) + ref.shape)
      u, ref = u.expand((2,) + u.shape[1:]), np.broadcast_to(ref, (2,) + ref.shape[1:])
    else:
      u, ref = u.reshape((int(np.prod(ref.shape)),)), ref.reshape(-1)
  np.testing.assert_allclose(run_shaped(u, [x]), ref)

def test_reduce_max_and_nested():
  x = np.random.randn(6, 5).astype(np.float32)
  u = buf(0, 30).reshape((6, 5))
  np.testing.assert_allclose(run_shaped(u.reduce(Ops.MAX, 1), [x.ravel()]), x.max(1))
  # x - max(x) broadcast back: the reduce becomes a sibling loop, computed once per row (no kernelizer needed)
  m = u.reduce(Ops.MAX, 1).reshape((6, 1)).expand((6, 5))
  np.testing.assert_allclose(run_shaped(u + -m, [x.ravel()]), x - x.max(1, keepdims=True))
  np.testing.assert_allclose(run_shaped(u.reduce(Ops.ADD, 2), [x.ravel()]), x.sum(), rtol=1e-5)

def test_stack_and_index_in_rangeify():
  x = np.random.randn(4).astype(np.float32)
  u = buf(0, 4)
  s = UOp(Ops.STACK, (u, u * u, u + u))
  np.testing.assert_allclose(run_shaped(s, [x]), np.stack([x, x * x, x + x]))
  np.testing.assert_allclose(run_shaped(s.index(UOp.const(1)), [x]), x * x)

def test_pad_flip_composites():
  x = np.random.randn(3, 4).astype(np.float32)
  u = buf(0, 12).reshape((3, 4))
  np.testing.assert_allclose(run_shaped(pad(u, ((1, 5), (2, 0))), [x.ravel()]), np.pad(x, ((1, 5), (2, 0))))
  np.testing.assert_allclose(run_shaped(flip_axis(u, 1), [x.ravel()]), x[:, ::-1])
  np.testing.assert_allclose(run_shaped(flip_axis(u, 0), [x.ravel()]), x[::-1])
  assert to_python(run(anti_eye(3, u.dtype))) == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
