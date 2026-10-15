"""Independent numerical and semantic checks for the executable book labs."""
from dataclasses import replace
import numpy as np
import pytest
from labs.attention import online_attention, attention_backward
from labs.ssa import Block, Instruction, diamond, dominators, liveness, verify
from labs.memory import Temporary, plan_memory, verify_plan
from labs.dynamic import GuardedJIT
from labs.numerics import Moments, moments, quantize_weights
from labs.irregular import csr_matvec, embedding_gradient


def dense_attention(q, k, v, causal=False, offset=0):
  scores = q @ k.T / np.sqrt(q.shape[1])
  if causal:
    scores[np.arange(len(k))[None, :] > (np.arange(len(q)) + offset)[:, None]] = -np.inf
  probability = np.exp(scores - scores.max(axis=1, keepdims=True))
  probability /= probability.sum(axis=1, keepdims=True)
  return probability @ v


@pytest.mark.parametrize("causal,offset", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize("tile", [1, 4, 32])
def test_attention_forward_and_directional_backward(causal, offset, tile):
  rng = np.random.default_rng(73)
  arrays = [rng.normal(size=shape) for shape in ((5, 3), (9, 3), (9, 4))]
  q, k, v = arrays
  output, log_normalizer = online_attention(q, k, v, tile, causal, offset)
  np.testing.assert_allclose(output, dense_attention(q, k, v, causal, offset), atol=1e-12)
  grad_output = rng.normal(size=output.shape)
  gradients = attention_backward(q, k, v, output, log_normalizer, grad_output, tile, causal, offset)
  # Each independent input is checked against finite differences of a dense oracle.
  for i, gradient in enumerate(gradients):
    direction = rng.normal(size=arrays[i].shape)
    plus, minus = arrays.copy(), arrays.copy()
    plus[i], minus[i] = arrays[i] + 1e-5*direction, arrays[i] - 1e-5*direction
    numerical = np.sum((dense_attention(*plus, causal, offset) - dense_attention(*minus, causal, offset)) * grad_output) / 2e-5
    assert np.sum(gradient * direction) == pytest.approx(numerical, rel=1e-7, abs=1e-8)


def test_attention_extreme_finite_logits_and_input_contract():
  rng = np.random.default_rng(0)
  q, k, v = rng.normal(size=(3, 4))*100, rng.normal(size=(5, 4))*100, rng.normal(size=(5, 2))
  with np.errstate(over="raise", invalid="raise"):
    out, _ = online_attention(q, k, v, tile=2, causal=True)
  np.testing.assert_allclose(out, dense_attention(q, k, v, True), atol=1e-12)
  with pytest.raises(ValueError): online_attention(q, k, v, tile=0)
  with pytest.raises(ValueError): online_attention(q, k, v[:2])


def test_ssa_phi_edges_and_dominance():
  cfg = diamond()
  verify(cfg, "entry", ("p", "x"))
  assert dominators(cfg, "entry")["merge"] == {"entry", "merge"}
  incoming, outgoing = liveness(cfg, "entry")
  assert outgoing["left"] == {"a"} and outgoing["right"] == {"b"}
  assert incoming["entry"] == {"p", "x"}
  cfg["merge"] = replace(cfg["merge"], instructions=(Instruction(None, ("a",)),))
  with pytest.raises(ValueError, match="does not dominate"): verify(cfg, "entry", ("p", "x"))


def test_ssa_loop_phi_and_malformed_programs():
  cfg = {
    "entry": Block((), ("head",)),
    "head": Block((Instruction(None, ("i",)),), ("body", "exit"),
                  (("i", (("entry", "initial"), ("body", "next"))),)),
    "body": Block((Instruction("next", ("i",)),), ("head",)),
    "exit": Block((Instruction(None, ("i",)),)),
    "unreachable": Block((Instruction(None, ("undefined",)),)),
  }
  verify(cfg, "entry", ("initial",))
  assert dominators(cfg, "entry")["body"] == {"entry", "head", "body"}
  assert "unreachable" not in dominators(cfg, "entry")
  cfg["body"] = replace(cfg["body"], instructions=(Instruction("next", ("missing",)),))
  with pytest.raises(ValueError, match="undefined"): verify(cfg, "entry", ("initial",))
  cfg = diamond()
  cfg["merge"] = replace(cfg["merge"], phis=(("z", (("left", "a"),)),))
  with pytest.raises(ValueError, match="incoming edges"): verify(cfg, "entry", ("p", "x"))


def test_memory_reuse_preserves_inclusive_lifetimes_and_alignment():
  temporaries = [Temporary("a", 1024, 0, 2), Temporary("b", 512, 2, 3), Temporary("c", 768, 3, 4)]
  allocations, peak = plan_memory(temporaries)
  verify_plan(temporaries, allocations)
  assert allocations["a"].offset != allocations["b"].offset
  assert allocations["a"].offset == allocations["c"].offset
  assert peak == 1536
  with pytest.raises(ValueError): plan_memory([Temporary("bad", 1, 4, 2)])


def test_random_memory_plans_do_not_clobber_live_buffers():
  rng = np.random.default_rng(42)
  for _ in range(30):
    temporaries = [Temporary(str(i), int(rng.integers(1, 500)), int(start), int(start+rng.integers(0, 5)),
                             int(rng.choice([16, 64, 256]))) for i, start in enumerate(rng.integers(0, 10, 20))]
    allocations, peak = plan_memory(temporaries)
    verify_plan(temporaries, allocations)
    arena = np.zeros(peak, dtype=np.uint8)
    # Execute synthetic writes and reads, detecting overlap without trusting the verifier.
    for step in range(15):
      for i, t in enumerate(temporaries):
        if t.first == step:
          start = allocations[t.name].offset
          arena[start:start+t.nbytes] = i+1
      for i, t in enumerate(temporaries):
        if t.first <= step <= t.last:
          start = allocations[t.name].offset
          assert np.all(arena[start:start+t.nbytes] == i+1)


def test_guarded_jit_reuses_runtime_length_and_falls_back_for_views():
  jit = GuardedJIT(64)
  for n in (1, 7, 31, 64, 7):
    x = np.linspace(-2, 2, n, dtype=np.float32)
    np.testing.assert_allclose(jit(x), x*x+1, rtol=1e-6)
  for x in (np.arange(30, dtype=np.float32)[::3], np.arange(8, dtype=np.float64)):
    np.testing.assert_allclose(jit(x), x*x+1)
  assert (jit.compilations, jit.fast_calls, jit.fallback_calls) == (1, 5, 2)
  with pytest.raises(ValueError): jit(np.zeros(65, dtype=np.float32))
  with pytest.raises(ValueError): jit(np.zeros(3, dtype=np.int32))


def test_mergeable_moments_near_constant_data_and_empty_partitions():
  values = 1e8 + np.random.default_rng(0).normal(size=1000)
  merged = moments(values[:431]).merge(moments(values[431:]))
  assert merged.variance() == pytest.approx(np.var(values), rel=1e-7)
  assert Moments().merge(merged) == merged.merge(Moments()) == merged
  with pytest.raises(ValueError): Moments().variance()


def test_quantization_zero_channels_and_error_bound():
  weights = np.random.default_rng(12).normal(size=(64, 8)); weights[:, 0] = 0
  q, scale = quantize_weights(weights)
  assert q.dtype == np.int8 and np.all(q[:, 0] == 0)
  assert np.all(np.abs(q*scale-weights) <= scale/2 + 1e-14)
  with pytest.raises(ValueError): quantize_weights([[np.inf]])


def test_csr_empty_rows_duplicates_and_invalid_offsets():
  rowptr, columns, values = [0, 3, 3, 4], [0, 2, 2, 1], [2., 3., 5., 4.]
  dense = np.zeros((3, 3))
  for row in range(3):
    for position in range(rowptr[row], rowptr[row+1]): dense[row, columns[position]] += values[position]
  vector = np.array([1., 2., 3.])
  np.testing.assert_array_equal(csr_matvec(rowptr, columns, values, vector), dense @ vector)
  with pytest.raises(ValueError): csr_matvec([0, 2, 1], [0], [1.], vector)
  with pytest.raises(ValueError): csr_matvec([0, 1], [3], [1.], vector)


def test_embedding_gradient_accumulates_repeated_indices():
  result = embedding_gradient([1, 1, 2], np.array([[1., 2.], [3., 4.], [5., 6.]]), 4)
  np.testing.assert_array_equal(result, [[0, 0], [4, 6], [5, 6], [0, 0]])
  with pytest.raises(ValueError): embedding_gradient([-1], np.ones((1, 2)), 4)
