"""Dynamic tensor compiler: independent NumPy oracles, guards, fusion, and reuse."""
import numpy as np
import pytest
from tensorjit import CompiledGraph, Dim, input
from tensorjit.graph import constant


def random(rng, shape): return rng.normal(size=shape).astype(np.float32)


def test_one_native_variant_runs_different_symbolic_matmul_shapes():
  b, k, n = Dim("B", 1, 32), Dim("K", 1, 64), Dim("N", 1, 64)
  x, w, bias = input("x", (b, k)), input("w", (k, n)), input("bias", (n,))
  expression = ((x @ w) + bias).relu().sum(1)
  fused = CompiledGraph((expression,))
  unfused = CompiledGraph((expression,), fuse=False)
  rng = np.random.default_rng(23)
  for batch, inner, width in ((1, 1, 1), (2, 7, 5), (7, 31, 17), (2, 7, 5)):
    arrays = {"x": random(rng, (batch, inner)), "w": random(rng, (inner, width)), "bias": random(rng, (width,))}
    expected = np.fmax(arrays["x"] @ arrays["w"] + arrays["bias"], 0).sum(axis=1)
    for compiled in (fused, unfused):
      actual, = compiled(**arrays)
      np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
  assert fused.compilations == unfused.compilations == 1
  assert fused.calls == 4
  assert len(fused.plan.kernels) == 2 < len(unfused.plan.kernels)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_broadcast_transpose_and_reduction(axis):
  m, n = Dim("M", 1, 8), Dim("N", 1, 9)
  x, row, column = input("x", (m, n)), input("row", (n,)), input("column", (m, 1))
  expression = ((x + row) * column).transpose().sum(axis)
  compiled = CompiledGraph((expression,))
  rng = np.random.default_rng(2)
  for shape in ((1, 1), (3, 7), (8, 9)):
    arrays = {"x": random(rng, shape), "row": random(rng, (shape[1],)), "column": random(rng, (shape[0], 1))}
    expected = ((arrays["x"] + arrays["row"]) * arrays["column"]).T.sum(axis=axis)
    actual, = compiled(**arrays)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_shared_subexpression_is_materialized_once():
  x = input("x", (4, 7))
  shared = (x*2 + 1).relu()
  compiled = CompiledGraph((shared.sum(0), shared.sum(1)))
  assert sum(kernel.node is shared for kernel in compiled.plan.kernels) == 1
  data = np.arange(28, dtype=np.float32).reshape(4, 7)-10
  outputs = compiled(x=data)
  expected = np.fmax(data*2+1, 0)
  np.testing.assert_array_equal(outputs[0], expected.sum(0))
  np.testing.assert_array_equal(outputs[1], expected.sum(1))


def test_actual_runtime_reuses_dead_matrix_storage():
  x, a, b, c = (input(name, (8, 8)) for name in ("x", "a", "b", "c"))
  first, second = x @ a, None
  second = first @ b
  third = second @ c
  compiled = CompiledGraph((third.sum(1),))
  rng = np.random.default_rng(0)
  arrays = {name: random(rng, (8, 8)) for name in ("x", "a", "b", "c")}
  actual, = compiled(**arrays)
  np.testing.assert_allclose(actual, ((arrays["x"] @ arrays["a"] @ arrays["b"]) @ arrays["c"]).sum(1), rtol=2e-5, atol=2e-5)
  memory = compiled.last_memory
  assert memory["temporary_bytes"] == 2*8*8*4
  assert memory["unreused_temporary_bytes"] == 3*8*8*4
  assert all(allocation.offset % 64 == 0 for allocation in memory["allocations"].values())


def test_retained_outputs_and_later_calls_cannot_be_clobbered():
  x, w = input("x", (4, 4)), input("w", (4, 4))
  intermediate = x @ w
  final = ((intermediate @ w) @ w).sum(1)
  compiled = CompiledGraph((intermediate, final))
  arrays = {"x": np.arange(16, dtype=np.float32).reshape(4, 4), "w": np.eye(4, dtype=np.float32)}
  first = compiled(**arrays)
  saved = [array.copy() for array in first]
  compiled(x=arrays["x"]+100, w=arrays["w"])
  for actual, expected in zip(first, saved): np.testing.assert_array_equal(actual, expected)
  assert not np.shares_memory(first[0], arrays["x"])


def test_returning_input_and_scalar_constants_has_owned_outputs():
  x = input("x", (3,))
  compiled = CompiledGraph((x, constant(2.5)))
  array = np.arange(3, dtype=np.float32)
  out, scalar = compiled(x=array)
  np.testing.assert_array_equal(out, array)
  assert scalar.shape == () and scalar.item() == 2.5
  out[0] = 99
  assert array[0] == 0


def test_scalar_input_broadcast_and_nan_relu_contract():
  x, scalar = input("x", (4,)), input("scalar", ())
  compiled = CompiledGraph(((x+scalar).relu(),))
  array = np.array([np.nan, -2., 0., 4.], dtype=np.float32)
  out, = compiled(x=array, scalar=np.array(1., dtype=np.float32))
  np.testing.assert_array_equal(out, np.fmax(array+1, 0))


def test_dynamic_guards_check_layout_dtype_bounds_and_equal_symbols():
  n = Dim("N", 1, 8)
  a, b = input("a", (n,)), input("b", (n,))
  compiled = CompiledGraph((a+b,))
  valid = np.zeros(4, dtype=np.float32)
  for invalid in (np.zeros(4, dtype=np.float64), np.zeros(8, dtype=np.float32)[::2], np.zeros(0, dtype=np.float32), np.zeros(9, dtype=np.float32)):
    with pytest.raises(ValueError): compiled(a=invalid, b=valid)
  with pytest.raises(ValueError, match="inconsistent"): compiled(a=valid, b=np.zeros(3, dtype=np.float32))
  with pytest.raises(ValueError): compiled(a=valid)
  assert compiled.calls == 0


def test_guard_failures_do_not_mutate_inputs_or_previous_outputs():
  n = Dim("N", 1, 4)
  a, b = input("a", (n,)), input("b", (n,))
  compiled = CompiledGraph((a*b,))
  x = np.arange(4, dtype=np.float32)
  previous, = compiled(a=x, b=x)
  saved = previous.copy()
  with pytest.raises(ValueError): compiled(a=x, b=np.zeros(5, dtype=np.float32))
  np.testing.assert_array_equal(previous, saved)
  np.testing.assert_array_equal(x, np.arange(4, dtype=np.float32))


def test_graph_rejects_unproved_shape_equalities_and_ambiguous_names():
  with pytest.raises(ValueError): input("a", (Dim("A"),)) + input("b", (Dim("B"),))
  with pytest.raises(ValueError): input("a", (2, 3)) @ input("b", (4, 5))
  with pytest.raises(ValueError): Dim("N", 0, 5)
  with pytest.raises(ValueError): input("a", (0,))
  with pytest.raises(ValueError): CompiledGraph((input("x", (2,)) + input("x", (2,)),))
  a, b = input("a", (Dim("N", 1, 4),)), input("b", (Dim("N", 1, 8),))
  with pytest.raises(ValueError, match="inconsistent bounds"): CompiledGraph((a.sum(), b.sum()))


def test_total_allocation_budget_is_checked_before_native_work():
  x = input("x", (16, 16))
  compiled = CompiledGraph((x+x,), max_bytes=1500)
  array = np.zeros((16, 16), dtype=np.float32)
  with pytest.raises(ValueError, match="budget"): compiled(x=array)
  assert compiled.calls == 0


def test_seeded_graph_family_fused_and_unfused_against_numpy():
  rng = np.random.default_rng(404)
  n = Dim("N", 1, 32)
  x, y = input("x", (n,)), input("y", (n,))
  for _ in range(8):
    a, b = map(float, rng.integers(-3, 4, 2))
    common = ((x*a+y) * b).relu()
    output = (common*common + x).sum()
    fused, unfused = CompiledGraph((output,)), CompiledGraph((output,), fuse=False)
    for length in (1, 7, 31):
      arrays = {"x": random(rng, (length,)), "y": random(rng, (length,))}
      reference = np.fmax((arrays["x"]*a+arrays["y"])*b, 0)
      expected = (reference*reference+arrays["x"]).sum()
      for compiled in (fused, unfused):
        actual, = compiled(**arrays)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-5)


def test_compiled_autodiff_matches_double_precision_directional_oracle():
  from tensorjit import gradients
  batch = Dim("B", 1, 16)
  x, w, bias = input("x", (batch, 3)), input("w", (3, 2)), input("bias", (2,))
  preactivation = x @ w + bias
  activated = preactivation.relu()
  loss = (activated*activated).sum(0).sum(0)
  dx, dw, db = gradients(loss, (x, w, bias))
  compiled = CompiledGraph((loss, dx, dw, db))
  rng = np.random.default_rng(53)
  for length in (1, 7, 13):
    arrays = {"x": random(rng, (length, 3)), "w": random(rng, (3, 2)), "bias": random(rng, (2,))}
    actual_loss, *actual_gradients = compiled(**arrays)
    double = {name: value.astype(np.float64) for name, value in arrays.items()}
    def oracle(values):
      value = np.fmax(values["x"] @ values["w"] + values["bias"], 0)
      return np.sum(value*value)
    assert actual_loss == pytest.approx(oracle(double), rel=2e-5, abs=2e-5)
    for name, gradient in zip(("x", "w", "bias"), actual_gradients):
      direction = rng.normal(size=arrays[name].shape)
      plus, minus = double.copy(), double.copy()
      plus[name], minus[name] = double[name]+1e-6*direction, double[name]-1e-6*direction
      expected = (oracle(plus)-oracle(minus))/2e-6
      assert float(np.sum(gradient.astype(np.float64)*direction)) == pytest.approx(expected, rel=2e-5, abs=2e-5)


def test_autodiff_broadcast_sum_transpose_and_unused_input():
  from tensorjit import gradients
  n = Dim("N", 1, 9)
  x, row, absent = input("x", (2, n)), input("row", (n,)), input("absent", (n,))
  total = ((x+row).transpose()*2).sum(1).sum(0)
  dx, dr, da = gradients(total, (x, row, absent))
  compiled = CompiledGraph((dx, dr, da))
  # Absent-input zero gradient carries N but does not read the absent buffer.
  assert {node.arg for node in compiled.plan.inputs} == set()
  # Shape-only graphs require explicit checked metadata, never a guessed N.
  with pytest.raises(ValueError, match="unbound"): compiled()
  for length in (1, 5, 9):
    gx, gr, ga = compiled.run_with_bindings({"N": length})
    np.testing.assert_array_equal(gx, np.full((2, length), 2, dtype=np.float32))
    np.testing.assert_array_equal(gr, np.full(length, 4, dtype=np.float32))
    np.testing.assert_array_equal(ga, np.zeros(length, dtype=np.float32))
  for invalid in ({"N": 10}, {"N": 0}, {"N": True}, {"unknown": 3}):
    with pytest.raises(ValueError): compiled.run_with_bindings(invalid)
  compiled = CompiledGraph((total, dx, dr, da))
  for length in (1, 5, 9):
    _, gx, gr, ga = compiled(x=np.ones((2, length), dtype=np.float32), row=np.ones(length, dtype=np.float32))
    np.testing.assert_array_equal(gx, np.full((2, length), 2, dtype=np.float32))
    np.testing.assert_array_equal(gr, np.full(length, 4, dtype=np.float32))
    np.testing.assert_array_equal(ga, np.zeros(length, dtype=np.float32))


def test_autodiff_explicit_broadcast_and_singleton_gradient():
  from tensorjit import gradients
  x = input("x", (2, 1))
  expanded = x.broadcast_to((2, 4))
  loss = (expanded*expanded).sum(0).sum(0)
  gradient, = gradients(loss, (x,))
  compiled = CompiledGraph((loss, gradient))
  actual_loss, actual_gradient = compiled(x=np.array([[2.], [-3.]], dtype=np.float32))
  assert actual_loss == pytest.approx(52)
  np.testing.assert_array_equal(actual_gradient, [[16.], [-24.]])


def test_dynamic_training_reuses_one_executable_and_updates_parameters():
  from examples.train_tensorjit import train
  result = train()
  assert result["final_loss"] < result["initial_loss"]*1e-5
  assert result["parameter_max_error"] < 1e-3 and result["bias_max_error"] < 1e-3
  assert result["compiled_variants"] == 1 and result["calls"] == 100


def test_explicit_bindings_cannot_override_actual_input_shape():
  n = Dim("N", 1, 8)
  x = input("x", (n,))
  compiled = CompiledGraph((x+1,))
  data = np.zeros(4, dtype=np.float32)
  with pytest.raises(ValueError, match="inconsistent"): compiled.run_with_bindings({"N": 3}, x=data)
  result, = compiled.run_with_bindings({"N": 4}, x=data)
  np.testing.assert_array_equal(result, np.ones(4, dtype=np.float32))
