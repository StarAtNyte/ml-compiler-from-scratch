"""Semantic and differential tests for the executable source-to-machine compiler."""
from copy import deepcopy
import textwrap
import numpy as np
import pytest
from minicc import compile_ir, compile_native, execute, lower
from minicc.analysis import dominance_frontiers, verify
from minicc.c_backend import NativeFunction
from minicc.ir import Instruction, Terminator, format_ir
from minicc.machine import allocate, destroy_phis, execute_allocated
from minicc.optimize import eliminate_dead_code, sccp
from minicc.ssa import to_ssa


SUM = """
def f(n):
    i = 0
    total = 0
    while i < n:
        if i < 3:
            total += i
        else:
            total += 2 * i
        i += 1
    return total
"""
SWAP = """
def f(n):
    a = 1
    b = 2
    i = 0
    while i < n:
        temporary = a
        a = b
        b = temporary
        i += 1
    return 10 * a + b
"""
CONTROL = """
def f(n):
    i = 0
    total = 0
    while i < n:
        i += 1
        if i == 2:
            continue
        if i == 7:
            break
        j = 0
        while j < i:
            total += j
            j += 1
    return total
"""
MEMORY = """
def f(a: buffer, b: buffer, n: int) -> int:
    i = 0
    total = 0
    while i < n:
        a[i] = b[i] + 1
        total += b[i]
        i += 1
    return total
"""


def reference(source):
  # Only tests' fixed/generated trusted sources are executed. The compiler never
  # executes incoming Python, and int/buffer annotations are names only here.
  env = {"buffer": object, "int": int}
  exec(source, env)
  return env["f"]


@pytest.mark.parametrize("source", [SUM, SWAP, CONTROL, """
def f(n):
    if n < 0:
        return 1
    else:
        return n + 3
"""])
def test_all_execution_stages_against_independent_source(source):
  mutable = lower(source)
  ssa = to_ssa(mutable)
  optimized = compile_ir(source)
  native, native_unoptimized = NativeFunction(optimized), NativeFunction(ssa)
  machines = [allocate(destroy_phis(optimized), registers=r) for r in (1, 2, 16)]
  oracle = reference(source)
  for n in (-4, 0, 1, 2, 3, 7, 10):
    expected = oracle(n)
    assert execute(mutable, n) == execute(ssa, n) == execute(optimized, n) == native(n) == native_unoptimized(n) == expected
    for machine in machines: assert execute_allocated(machine, n)[0] == expected


def test_loop_phi_parallel_copies_and_real_spills():
  function = compile_ir(SWAP)
  machine = allocate(destroy_phis(function), registers=1)
  assert machine.slots > 0
  for n in range(12):
    actual, counts = execute_allocated(machine, n)
    assert actual == (12 if n % 2 == 0 else 21)
    assert counts["spill_reads"] > 0 and counts["spill_writes"] > 0
  assert all(not block.phis for block in machine.function.blocks.values())


def test_iterated_frontiers_and_dead_phi_pruning():
  function = lower(SUM)
  frontiers = dominance_frontiers(function)
  assert any(name in frontier for name, frontier in frontiers.items())  # loop header's self frontier
  ssa = to_ssa(function)
  verify(ssa)
  assert sum(len(block.phis) for block in ssa.blocks.values()) >= 3
  no_phi = compile_ir("""
def f(n):
    if n:
        dead = 1
    else:
        dead = 2
    return 7
""", optimized=False)
  assert not any(block.phis for block in no_phi.blocks.values())


@pytest.mark.parametrize("body", [
  "if n:\n    x = 1\nreturn x",
  "while n:\n    x = 1\n    n -= 1\nreturn x",
  "x = x + 1\nreturn x",
])
def test_definite_assignment_rejects_branch_loop_and_self_uses(body):
  source = "def f(n):\n" + textwrap.indent(body, "    ") + "\n"
  with pytest.raises(ValueError, match="uninitialized"): compile_ir(source)


def test_sccp_distinguishes_unexecuted_edges_from_unknown_values():
  source = """
def f(n):
    flag = 1
    if flag:
        x = 5
    else:
        x = n
    return x * 2
"""
  unoptimized = compile_ir(source, optimized=False)
  optimized = compile_ir(source)
  assert len(optimized.blocks) < len(unoptimized.blocks)
  assert execute(optimized, -100) == 10
  assert not any(block.phis for block in optimized.blocks.values())
  assert all(not block.instructions for block in optimized.blocks.values())
  assert "return 10" in format_ir(optimized)
  unknown = compile_ir("""
def f(n):
    if n:
        x = 2
    else:
        x = 3
    return x
""")
  assert execute(unknown, 0) == 3 and execute(unknown, 1) == 2
  assert any(block.phis for block in unknown.blocks.values())


def test_sccp_loop_value_becomes_overdefined():
  source = """
def f(n):
    x = 0
    while x < n:
        x += 1
    return x
"""
  function = compile_ir(source)
  assert any(block.phis for block in function.blocks.values())
  assert [execute(function, n) for n in (0, 1, 4)] == [0, 1, 4]


def test_sccp_does_not_mutate_its_input():
  source = compile_ir(SUM, optimized=False)
  before = format_ir(source)
  optimized = eliminate_dead_code(sccp(source))
  assert format_ir(source) == before
  assert optimized is not source


@pytest.mark.parametrize("aliased", [False, True])
def test_memory_effects_and_aliases_survive_all_backends(aliased):
  optimized = compile_ir(MEMORY)
  native = NativeFunction(optimized)
  machine = allocate(destroy_phis(optimized), 2)
  expected_a = [2, 4, 6, 8]
  expected_b = expected_a if aliased else [10, 20, 30, 40]
  expected = reference(MEMORY)(expected_a, expected_b, 4)
  for runner in (lambda *a: execute(optimized, *a), native, lambda *a: execute_allocated(machine, *a)[0]):
    a = np.array([2, 4, 6, 8], dtype=np.int32)
    b = a if aliased else np.array([10, 20, 30, 40], dtype=np.int32)
    assert runner(a, b, 4) == expected
    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)


@pytest.mark.parametrize("index", [-1, 3])
def test_unused_load_remains_an_observable_bounds_error(index):
  source = """
def f(a: buffer, n: int):
    dead = a[n]
    return 9
"""
  function = compile_ir(source)
  assert any(i.op == "load" for b in function.blocks.values() for i in b.instructions)
  native = NativeFunction(function)
  machine = allocate(destroy_phis(function), 1)
  for runner in (lambda *a: execute(function, *a), native, lambda *a: execute_allocated(machine, *a)[0]):
    with pytest.raises(IndexError): runner(np.arange(3, dtype=np.int32), index)


def test_unreachable_trap_is_removed_but_stores_remain_ordered():
  source = """
def f(a: buffer):
    if 0:
        x = a[100]
    a[0] = 3
    a[0] = a[0] + 4
    return a[0]
"""
  native = compile_native(source)
  a = np.zeros(1, dtype=np.int32)
  assert native(a) == 7 and a[0] == 7


def test_trap_retains_prior_stores_and_skips_later_stores():
  source = """
def f(a: buffer):
    a[0] = 7
    dead = a[3]
    a[0] = 99
    return 1
"""
  function = compile_ir(source)
  for runner in (lambda a: execute(function, a), NativeFunction(function)):
    array = np.zeros(1, dtype=np.int32)
    with pytest.raises(IndexError): runner(array)
    assert array[0] == 7


@pytest.mark.parametrize("a,b", [(2**31-1, 1), (-(2**31), -1), (65536, 65536), (2**40+7, 3)])
def test_explicit_i32_wrap_semantics(a, b):
  source = """
def f(a, b):
    return (a + b) * (a - b)
"""
  function = compile_ir(source)
  native = NativeFunction(function)
  def wrap(x): return (x + 2**31) % 2**32 - 2**31
  expected = wrap(wrap(wrap(a)+wrap(b)) * wrap(wrap(a)-wrap(b)))
  assert execute(function, a, b) == native(a, b) == expected
  assert execute_allocated(allocate(destroy_phis(function), 1), a, b)[0] == expected


def test_ssa_verifier_rejects_bad_edges_and_non_dominating_use():
  function = compile_ir(SUM, optimized=False)
  broken = deepcopy(function)
  phi = next(phi for b in broken.blocks.values() for phi in b.phis)
  phi.incoming.pop(next(iter(phi.incoming)))
  with pytest.raises(ValueError, match="edge mismatch"): verify(broken)
  broken = deepcopy(function)
  branches = [(name, b) for name, b in broken.blocks.items() if name.startswith("then")]
  value = branches[0][1].instructions[0].dest
  exit_block = next(b for b in broken.blocks.values() if b.terminator.op == "return")
  exit_block.terminator.args = (value,)
  with pytest.raises(ValueError, match="does not dominate"): verify(broken)
  broken = deepcopy(function)
  broken.blocks[broken.entry].instructions.append(Instruction("copy", "n", (1,)))
  with pytest.raises(ValueError, match="duplicate SSA"): verify(broken)


@pytest.mark.parametrize("source", [
  "def f(n):\n    return abs(n)",
  "def f(n):\n    return n / 2",
  "def f(n):\n    return n and 3",
  "def f(n):\n    return 1 < n < 9",
  "def f(n):\n    for i in range(n):\n        pass\n    return 0",
  "def f(a: buffer):\n    a = 3\n    return 0",
  "def f(n):\n    if n:\n        return 1",
])
def test_unsupported_or_ambiguous_language_features_are_rejected(source):
  with pytest.raises(ValueError): compile_ir(source)


def test_native_layout_and_argument_guards():
  native = compile_native("def f(a: buffer):\n    return a[0]")
  for invalid in (np.zeros(3, dtype=np.float32), np.zeros(6, dtype=np.int32)[::2], [1, 2]):
    with pytest.raises(ValueError): native(invalid)
  with pytest.raises(ValueError): native()
  array = np.zeros(3, dtype=np.int32); array.flags.writeable = False
  with pytest.raises(ValueError): native(array)


def test_execution_step_limit_catches_nontermination():
  source = "def f(n):\n    while n:\n        pass\n    return 0"
  function = compile_ir(source)
  with pytest.raises(RuntimeError, match="step limit"): execute(function, 1, max_steps=10)
  with pytest.raises(RuntimeError, match="step limit"):
    execute_allocated(allocate(destroy_phis(function), 1), 1, max_steps=10)


def test_seeded_program_family_against_source_oracle():
  rng = np.random.default_rng(2026)
  for _ in range(20):
    a, b, c, limit = map(int, rng.integers(1, 7, 4))
    source = f"""
def f(n):
    i = 0
    x = {a}
    y = {b}
    while i < n:
        if i < {limit}:
            x = x + y + {c}
        else:
            y = y - {b}
        i += 1
    return x + 3 * y
"""
    function = compile_ir(source)
    native = NativeFunction(function)
    machine = allocate(destroy_phis(function), registers=2)
    oracle = reference(source)
    for n in (0, 1, 5, 9):
      expected = oracle(n)
      assert execute(function, n) == native(n) == execute_allocated(machine, n)[0] == expected
