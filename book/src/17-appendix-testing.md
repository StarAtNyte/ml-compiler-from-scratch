# Appendix B. Testing, Debugging, and Slop Management

A compiler bug rarely announces itself where it lives. A wrong rewrite rule in week 1 does not fail a week-1
test; it produces an index that is off by a few elements, which becomes an out-of-bounds load in a week-5
kernel, which corrupts the heap, which aborts the Python process during a GPT training test. That exact
chain happened in this appendix, on purpose. The course syllabus calls itself "an exercise in slop
management": code that is a little wrong, or a little unclear, compounds every week. This appendix is the
defense: how the reference suite is organized, which oracles it trusts, how to write the tests it is missing
(fuzzers, property tests, structural tests), how to *test the tests* by re-introducing old bugs, how to debug
when something fails anyway, and, for every bug in `book/NOTES.md`, the test that catches it.

**Where this fits:** it complements every chapter's Pitfalls section. The op reference is
[Appendix A](16-appendix-reference.md); the vocabulary is in [Appendix C](18-appendix-glossary.md).

> **Labeling convention.** Listings with an `@include` label are in the repository. Code in plain fenced
> blocks marked *scratch* was written for this appendix and run from outside the repository (the extra tests
> and the mutation harness live in a scratch directory); every output shown is real.

## B.1 The shape of the suite

The reference suite has one file per week, named `tests/test_wNN_topic.py`, and a 15-line helper module. With
a warm kernel cache it runs in a few seconds:

```text
........................................................................ [ 84%]
.........s...                                                            [100%]
============================= slowest 8 durations ==============================
2.63s call     tests/test_w09_models.py::test_tiny_gpt_learns
0.29s call     tests/test_w01_uops.py::test_deep_graph_no_recursion_error
0.26s call     tests/test_w05_tensor.py::test_training_reduces_loss
0.11s call     tests/test_w09_models.py::test_attention_is_causal
0.05s call     tests/test_w06_opt.py::test_gemm_opts_correct[opts5]
0.03s call     tests/test_w05_tensor.py::test_adamw_steps_reuse_one_plan
0.03s call     tests/test_w06_opt.py::test_fused_attention_reduces_are_not_nested
0.03s call     tests/test_w05_tensor.py::test_gradients[matmul-<lambda>-<lambda>]
84 passed, 1 skipped in 3.87s
```

(The skipped test runs the shared-memory GEMM on a real GPU; the reference machine has none, NOTES 25.) From
an empty kernel cache the same suite took 33.5 s and left 240 compiled `.so` files behind; the second run
took 9.7 s. That ratio is why `runtime.compile_c` caches by content hash, and why the cache directory is
configurable (`UOPC_CACHE`, Appendix A).

Per file:

| File | Tests | Time | What it checks | Oracle |
|---|---|---|---|---|
| `test_w01_uops.py` | 16 | 1.1 s | hash consing, inference errors, folding semantics (including C-matching f32 and i32 folding), symbolic identities, CALL scoping (including inlining with a nested CALL still present), no float reassociation, `.uir` round trip | hand-computed values; C semantics |
| `test_w02_render.py` | 5 | 0.2 s | loopless C, ALLOC/AFTER ordering, floor division with negatives, casts, buffer dtype checks in `run_c` | numpy; hand values |
| `test_w03_loops.py` | 4 | 0.2 s | hand-written GEMM and conv1d; loop-invariant hoisting; a RANGE without END is an error | numpy (`@`, `np.correlate`); schedule position |
| `test_w04_rangeify.py` | 10 | 0.2 s | GEMM from views, the syllabus non-GEMM, random movement chains, reductions, STACK/INDEX, PAD/FLIP | numpy |
| `test_w05_tensor.py` | 24 | 1.2 s | Tensor ops, gradients, kernel counts, training, SGD momentum, kernel arguments after deleted reads, int/float promotion, `item()` and `detach()`, one plan across AdamW steps | numpy; finite differences; closed-form momentum |
| `test_w06_opt.py` | 15 | 0.5 s | every `Opts` combination on GEMM, cache blocking (`Opts.tile`), shared reduce loops, unrolling, layout pass, fused attention loop depth | numpy; structure of the C |
| `test_w07_gpu.py` | 7 + 1 skipped | 0.4 s | GPU lowering through the CPU emulator, launch dims, CUDA text | numpy via emulator; text checks |
| `test_w09_models.py` | 3 | 8.2 s | LayerNorm, causal masking, a tiny GPT learning | numpy; invariance; loss curve |

@diagram 17-test-pyramid

The pyramid is the classical one, adapted. The base is cheap and *sharp*: when an IR or algebra test fails,
it names the function that is wrong. The apex is expensive and *broad*: when the GPT test fails, the bug can
be anywhere, but nothing else would have noticed that, say, AdamW's update has the wrong sign. You need both,
and the middle layers exist to make apex failures rare. A useful rule of thumb from building `uopc`: every
time an apex test caught a bug, a smaller test was written that catches it lower down. That is the last arrow
in the triage diagram of B.7.

## B.2 Oracles

A test needs an *oracle*: an independent way to know the right answer. A compiler is unusual in having
several good ones.

**numpy.** The workhorse. Every kernel-level test lowers a graph, runs it, and compares with the numpy
expression for the same math. The whole harness is one helper:

@include tests/helpers.py#run_shaped

**Finite differences.** For autodiff, the oracle is calculus itself: `(f(x + ε) − f(x − ε)) / 2ε`, computed in
float64 with numpy, entry by entry:

@include tests/test_w05_tensor.py#numgrad

Note the tolerances in `test_gradients` (`rtol=2e-2, atol=2e-3`): the oracle has truncation error `O(ε²)` and
the system under test computes in f32. Too tight and the test is flaky; too loose and a missing factor of 2
slips through. A factor-of-2 bug is far outside these tolerances; a sign error is outside any tolerance.

**The interpreter.** Week 1's "run by rewriting" (`symbolic.run`) is an independent implementation of the
same semantics as the compiled C. For closed programs (no buffers) you can compare three independent paths
through the compiler. A scratch harness:

```python
# scratch: evaluate one closed .uir program three independent ways
import sys; sys.path.insert(0, "tests")
import numpy as np
from uopc.uir import parse
from uopc.symbolic import run, to_python
from uopc.tensor import Tensor
from helpers import run_shaped

g = parse("""%0 = stack 1.0, -2.0, 3.0
%1 = stack %0, %0
%2 = permute %1, (1, 0)
%3 = reduce %2 : op=max pop=1
%4 = mul %3, %3""")
paths = {
  "rewrite (symbolic.run)":   lambda: np.array(to_python(run(g))),
  "rangeify + C (lower)":     lambda: run_shaped(g, []),
  "kernelize + opt (Tensor)": lambda: Tensor(g).numpy(),
}
for name, f in paths.items(): print(f"{name:26s}", f().tolist())
```

```text
rewrite (symbolic.run)     [1.0, 4.0, 9.0]
rangeify + C (lower)       [1.0, 4.0, 9.0]
kernelize + opt (Tensor)   [1.0, 4.0, 9.0]
```

When the three disagree, the disagreement *localizes* the bug: if `run` and `lower` agree but the Tensor path
does not, look in `kernelize` or `opt`. This is differential testing, and it is the core of B.7's bisection.

**The emulator.** A GPU kernel renders twice: once as CUDA, once with the plain C renderer, where `GLOBAL` and
`LOCAL` ranges become ordinary loops. The index arithmetic is identical, so the emulated kernel tests the
mapping without a GPU:

@include tests/test_w07_gpu.py#test_gpu_gemm_emulated

Its blind spot is anything that depends on threads running *concurrently*: barriers and shared memory
(NOTES 25). That is why `smem_gemm_ir` is only render-tested on the reference machine.

**Invariants instead of values.** Some properties are easier to state than to compute. `test_attention_is_causal`
changes future tokens and asserts that past outputs do not move; `test_tiny_gpt_learns` asserts the loss falls
by 4x; `test_uir_roundtrip_and_syllabus_example` asserts `parse(to_uir(g)) is g`. Hash consing makes that last
one strong: `is` means the two graphs are structurally identical, node for node.

**Floating-point tolerance.** The default `CFLAGS` include `-fassociative-math`, so gcc may reorder float
additions (NOTES 20). Results are deterministic for a given binary but differ from numpy's summation order at
the level of f32 rounding. Never compare floats bitwise across implementations; use `assert_allclose` with
tolerances scaled to the reduction length.

## B.3 Fuzzing movement chains

Movement ops compose in combinatorially many ways, and each composition is an index map that rangeify must get
exactly right. The suite does not enumerate cases; it *generates* them:

@include tests/test_w04_rangeify.py#test_random_movement_chains

Six random steps from {permute, shrink, expand-a-new-axis, flatten}, applied in lockstep to a UOp and a numpy
array, then compared. Five seeds run in milliseconds and exercise hundreds of index-map compositions. Three
design points carry over to any fuzzer you write:

1. **Mirror the operation in the oracle step by step.** The generator builds the numpy reference alongside the
   graph, so there is no separate "compute the expected answer" logic to get wrong.
2. **Seed everything** (`np.random.default_rng(seed)`, parametrized over seeds). A failure is then a test ID you
   can rerun, not a flake.
3. **Bias toward edge cases.** The shrink step picks `begin` anywhere and `end` above it, so length-1 axes appear
   often; expand only grows size-1 axes, which is exactly the case broadcasting bugs live in.


## B.4 Property tests for the symbolic engine

`symbolic.py` has about forty rewrite rules, and each one is a claim: "this rewrite never changes the value".
Example tests check the claims you thought of. A *property test* checks the claim itself, on expressions you
did not think of. The property for `simplify` is: for every integer expression `e` over loop variables and
every assignment of values to those variables, `e` and `simplify(e)` evaluate to the same number, and that
number lies within `[vmin(e), vmax(e)]`. Here is one written without any library (scratch):

```python
# scratch: test_symbolic_props.py
"""Property tests for uopc.symbolic: simplify() must not change the value of any
integer expression, and vmin/vmax must bound every value it can take."""
import random, pytest
from uopc.uop import UOp, Ops, range_
from uopc.symbolic import simplify, vmin, vmax, run

def rand_expr(rng: random.Random, rngs: list[UOp], depth: int) -> UOp:
  if depth == 0 or rng.random() < 0.25:
    return rng.choice(rngs) if rng.random() < 0.7 else UOp.const(rng.randint(-8, 16))
  a, b = rand_expr(rng, rngs, depth - 1), rand_expr(rng, rngs, depth - 1)
  op = rng.choice(["add", "mul", "div", "mod", "lt"])
  if op == "add": return a + b
  if op == "mul": return a * rng.randint(-3, 5)          # keep it affine-ish, like real indices
  if op == "div": return a // rng.randint(1, 9)
  if op == "mod": return a % rng.randint(1, 9)
  return (a < b).where(a, b)

def evaluate(e: UOp, env: dict[UOp, int]) -> int:
  return run(e.substitute({r: UOp.const(v) for r, v in env.items()})).arg

@pytest.mark.parametrize("seed", range(200))
def test_simplify_preserves_value_and_bounds(seed):
  rng = random.Random(seed)
  rngs = [range_(rng.randint(1, 12), k) for k in range(3)]
  e = rand_expr(rng, rngs, 4)
  s = simplify(e)
  for _ in range(10):
    env = {r: rng.randrange(r.src[0].arg) for r in rngs}
    want = evaluate(e, env)
    assert evaluate(s, env) == want, f"seed {seed}: {e} -> {s} at {env}"
    assert vmin(e) <= want <= vmax(e), f"seed {seed}: bounds [{vmin(e)}, {vmax(e)}] miss {want}"
```

```text
........................................................................ [ 36%]
........................................................................ [ 72%]
........................................................                 [100%]
200 passed in 1.29s
```

Three details make it work. The generator produces the expressions rangeify actually produces: sums of loop
variables times small constants, floor-divided and taken modulo small constants, with an occasional `WHERE`.
The evaluator is *not* the system under test's C path; it substitutes constants and lets constant folding
(`exec_alu`, the reference semantics) compute the value. And it checks bounds too, because a wrong `vmax`
silently turns a correct `floordiv_i32` into a wrong `/` in the renderer (NOTES 4).

Does it find anything? Not on the reference code. So break the code and see. This mutant forgets one check in
`_divides` (scratch, applied by monkeypatching, never by editing `uopc/`):

```python
# scratch: a plausible bug in symbolic._divides
def m_divides():
  import uopc.symbolic as s
  from uopc.uop import Ops
  old = s._divides
  def divides(u, c):                                   # forget to check that the coefficient divides c
    if u.op is Ops.MUL and u.src[1].op is Ops.CONST: return u.src[0] * (u.src[1].arg // c)
    return old(u, c)
  s._divides = divides
```

With the mutant applied:

- all 13 week-1 tests **pass**, as do weeks 2 and 3;
- the property test fails **21 of 200** seeds, each with the offending expression in the message, for
  example `seed 175: ... -> UOp(Ops.CONST, arg=0)` where the right answer was `6`;
- the full suite fails **36 tests**, all from week 4 on: movement ops, Tensor ops, gradients, the GPU emulator
  and the GPT, every one of them far from `_divides`. (Before `kernelize` was fixed to pass only the buffers a
  kernel still reads, the same mutant made a generated kernel write out of bounds and the process died with
  `Fatal Python error: Aborted` inside `test_tiny_gpt_learns`.)

The property test is the only check that fails *in the right place*. That is what the pyramid is for.

> **Key idea:** a rewrite rule is a theorem. Example tests check instances; property tests check the theorem
> on instances you did not choose. For an algebra engine, write the property test first.

## B.5 Testing the tests: mutation testing

How do you know a test suite would catch a bug? Put the bug back and run it. *Mutation testing* does exactly
that: apply a small, plausible defect (a *mutant*), run the suite, and see whether any test fails (the mutant
is *killed*) or not (it *survives*). The bugs in `book/NOTES.md` are perfect mutants: they really happened.

The harness re-introduces each one by monkeypatching before pytest imports the tests, so the repository is
never edited (scratch, abridged to four mutants):

```python
# scratch: mutants.py -- re-introduce bugs from book/NOTES.md, then run pytest
import sys, pytest

def m_argkey():
  import uopc.uop as m
  m._argkey = lambda arg: arg                      # NOTES 1: forget the type in the cache key

def m_trunc_div():
  import uopc.render_c as rc
  old = rc._binop
  def binop(op, a, b, ua, ub, d):                  # NOTES 4: C '/' and '%' are floor division, right?
    from uopc.uop import Ops
    if op is Ops.FLOORDIV: return f"({a}/{b})"
    if op is Ops.FLOORMOD: return f"({a}%{b})"
    return old(op, a, b, ua, ub, d)
  rc._binop = binop

def m_f64_fold():
  import uopc.symbolic as s
  old = s.truncate
  s.truncate = lambda d, x: x if d.name == "f32" else old(d, x)   # NOTES 5: fold floats in f64

def m_const_like():
  from uopc.uop import UOp
  UOp.const_like = lambda self, v: UOp.const(v, self.dtype)       # NOTES 13: forget to broadcast

MUTANTS = {k[2:]: v for k, v in globals().items() if k.startswith("m_")}

if __name__ == "__main__":
  name, *args = sys.argv[1:]
  if name != "none": MUTANTS[name]()
  sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "-rN", *args]))
```

Results against the reference suite as it stood with 76 tests (all measured; "+ new test" refers to the tests in B.4 and B.6):

| Mutant (NOTES) | Suite result | Killed by |
|---|---|---|
| none (baseline) | 76 passed, 1 skipped | — |
| `_argkey` ignores type (1) | **66 failed** | `test_hash_consing` and almost everything downstream |
| naive `inline_call` (2) | 76 passed | **survived**; now killed by `test_inline_call_does_not_enter_nested_bodies` |
| recursive `graph_rewrite` (3) | 1 failed | `test_deep_graph_no_recursion_error` |
| `/` and `%` for floor ops (4) | 1 failed | `test_floordiv_rendering_negative` |
| f32 folding in f64 (5) | 2 failed | `test_const_fold_semantics`, `test_folding_matches_c_semantics` |
| accumulators at kernel top (9) | 76 passed at `-O3`; **3 failed at `-O0`** | `test_gemm_opts_correct[opts1]`, `[opts3]`, `[opts5]` |
| `const_like` without broadcast (13) | 8 failed | `test_reduce_max_and_nested`, 4 gradient tests, training, emulated softmax, tiny GPT |
| AdamW constants as Python floats (17) | 76 passed, **77 s instead of 11 s** | **survived**; now killed by `test_adamw_steps_reuse_one_plan` |
| `_divides` without divisibility check | 36 failed | the new property test (21/200 seeds) is the only week-1 failure |

Three survivors teach three lessons.

**The CALL scoping mutant survives** because `run` rewrites bottom-up: by the time the outer CALL is inlined,
the inner one is already gone, so the naive substitution never sees a nested body. NOTES 2 says as much
("bottom-up rewriting also inlines inner calls first"). The bug is real for any *other* caller of
`inline_call`, so the fix is a unit test that calls it directly (B.6).

**The accumulator mutant survives at `-O3`** and dies at `-O0`. Declaring `float acc0;` at the top of the
function makes it shared between OpenMP threads: a data race, which is undefined behavior. At `-O3` gcc keeps
`acc0` in a register inside each thread's loop, so the race never touches memory and every answer is right.
At `-O0` every access goes to the shared variable:

```python
# scratch: alloc_check.py -- the accumulator mutant on a 256x256 GEMM with a THREAD axis, 20 runs
import sys; sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import mutants; mutants.m_alloc_top()          # moves every ALLOC to the top of the kernel
import numpy as np
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.runtime import Program
from test_w06_opt import gemm_graph
from helpers import buf
M, N, K = 256, 256, 256
src = render_c(lower(gemm_graph(M, N, K), buf(2, M * N), Ctx(opts=Opts(thread=0))), "k")
a, b = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32)
bad = 0
for t in range(20):
  c = np.zeros(M * N, np.float32); Program(src, "k")(a, b, c)
  bad += not np.allclose(c.reshape(M, N), a @ b, atol=1e-3)
print("wrong results in", bad, "of 20 runs")
```

With the default flags it prints `wrong results in 0 of 20 runs`; with `CFLAGS="-O0 -fopenmp"`:

```text
wrong results in 20 of 20 runs
```

The kernel it runs declares the accumulator outside the parallel loop:

```c
void k(float* restrict data0, float* restrict data1, float* restrict data2) {
  float acc0;
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 256; ridx0++) {
    int alu1 = (ridx0*256);
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 256; ridx2++) {
        float val2 = acc0;
        float val3 = data0[(alu1+ridx2)];
        float val4 = data1[((ridx2*256)+ridx1)];
        acc0 = (val2+(val3*val4));
      }
      float val5 = acc0;
      data2[(alu1+ridx1)] = val5;
    }
  }
}
```

The lesson: **run the suite under more than one set of compiler flags.** Optimizers hide undefined behavior
as often as they expose it. Because `CFLAGS` is part of the kernel-cache key, the two configurations never
share binaries.

**The AdamW mutant survives** because recompiling every step is not *wrong*, only slow. The suite's wall time
went from 11 s to 77 s and nothing failed. Performance bugs need tests that assert on *how* the answer was
computed. Those are next.

## B.6 Structural tests: assert on the code, not the answer

Some properties of a compiler are about its output program, not its output values: how many loops, which
loop a value is computed in, how many kernels, how many compiled plans. The suite already has a few:

@include tests/test_w03_loops.py#test_loop_invariant_code_is_hoisted

@include tests/test_w06_opt.py#test_upcast_shares_reduce_loop

Counting `"for ("` in generated C is crude, and that is fine: it is cheap, it is stable as long as the
renderer's output format is stable, and it encodes a design decision ("an upcast GEMM shares one reduce loop")
that a refactor could silently undo. Here are the tests the suite *was* missing, each tied to a NOTES entry. They were written for this appendix
as scratch files, run against mutants (below), and then adopted into the real suite. They are included
here straight from `tests/`:

@include tests/test_w01_uops.py#test_inline_call_does_not_enter_nested_bodies

@include tests/test_w01_uops.py#test_cast_inf_to_int_is_not_folded

@include tests/test_w01_uops.py#test_float_constants_do_not_reassociate

@include tests/test_w06_opt.py#max_loop_depth

@include tests/test_w06_opt.py#test_fused_attention_reduces_are_not_nested

@include tests/test_w05_tensor.py#test_adamw_steps_reuse_one_plan

On the reference code all of them pass. Against the mutants, the scratch versions said (paths shortened):

```text
== mutant: call_scope   (test_extra.py)
FAILED .../test_extra.py::test_inline_call_does_not_enter_nested_bodies
1 failed, 2 passed in 0.02s
== mutant: adamw_consts (test_structure.py)
.../test_structure.py:33: AssertionError: assert 7 == 4
FAILED .../test_structure.py::test_adamw_steps_reuse_one_plan
1 failed, 1 passed in 1.30s
```

The plan-count test turns a two-minute slowdown into a one-line failure with a number in it: three more steps
made three more plans. The loop-depth test pins the structure that NOTES 12's O(N³) bug destroyed. Its strength is limited:
re-creating that bug by simply disabling the `simplify` call at `RESHAPE` did *not*
reproduce the old nesting in the current code (other rules now clean up the indices), so this test is a guard
on the property, not a demonstrated catch.

> **Key idea:** correctness tests catch wrong answers; structural tests catch right answers computed the wrong
> way. A compiler course is graded on both, and most performance regressions are invisible to the first kind.

## B.7 Debugging: a triage procedure

@diagram 17-debug-flow

**Crashes while building a graph.** `UOpError` is raised in the constructor of the first malformed node, and
its message names the op, the dtypes and the shapes (the full list is in Appendix A.3). Read the traceback up
to *your* line: that is where the bad node was built, not where it was used. `LinearizeError` means the
schedule deadlocked (it prints the open loops and the first stuck nodes with their target scopes) or that a
loop was opened and never closed (`RANGE without END: loops [...] are never closed`). A
`CompileError` from gcc includes the whole C source with line numbers, so the gcc message and the source line
are side by side.

**Wrong numbers.** Shrink first, then bisect.

1. *Shrink the input.* The smallest shape that still fails (often 2×3) makes the generated C readable. Make
   sizes distinct (2, 3, 5) so a swapped axis cannot hide behind a square shape.
2. *Turn off optimizations.* `OPT=0` removes the layout pass and all `Opts`. If the answer becomes right, the bug
   is in `opt.py` or in how `rangeify` applies upcasts and unrolls.
3. *Bisect the pipeline* with the three paths of B.2: `symbolic.run` (no rangeify), `helpers.run_shaped`
   (rangeify, no kernelize), `Tensor(...).numpy()` (everything). The first path that disagrees with numpy
   contains the bug.
4. *Read the C* (`DEBUG=4`, or `render_c(...)` directly). Check one output element by hand: pick indices, follow
   the index expression, and see which input elements it reads.
5. *Change the compiler flags.* If `CFLAGS="-O0 -fopenmp"` changes the answer, suspect a data race or other
   undefined behavior (B.5).

**Too slow.** Measure before guessing. `DEBUG=2` prints every kernel's time and GFLOP/s; `DEBUG=3` runs each
kernel a second time, which separates first-touch page faults from real work. A kernel with a low GFLOP/s and a
deep loop nest is usually a fusion problem (recompute through an `EXPAND`, NOTES 15) or a layout problem (a
strided reduce axis, NOTES 19); read its C. If kernels are fast but steps are slow, the time is in Python. Here
is a cProfile of the first, compiling, training step of a small GPT (2 layers, `dim` 64, context 32, batch 8),
scratch:

```python
# scratch: where does the first (compiling) GPT training step spend its time?
import cProfile, pstats, sys
sys.path.insert(0, "examples")
import numpy as np
from gpt import GPT
from uopc.tensor import Tensor
from uopc.nn import AdamW, parameters

Tensor.manual_seed(0)
model = GPT(vocab=65, ctx=32, dim=64, heads=4, layers=2)
opt = AdamW(parameters(model), lr=1e-3)
x = Tensor(np.random.randint(0, 65, (8, 32)).astype(np.int32))
y = Tensor(np.random.randint(0, 65, (8, 32)).astype(np.int32))
def step():
  loss = model(x).reshape(-1, 65).cross_entropy(y.reshape(-1))
  opt.zero_grad(); loss.backward(opt.params); opt.step(loss)
cProfile.run("step()", "step.prof")
pstats.Stats("step.prof").sort_stats("cumulative").print_stats(14)
```

```text
         28342210 function calls (26311338 primitive calls) in 38.837 seconds
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.004    0.004   39.079   39.079 uopc/kernelize.py:178(realize)
        1    0.097    0.097   22.546   22.546 uopc/kernelize.py:111(make_plan)
      226    0.003    0.000   16.455    0.073 uopc/runtime.py:47(__init__)
      226    0.008    0.000   16.414    0.073 uopc/runtime.py:27(compile_c)
       97    0.002    0.000   16.307    0.168 subprocess.py:506(run)
```

and the same profile sorted by self time:

```text
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       97   16.166    0.167   16.166    0.167 {method 'poll' of 'select.poll' objects}
2108782/612269    3.785    0.000    4.796    0.000 uopc/rewrite.py:36(match)
   237157    1.490    0.000    2.613    0.000 uopc/uop.py:229(_infer)
     2120    1.419    0.001   12.010    0.006 uopc/rewrite.py:90(graph_rewrite)
    10329    1.339    0.000    2.574    0.000 uopc/uop.py:130(toposort)
   522574    0.999    0.000    6.627    0.000 uopc/uop.py:110(__new__)
```

(Paths shortened; rows for the profiler's own frames omitted.) Read it top down: the step built 226 kernels,
97 of which were new and went through gcc (16.2 s waiting on subprocesses); the other 129 were already in the
disk cache. Of the 22.5 s in `make_plan`, most is pattern matching (`UPat.match`, 2.1 million calls) inside
`graph_rewrite`. The next optimization is therefore not in gcc but in the matcher: fewer rules tried per node,
or memoizing more. That is the same reasoning that took GPT compilation from 163 s to 47 s (NOTES 22), and it is
project 28 in [Chapter 15](15-beyond.md).

**Freeze it.** Whatever the bug was, the last step is a test that fails without the fix and passes with it,
placed at the lowest layer of the pyramid that can see it. If the bug was in `symbolic.py` and you found it with
GPT, the regression test belongs in week 1.

## B.8 Style: "can it be simpler?"

The course grades style at 15%, with one question: *did you write slop?* Slop is code that works today and
costs you next week: dead branches, duplicated logic, clever one-liners nobody can check, comments that
narrate instead of explain. `uopc` is under 2,800 lines (`wc -l uopc/*.py`) for a compiler that trains a GPT; these are the rules
that kept it there.

1. **One idea per function, one screen per function.** `graph_rewrite` is 26 lines. The two longest functions
   are `_infer` (78 lines, but it is a table: one short case per op) and `linearize` (65 lines, because
   scheduling is the hardest problem in the compiler).
2. **Rules as data.** Algebra lives in `PatternMatcher` tables, one rule per line, each with a comment for the
   non-obvious ones ("ints only: float addition isn't associative"). Adding a rule is a one-line diff that can
   be reviewed in isolation and property-tested (B.4).
3. **Fail at construction, with a message.** Every `_check` in `_infer` names the op and the offending values.
   A good error message is worth ten debugging sessions.
4. **No recursion over graphs.** Graphs get deep (20,000 nodes in `test_deep_graph_no_recursion_error`);
   `toposort`, `graph_rewrite`, `substitute` and `call_params` are all iterative (NOTES 3).
5. **Comments say why, not what.** Compare `# C division truncates; it equals floor division when both sides
   are non-negative` (why) with `# divide a by b` (what). Every comment in `render_c._binop` is the first kind.
6. **No dead code.** An expression like `X if False else Y` is `Y` with extra steps. Two of them crept into
   the reference tests while it was being written and were removed in review; `grep -n "if False" -r tests uopc
   examples` now prints nothing. Grep for leftovers like this before every submission.
7. **Measure size.** `wc -l uopc/*.py` per week. If a module doubles in a week, ask why before moving on.
8. **Name things after what they mean in the course.** `Opts.upcast`, `AxisType.THREAD`, `STAGE`: the names in
   the code are the names in the lectures and in this book, so a reader can go from either to the other.

## B.9 A checklist per week

Before submitting each week, answer every question with a test, not with a belief.

| Week | Must have a test for |
|---|---|
| 1 | hash consing of `1`, `1.0`, `True`, `0.0`, `-0.0`; every `_infer` error you rely on; floor vs truncating division in folding; i32 wraparound; f32 rounding in folding; CALL with nested CALL; a 20,000-node graph; `.uir` round trip; a symbolic property test |
| 2 | loopless C for every ALU op against numpy; ALLOC + AFTER ordering; negative operands to floor ops in C; casts, including negative floats to int |
| 3 | GEMM and a convolution in the loop IR; sibling loops (`x - max(x)`); loop-invariant hoisting; an accumulator inside a parallel loop |
| 4 | each movement op alone; random chains of them (fuzz); REDUCE over 1 and all axes, ADD and MAX; STACK inside rangeify; PAD and FLIP; a real GEMM from views (and the syllabus non-GEMM) |
| 5 | each Tensor op vs numpy; a gradient check per autodiff rule; kernel counts for fusion decisions; the plan cache across training steps; a loss that falls |
| 6 | every `Opts` combination against the unoptimized result; structure (shared reduce loops, unrolled bodies); the layout pass; the suite under a second set of `CFLAGS` |
| 7–8 | every GPU lowering through the emulator; launch dimensions; coalescing (`.x` on the contiguous axis); CUDA text for shared memory and barriers; the real-GPU test where hardware exists |
| 9–10 | layer-level checks (LayerNorm, attention causality); a tiny model that learns; plan reuse across steps and across sampling; a step-time budget |

## B.10 Every pitfall and the test that catches it

The final table maps each entry in `book/NOTES.md` to the test that catches it. "Measured" means the bug was
re-introduced as a mutant (B.5) and the named test failed; "existing" means a suite test targets it but it
was not mutated here; "proposed" means no test exists yet and the cell says what to write.

| NOTES | Bug | Catching test | Status |
|---|---|---|---|
| 1 | hash-consing merges `1`, `1.0`, `True` | `test_hash_consing` (50 tests fail) | measured |
| 2 | CALL inlining rewrites nested bodies | `test_inline_call_does_not_enter_nested_bodies` (B.6); the older call test cannot see it | in suite |
| 3 | recursion depth on deep graphs | `test_deep_graph_no_recursion_error` | measured |
| 4 | C `/` is not floor division | `test_floordiv_rendering_negative`; `test_folding_matches_c_semantics` also pins `INT_MIN // -1` as not folded | measured |
| 5 | f32 constant folding in f64 | `test_const_fold_semantics`, `test_folding_matches_c_semantics` | measured |
| 6 | folding `(int)inf` | `test_cast_inf_to_int_is_not_folded` (B.6) | in suite |
| 7 | float reassociation | `test_float_constants_do_not_reassociate` (B.6) | in suite |
| 8 | linearizer deadlocks and misplaced inits | `test_gemm_loops`, `test_conv1d_loops`, `test_reduce_max_and_nested` (sibling loops), `test_loop_invariant_code_is_hoisted`, `test_missing_end_is_an_error` | existing |
| 9 | shared accumulators across threads | `test_gemm_opts_correct[opts1]`, `[opts3]`, `[opts5]`, **only under `CFLAGS="-O0 -fopenmp"`** | measured |
| 10 | `os.replace` across filesystems | run the suite with `UOPC_CACHE` on a different filesystem from the system temp dir | proposed |
| 11 | the syllabus GEMM is not a GEMM | `test_syllabus_gemm_is_not_a_gemm` | existing |
| 12 | unsimplified indices nest reductions | `test_fused_attention_reduces_are_not_nested` (B.6) | in suite (guard) |
| 13 | `const_like` without broadcasting | `test_reduce_max_and_nested` and 7 more | measured |
| 14 | no distribution, `//` chains in inner loops | a structural test that the inner loop of a GPT attention kernel contains no `/` or `%` | proposed |
| 15 | fusion recomputes through `EXPAND` | a test that `make_plan` of LayerNorm → matmul contains a STAGE and that the plan's total `flops` is within 2x of the unfused count | proposed |
| 16 | `substitute` wraps the original node | a test where a staged node contains another staged node, asserting both STAGEs survive `stage_broadcast_compute` | proposed |
| 17 | AdamW constants recompile every step | `test_adamw_steps_reuse_one_plan` (B.6) | in suite |
| 18 | sampling lengths recompile | the same plan-count pattern across 10 sampling steps | proposed |
| 19 | strided GEMM, threads slower | `test_layout_pass_stages_transposed_operand` (layout); GFLOP/s gates as a separate, opt-in benchmark run | existing + proposed |
| 20 | no vectorization without `-fassociative-math` | a benchmark gate; or assert the flag is in `runtime.CFLAGS` | proposed |
| 21 | layout pass stages the broadcast product | `test_layout_pass_stages_transposed_operand` (stages only the operand) | existing |
| 22 | Python overhead in compilation | a compile-time budget for one GPT step, run with a warm `.so` cache | proposed |
| 23 | uncoalesced `.x` mapping | assert in `render_cuda` output that `lidx0` (the `.x` index) is multiplied by 1 in the store address | proposed |
| 24 | `__shared__` declared inside a loop | `test_smem_gemm_renders` (checks the declaration; extend it to check it is the first line of the body) | existing |
| 25 | barriers cannot be emulated | `gpu.emulate` now refuses kernels with a `BARRIER` (`NotImplementedError`); `test_cuda_smem_gemm_on_gpu` runs them, skipped without a GPU | existing |

Bugs found while this book was written, after NOTES was closed, got regression tests of their own:
`test_run_c_checks_buffer_dtypes` (a numpy array of the wrong dtype or size passed to `run_c`),
`test_deleted_reads_do_not_shift_kernel_arguments` (a read removed by simplification shifted the remaining
kernel arguments, so a kernel wrote into its input), `test_python_float_promotes_int_tensors`, and
`test_item_before_backward_and_detach`.

Nine NOTES bugs were re-introduced as mutants. The reference suite killed five on its own (1, 3, 4, 5, 13),
one only under different compiler flags (9), two survived until a new test was added (2, 17), and one could
not be re-created in the current code (12). That is a good suite, and it is not a complete one. Neither is yours. Mutation testing is
how you find out which.

## Further reading

- The pytest documentation (docs.pytest.org): parametrization, fixtures, `--durations`, `-k` and `-x`.
- K. Claessen, J. Hughes. *QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs.* ICFP 2000.
  The origin of property-based testing; Hypothesis is its best-known Python descendant.
- X. Yang, Y. Chen, E. Eide, J. Regehr. *Finding and Understanding Bugs in C Compilers.* PLDI 2011 (Csmith).
  Differential testing of compilers at scale.
- Y. Jia, M. Harman. *An Analysis and Survey of the Development of Mutation Testing.* IEEE Transactions on
  Software Engineering, 2011.
- The tinygrad test suite (github.com/tinygrad/tinygrad, `test/`), for how a production ML compiler organizes
  unit, op-level and model-level tests.
