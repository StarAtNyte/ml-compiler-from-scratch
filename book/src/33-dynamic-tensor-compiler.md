# 33. A Dynamic Tensor Compiler with Fusion and Planned Memory

The guarded square kernel in Chapter 23 showed a dispatch contract, but its program was fixed.
`tensorjit` now compiles a tensor expression graph into float32 C loops with symbolic dimensions,
pointwise fusion, materialization boundaries, and actual temporary-buffer reuse. It is a second
executable compiler path, designed to connect the earlier isolated labs into one runtime.

```bash
.venv/bin/python -m tensorjit
.venv/bin/python -m pytest tests/test_tensorjit.py -q
```

A matrix multiplication, bias, ReLU, and row sum run for several `(batch,inner,width)` shapes using
one compiled library. The example emits two kernels: a matrix product and a reduction that inlines
its bias and ReLU producers. It does not recompile when a runtime dimension changes within bounds.

## 33.1 Symbols are identities with constraints

A `Dim("N",1,64)` represents one dimension in `[1,64]`. Reusing N in multiple input positions asserts
that their actual dimensions agree. Two symbols with different names are not assumed equal merely
because they have equal bounds. Symbols with the same name but inconsistent bounds are rejected.
This makes shape assumptions inspectable rather than relying on an accidental example shape.

Broadcasting aligns trailing axes. Equal dimensions match; a static dimension of one repeats along
the other extent. Otherwise graph construction fails. Matrix multiplication requires rank two and
identical reduction dimensions. These are deliberately conservative proofs: runtime-disjunctive
broadcasting, symbolic affine equalities, zero-length dimensions, and arbitrary-strided inputs are
not yet supported.

Nodes are immutable, and shared subexpressions are shared by object identity. The compiler does
not yet hash-cons distinct structurally equal nodes. Reusing a node expresses intended sharing;
constructing an equal-looking expression twice may duplicate work. That is a useful experiment for
a future common-subexpression pass.

@include tensorjit/graph.py

## 33.2 Materialization and fusion must agree

Walk the graph topologically and count uses. Inputs are external buffers. Constants are inline
literals. Reductions and matrix products become materialization boundaries. A pointwise node is
also materialized when it has several consumers or is requested as an output. Other pointwise
nodes are recursively inlined into their consumer's scalar expression.

This is a heuristic, not a cost model. Materializing a multi-use node avoids recomputation but adds
memory traffic and a launch. Inlining it might still win when the computation is cheap or cache
reuse is good. Keep `fuse=False` as a correctness and measurement baseline.

For `sum(relu(matmul(X,W)+bias), axis=1)`, matmul is one stage. The next stage iterates output rows,
reduces over columns, and reads `matmul_result[row,column] + bias[column]`, applying ReLU before
accumulating. It never allocates separate bias-add or ReLU matrices.

For `(X+row)*column`, index maps express broadcasting by selecting zero for a repeated singleton
axis. Transpose reverses matrix coordinates. These are logical mappings into a producer, so an
inlined transpose does not require an intermediate transpose buffer.

## 33.3 Render dynamic loops and wide addresses

Each kernel receives the input buffers it actually reads, one output buffer, and the dimension
bindings. Loop bounds refer to runtime dimension arguments. Flattened offsets use int64 arithmetic;
using int32 for a product of dimensions could overflow even when each dimension fits int32.
Allocation checks bound the size of materialized tensors before a native call.

The C emitter uses ordinary sequential float32 accumulation, disables contraction, and avoids
fast-math. ReLU follows `fmaxf(x,0)` semantics: a NaN input selects the numeric zero operand. This
choice is tested explicitly against `np.fmax`, not silently compared with a different NaN policy.
BLAS matrix products can still differ in rounding because they use different accumulation orders.

@include tensorjit/compiler.py

**Trace:** print `compiled.source` for the example and mark which graph nodes correspond to a
buffer, a loop, an address expression, and an inline scalar expression. Count multiplication and
addition work separately from intermediate bytes written. Then disable fusion and compare source
and kernel count before timing either implementation.

## 33.4 Guard inputs before allocating outputs

The runtime checks the complete input-name set, dtype, rank, contiguity, alignment, static extents,
symbol bounds, and equal-symbol bindings. Guard failure occurs before native work. A mismatched
reduction dimension cannot be repaired by accidentally reusing a prior binding.

Input bindings, or checked explicit metadata for shape-only graphs, determine every output and temporary size. Python integer multiplication computes
sizes without silent machine overflow, and the configured byte budget rejects excessive payload
and workspace requests. The byte accounting covers logical input/output payload plus the planned
arena extent; Python object overhead, compiled code, and up to 63 extra arena-alignment bytes are
outside that accounting. It is a local allocation guard, not a process memory limit.

The runtime compiles one library at construction. Calls with compatible shapes allocate new output
and temporary storage but reuse that executable. Compilation count means library builds for this
object; there is no persistent cross-process cache. Calls serialize around the object's diagnostics;
this is not a multi-stream or concurrent-serving runtime.

## 33.5 Derive lifetimes from the fused schedule

After fusion, the graph's original edges no longer fully describe temporary consumption. A reduction
may directly read a distant materialized producer through several inlined expressions. Each rendered
kernel therefore records its actual buffer dependencies. Lifetime starts at its producer kernel and
ends at the last kernel whose rendered expression reads that buffer.

For three matrix products followed by a reduction, a is consumed by b and b by c. The first matrix's
slot can be reused for c after b finishes, but a and b cannot share storage: b's kernel reads a while
writing b. The planner uses inclusive intervals and the strict reuse condition from Chapter 24.

Requested outputs are retained separately. They cannot enter the reuse pool, and a later call
allocates fresh outputs so it cannot overwrite values returned by an earlier call. An input requested
as an output is copied into owned output storage. The arena base is aligned to 64 bytes, and each
planned offset respects the same alignment contract.

@include tensorjit/runtime.py

**Experiment:** compile a chain of three 8×8 matrix products and a reduction. Three independent
matrix temporaries would require 768 payload bytes. The planner uses a 512-byte arena because the
third matrix reuses the first matrix's slot. Retain the first matrix as an additional output and
check that no later kernel or invocation changes it. The test suite evaluates both the numerical
results and the actual runtime memory report.

## 33.6 A measured baseline, including its losses

Run the benchmark as a module so imports resolve from the repository root:

```bash
.venv/bin/python -m examples.bench_tensorjit --sizes 16 32 64 --repeat 10 --output book/outputs/tensorjit.json
```

The checked-in run uses a three-product chain, bias, ReLU, and row sum. It records raw end-to-end
call samples, separate compilation time, environment metadata, numerical error, kernel count, and
planned temporary bytes. These are CPU measurements that include guards, planning, allocation,
Python dispatch, and native execution. They are not isolated kernel timings.

| Square size | Unfused median ms | Fused median ms | Kernels before/after | Planned temporary bytes |
|---|---|---|---|---|
| 16 | 0.275 | 0.193 | 6 / 4 | 2,048 |
| 32 | 0.307 | 0.230 | 6 / 4 | 8,192 |
| 64 | 0.993 | 0.828 | 6 / 4 | 32,768 |

These particular samples show a fusion benefit while remaining much slower than the NumPy expression
on this machine. The implementation has naive matrix multiplication and substantial per-call Python
work. This is a useful baseline with identifiable bottlenecks, not competitive GEMM. Timing spread
and machine conditions matter; rerun before making a performance claim.

In this chain, fusion reduces total temporary payload produced but does not reduce peak arena bytes:
the matrix products already need two overlapping buffers. Reporting only intermediate allocation
count would conceal that distinction. Compilation in the recorded run took roughly 166 ms unfused
and 132 ms fused; those costs must be amortized separately from steady-state calls.

The raw report is `book/outputs/tensorjit.json`; the harness is included below so the measurements
are reproducible rather than detached prose.

@include examples/bench_tensorjit.py

## 33.7 Integration tests and the next bottleneck

Tests compare fused and unfused results with independent NumPy expressions across dynamic and
irregular shapes, all supported reduction axes, broadcast rows and columns, shared nodes, scalar
inputs, and retained outputs. They also exercise guard rejection and byte-budget failure before
execution. The original compiler's tests remain a separate suite; passing this suite does not
establish integration with `uopc` autodiff or its CUDA device runtime.

The reverse-mode graph and reusable training executable are implemented next in
[Chapter 34](34-dynamic-training.md). Further tasks are explicit: cache shape-specific memory plans
under a bounded policy; tile/vectorize matmul;
add more general symbolic relations; and lower the same graph to a real accelerator. Measure those
changes against this baseline and keep the guard, lifetime, and numerical contracts intact.
