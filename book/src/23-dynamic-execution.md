# 23. Symbolic Shapes, Guards, Mutation, and Graph Capture

A program with a different sequence length should not necessarily require a new executable. But
removing sizes from a cache key without changing the generated program is incorrect. Dynamic
execution requires a consistent contract across shapes, proofs, generated loops, and dispatch.

## 23.1 A symbolic dimension is a constrained value

Represent a dimension N as an expression with constraints such as `1 <= N <= 4096`. Separate its
identity and bounds from its current runtime binding. For a contiguous `(B,N)` tensor, address
`b*N+j` includes runtime N; its stride is dynamic too. An allocation may use a maximum capacity
while logical sizes remain smaller.

For `0 <= j < N`, intervals alone may say `j <= 4095` and `N >= 1`, which cannot prove `j < N`.
That fact comes from the loop relation. Keep facts scoped to the region where they hold. Never
reuse an inner-loop assumption outside that loop. Check overflow of index arithmetic before using
identities that assume mathematical integers.

**Lab:** implement bounded dimensions, runtime bindings, and dynamic indexing for vector operations
and one reduction. Execute lengths 1, 7, 31, and 64 with one plan. Log the actual plan count and inspect
that generated loop bounds depend on runtime arguments. Reject bindings outside declared bounds.

## 23.2 Guards and specialization

A guard is an executable check for a compile-time assumption. Suppose a fast kernel assumes
contiguous f32 input, aligned addresses, and `N % 16 == 0`. Its dispatch contract must check all
relevant assumptions, then select a generic fallback when they do not hold. A shape guard alone
cannot establish strides or alignment.

A useful cache key includes graph semantics, target architecture, dtypes, layout class, numerical
policy, compiler version, and specialization constraints. Runtime values belong in the key only
when the generated code specializes on them. Guard failure may trigger another compiled variant;
control variant growth to avoid compiling one executable per request.

**Lab:** build a guarded dispatcher with a generic implementation and one divisible-by-16 fast path.
Call it with repeated and alternating shapes and strided inputs. Record compile count, guard failures,
and which path executed. Verify every output against the eager reference. Corrupt one guard deliberately
and demonstrate a test that detects the resulting wrong dispatch.

## 23.3 Mutation and views

For `y = view(x); x[...] = new_value; use(y)`, whether y sees the update depends on the language's
aliasing semantics. A pure expression graph cannot represent this merely by rebinding the Python
variable x. Introduce explicit effects or transform mutation into functional values while preserving
aliases and ordering.

One approach versions memory: a store consumes memory state M0 and produces M1, and later loads
consume M1. Another functionalizes mutation by generating new values and rebuilding affected views.
Both require an explicit treatment of overlapping slices and external aliases.

**Lab:** define mutation semantics before implementation. Test disjoint slices, overlapping slices,
a view surviving an assignment, and a parameter update followed by a new forward pass. Ensure host
and device copies cannot silently diverge. Tie in-place reuse to an alias and liveness proof.

## 23.4 Capturing a host-language program

Lazy tensor APIs capture operations because every tensor method records a node. Capturing an eager
host program also encounters branches, loops, scalar conversions, side effects, and unsupported calls.
Tracing one branch records that execution; it does not automatically capture the other branch.
A branch on tensor data needs representable control flow, a graph break, or a correct fallback.

Start with a restricted function tracer that supports tensor arithmetic and fixed control flow.
Document exactly which operations are captured and reject unsupported effects rather than dropping
them. Then add shape-dependent branching with guards. General Python bytecode capture is a later
project with a much larger semantic surface.

**Lab:** trace a two-layer model and compare eager and captured forward/gradient results. Exercise
both outcomes of a shape-dependent branch. Add a tensor-data branch and demonstrate the chosen
fallback. Record graph breaks and recompilations as part of the result, not only kernel time.

## 23.5 A true KV cache

Allocate K and V storage per layer with shape `(batch, heads, capacity, head_dim)`. At position p,
write the new key and value into slot p, then attend only to `0..p`. The cache needs a length and
ownership policy; reset it between independent sequences and reject capacity overflow.

A fixed-capacity masked implementation can be correct while doing work proportional to capacity
on every token. A dynamic loop over the populated prefix avoids this wasted work. Its GEMM shape,
parallel reduction strategy, and compilation behavior still need evaluation.

**Exit gate:** compare cached and uncached logits at every decoding step, verify one reusable plan
where intended, and plot latency against populated length and capacity separately. Demonstrate cache
reset, batch isolation, and an out-of-capacity error.

## 23.6 Worked lab: one C kernel for many lengths

A dynamic length can be data without changing a function's code. The lab passes N in an int32 buffer
and generates a C loop whose bound reads that buffer. Contiguous f32 vectors take the compiled path;
other floating-point vectors use a NumPy fallback preserving their dtype and logical layout.
Lengths outside the declared bound are rejected rather than silently extending a compiled assumption.

```bash
.venv/bin/python -m labs.dynamic
```

Expected counters after the example are `variants built=1`, `compiled calls=5`, and
`layout fallbacks=1`. “Variants built” counts Program construction in this instance; the runtime's
on-disk cache can make the actual C compiler invocation unnecessary. This distinction matters when
measuring cold and warm compilation.

@include labs/dynamic.py

A noncontiguous slice `x[::2]` has a larger physical stride. Passing its pointer to this C kernel
would read adjacent physical values instead of adjacent logical values. The layout guard prevents
that error. A general implementation could instead pass a runtime stride; then the generated address
would become `x[i*stride]`, and vectorization assumptions would need their own guards.

**Experiment:** pass a reversed view, a float64 vector, and lengths 1, 7, 31, and 64. Check outputs
and counters. Then remove the contiguity check in a scratch copy and observe a failing strided-input
test. Do not turn the failure into an unconditional copy without measuring the copy's cost.

**Next implementation:** replace the fixed kernel with lowering from symbolic tensor expressions.
Keep expression identity, bounds, target, numerical policy, and layout specialization in the plan
contract. This small dispatcher does not provide general symbolic shape inference or Python capture.


## 23.7 Continue into a dynamic tensor graph compiler

[Chapter 33](33-dynamic-tensor-compiler.md) goes beyond the fixed-kernel dispatcher. Its `tensorjit`
compiler emits runtime-sized loops for tensor expressions, checks symbolic dimension bindings,
fuses pointwise producers into consumers, and runs the result with planned temporary storage.
It implements a bounded subset; general Python capture and arbitrary symbolic relations remain
separate projects.
