# 32. Executing SSA, Destroying Phis, and Allocating Registers

A verified optimized graph still needs an execution strategy. This chapter implements three:
an SSA interpreter for debugging, a constrained teaching machine for register allocation, and a
native C backend. Each exposes different errors and delegates different responsibilities.

## 32.1 Phi evaluation is simultaneous

On entry to a block, select each phi operand using the predecessor actually taken. Read all selected
values before writing any phi result. For a loop that swaps a and b, sequentially assigning `a=b`
and then `b=a` loses the old a. A dictionary of pending results followed by one environment update
preserves the parallel semantics.

The interpreter bounds-checks accesses before indexing so Python's negative-index convention cannot
leak into the language. It normalizes loaded and stored integers to i32 and executes instructions
in source-effect order. A block-step budget catches nontermination during experiments; this is a
debugging facility, not a proof of termination.

@include minicc/interpreter.py

**Experiment:** compile a loop initialized with a=1 and b=2 that swaps them each iteration. Return
`10*a+b`. Even iteration counts must return 12, odd counts 21. This small program exposes phi-copy
bugs that ordinary sums rarely detect.

## 32.2 Phi destruction belongs on edges

An SSA phi in block merge says which assignment happens for each incoming edge. Lower it to copies
on those edges. Putting a predecessor-specific copy unconditionally at the end of a branching
predecessor can change the value on the wrong path. Putting all copies at the merge loses the identity
of the incoming edge.

The implementation creates an edge block for every incoming edge to a phi-bearing block, redirects
that branch or jump, saves all phi sources into fresh temporaries, then writes all destinations.
Saving every source is conservative but handles arbitrary copy cycles. A later parallel-copy scheduler
could emit acyclic copies directly and use one scratch value only when breaking a cycle.

This transformation leaves the graph in a non-SSA form: the same destination may be assigned on
several incoming edges. Re-run definite assignment and structural verification under that form.
Do not incorrectly apply the single-definition verifier after intentionally destroying SSA.

## 32.3 Liveness gives an interference graph

At each program point, two distinct live values need different locations unless a separate value-
equality argument permits sharing. Compute block live-in/live-out sets, then scan instructions
backward. For a definition d, connect d to every value live after its definition, remove d from
that live set, and add the instruction's operands.

Parameters are installed before entering the function. This implementation connects all scalar
parameters to one another, including unused ones, so initialization cannot overwrite a live parameter.
A better allocator could avoid binding dead parameters, but omitting both protections is incorrect.
Buffers remain separate typed resources and never receive scalar-register colors.

Color the interference graph greedily in descending degree order. Pick an available register color;
if none is available, assign a distinct spill slot. This policy is simple and deterministic but not
optimal. It does not perform coalescing, live-range splitting, rematerialization, or spill-slot reuse.
Those are subsequent compiler projects with measurable effects on this baseline.

## 32.4 Execute the allocation, not just inspect it

The teaching machine reads and writes the allocated physical locations. If the allocator assigns
interfering values the same register, the computation is actually corrupted and the source-oracle
test fails. A test that only checks the allocator's own graph would be weaker: the graph and allocator
could agree on the same incorrect assumptions.

Spilled values live in a frame and their reads and writes are counted. The teaching instruction set
permits spill-slot operands directly. A register-only hardware ISA would require explicit reload and
spill instructions, scratch-register management, stack layout, and calling-convention support.
The counts are algorithmic traffic, not measured CPU or GPU cycles.

@include minicc/machine.py

**Experiment:** run the same compiled function with register budgets 1, 2, 4, and 16. Check semantic
agreement before comparing spill slots and traffic. Then implement a copy-propagation or coalescing
pass and determine whether it reduces actual dynamic spill traffic, not only graph size.

## 32.5 Native C needs a declared numerical and memory contract

A naive C emitter prints signed `a+b`, but overflowing signed arithmetic is undefined in C. The
backend converts operands to uint32, computes modulo arithmetic there, and copies the resulting bit
pattern into int32 with `memcpy`. Comparisons still use signed operands. The minimum i32 literal is
emitted without depending on a positive 2147483648 having int type.

Every buffer parameter has a pointer and explicit length. Every access checks its signed index
before dereferencing. An error pointer communicates a trap to the wrapper, which raises an
`IndexError` after the native function returns. Earlier writes remain visible. Scalar results are
not mistaken for error codes; a legitimate result of zero remains distinguishable from failure.

The wrapper declares every `ctypes` argument type and return type. It accepts only aligned, writable,
contiguous rank-one int32 buffers. Rejecting a noncontiguous view avoids treating a logical stride
as physical adjacency. Writable buffers simplify the contract even for functions that only read;
relaxing that guard requires per-parameter effect information.

Phi assignments in C are again emitted with parallel-copy semantics on branch edges. All SSA scalar
names receive safe generated C identifiers, so source identifiers cannot inject generated code.
The original source is parsed structurally, never interpolated as C statements.

@include minicc/c_backend.py

## 32.6 What these backends establish

The native backend delegates instruction selection, scheduling, register allocation, assembly, and
linking to GCC. The register-machine backend exposes allocation but does not emit a physical ISA.
Together they make two useful boundaries visible instead of pretending the C emitter performs every
backend task.

Native execution has no interpreter step budget; do not run a possibly nonterminating test in-process
without a process timeout. The generated library and its temporary source remain owned by the
`NativeFunction` object for its lifetime. This is an experimental JIT, not a persistent executable
cache or a sandbox for adversarial programs.

**Acceptance:** execute a program through mutable IR, SSA, optimized SSA, the allocated machine,
and native C. Compare returns, final buffers, and failures against independent expected behavior.
Keep the generated C when diagnosing a native-only mismatch, and identify the first stage that
changes the specified semantics.
