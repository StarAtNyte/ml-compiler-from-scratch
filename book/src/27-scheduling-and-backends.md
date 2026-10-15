# 27. Scheduling, Autotuning, and Backend Construction

A compiler must choose a legal implementation, predict whether it is useful, and lower it to the
machine. This chapter extends the scalar analysis in Chapter 21 and GPU experiments in Chapter 22
into a compiler-driven search rather than a collection of hand-selected kernels.

## 27.1 Iteration domains and dependencies

For GEMM, an iteration domain is `{(i,j,k): 0<=i<M, 0<=j<N, 0<=k<K}`. Access maps send an iteration
to `A[i,k]`, `B[k,j]`, and `C[i,j]`. A schedule maps iterations to execution order. Tiling introduces
outer tile coordinates and inner coordinates, with guards for partial tiles.

Dependences constrain schedules. A producer must precede its consumer unless a transformation
replaces the dependence with an equivalent mechanism. Reduction parallelization introduces partial
accumulators and a combine step; it changes floating-point order. For affine bounds and accesses,
integer-set reasoning can establish useful properties. Data-dependent indices require other methods
or conservative fallback. A polyhedral representation is powerful within its supported domain,
not a proof engine for arbitrary programs.

**Lab:** represent a two-dimensional iteration domain and affine access maps. Implement interchange
and tiling, then reject an interchange that violates a dependence in an in-place recurrence. Print
both the legality reason and resulting schedule.

## 27.2 Layout as a compiler decision

Logical tensor shape does not determine lane ownership, register arrangement, shared-memory swizzle,
or vector access width. A matrix-instruction operand needs an exact mapping from logical elements to
participating lanes and registers. A layout conversion may be register shuffles, shared-memory
traffic, or a combination. Its cost can erase an arithmetic speedup.

Track layouts explicitly through the kernel IR. Require a verifier to check that consumers accept
the producer layout or that a legal conversion is present. Estimate conversion bytes and instructions
alongside compute. Test layout maps for injectivity where each element needs a unique owner and
account for intentional replication separately.

**Lab:** define two lane layouts for a small tile, implement their conversion, and round-trip random
values. Count communication and compare with a kernel that keeps one layout throughout. Investigate
one existing compiler's printed layout IR at a pinned revision.

## 27.3 Search and cost models

Define schedule parameters with legality constraints: tile sizes, warps, vector widths, pipeline
stages, and shared-memory budgets. Prune impossible configurations before compilation. Record compiler
failures and correctness failures separately from slow valid schedules. Keep a generic fallback.

An analytical model can estimate arithmetic intensity and resource limits. A learned model can rank
candidates from features and measurements, but its quality depends on the training distribution.
Split evaluation by workloads or shape families, not merely repeated timings of the same kernels.
Include compilation and search cost when judging whether tuning pays off.

**Lab:** compare a heuristic, random search, and a ranked search on a fixed tuning budget. Evaluate
on held-out shapes. Report best latency, total tuning time, compilation failures, and variance.
Cache winners with target, compiler version, dtype, layout, and numerical policy in the key.

## 27.4 Direct backend essentials

A backend legalizes unsupported operations, selects instructions, schedules them, allocates registers,
and emits executable code or an intermediate format accepted by another toolchain. Start with a
small virtual target so each responsibility is visible. Define registers, loads/stores, branches,
calling convention, and instruction encoding before writing an emitter.

Liveness and interference constrain register reuse. Spills need stack or local-memory slots and
additional instructions. Phi elimination introduces edge copies, whose parallel-copy semantics must
be preserved even when values form a swap cycle. Instruction scheduling must honor data and effect
dependences; latency hiding cannot violate the target's memory model.

**Lab:** lower the scalar compiler into a tiny register machine, allocate a deliberately small number
of registers, resolve phi copies, and execute the resulting program in an emulator. Add a disassembler
and compare pre/post-allocation behavior. Then inspect a real backend's code for the same example.

## 27.5 Compilation as an observable service

Track time and memory per pass, IR sizes, cache hit rates, and diagnostic context. Incremental
compilation should reuse results only when their dependencies remain valid. A parallel compiler must
keep shared caches and contexts thread-safe and preserve deterministic artifacts where required.
Persistent executable caches need atomic writes, compatibility checks, and a policy for untrusted
or corrupted artifacts; do not load arbitrary cached machine code solely because its filename matches.

**Exit gate:** legal schedule transformations, verified layouts, a held-out tuning evaluation, a
small functioning backend, and a compiler profile identifying where compilation time is spent.
