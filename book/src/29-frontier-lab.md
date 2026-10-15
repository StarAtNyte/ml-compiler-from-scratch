# 29. A Frontier-Lab Preparation Map

This track targets ML compiler, GPU kernel, and training/inference systems engineering. It does not
claim to cover every role in a research lab, and completing a book does not guarantee hiring or
research success. The objective is demonstrable independence: turn an unfamiliar workload into a
correct implementation, diagnose its limits, improve it, and make the result reproducible.

## 29.1 Common foundation and specialization

Finish the prerequisite labs and core compiler, then work through Chapters 21–28. Choose one primary
specialization for deep projects while retaining working knowledge of the others.

| Track | Core depth | Evidence of independence |
|---|---|---|
| ML compiler | SSA/effects, lowering, symbolic proofs, layouts, scheduling, cost models | Nontrivial verified pass in a larger compiler; legality and performance analysis |
| GPU kernels | Memory hierarchy, matrix instructions, asynchronous pipelines, numerics | Forward/backward kernel suite with hardware profiles and diverse shape results |
| Training systems | Autodiff, optimizers, memory, parallelism, checkpointing, reliability | Resumable distributed training experiment with scaling and failure analysis |
| Inference systems | Dynamic shapes, KV memory, quantization, batching, tail latency | End-to-end serving experiment under a finite cache and latency budget |
| Hardware/compiler codesign | Target models, backend legality, layouts, sparse workloads | Executable backend or target port, microbenchmarks, and model-versus-device analysis |

Read related mathematical material when a lab demands it: linear algebra for tensor transformations,
calculus for derivatives, probability for stochastic training, numerical analysis for precision,
and statistics for experiments. For systems depth, study concurrency, virtual memory, networking,
storage, and distributed failure models. A gap exposed by a project becomes required study, not an
excuse to extrapolate from a toy implementation.

## 29.2 Distributed training beyond two workers

Chapter 24 establishes parallel algebra. Extend it with gradient accumulation, data/tensor/pipeline
partitioning, optimizer-state sharding, and activation recomputation. For pipeline parallelism, draw
a microbatch timeline and account for idle stages, activation lifetime, and weight-version consistency.
For sharding, state which worker owns each parameter, gradient, and optimizer state at each phase.

Build a per-step memory budget: parameters, gradients, optimizer states, activations, communication
buffers, allocator fragmentation, and temporary workspace. Build a communication budget by collective,
message size, topology, and frequency. Measure overlap; asynchronous APIs alone do not demonstrate it.

**Lab:** implement two partitioning strategies on a bounded model. Compare with a single-device
reference, report strong-scaling efficiency, and account for global batch and optimizer semantics.
Restore after an injected worker failure from a consistent checkpoint. Verify dataset position,
optimizer state, random state, and parameter consistency. Simulated ranks establish logic; hardware
experiments establish scaling.

## 29.3 End-to-end diagnosis

A model slowdown may come from data loading, host dispatch, compiler misses, kernels, transfers,
collectives, or synchronization. Capture a timeline before optimizing individual kernels. Derive a
critical path and estimate the maximum benefit of the proposed change. If a kernel occupies 10% of
step time, even eliminating it entirely cannot yield a 2× end-to-end speedup.

**Lab:** create one deliberate bottleneck in each layer: input pipeline, compilation, launch overhead,
memory traffic, compute, and communication. Diagnose each from evidence rather than knowing which
bottleneck was injected. Maintain a short investigation record: observation, hypothesis, experiment,
result, and next action.

## 29.4 Production reliability and review

A production runtime needs resource ownership, bounded caches, cancellation, timeouts, useful errors,
and cleanup after partial failure. Asynchronous errors can surface long after their cause; attach
kernel and graph context to diagnostics. Validate imported model formats and allocation sizes before
trusting them. Define how the system behaves when memory is exhausted or a device disappears.

**Lab:** inject allocation failure, corrupted cache metadata, invalid model dimensions, interrupted
checkpoint writes, and failed collectives. Verify clean failure or recovery with no silently wrong
outputs. Review an optimization for semantic assumptions, numerical impact, concurrency, cache
compatibility, and regressions—not just code style.

## 29.5 Research literacy and original experiments

Read a paper in three passes: identify the problem and claimed contribution; reconstruct the algorithm
and assumptions; examine the evidence and missing comparisons. Separate algorithmic changes from
hardware, precision, compiler, and workload differences. Implement the smallest experiment capable
of falsifying the central claim before attempting a full reproduction.

A useful new result may be a negative result, a counterexample, a better measurement, or a narrower
algorithmic improvement. State the hypothesis in advance, use held-out workloads for tuning decisions,
and report uncertainty and failed runs. Keep datasets, seeds, revisions, environment, and commands
sufficient for another engineer to reproduce the conclusion.

**Lab:** reproduce one bounded result, identify one limitation, and test an extension that addresses
it. Write a short paper-style report: question, method, baselines, results, ablations, limitations,
and reproduction instructions. Obtain technical review from another person when possible and resolve
their strongest counterargument with an experiment.

## 29.6 Portfolio and readiness gates

Produce three substantial artifacts: a compiler change with semantic tests; a real-device kernel
study with matched baselines; and an end-to-end training or inference system with memory and failure
analysis. At least one should work inside an established external codebase at a pinned revision.
The projects may share a model but must establish distinct claims.

You are ready to attempt independent work when you can explain unfamiliar IR, minimize a wrong-code
bug, reason about numerical differences, find a measured bottleneck, design a legal optimization,
and communicate a reproducible result under review. Continue learning through work with larger
systems and experienced reviewers. The curriculum supplies the preparation and the tests of progress;
real projects supply the remaining depth.


## 29.7 What you can execute now

The [worked compiler analysis](21-compiler-foundations.md#216-worked-lab-why-phi-operands-live-on-edges),
[attention forward/backward](22-gpu-engineering.md#227-worked-lab-derive-and-execute-tiled-attention-backward),
[guarded C execution](23-dynamic-execution.md#236-worked-lab-one-c-kernel-for-many-lengths),
[buffer planner](24-runtime-and-scale.md#246-worked-lab-buffer-reuse-with-observable-lifetimes),
[numerical methods](26-numerical-analysis.md#265-worked-lab-mergeable-variance-and-quantization-error),
and [irregular operation references](28-irregular-and-heterogeneous.md#285-worked-lab-duplicate-indices-change-the-operation)
are executable baselines with independent checks. Run their modules from the repository root and
read the corresponding tests before extending them.

The next milestone for each is integration: SSA construction and machine lowering; an actual GPU
attention kernel; symbolic tensor compilation; runtime buffer binding; model-level precision
validation; and parallel irregular operations. The later research and scale assignments remain
substantial work. Keep the [completion ledger](20-syllabus-contract.md#206-completion-ledger) honest
about that distinction while building your portfolio.
