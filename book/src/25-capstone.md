# 25. From Course Completion to Independent Engineering

Expertise means handling unfamiliar programs, locating errors, and defending performance claims.
Use this chapter as an assessment and a repeatable research workflow. There is no fixed number of
weeks that guarantees mastery; progress depends on prior background, available hardware, and depth.

## 25.1 A capstone with a bounded question

Choose one project with a specific hypothesis. Examples: an asynchronous pipeline improves long-K
GEMM on a named GPU; symbolic shapes eliminate recompilation for a bounded family of sequence lengths;
a memory planner reduces GPT peak temporary allocation; an attention backward avoids quadratic saved
state. State the input domain and numerical contract before coding.

Deliver the implementation, correctness suite, baseline, benchmark harness, raw results, analysis,
and a short reproduction guide. Include at least one negative result or limitation. A hardware port
must execute on the target; a paper reproduction must identify which algorithmic features and reported
experiments it actually reproduces.

## 25.2 Correctness beyond examples

Generate random small legal graphs, evaluate them with an independent reference, and compare compiled
results. Bias the generator toward edge cases: singleton dimensions, irregular tiles, negative integer
division, shared subexpressions, views, repeated reductions, and deep graphs. Restrict operations to
well-defined inputs or explicitly check the error contract.

When a failure appears, save its seed and shrink the program by removing operations, reducing shapes,
and simplifying constants while preserving the failure. A three-node counterexample is more useful
than a thousand-node crash. Add the minimized case as a regression test.

Competitive testing can be useful: contributors submit valid programs intended to expose compiler
bugs. Validate each test with an independent reference before counting failures. Keep correctness
scoring separate from performance scoring and cap execution resources for each submitted case.

**Lab:** build a differential fuzzer and reducer. Introduce a known bad rewrite and show the fuzzer
finds it. Then test an unmodified compiler and report findings, including “no failure found” honestly.

## 25.3 Reading and changing a larger compiler

Pick a codebase aligned with the question: a tensor compiler for graph scheduling, a kernel compiler
for layout lowering, or LLVM/MLIR for analysis and intermediate representations. Pin a revision.
Start from a tiny running example and trace one operation from public API to generated kernel.
Read tests and print pass outputs before trying to understand the entire repository.

Make one bounded change: fix a minimized correctness case, add a missing legal lowering, or improve
a measured schedule. Run the project's relevant checks and document both favorable and unfavorable
workloads. An upstream-ready patch is a useful artifact even if you choose not to submit it.

## 25.4 Closed-book oral assessment

Explain these with a whiteboard and a short program, without reading the implementation:

1. Why does hash consing simplify common-subexpression sharing, and why is effect ordering different?
2. Which index transformation implements transpose followed by reshape, and when is a copy needed?
3. Why can integer and floating-point rewrites require different legality rules?
4. Where must a phi appear in a branch or loop, and what does dominance establish?
5. When does fusion save memory traffic but increase compute or register pressure?
6. Why can a correct barrier kernel fail under partial blocks?
7. Why can a lower-occupancy GEMM be faster, and which measurements distinguish explanations?
8. Derive the online softmax recurrence and its backward gradient.
9. What belongs in a compiled-plan key, and which assumptions require guards?
10. Under which ordering and alias conditions can two temporaries share storage?
11. How does gradient averaging change with unequal local batch sizes?
12. Which evidence distinguishes correct small-GPT training from competitive large-model training?

For each answer, produce a counterexample to an incorrect simplification. If you cannot, revisit the
corresponding lab before adding more features.

## 25.5 Final acceptance report

| Dimension | Required evidence |
|---|---|
| Semantics | Written operation, type, shape, effect, and numerical contracts |
| Correctness | Independent references, gradient checks, minimized regressions |
| Compiler understanding | Pass-by-pass trace and legality argument for one nontrivial optimization |
| GPU understanding | Actual device execution, profiler evidence, synchronization validation |
| Performance | Matched baselines, diverse shapes, raw samples, compile/runtime separation |
| Runtime | Memory accounting, cache/guard behavior, checkpoint resume |
| Scale | Clearly bounded model and device configuration actually exercised |
| Maintainability | Small reviewable modules, useful diagnostics, reproducible commands |

A completed report supports concrete claims about the work you performed. Keep expanding its scope
through new architectures, workloads, and external contributions rather than treating “hero” as a
permanent certification.
