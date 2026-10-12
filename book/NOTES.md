# Development notes: real bugs and lessons from building uopc

These happened while writing the reference compiler. Use them in "Pitfalls" sections; they are true.
Numbers are from an Intel i7-13620H laptop (6P+4E cores, 16 threads), gcc 15, Python 3.12.

## Week 1 — IR, rewriting, symbolic
1. **Hash-consing collisions.** Python says `1 == 1.0 == True` and `0.0 == -0.0`, and they hash equal. A cache
   keyed on `(op, src, arg)` merged `CONST 1` (i32), `CONST 1.0` (f32) and `CONST True` (bool) into one node.
   Fix: `_argkey` in `uopc/uop.py` adds the type, and uses `repr` for floats.
2. **CALL scoping.** PARAMs are hash-consed, so `PARAM slot=0` in an inner CALL body is the *same object* as
   in the outer body. Naive substitution when inlining the outer call rewrote the inner body too. Fix:
   `call_params` / `inline_call` never descend into a nested CALL's `src[0]`. (Bottom-up rewriting also inlines
   inner calls first.)
3. **Recursion depth.** A chain of 20,000 ADDs blew Python's recursion limit in a recursive rewriter. Fix:
   `graph_rewrite` and `toposort` are iterative with explicit stacks (test: `test_deep_graph_no_recursion_error`).
4. **Floor vs truncation.** FLOORDIV/FLOORMOD use floor semantics (Python), C `/` truncates. Benefit: the rewrite
   rules `(a*c + b)//c -> a + b//c` and `(a*c+b)%c -> b%c` are *unconditionally* true for floor division (not
   for truncation when b<0). Cost: the renderer must emit `/` only when bounds analysis proves the numerator is
   non-negative (`vmin(x) >= 0`), else call `floordiv_i32`. Real indexing code is almost always provably >= 0.
5. **f32 constant folding** must round through `struct.pack('f')`, otherwise folding in Python's f64 gives
   different answers than the C program would (`16777216.0 + 1.0` stays `16777216.0` in f32).
6. **Casting inf/nan to int is UB in C**: the folder refuses to fold it (`exec_alu` returns None).
7. Float reassociation `(x+1.0)+2.0 -> x+3.0` is not exact; restricted to ints. (`x + 0.0 -> x` is kept and
   documented as not exact for -0.0.)

## Week 2-3 — rendering and loops
8. The linearizer took four tries (`uopc/linearize.py`):
   a. "Emit a node when its scope == the open loops": a GEMM node indexing `A[i*K+k]` has scope {i,k}, but k
      is nested inside j, so the stack is (i,j,k) and the node could never be emitted -> deadlock.
   b. "Emit when scope is a subset of open loops": works, but correctness relied on scheduling luck (an
      accumulator init could be emitted inside the reduce loop and reset it every iteration).
   c. Sibling loops: in `x - max(x)` the output loop j and the reduce loop k are siblings inside i, and the
      j-loop nodes read the k-loop's result through its END. Opening j first and then k inside it reset
      the accumulator wrongly / deadlocked. First fix attempt ordered loops by *any* data dependency, which
      wrongly made `k*N` (pure arithmetic of k) force k outermost.
   d. Final design: close each scope under "encloses" (if ranges a<b ever share a scope, b is inside a, so
      every node using b also sits inside a) — scopes become paths in the loop tree and exact matching works;
      sibling order is decided only by END dependencies.
9. Accumulators must be declared in the deepest scope common to their users, so each OpenMP thread gets a
   private copy (`ALLOC` placement in `linearize`).
10. `os.replace` across filesystems fails (`/tmp` -> `~/.cache`): compile in a temp dir *inside* the cache dir.

## Week 4 — rangeify
11. **The syllabus's own example is not a GEMM.** `gemm_256x256.uir` reshapes A to (256,256,1) and B to
    (256,1,256), multiplies and reduces the last axis: that computes `A * rowsum(B)`. We have a test proving it
    (`test_syllabus_gemm_is_not_a_gemm`). Correct version: A -> (N,1,K), B^T -> (1,N,K), reduce last axis.
12. **Unsimplified indices change the loop structure.** Rangeify of a RESHAPE produces `((i*N + j)//N) % N`
    — mathematically `i`, syntactically mentions `j`. The reduce lowering puts the accumulator in the loops its
    index *mentions*, so max/sum reductions in fused attention were nested inside the j loop: O(N^3).
    Fused attention at N=512 took 1787 ms; simplifying indices at each RESHAPE made it 7 ms (250x).
13. `UOp.const_like` must broadcast to the node's shape, or `-x` on a shaped UOp fails type checking.
14. Distributing multiplication over addition (`(a+b)*c -> a*c + b*c`, ints) was needed before div/mod rules
    could see individual terms; without it GPT attention kernels had `((x/4096)%4)` chains in inner loops.

## Week 5 — kernels, autodiff
15. **The fusion/recompute trap.** Fusing everything between reductions sounds free, but an elementwise
    result that is EXPANDed into a reduce (LayerNorm output feeding a matmul) is recomputed for every output
    column. A GPT forward/backward step spent 374 ms in kernels; staging broadcast computations
    (`stage_broadcast_compute`) brought it to 125 ms. Only stage if the recompute is expensive (cost model
    `_fused_work`): staging softmax's tiny `1/sum` added two useless kernels.
16. **`substitute` vs bottom-up wrap.** `graph.substitute({x: STAGE(x)})` wraps the *original* x, so anything
    inside x that should also be staged is not. Fix: `UOp.wrap` rebuilds bottom-up and wraps the rewritten node.
17. **JIT cache misses from Python floats.** AdamW's bias corrections `1/(1-b1^t)` change every step; baked in as
    CONSTs they make every step's graph structurally different and recompile everything. Passing them as
    1-element tensors keeps one compiled plan.
18. Sampling from the GPT pads the context to a fixed length so every forward pass has the same graph.

## Week 6 — performance
19. GEMM 1024^3 with B read down columns (stride N): naive 1.9 GFLOP/s, *threads made it slower* (1.1) —
    every thread streams a column, touching a new cache line and 4 KB page per iteration. Register tiling
    alone barely helped. With B^T (k contiguous for both operands): naive 8.0, threads 35, 4x4 tile + threads
    114 GFLOP/s; numpy/OpenBLAS 132. See `book/outputs/bench_gemm_1024.txt`.
20. gcc will not vectorize a float reduction without `-fassociative-math` (it would reorder additions).
    That flag is the single biggest switch: 17 -> 170 GFLOP/s in one experiment. We keep NaN/inf semantics.
21. The layout pass first staged a whole (128,784,128) broadcast product (21 ms kernel). Fixes: push movement
    ops through elementwise ops to the leaves (`push_views`) so view chains end at real data, and only stage
    operands strictly smaller than the reduce input.
22. Python overhead: `UOp.__eq__/__hash__` written in Python and Enum's Python `__hash__` made compilation of
    the GPT graph take 163 s for 12 steps; using object identity (C) and memoizing `simplify` -> 47 s.
    First-step GPT compile is now ~9 s, then cached.

## Weeks 7-8 — GPUs
23. CUDA's `.x` dimension is the fastest-varying across a warp: map the *last* (contiguous) axis to `.x` or
    loads/stores are uncoalesced.
24. `__shared__` arrays are declared at kernel top (linearizer pins SHARED ALLOCs to the empty scope).
25. Barriers can't be emulated by running threads one after another; the CPU emulator only covers kernels
    without shared memory. The reference machine had no working NVIDIA driver, so CUDA code is render-tested
    only; `tests/test_w07_gpu.py::test_cuda_smem_gemm_on_gpu` runs when a GPU is present.

## Found while writing the book (all fixed, each with a regression test)
The chapter authors ran every snippet they printed, and that alone surfaced these. "Run what you publish"
is a testing strategy.
26. **Kernel arguments shifted when a read was simplified away.** `Tensor([1,2,3]) * 0` simplifies to a
    constant, so the kernel's C signature lost its input parameter, but the plan still passed the input array
    first: the kernel wrote its output *into the input*. Fix: bind only the buffer slots that survive lowering.
    (`test_deleted_reads_do_not_shift_kernel_arguments`)
27. **f32 folding rounded the result but not the inputs**: `16777217.0 + 2.0` folded to 16777220, C gives
    16777218 (the literal itself rounds). Fix: `exec_alu` rounds f32 inputs. (`test_folding_matches_c_semantics`)
28. **Reassociation built out-of-range i32 constants**: `(x*65536)*65536 -> x*4294967296` (a `long` in C).
    Fix: wrap folded constants to int32; the result is `0`, which is exactly right mod 2^32.
29. **`INT_MIN // -1` was folded**; in C it traps. A folder must never be more defined than the target.
30. **A RANGE without END rendered unbalanced C**; gcc's syntax error was the first symptom. Now a
    `LinearizeError`. (`test_missing_end_is_an_error`)
31. **`run_c` accepted float64 arrays for `float*`** and the kernel reinterpreted bytes. Now a `TypeError`.
32. **`.uir` literals in WHERE took the condition's dtype (bool).** Fix: type them from the branches.
33. **Python float scalars didn't promote int tensors**: `Tensor([1,2,3]).mean()` was 0. (`test_python_float_promotes_int_tensors`)
34. **`loss.item()` before `loss.backward()` gave zero gradients**: realizing rebound the loss to a buffer,
    cutting the graph. Fix: realized tensors keep `_graph`. **`detach()` didn't detach**: now
    `STAGE(arg="detach")`, which autodiff stops at and kernelize strips. (`test_item_before_backward_and_detach`)
35. **"Fused" flash attention still wrote an N*N buffer**: `stage_broadcast_compute` staged P. `fuse_reduce`
    now disables it — and the honest result is that recompute makes it ~3x slower than unfused on a CPU.
36. **`match_gemm` matched a sum of squares** (it checked shapes, not operands). **`emulate` crashed with
    a TypeError on barriers** instead of saying why it can't run them.
37. Temp buffers were reallocated every step (first-touch page faults: 9.7 ms vs 1.1 ms); plans now keep them.
