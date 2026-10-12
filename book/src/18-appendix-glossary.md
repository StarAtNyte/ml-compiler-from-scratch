# Appendix C. Glossary and Quiz Bank

The first half of this appendix defines the vocabulary of the book, as `uopc` uses it. Where a term has a
broader meaning in the literature, the definition gives the general sense first and the `uopc` specifics
second, with the module where it lives. The second half collects quiz questions for every week of the course,
in the style of the in-class quizzes, with answers in collapsed blocks.

**Where this fits:** op-level details are in [Appendix A](16-appendix-reference.md); testing practice is in
[Appendix B](17-appendix-testing.md); the last appendix, [Appendix D](19-appendix-source.md), lists the full
source.

## C.1 Glossary

**Accumulator.** Scratch memory a reduction adds (or maxes) into. Rangeify creates one size-1 `ALLOC` per
reduction, initialized to the identity, updated inside the reduce loop, and read after it. The C renderer turns
a size-1 LOCAL accumulator into a plain scalar (`float acc0;`).

**`AddrSpace`.** The third inferred property of a UOp (after dtype and shape): `ALU` for values, `GLOBAL` for
buffers passed to a kernel, `LOCAL` for call-scope scratch memory, `SHARED` for GPU shared memory. `LOAD` and
`STORE` require a non-ALU source (`uopc/dtype.py`).

**`AFTER`.** `AFTER(mem, *deps)` is `mem`, but only once `deps` have executed. It is how an SSA graph, which
has no inherent order, expresses "read this memory after that write" or "this accumulator lives inside that
loop". It renders as nothing.

**All-reduce.** A collective operation in which every device contributes an array and every device receives
their sum (or another reduction). Data-parallel training all-reduces gradients. A ring all-reduce moves about
`2N` elements per device regardless of the device count.

**`ALLOC`.** Call-scope storage, `float a[n];` in C. Created by rangeify for accumulators or written by hand in
loop IR. The linearizer declares it in the deepest scope shared by all its users; SHARED allocs go at the top
of the kernel.

**ALU op.** An op that computes a value from values: `CAST`, `RECIP`, `EXP2`, `LOG2`, `SQRT`, `SIN`, `ADD`,
`MUL`, `MAX`, `CMPLT`, `CMPNE`, `FLOORDIV`, `FLOORMOD`, `WHERE` (`uopc.uop.ALU`). Elementwise on shaped values.

**Arithmetic intensity.** FLOPs performed per byte (or per element) moved from memory. Low intensity means
memory-bound. Register tiling raises it by reusing each loaded value many times.

**Autodiff (reverse mode).** Computing the gradient of a scalar with respect to many inputs by walking the
graph from the output back to the inputs, applying a local rule at each node and summing contributions where
paths merge. In `uopc`, `autodiff.gradient` builds a new *graph* for the gradient, which the compiler then
compiles like any other.

**`AxisType`.** The kind of a `RANGE`: `LOOP`, `REDUCE`, `UPCAST`, `THREAD`, `GLOBAL`, `LOCAL`. It decides how
the renderer emits the loop: a `for`, an OpenMP parallel `for`, or a GPU block/thread index.

**Bank conflict.** On a GPU, shared memory is split into banks; when threads of one warp access different
addresses in the same bank in one instruction, the accesses serialize.

**Barrier.** A point every thread of a GPU block must reach before any continues (`__syncthreads()`).
`Ops.BARRIER` in the loop IR; it cannot be emulated by running threads one after another.

**Beta reduction.** Replacing a function application by the function's body with arguments substituted for
parameters. `inline_call` performs it on `CALL` nodes, scoped so nested CALL bodies are untouched.

**bf16 (bfloat16).** A 16-bit float with f32's 8-bit exponent and a 7-bit mantissa: f32's range at reduced
precision. Not in `uopc`.

**Bounds analysis.** Computing, for every integer expression, an interval `[vmin, vmax]` guaranteed to contain
its value (`symbolic.bounds`). Drives div/mod simplification, the choice of `/` vs `floordiv_i32` in the
renderer, and constant-folding of comparisons.

**Broadcasting.** Making shapes compatible by adding leading size-1 axes (`RESHAPE`) and stretching size-1 axes
(`EXPAND`). The IR never broadcasts implicitly; the Tensor API and the `.uir` parser insert the ops.

**`BUFFER`.** Global, persistent, flat memory: a kernel argument. In the Tensor runtime a BUFFER UOp owns its
numpy array.

**`CALL` / `PARAM`.** A function application and its formal parameters. `CALL(body, *args)`; the body's PARAMs
are numbered by slot. Week 1 only: they are eliminated by inlining, never rendered.

**Canonical form.** A chosen representative among equivalent expressions, so equal things look equal. The
symbolic rules move constants to the right of commutative ops; hash consing then makes equal canonical forms
the same object.

**Coalescing.** When the threads of a warp access consecutive addresses, the hardware combines the accesses
into few memory transactions. Map the contiguous axis to `threadIdx.x` (NOTES 23).

**Common subexpression elimination (CSE).** Computing a repeated expression once. Hash consing gives it for
free: building the same subexpression twice returns the same node.

**Confluence.** A rewrite system is confluent if the order in which rules are applied does not change the
final normal form. Real rule sets, including `uopc`'s, are designed to be close to confluent but are not
proven so.

**Constant folding.** Evaluating an op whose inputs are all constants at compile time. `exec_alu` is the
reference semantics; it refuses to fold undefined behavior (casting `inf` to int).

**Data parallelism.** Splitting a batch across devices that each hold a full copy of the model, then
all-reducing gradients.

**`DEBUG`.** Environment variable controlling diagnostics in `kernelize.realize`: 1 schedule summary,
2 per-kernel timing, 3 warm re-run timing, 4 generated C.

**`DType`.** A data type record (`name`, `itemsize`, C type, numpy format): `void`, `bool`, `i32`, `f32`,
`f16`.

**Emulator.** `gpu.emulate`: rendering a GPU kernel with the C renderer, where `GLOBAL` and `LOCAL` ranges are
ordinary loops, so its index mapping can be tested on a CPU.

**`END`.** `END(x, *ranges)` closes loops: nodes that depend on it run after those loops finish.

**`EXPAND`.** A movement op that stretches size-1 axes; reading an expanded axis reads index 0. Its gradient is
a sum over the expanded axes.

**f16.** IEEE half precision: 5-bit exponent, 10-bit mantissa, maximum 65504. `_Float16` in the generated C,
`half` in generated CUDA.

**Finite differences.** Approximating a derivative by `(f(x+ε) − f(x−ε)) / 2ε`. The oracle for gradient tests.

**Fixed point.** A state that a transformation no longer changes. `graph_rewrite` rewrites until no rule
fires.

**Floor division.** Division rounding toward negative infinity (Python's `//`), as opposed to C's truncation
toward zero. `FLOORDIV`/`FLOORMOD` use floor semantics, which makes `(a*c + b) // c = a + b // c`
unconditional.

**Fusion.** Computing several ops in one kernel without writing intermediates to memory. Its cost is
*recompute* when a fused value is consumed through an `EXPAND`.

**`fuse_reduce`.** A kernelize mode in which `REDUCE`s are not materialization points, so reductions nest
inside each other (used for fused attention); it also skips `stage_broadcast_compute`).

**Guard.** A run-time check that the assumptions a compiled program was specialized on still hold (a shape, a
bound on a symbolic size). Failing a guard means recompiling or rejecting the input.

**Gradient check.** Comparing an autodiff gradient against finite differences.

**`graph_rewrite`.** The rewrite engine: bottom-up, greedy, to a fixed point, iterative, memoized by node
(`uopc/rewrite.py`).

**Hash consing.** Constructing structurally identical objects only once, so identity is structural equality.
`UOp.__new__` looks up `(op, src, _argkey(arg))` in a weak-value cache.

**Identity element.** The value a reduction starts from: 0 for `ADD`, `-inf` (or `INT_MIN`) for `MAX`.

**im2col.** Rewriting a convolution as a matrix multiply by gathering each receptive field into a row.
`Tensor.conv2d` builds the columns from `SHRINK`s.

**`INDEX`.** Indexing: on a value it selects a sub-array (popping leading axes); on memory it computes an
address that `LOAD` and `STORE` use.

**Interval arithmetic.** Computing bounds of an expression from bounds of its parts (`[a0,a1] + [b0,b1] =
[a0+b0, a1+b1]`). Sound but unable to see relations between variables (it cannot prove `j < n` inside a loop
to `n`).

**JIT cache (structural cache).** `kernelize._plans`, keyed by `_structure_key`: the graph's structure with
buffers anonymized. A training step with the same structure as the previous one reuses the compiled plan.

**Kernel.** One compiled function that reads buffers and writes one output buffer. A *plan* is a list of
kernels plus the temp buffers between them.

**Kernelize.** Cutting a shape graph into kernels at materialization points, and building the plan
(`uopc/kernelize.py`).

**KV cache.** In autoregressive decoding, the stored key and value vectors of all past tokens, per layer, so
each new token only computes its own.

**Layout pass.** `opt.layout_pass`: copying an operand whose reduce axis is strided into a contiguous temp
(a `STAGE`) before a reduction reads it, after pushing movement ops down to the data (`push_views`).

**Linearizer.** The pass that turns a graph into an ordered list of UOps respecting loop scopes
(`uopc/linearize.py`), by list scheduling over the loop tree.

**`LOAD` / `STORE`.** Read a value from memory / write a value to memory, through an `INDEX` address.

**Loop-invariant hoisting.** Computing an expression outside the loops it does not depend on. The linearizer
does it automatically: a node is emitted in the scope of its sources, not of its users.

**`lower`.** `rangeify.lower(root, out, ctx)`: turn one shaped value into a `SINK` of loops that store it into
a buffer.

**Materialization point.** A node whose value is written to memory, ending a kernel: every root, every `STAGE`,
and every `REDUCE` unless `fuse_reduce` is set.

**Mixed precision.** Storing values in a low-precision type (f16, bf16) and accumulating in a higher one (f32).

**Movement op.** `RESHAPE`, `EXPAND`, `PERMUTE`, `SHRINK`: ops that change how indices map to data without
computing anything.

**Mutation testing.** Measuring a test suite by injecting small defects (mutants) and checking that some test
fails for each.

**Normal form.** An expression no rule can rewrite further.

**OpenMP.** A C extension for shared-memory parallelism; `#pragma omp parallel for` splits a loop's iterations
across threads. Emitted for `THREAD` ranges.

**`OPT`.** Environment variable selecting the kernel optimizer in `tensor.py`: 0 none, 1 `HandOpt` (default),
2 `SearchOpt`.

**`Opts`.** The per-kernel schedule: `upcast` factors, a `thread` axis, `unroll` factor, `tile` (cache block)
sizes, and the GPU flag with `local` sizes (`rangeify.Opts`).

**`PatternMatcher` / `UPat`.** A list of (pattern, function) rewrite rules, indexed by root op; `UPat` matches
UOps structurally, binds names, and with a list `src` tries all orders of a commutative op's sources.

**`PERMUTE`.** A movement op that reorders axes. Its gradient is the inverse permutation.

**Placeholder buffer.** A `BUFFER` with a negative slot that `kernelize` substitutes for a materialized value
while building each kernel, later bound to a temp, input or output.

**Property test.** A test that checks a general claim on many generated inputs (for example, that `simplify`
preserves values) rather than on a few chosen examples.

**Rangeify.** Lowering a shape graph to scalar loop code by giving every output axis a `RANGE` and pushing the
index tuple down through every op (`uopc/rangeify.py`).

**`RANGE`.** A loop: its value is the loop counter, its source is the trip count, its arg is `RangeArg(idx,
kind)`. Loops nest in increasing `idx`.

**`ravel` / `unravel`.** Converting a multi-dimensional index to a flat offset in row-major order, and back.

**Recompute.** Computing the same value more than once because it was fused rather than materialized. The
trade-off of fusion; `stage_broadcast_compute` bounds it with a cost rule.

**`REDUCE`.** `REDUCE(x, (op, n))` reduces the last `n` axes with `ADD` or `MAX`.

**Register tiling (upcasting).** Computing a small tile of outputs per loop iteration, held in registers, so
each loaded input is reused across the tile.

**`RESHAPE`.** A movement op that changes the shape without changing the element count or order.

**Roofline model.** Attainable performance = min(peak FLOP/s, arithmetic intensity × memory bandwidth).

**Scope.** The set of loops a node must sit inside: every `RANGE` it depends on, minus those closed by an
`END` on the path. The linearizer closes scopes under "encloses" so each is a path in the loop tree.

**Shape graph.** A graph of shaped UOps (with movement ops and reductions), before rangeify.

**Shared memory.** Fast, per-block, programmer-managed GPU memory (`__shared__`). `AddrSpace.SHARED`.

**`SHRINK`.** A movement op that keeps a sub-box `[begin, end)` per axis. Its gradient is a zero-pad.

**SIMT.** Single instruction, multiple threads: the GPU execution model in which a warp of threads executes
one instruction stream on different data.

**`SINK`.** The root of a program: groups side effects.

**SSA.** Static single assignment: every value is defined once. A UOp graph is SSA by construction.

**`STACK`.** Creates a new leading axis from same-shaped sources. Rangeify turns it into a `WHERE` chain.

**`STAGE`.** "Materialize this into a temp buffer": a user- or pass-inserted materialization point. With
`arg="detach"` it is instead a stop-gradient (`Tensor.detach()`), removed by `kernelize` before scheduling.

**Symbolic shape.** A shape entry known only at run time, represented as an expression (usually a variable with
declared bounds) rather than an integer. Not in `uopc`; see Chapter 15.

**Tensor core.** GPU hardware that performs a small matrix multiply-accumulate (for example 16×16×16 with f16
inputs and f32 accumulation) per instruction, issued cooperatively by a warp.

**Term rewriting system.** A set of rules `lhs → rhs` applied to expression trees until none applies. The
basis of `uopc`'s optimizer and its week-1 interpreter.

**Termination.** A rewrite system terminates if no infinite sequence of rewrites exists. `graph_rewrite` has a
step limit as a safety net, not a proof.

**`THREAD`.** The `AxisType` of a loop run in parallel across CPU threads with OpenMP.

**Toposort.** An ordering of graph nodes with every node after its sources. `UOp.toposort` is iterative.

**UOp.** The IR node: `UOp(op, src, arg)`, immutable and hash-consed, with inferred `dtype`, `shape` and
`addrspace`.

**`.uir`.** The course's text interchange format: one SSA statement per line, parsed by `uir.parse` and
printed by `uir.to_uir`.

**Unroll.** Replicating a loop body so fewer iterations run; in `uopc`, `Opts.unroll` replicates the innermost
reduce body and combines the copies as a balanced tree.

**`UPCAST`.** The `AxisType` for a fully unrolled axis. In `uopc`, upcasting is done during rangeify by
instantiating the body per offset, so no UPCAST range is ever created.

**`vmin` / `vmax`.** The lower and upper bounds computed by bounds analysis.

**Warp.** A group of 32 GPU threads that execute in lockstep.

**`WHERE`.** Select: `WHERE(c, a, b)` is `a` where `c` is true, else `b`.

**`WMMA`.** Warp matrix multiply-accumulate: CUDA's API for tensor cores (`nvcuda::wmma`). `Ops.WMMA` is
reserved in `uopc`; tensor-core kernels are produced by pattern (`gpu.wmma_gemm`).

## C.2 Quiz bank

Questions are grouped by week. Answers refer to `uopc` and to measured numbers from `book/NOTES.md` where
relevant.

### Week 1: UOps, rewriting, symbolic

1. Why does `uopc`'s hash-consing cache key use `_argkey(arg)` rather than `arg`?
2. What is `-7 // 2` under `FLOORDIV`, and what does C's `-7 / 2` give?
3. Why is the rule `(x + c1) + c2 → x + (c1 + c2)` restricted to integers?
4. In `graph_rewrite`, why is the result of a rule that fired rewritten again?
5. `x` is an `i32` PARAM. What does `simplify((x*6 + 4) % 2)` return, and why is it valid for every `x`?
6. Why are `toposort` and `graph_rewrite` written with explicit stacks?
7. What are the shape and value of `INDEX(STACK(STACK(1,2,3), STACK(4,5,6)), CONST 1)`?
8. When inlining an outer CALL, why must the substitution not descend into a nested CALL's body?
9. What does folding `16777216.0 + 1.0` in f32 give, and why must the folder round through f32?

<details>
<summary>Answers, week 1</summary>

1. Python treats `1 == 1.0 == True` and `0.0 == -0.0` as equal with equal hashes, so a cache keyed on `arg`
   would merge an `i32`, an `f32` and a `bool` constant into one node. `_argkey` adds the type and uses `repr`
   for floats (NOTES 1).
2. `-4` (floor). C gives `-3` (truncation toward zero). That is why the renderer uses `floordiv_i32` unless
   bounds prove the numerator non-negative.
3. Float addition is not associative: `(x + 1.0) + 2.0` and `x + 3.0` can round differently, so the rewrite
   would change results (NOTES 7).
4. The result may itself match rules (or create new matches with its parent's context), and the engine
   promises a normal form: everything is rewritten to a fixed point.
5. `CONST 0`. `6x + 4 = 2·(3x + 2)`, and with floor semantics `(a·c + b) % c = b % c` holds for all integers,
   so the whole sum is divisible by 2.
6. Graphs get deep: a chain of 20,000 `ADD`s exceeded Python's recursion limit in a recursive rewriter
   (NOTES 3, `test_deep_graph_no_recursion_error`).
7. Shape `(3,)`, value `[4, 5, 6]`: `INDEX` pops the leading axis.
8. PARAMs are hash-consed, so `PARAM slot=0` in the inner body is the *same object* as in the outer body; a
   naive substitution would rewrite the inner function's parameter too (NOTES 2).
9. `16777216.0`: `2^24 + 1` is not representable in f32. Folding in Python's f64 would give `16777217.0`, a
   different answer from the compiled C program (NOTES 5).

</details>

### Week 2: memory, rendering, runtime

10. What does `AFTER(mem, dep)` express, and why does an SSA graph need it?
11. Exactly when does the C renderer emit `/` for a `FLOORDIV`?
12. Which UOps does the renderer inline as expressions, and which always get a named variable?
13. Name one error that address-space inference catches and dtype/shape inference cannot.
14. Why does `compile_c` compile in a temporary directory *inside* the cache directory?

<details>
<summary>Answers, week 2</summary>

10. "`mem`, but only after `dep` has executed." A graph has no order except data dependence; a read of memory
    after a write to it has no data edge to the write unless one is added. `AFTER` is that edge.
11. When `vmin(numerator) >= 0` and the divisor is a positive constant; then truncation equals floor division.
    Otherwise it calls `floordiv_i32`.
12. Inlined: `CONST`, `INDEX`, `AFTER` (which renders as its memory), and ALU ops used exactly once. Named:
    every `LOAD` (`valN`), ALU ops used more than once (`aluN`); `ALLOC`s and `RANGE`s get declarations.
13. Loading from (or storing into) a plain value, for example `CONST.load()`: "LOAD needs memory".
14. `os.replace` is atomic only within one filesystem; renaming from `/tmp` into `~/.cache` failed across
    filesystems (NOTES 10). Same-filesystem rename also makes concurrent compiles of one kernel safe.

</details>

### Week 3: loops and the linearizer

15. Define the *scope* of a node.
16. Why did the first linearizer design ("emit a node when its scope equals the open loops") deadlock on GEMM?
17. Where is an accumulator `ALLOC` declared, and what goes wrong with OpenMP if it is declared at the top of
    the kernel?
18. In `x - max(x)` computed per row, which two loops are siblings, and which must run first?
19. How does the loop-IR GEMM in `tests/test_w03_loops.py` keep its accumulator inside the `i` and `j` loops but
    outside `k`?

<details>
<summary>Answers, week 3</summary>

15. The set of `RANGE`s the node must sit inside: the union of its sources' scopes, plus itself for a `RANGE`,
    minus the ranges an `END` closes.
16. `A[i*K + k]` has scope `{i, k}`, but `k` is nested inside `j`, so the open loops are `(i, j, k)` whenever `k`
    is open; the scope never equals the open set and the node is never emitted. The fix closes scopes under
    "encloses" so scopes are paths in the loop tree (NOTES 8).
17. In the deepest scope common to all its users (inside the parallel loop), so each thread has a private copy.
    Declared at the top it is shared: a data race. At `-O3` gcc happened to keep it in a register and results
    were right; at `-O0` 20 of 20 runs were wrong (Appendix B).
18. Inside the row loop `i`: the reduce loop over `k` (computing the max) and the output loop over `j`. The `k`
    loop must run first, because the `j`-loop nodes read its result through its `END`.
19. `acc.after(i, j)` places the accumulator in scope `{i, j}`; its init store is there too; the update reads
    `acc.after(init, k)` inside `k`; the final read is `acc.after(upd.end(k))`, after the `k` loop closes.

</details>

### Week 4: movement ops and rangeify

20. What does the syllabus's `gemm_256x256.uir` compute? Give the correct version.
21. State the index map rangeify uses for `EXPAND`, `SHRINK` and `PERMUTE`.
22. Why does rangeify simplify the index at every `RESHAPE`? Quote the measured effect.
23. `REDUCE` pops which axes? How do you sum axis 0 of a `(3, 4)` tensor?
24. What does rangeify turn a `STACK` into?
25. How is `PAD` built without a pad primitive?

<details>
<summary>Answers, week 4</summary>

20. `A * rowsum(B)`: reshaping `A` to `(N, N, 1)` and `B` to `(N, 1, N)` and reducing the last axis sums `B`'s
    rows. A GEMM reshapes `A` to `(N, 1, K)`, `B` *transposed* to `(1, N, K)`, multiplies and reduces the last
    axis (NOTES 11).
21. `EXPAND`: an axis that was size 1 reads index 0. `SHRINK`: add each axis's `begin`. `PERMUTE`: place each
    output index at the input axis it came from (the inverse permutation).
22. Ravel-then-unravel produces expressions like `((i*N + j) // N) % N` that equal `i` but mention `j`; the
    reduce lowering places accumulators in the loops an index *mentions*, so reductions were nested in `j`,
    making fused attention O(N³). Simplifying took N=512 fused attention from 1787 ms to 7 ms (NOTES 12).
23. The last `n`. Permute the axis to the end first: `x.permute((1, 0)).reduce(Ops.ADD, 1)`, shape `(4,)`.
24. A chain of `WHERE`s on the first index: `where(i < 1, a[rest], where(i < 2, b[rest], c[rest]))`.
25. Concatenate the tensor with zeros along the axis (a `STACK` moved next to the axis and reshaped), then
    `SHRINK` back to the padded size (`composite.pad_axis`). Rangeify plus symbolic turn it into a bounds-checked
    `WHERE`.

</details>

### Week 5: kernelize, autodiff, the Tensor library

26. What are the materialization points of `kernelize`?
27. Why can fusing an elementwise result into a reduce through an `EXPAND` be a bad idea? What rule does `uopc`
    use, and what did it save on GPT?
28. Why does `make_plan` renumber buffers `0..n-1` within each kernel?
29. Why does AdamW pass its bias corrections as one-element tensors rather than Python floats?
30. Give the gradient rules for `EXPAND`, `SHRINK` and `PERMUTE`.
31. How does `uopc` split the gradient of `REDUCE(MAX)` among tied maxima?
32. Why does autodiff refuse `STORE`, `CALL`, `AFTER`, `RANGE` and `END`?
33. Why is `nn.Embedding` a one-hot matmul?

<details>
<summary>Answers, week 5</summary>

26. Every root (unless it is already a contiguous buffer), every `STAGE` (except `STAGE(x, "detach")`,
    which is stripped), and every `REDUCE` unless `fuse_reduce` is on.
27. The fused value is recomputed for every element it is broadcast to: a LayerNorm output feeding a matmul is
    recomputed per output column. `stage_broadcast_compute` stages an ALU value expanded by 4x or more when the
    recompute cost (`_fused_work`) is at least 4096; it took a GPT step's kernel time from 374 ms to 125 ms
    (NOTES 15).
28. So structurally identical kernels render byte-identical C, which hits the content-addressed `.so` cache.
29. They change every step; as `CONST`s they make each step's graph structurally different, missing the plan
    cache and recompiling everything (NOTES 17). As data, the graph is identical every step.
30. `EXPAND`: sum the gradient over the expanded axes (`sum_to`). `SHRINK`: pad the gradient with zeros back to
    the input shape. `PERMUTE`: apply the inverse permutation.
31. It builds a mask `x == max`, counts the ties, and gives each tied element `grad / count`.
32. Those ops carry effects and control structure, not mathematical functions of their inputs; autodiff works
    on pure shape graphs, before rangeify introduces them.
33. A data-dependent `INDEX` (a gather) has no gradient rule in `uopc`; a one-hot matrix times the embedding
    table is differentiable with existing rules.

</details>

### Week 6: fast CPU kernels

34. Why does register tiling raise arithmetic intensity? Use the 4×8 tile from `opt.py`.
35. Why did adding threads make the strided 1024³ GEMM *slower*?
36. What does `-fassociative-math` allow, and what is the cost?
37. With `Opts(upcast={0: 2, 1: 2})`, how many `for` loops and accumulators does a GEMM kernel have?
38. What does the layout pass do, and when does it stage an operand?
39. How do `hand_opt` and `search_opt` differ?

<details>
<summary>Answers, week 6</summary>

34. Each step of `k` loads 4 values of A and 8 of B (12 loads) and performs 32 multiply-adds into 32
    accumulators held in registers: about 2.7 multiply-adds per load, versus 1/2 for the naive loop.
35. Each thread streamed down a column of B (stride N), touching a new cache line and a new 4 KB page every
    iteration; more threads meant more concurrent cache and TLB misses: 1.9 → 1.1 GFLOP/s (NOTES 19).
36. Reordering float additions, which lets gcc vectorize reductions (17 → 170 GFLOP/s in one experiment,
    NOTES 20). Results differ from the strict left-to-right sum at the rounding level.
37. Three loops (two output loops and one shared reduce loop) and four accumulators
    (`test_upcast_shares_reduce_loop`).
38. It pushes movement ops through elementwise ops to the data (`push_views`), finds operands whose innermost
    reduce axis is strided, and stages them contiguously, but only if the staged view is no larger than the
    base data and strictly smaller than the reduce input (NOTES 21).
39. `hand_opt` applies fixed rules (a 4×4 tile on the last two axes, threads if the work is at least 2¹⁶,
    unroll 4 for small reduces); `search_opt` compiles candidate `Opts` and keeps the one that runs fastest on
    random data.

</details>

### Weeks 7–8: GPUs, shared memory, tensor cores

40. Which kernel axis gets `threadIdx.x`, and why?
41. Why does `smem_gemm_ir` need two barriers per tile step?
42. What does `gpu.emulate` let you test without a GPU, and what can it not test?
43. By what factor does T×T tiling in shared memory cut global-memory traffic of a GEMM?
44. What are the operand and accumulator types of a typical tensor-core MMA, and what tile does `wmma_gemm`
    use?

<details>
<summary>Answers, weeks 7–8</summary>

40. The last (fastest-varying, contiguous) axis, so neighboring threads of a warp touch neighboring addresses
    and their accesses coalesce (NOTES 23).
41. The first ensures the whole tile is loaded before any thread reads it; the second ensures every thread has
    finished reading before the next step's copy overwrites the tile.
42. The index mapping of `GLOBAL`/`LOCAL` ranges (by running them as loops in C). Not anything that needs threads
    to run concurrently: barriers and shared memory (NOTES 25).
43. A factor of T: each element loaded into shared memory is reused by T threads.
44. f16 (or bf16) inputs with f32 accumulation; `wmma_gemm` uses 16×16×16 fragments, one warp per 16×16 output
    tile.

</details>

### Weeks 9–10: real models

45. Why does `examples/gpt.py` pad the sampling context to a fixed length?
46. How does `examples/gpt.py` make attention causal?
47. What state does online softmax keep per query row?
48. What were the measured effects, on GPT, of `stage_broadcast_compute` and of identity-based hashing plus
    memoized `simplify`?

<details>
<summary>Answers, weeks 9–10</summary>

45. So every forward pass has the same shapes, the same graph structure, and therefore one compiled plan
    (NOTES 18).
46. It adds a precomputed mask, `-1e9` above the diagonal (`np.triu(..., 1)`), sliced to `[:T, :T]`, to the
    scores before the softmax.
47. A running maximum `m`, a running sum `l` of exponentials, and the output accumulator `o[D]`, all rescaled by
    `exp(m_old - m_new)` when the maximum grows.
48. Kernel time per step 374 → 125 ms (NOTES 15); compile time for 12 steps 163 → 47 s, first step about 9 s
    (NOTES 22).

</details>

### Beyond (Chapter 15)

49. Give the KV-cache size formula and evaluate it for 2 layers, `dim` 64, 64 tokens, f32.
50. What do *declared bounds* on a symbolic size buy the compiler?
51. In data-parallel training, what is communicated each step?

<details>
<summary>Answers, Chapter 15</summary>

49. `2 · layers · T · dim · bytes`: 2 · 2 · 64 · 64 · 4 = 65,536 bytes (64 KiB).
50. Bounds analysis keeps working: index expressions cannot overflow, div/mod rules fire, the renderer can use
    `/` instead of `floordiv_i32`, and temp buffers can be allocated at the upper bound. Without bounds a loaded
    size has full `i32` range and `bounds()` gives up.
51. Gradients, summed (or averaged) across devices with an all-reduce.

</details>
