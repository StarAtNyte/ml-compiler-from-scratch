# 22. GPU Engineering: Correctness, Throughput, and Attention

Chapters 12–13 explain GPU mapping and shared-memory kernels. This chapter turns those ideas into
a hardware investigation. Every performance result here is an assignment to measure on your GPU;
there are no implied benchmark results. Start with the architecture sections of the official
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) and record the
GPU model, supported compute capability, compiler, and driver used for your experiment.

## 22.1 Execution, memory, and synchronization

A block contains threads that execute in warps. Blocks may execute in any order; a conventional
kernel cannot depend on a different block reaching a software barrier. Use separate launches or a
supported cooperative mechanism when global coordination is required.

A block barrier requires participating threads to reach it consistently. This is unsafe when the
last block is partial:

```text
if global_index < n:
    shared[local_index] = input[global_index]
    block_barrier()
```

Move the barrier outside the divergent condition and initialize invalid lanes to the operation's
identity. For a sum use zero; for a maximum use negative infinity. Ensure consumers cannot read an
uninitialized slot. Warp-level primitives have their own active-mask and participation requirements;
lockstep intuition is not a substitute for the specified synchronization contract.

**Lab:** implement vector addition, a block reduction, and a multi-block reduction. Test lengths
1, 31, 32, 33, and 1003. Run the available memory, race, and synchronization checking tools on real
hardware. Keep a deliberately broken version to demonstrate which check catches the error.

## 22.2 Coalescing and shared-memory banks

For a row-major matrix, adjacent threads loading adjacent columns typically access nearby addresses.
If adjacent threads instead stride across rows, the warp may need many more memory transactions.
Logical bytes read and physical bytes transferred can therefore differ substantially.

A tiled transpose loads contiguous rows into shared memory and stores contiguous rows of the
transposed result. Shared-memory layout can create bank conflicts. Under the common teaching model
of 32 banks and 4-byte words, bank index is `(byte_address/4) mod 32`; a 32×32 float tile accessed
by column maps those accesses to the same bank. Padding its row stride to 33 spreads those accesses.
Check the actual architecture's rules, access width, and broadcast behavior before generalizing.

**Lab:** compare naive transpose, tiled transpose, and padded tiled transpose. Handle rectangular
sizes and edge tiles. Report useful bandwidth `2*M*N*sizeof(element)/time` and relevant profiler
transaction and bank-conflict evidence. Explain any mismatch between predicted and measured behavior.

## 22.3 The GEMM optimization ladder

For a block output tile `BM × BN` and reduction chunk BK, load `BM*BK + BK*BN` elements into
shared memory, then perform approximately `2*BM*BN*BK` operations. With f32 inputs, the idealized
input arithmetic intensity per chunk is:

```text
2*BM*BN*BK / (4*(BM*BK + BK*BN)) FLOPs per byte
```

This ignores output traffic, cache effects, and repeated block loads, so it is a model to check.
Each thread can retain several output accumulators in registers. That increases reuse but also
register demand. Shared memory per block, registers per thread, thread count, and architectural
limits jointly constrain residency. High occupancy alone does not imply high throughput.

Build and retain each rung: naive global loads; shared-memory tiling; per-thread register tiles;
vectorized loads where alignment permits; tensor-core operations; pipelined memory transfer.
For each rung, measure correctness, time, register use, shared-memory use, spills, and occupancy.
A larger tile that spills can lose to a smaller tile with less nominal reuse.

For tensor cores, match supported tile shapes, storage and accumulator types, layouts, alignment,
and warp participation rules. WMMA fragments are implementation-defined distributed register
representations; never assume each thread owns a simple contiguous subsection of the matrix.
Compare f16-input/f32-accumulate output against a reference using the same quantized inputs.

**Lab:** benchmark square and rectangular GEMMs with aligned and irregular dimensions. For each
optimization predict its benefit, then inspect generated code and profiler evidence. Demonstrate
at least one configuration where an apparently stronger optimization loses and explain why.

## 22.4 Asynchronous pipelines

A sequential tile loop loads tile t, synchronizes, computes it, then starts t+1. A double-buffered
pipeline can overlap loading t+1 with computing t. It needs separate storage for tiles in flight,
a completion mechanism before consumption, and protection against reusing storage too early.

Write a timeline before code:

```text
prologue: begin load tile 0
iteration t: wait tile t; begin load t+1 into other buffer; compute tile t
boundary: ensure all readers finished before reusing tile t's buffer
```

Exact wait and barrier operations depend on the target instruction and architecture. A block barrier
does not automatically replace an asynchronous-copy completion protocol. More stages consume more
shared memory, potentially reducing residency enough to remove the benefit.

**Lab:** add a supported asynchronous-copy pipeline to your tiled GEMM. Validate the final partial
K tile, buffer reuse, and all synchronization paths. Compare two and three stages across several K
sizes. If the hardware lacks the feature, retain the design and mark the device experiment incomplete.

## 22.5 FlashAttention beyond the formula

For one query row, process scores in tiles and maintain running maximum m, normalizer l, and
unnormalized weighted-value accumulator o. Starting with `m=-infinity, l=0, o=0`, a new score tile S
updates:

```text
m_new = max(m, max(S))
p = exp(S - m_new)
alpha = exp(m - m_new)
l_new = alpha*l + sum(p)
o_new = alpha*o + p @ V_tile
output = o / l
```

Rescaling old state is what makes the tiled result match a stable full softmax in real arithmetic.
Floating-point reduction order still changes rounding. Causal masking must exclude future positions
before they contribute to maxima and sums. Fully masked rows require explicit semantics to avoid
`-infinity - -infinity` producing NaN.

For backward, with `P=softmax(S)` and output `O=P*V`, compute `dV=P^T*dO`, `dP=dO*V^T`, and
`dS=P*(dP-row_sum(dP*P))`. Then apply the score scale when forming dQ and dK. A memory-efficient
backward recomputes needed probabilities from scores and saved row statistics instead of storing
all of P. Its scheduling and race-free gradient accumulation are substantial implementation work.

**Lab:** implement forward and backward against an unfused reference. Test causal and noncausal
attention, irregular sequence lengths, several head dimensions, and extreme logits. Run finite
differences on tiny double-precision cases. Measure peak memory and latency as sequence length grows;
report both forward and backward. Add dropout only after defining reproducible masks.

## 22.6 Framework comparison and backend portability

Use the official [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
as a second implementation route for vector operations, reductions, matrix multiplication, and attention.
Compare the schedule expressed by a block program with the thread and shared-memory schedule your
CUDA renderer emits. Read the intermediate representations for one kernel.

For benchmarks, warm up compilation and caches, measure repeated device work with synchronization or
device events, and report median plus spread. Host wall time without waiting often measures enqueue
latency. Keep transfers and cold compilation as separate end-to-end measurements. Match PyTorch's
precision settings and verify that the compared operation performs equivalent work.

For HIP or another backend, list the portable semantics and the architecture-specific assumptions:
subgroup width, matrix instructions, memory spaces, barriers, launch limits, and runtime APIs. A
textual rename of CUDA builtins is insufficient. Port a reduction before attempting tensor cores.

**Exit gate:** a correctness suite on real hardware, a transpose and GEMM optimization notebook, an
attention forward/backward implementation, and a workload-wide comparison. Competitive performance
is earned separately for each tested workload and device.

## 22.7 Worked lab: derive and execute tiled attention backward

Let scores be `S=Q*K^T/sqrt(d)` and probabilities `P=softmax(S)`. For a loss gradient G on
`O=P*V`, the chain rule gives `dV=P^T*G` and `dP=G*V^T`. A single softmax row has Jacobian
`diag(P)-P*P^T`, so its vector-Jacobian product is `dS=P*(dP-sum(dP*P))`.

The row correction can be computed without materializing dP or P:

```text
sum_j dP[i,j]*P[i,j]
= sum_j sum_c G[i,c]*V[j,c]*P[i,j]
= sum_c G[i,c]*O[i,c]
```

That identity lets backward save output O and one log-normalizer per query row, then recompute each
probability tile as `exp(scores-log_normalizer)`. For a score tile of width T, temporary score and
probability storage is proportional to `queries*T`, rather than `queries*keys`. Q, K, V, output,
and gradients still occupy memory: the algorithm removes quadratic score storage, not all storage.

For a scalar online-softmax example, process scores 1000 and 1001 in separate tiles. After the
first tile, m=1000 and l=1. After the second, m=1001 and l=`exp(-1)+1`. The first contribution to
the weighted-value sum is also rescaled by `exp(-1)`. Forgetting that rescaling produces wrong
results even though the normalizer looks stable.

Run the implementation and its numerical checks:

```bash
.venv/bin/python -m labs.attention
.venv/bin/python -m pytest tests/test_advanced_labs.py -q -k attention
```

@include labs/attention.py

This lab is a CPU implementation of the algorithm. It supports one head, finite floating-point
inputs, causal masking, and a nonnegative query-position offset for decoding-style subproblems.
Its loops do not establish GPU synchronization correctness or performance. Finite inputs can still
overflow a dot product at extreme magnitudes; the tested numerical domain must be reported.

**Experiment:** vary tile size through 1, 4, and larger than the key length. Forward results should
agree within rounding error. Perturb Q, K, and V separately in random directions, comparing the dense
loss finite difference with the analytic gradient's directional inner product. The test suite uses
an independent dense computation, so a shared error in tiled forward and backward cannot validate itself.

**Next implementation:** lower the tile loop into a real GPU kernel, then handle parallel dK/dV
accumulation without races. Compare memory and device time with the dense baseline at matched precision.
The serial CPU loop's safe accumulation order is not an automatic GPU scheduling solution.
