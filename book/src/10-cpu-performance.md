# 10. Fast CPU kernels

At the end of [Chapter 9](09-tensor-mnist.md) the compiler was correct and general. It was not fast.
Here is the opening position for this chapter, a 1024x1024x1024 matrix multiply in `f32`, compiled
straight from the shape graph, against NumPy (which calls OpenBLAS, decades of hand-tuned assembly):

```text
  B strided  naive                     5.2 GFLOP/s   max err 2.1e-04
  numpy                              449.4 GFLOP/s
```

Eighty-six times slower, for the same 2.15 billion floating-point operations. By the end of the chapter the
same compiler, with a few hundred lines of new code, produces a kernel that ran at 238 GFLOP/s in that
measurement session and has measured between 114 and 517 GFLOP/s in others, within 5% of NumPy on a quiet
machine. Nothing about the math changes. What changes is *where the data is* when the arithmetic needs it:
which cache level, which register, which core. This chapter teaches you to reason about that
quantitatively (FLOP counts, bytes, arithmetic intensity, the roofline model), then implements the
transformations that close the gap: a layout pass, multithreading, register tiling and cache blocking.

**What you'll build**

- `Opts` in `uopc/rangeify.py`: `upcast` (register tiles), `unroll` (reduce unrolling), `thread`
  (OpenMP) and `tile` (cache blocks of the output axes), applied *during* rangeify, plus `merge_ends` to
  let unrolled copies share one loop.
- `uopc/opt.py`: the layout pass (`push_views`, `_last_axis_stride`, `layout_pass`), the hand-written
  heuristic `hand_opt_choice`, and a tiny autotuner, `search_opt`.
- `examples/bench_gemm.py`: a benchmark that walks the GEMM from 5 to over 200 GFLOP/s step by step.

**Where this fits.** [Chapter 9](09-tensor-mnist.md) built a frontend whose kernels this chapter makes
fast. [Chapter 11](11-fusion-flash.md) applies the same reasoning to attention, where the question is not
how to tile one kernel but how many kernels to have; [Chapter 12](12-gpus.md) maps the same `Opts` idea onto
GPUs.

## 10.1 The benchmark

`examples/bench_gemm.py` builds the GEMM as a shape graph by hand, lowers it with different `Opts`,
renders and runs each version, and checks it against NumPy:

@include examples/bench_gemm.py

Two layouts are compared. "B strided" is the textbook `C = A @ B` with `B` stored row-major as
`(K, N)`: the reduction over `k` walks *down a column* of `B`. "B^T" stores `B` transposed as `(N, K)`,
which is the same math with both operands contiguous along `k`. The last `B^T` row adds cache blocks
to the tiled kernel; 10.11 explains it. The captured run (`book/outputs/bench_gemm_1024.txt`) is the
table this chapter explains, row by row:

```text
N=1024  (2.15 GFLOP per GEMM)
  B strided  naive                     5.2 GFLOP/s   max err 2.1e-04
  B strided  threads                   7.1 GFLOP/s   max err 2.1e-04
  B strided  4x4 tile + threads       65.1 GFLOP/s   max err 2.1e-04
  B^T        naive                    28.0 GFLOP/s   max err 8.4e-05
  B^T        threads                 149.1 GFLOP/s   max err 8.4e-05
  B^T        4x4 tile                 85.3 GFLOP/s   max err 8.4e-05
  B^T        4x4 tile + threads      237.8 GFLOP/s   max err 8.4e-05
  B^T        + 64x64 cache blocks    288.5 GFLOP/s   max err 8.4e-05
  Tensor A@B (layout pass+opt)       321.4 GFLOP/s   (end to end, includes the transpose copy)
  numpy                              449.4 GFLOP/s
```

Three things should surprise you. Sixteen hardware threads made the strided version only 1.4 times
faster. Transposing `B`, which does not change a single FLOP, made the naive kernel five times faster.
And register tiling without threads (85.3) is slower than threads without tiling (149.1). By the end of
the chapter each of these will be obvious.

**A word on measurement.** These numbers come from a laptop (Intel i7-13620H, gcc 15, see
`book/outputs/machine.txt`), and laptop numbers vary a lot between runs. This table was captured on AC
power while other processes kept the CPU busy (load average 6 to 9). Across the sessions in which this
chapter was written, the `B^T 4x4 tile + threads` row at N=1024 measured anywhere from 114 GFLOP/s (on
battery) to 517 GFLOP/s (on AC, with a quiet machine), and NumPy from 132 to 542. In one run NumPy came
out *slower* than our kernel (229 against 475 GFLOP/s), because the two rows ran a few seconds apart and
the machine's state changed in between. The reasons:

- **Power source.** On battery the firmware holds the cores at low clocks. The peak-FLOP measurement in
  10.3 found one core doing 59 GFLOP/s on battery and 117 on AC.
- **Thermal limits.** A laptop can sustain its maximum turbo clock for seconds, not minutes. A long
  benchmark slows down as the chip heats up, and the first rows of a table get a cooler chip than the
  last.
- **Hybrid cores.** The i7-13620H has 6 performance cores (P-cores) and 4 slower efficiency cores
  (E-cores). Which cores the OS gives a thread, and whether a P-core's second hardware thread is busy with
  another process, changes from run to run.
- **Background load.** Other processes take cores away from OpenMP's sixteen threads. One thread that
  shares its core with a browser holds up the whole parallel loop.

So the absolute numbers on a laptop are not stable. Large ratios between rows (5x from the layout,
several x from threads and from tiling) survive every run; differences of 20 or 30% do not, and the
`+ 64x64 cache blocks` and `Tensor A@B` rows are in that range (10.11 shows the cache-block row changing
sign between runs). `bench` in the script takes the best of five runs after a warm-up, which removes some
of the noise but not all of it. Compare variants within one session, repeat the session before believing
a small difference, and report the machine state with the numbers.

## 10.2 Counting work: FLOPs, bytes, intensity

Performance engineering starts with two counts.

**FLOPs.** A GEMM of `(M, K) @ (K, N)` performs `M*N*K` multiplies and `M*N*K` adds: `2*M*N*K` floating
point operations. For `N = M = K = 1024` that is 2,147,483,648, the "2.15 GFLOP per GEMM" in the header.
A fused multiply-add (FMA, `a*b + c` in one instruction) counts as two. The kernelizer estimates FLOPs
for `DEBUG=2` output the same way, one per element of every ALU node plus one per reduced element:

@include uopc/kernelize.py#_flops

**Bytes.** The *compulsory* memory traffic is what must cross between DRAM and the chip at least once:
for the GEMM, read `A` and `B` and write `C`, `3 * 1024^2 * 4` bytes, about 12.6 MB. An elementwise
`out = a * 2` over `n` floats reads `4n` bytes and writes `4n` bytes.

**Arithmetic intensity** (AI) is the ratio, FLOPs per byte of DRAM traffic:

| kernel | FLOPs | compulsory bytes | AI (FLOP/byte) |
|---|---|---|---|
| `out = a * 2` | `n` | `8n` | 1/8 |
| row sum `sum(a, 1)` | `n` | `4n` | 1/4 |
| GEMM, `N^3` | `2N^3` | `12N^2` | `N/6` = 171 at N=1024 |
| AdamW update for one weight | about `10n` | `16n` (read p, m, v, write p) | about 0.6 |

Elementwise kernels have constant, tiny intensity: no matter how cleverly you write them, each byte
fetched feeds at most a handful of operations. A GEMM's intensity *grows with N*: every element of `A`
is used `N` times. That reuse is the opportunity, and the whole chapter is about cashing it in.

## 10.3 This machine: peak compute and peak bandwidth

To know whether 5.2 GFLOP/s is bad, we need the machine's limits. The i7-13620H has 6 performance cores
(two hardware threads each) and 4 efficiency cores, 16 hardware threads in all. A performance core
issues AVX2 instructions on 256-bit registers holding 8 `f32` lanes, and can start two FMA instructions
per cycle. One FMA on 8 lanes is 16 FLOPs, so one P-core peaks at 32 FLOP/cycle: 157 GFLOP/s at its
4.9 GHz maximum turbo. That number is a ceiling written on a spec sheet; let us measure what we can
actually get.

### Measuring peak FLOP/s

A microbenchmark must keep both FMA units busy every cycle. An FMA has a latency of about four cycles
(its result is available four cycles after it starts), and two can start per cycle, so at least eight
*independent* chains of FMAs must be in flight. This scratch program (not part of the repository) runs
twelve, on as many threads as asked:

```c
// peak.c: how many f32 FLOP/s can this machine do? 12 independent chains of 8-wide FMAs.
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  int threads = atoi(argv[1]);
  long iters = 100000000;
  double t0 = omp_get_wtime(); float sink = 0;
  #pragma omp parallel num_threads(threads) reduction(+:sink)
  {
    __m256 x = _mm256_set1_ps(0.999f), y = _mm256_set1_ps(0.001f);
    __m256 a0 = _mm256_set1_ps(0.f), a1 = _mm256_set1_ps(1.f), a2 = _mm256_set1_ps(2.f), a3 = _mm256_set1_ps(3.f),
           a4 = _mm256_set1_ps(4.f), a5 = _mm256_set1_ps(5.f), a6 = _mm256_set1_ps(6.f), a7 = _mm256_set1_ps(7.f),
           a8 = _mm256_set1_ps(8.f), a9 = _mm256_set1_ps(9.f), a10 = _mm256_set1_ps(10.f), a11 = _mm256_set1_ps(11.f);
    for (long i = 0; i < iters; i++) {   // 12 independent dependency chains hide the 4-cycle FMA latency
      a0 = _mm256_fmadd_ps(a0, x, y); a1 = _mm256_fmadd_ps(a1, x, y); a2 = _mm256_fmadd_ps(a2, x, y);
      a3 = _mm256_fmadd_ps(a3, x, y); a4 = _mm256_fmadd_ps(a4, x, y); a5 = _mm256_fmadd_ps(a5, x, y);
      a6 = _mm256_fmadd_ps(a6, x, y); a7 = _mm256_fmadd_ps(a7, x, y); a8 = _mm256_fmadd_ps(a8, x, y);
      a9 = _mm256_fmadd_ps(a9, x, y); a10 = _mm256_fmadd_ps(a10, x, y); a11 = _mm256_fmadd_ps(a11, x, y);
    }
    __m256 s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;
    sink += s[0];
  }
  double dt = omp_get_wtime() - t0;
  printf("%2d threads: %7.1f GFLOP/s\n", threads, 2.0 * 8 * 12 * iters * threads / dt / 1e9);
  return sink == 12345.f;
}
```

Compiled with `gcc -O2 -march=native -fopenmp` and run on AC power on the same day as the captured
benchmark:

```text
 1 threads:   117.0 GFLOP/s
 6 threads:   602.6 GFLOP/s
10 threads:   727.7 GFLOP/s
16 threads:   776.8 GFLOP/s
```

Two lessons hide in this tiny program. The first draft initialized every chain to the same two
values, and gcc noticed that `a0`, `a2`, `a4`... computed identical things and merged them: the
disassembly had two FMAs per iteration, not twelve, and the "measurement" was off by a factor of six.
Always read the assembly of a microbenchmark. Second, 117 GFLOP/s on one core is below the 157 of the
spec sheet: 117.0 / 32 FLOP per cycle is about 3.7 GHz, not 4.9. The clock also moves between runs. A
second run a minute later printed 74.5 GFLOP/s for one thread and 527.0 for sixteen, and an earlier
session on battery measured 59.4 and 293.6 (about 1.9 GHz). A roof is an upper bound, so we use the best
values measured on AC power: 777 GFLOP/s for the whole chip, 117 for one core.

### Measuring bandwidth

For bandwidth we use our own compiler: a `uopc` kernel that scales a 64 Mi-element array (256 MB, far
larger than any cache), and one that sums it, each with and without threads:

```python
# bw.py: memory bandwidth as seen by uopc kernels, on 64 Mi floats (256 MB per array)
import time, numpy as np
from uopc.uop import UOp, Ops, ParamArg
from uopc.dtype import dtypes
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.runtime import Program
N, R = 1 << 26, 256
A, OUT = (UOp(Ops.BUFFER, (), ParamArg(s, dtypes.f32, (N,))) for s in (0, 1))
ROWS = UOp(Ops.BUFFER, (), ParamArg(1, dtypes.f32, (R,)))
a, out, rows = np.random.randn(N).astype(np.float32), np.zeros(N, np.float32), np.zeros(R, np.float32)
tests = [("scale  out=a*2", A.reshape((R, N // R)) * 2.0, OUT, out, 8),      # read 4 B + write 4 B per element
         ("rowsum sum(a)", A.reshape((R, N // R)).reduce(Ops.ADD, 1), ROWS, rows, 4)]   # read 4 B per element
for name, g, o_buf, o_arr, bytes_per in tests:
  for tname, o in [("1 thread", Opts()), ("threads", Opts(thread=0))]:
    prg = Program(render_c(lower(g, o_buf, Ctx(opts=o)), "k"), "k")
    prg(a, o_arr); ts = []
    for _ in range(10):
      t0 = time.perf_counter(); prg(a, o_arr); ts.append(time.perf_counter() - t0)
    print(f"{name:15s} {tname:9s} {bytes_per * N / min(ts) / 1e9:6.1f} GB/s")
```

```text
scale  out=a*2  1 thread    21.3 GB/s
scale  out=a*2  threads     40.8 GB/s
rowsum sum(a)   1 thread    21.2 GB/s
rowsum sum(a)   threads     55.9 GB/s
```

About 56 GB/s when every core pulls (a run under heavier background load gave 39 GB/s; on battery it was
17.7). One core alone gets about 21 GB/s: a single core cannot keep enough cache misses in flight to
saturate the memory system. The scale kernel's 40.8 GB/s under-counts: writing a cache line the CPU does
not have first *reads* it (write-allocate), so the true traffic is 12 bytes per element, not 8, and
40.8 x 12 / 8 = 61 GB/s, close to the sum kernel.

## 10.4 The roofline model

The roofline model (Williams, Waterman and Patterson, 2009) combines the two limits into one picture. A
kernel with arithmetic intensity `AI` cannot run faster than either limit:

`attainable GFLOP/s = min(peak GFLOP/s, AI * peak GB/s)`

Plotted on log-log axes this is a slanted line (memory-bound region) meeting a flat line (compute-bound
region) at the **ridge point**, `AI = 777 / 55.9 = 14` FLOP per byte on this machine. A kernel to the
left of the ridge can only go faster by moving fewer bytes; a kernel to the right can only go faster by
doing its arithmetic faster. Here is the machine with the measured kernels:

@diagram 10-roofline

(The roofs are the best values from 10.3, measured on AC power; the GEMM points are the captured table
from 10.1, measured on the same day under background load. On a quieter run the tiled point moves up
toward 500.)

The two elementwise kernels sit on the slanted roof: they are as fast as memory allows, and no
transformation in this chapter will help them. (Fusing them into their neighbours, so the data is not
written and re-read, is the only lever, and that is [Chapter 7](07-kernelize.md)'s business.) The GEMM
is far to the right of the ridge: with 171 FLOPs per byte it could, in principle, run at the full 777
GFLOP/s. The naive version runs 150 times below its roof.

> **Key idea:** the DRAM roofline says the naive GEMM is not limited by memory bandwidth. Its problem is
> lower in the hierarchy: how often it touches the caches, how often registers are reloaded, and how
> much of the core's parallelism (SIMD lanes, FMA pipelines, cores) it uses. The same roofline idea,
> applied per level (L1, L2, registers), explains every row of the benchmark.

The roofline also explains the MNIST kernels from [Chapter 9](09-tensor-mnist.md). The AdamW update of
`W1` touches about 1.6 MB with an intensity near 0.6, so its roof is about 34 GFLOP/s; in the
Chapter 9 run it reached 1.5 GFLOP/s. The gap is not the kernel's fault. Its output is a new parameter buffer, freshly allocated
every step, so the kernel pays a page fault the first time it touches each 4 KB page. `DEBUG=3` runs each
kernel twice to expose this: `e_784_128  0.603 ms ... (again: 0.359 ms)` for the `W1` update in a warm
step. (Temporaries used to pay the same price on every step; the plan now keeps its temporary arrays
between calls, see [Chapter 11](11-fusion-flash.md). Outputs cannot be reused that way, because they
become the new parameters.) Small kernels are dominated by such
fixed costs (page faults, waking OpenMP threads, the ctypes call), which is why `hand_opt` only uses
threads when there is enough work.

## 10.5 Caches, lines, pages and the TLB

The memory hierarchy of this machine, as the C library reports it:

```text
LEVEL1_DCACHE_SIZE                  49152
LEVEL1_DCACHE_LINESIZE              64
LEVEL2_CACHE_SIZE                   1310720
LEVEL2_CACHE_LINESIZE               64
LEVEL3_CACHE_SIZE                   25165824
LEVEL3_CACHE_LINESIZE               64
```

That is 48 KB of L1 data cache and 1.25 MB of L2 per performance core, and 24 MB of L3 shared by all
cores. Three facts govern everything below:

1. **Memory moves in 64-byte lines.** Reading one `float` brings in its 15 neighbours. If you use them
   next, they are free; if you do not, you paid for 16 floats and used one.
2. **Caches are small.** A 1024-float row is 4 KB; 48 KB of L1 holds twelve of them. A whole 1024x1024
   matrix (4 MB) fits in L3 but not in L2.
3. **Addresses are translated in 4 KB pages.** Every load needs its virtual page translated to a physical
   one, and the translation lookaside buffer (TLB) caches only on the order of a hundred recent
   translations at its fastest level. Touching a new page on every access misses there constantly.

Now look at the naive strided GEMM's inner loop again. In C, `data1[((ridx2*1024)+ridx1)]`: as `k`
(`ridx2`) increases, the address jumps by 4096 bytes. gcc is clever about it and vectorizes *across*
`j` instead (eight neighbouring `j` share every cache line), producing this inner loop:

```text
.L3:
	vbroadcastss	(%rdx), %ymm0
	vmulps	(%rax), %ymm0, %ymm0
	addq	$4096, %rax
	addq	$4, %rdx
	cmpq	%rax, %rcx
	vaddps	%ymm0, %ymm1, %ymm1
	jne	.L3
```

One `A` value is broadcast to all lanes, multiplied by eight consecutive `B` values, accumulated. But
`addq $4096, %rax`: every iteration reads from the next row of `B`, which is a new cache line *and* a new
4 KB page. The 1024 lines of one column strip (64 KB) do not fit in L1, so by the time the next strip of
`j` comes around they are gone. The loop spends its time waiting for L2 and the TLB, not computing.

@diagram 10-layout

Why do sixteen threads make this only 1.4 times faster (5.2 to 7.1 GFLOP/s)? On battery they even made
it *slower*, 1.9 to 1.1 GFLOP/s. Each thread runs the same column walk on different rows of `A`, so
sixteen hardware threads now each stream a 4 KB-stride pattern, and pairs of them share a core's L1, L2
and TLB. More threads means more simultaneous page walks and more cache lines competing for the same
small structures, while the arithmetic was never the bottleneck. This is NOTES 19 in `book/NOTES.md`, and
its lesson generalizes: **parallelism multiplies whatever your kernel does,
including its bad memory behaviour.** Fix locality first, then add threads.

## 10.6 Fix 1: layout

If the reduction walks down a column, transpose the matrix so that it walks along a row. With `B^T`
stored `(N, K)` the inner loop becomes a dot product of two contiguous rows,
`data0[(alu1+ridx2)] * data1[((ridx1*1024)+ridx2)]`, and gcc vectorizes it along `k`:

```text
.L3:
	vmovups	(%rcx,%rax), %ymm0
	vmulps	(%rdx,%rax), %ymm0, %ymm0
	addq	$32, %rax
	cmpq	$4096, %rax
	vaddps	%ymm0, %ymm1, %ymm1
	jne	.L3
```

Eight `k` per instruction, 32 bytes per step, and every byte of every fetched cache line is used. Naive single
thread goes from 5.2 to 28.0 GFLOP/s. Why not more? Look at `%ymm1`: there is *one* accumulator, and every
`vaddps` must wait for the previous one to finish. The loop is bound by the latency of a single
dependency chain, not by the machine's throughput; we will fix that with register tiling.

Users should not have to transpose their matrices. The compiler should do it, and only when it pays.
That is the **layout pass** in `opt.py`. It runs on the shape graph before kernelization and asks, for
every REDUCE: which operands are read with a non-unit stride along the reduced axis? Those are copied
into a contiguous temporary (a STAGE, which [Chapter 7](07-kernelize.md) turns into a kernel boundary)
*in transposed order*, so the reduce reads them contiguously.

Three helpers do the work. First, movement ops are pushed through elementwise ops down to the leaves, so
that every chain of views ends at real data:

@include uopc/opt.py#push_views

`permute(a * b)` is `permute(a) * permute(b)`, and the right-hand form exposes each operand's own view
chain. Second, `_views` collects, for the input of a REDUCE, every movement chain that ends at a
materialized node (a BUFFER, another REDUCE, or a STAGE):

@include uopc/opt.py#_views

Third, `_last_axis_stride` measures how far apart in memory `view[..., k]` and `view[..., k+1]` are. It
does not analyze the movement ops symbolically; it *rangeifies* the view against a fake buffer and
simplifies the resulting index expression at `k = 0` and `k = 1`:

@include uopc/opt.py#_last_axis_stride

Reusing rangeify as an oracle is the simplest correct stride analysis: whatever chain of reshapes,
permutes, expands and shrinks the view has, rangeify already knows how to turn it into an index, and
`simplify` turns the index into a constant when the view is affine. With those, the pass is short:

@include uopc/opt.py#layout_pass

For each strided operand it strips any EXPANDs (we copy the operand *before* it is broadcast, so the
copy is `N*K` floats, not `M*N*K`), and keeps the candidate only if staging it really yields stride 1.
The guard `prod(c.shape) > prod(base.shape) or prod(c.shape) >= prod(r.src[0].shape)` exists because of
a bug. The first version of the pass staged whatever view it found, and in the MNIST backward pass it
once materialized a whole `(128, 784, 128)` broadcast product, a 21 ms kernel that was larger than the
reduce it was meant to speed up (NOTES 21). Two fixes followed: `push_views`, so that views are found at
the leaves rather than around big computed products, and the size guard, so that only operands strictly
smaller than the reduce input are ever copied.

The tests check both directions: a strided GEMM gets a STAGE, an already-transposed one does not.

@include tests/test_w06_opt.py#test_layout_pass_stages_transposed_operand

Here is what `Tensor(a) @ Tensor(b)` turns into for 64x64 inputs with the default optimizer (`DEBUG=4`,
headers omitted): a copy kernel that writes `B^T`, and the GEMM reading it contiguously.

```c
void e_1_64_64(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      float val1 = data0[((ridx1*64)+ridx0)];
      data1[((ridx0*64)+ridx1)] = val1;
    }
  }
}
```

At 1024x1024 the copy costs about 0.7 ms against about 5 ms for the GEMM (`DEBUG=2`, steady state, on AC
power):

```text
  e_1_1024_1024                     0.635 ms     0.00 GFLOP/s
  r_1024_1024                       4.699 ms   456.98 GFLOP/s
realize: 2 kernels, schedule 0.0 ms, run 5.4 ms
```

An `O(N^2)` copy buys an `O(N^3)` kernel a factor of several; the larger the matrix, the better the deal.
This is the "Tensor A@B" row of the benchmark: 321.4 GFLOP/s end to end, copy included. In the captured
run it came out faster than the hand-built `4x4 tile + threads` row (237.8), although it runs the same
kernel plus a copy. That ordering is noise: the rows ran several seconds apart under changing background
load.

## 10.7 Fix 2: threads

Sixteen hardware threads are sitting idle. The output of a GEMM is embarrassingly parallel: each `C[i, j]`
is independent. The cheapest way to use the cores is OpenMP: put `#pragma omp parallel for` on the
outermost loop and let the runtime split its iterations among threads.

In the IR, a loop's behaviour is recorded in its RANGE's `AxisType` (from the week-6 syllabus: LOOP,
UPCAST or THREAD). The optimizer's choices travel to rangeify in an `Opts` record:

@include uopc/rangeify.py#Opts

`lower()` gives the THREAD axis the lowest range index, so the linearizer places it outermost, and the
renderer emits the pragma for it:

@include uopc/render_c.py#CRenderer.render_range

```c
void gemm(float* restrict data0, float* restrict data1, float* restrict data2) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 1024; ridx0++) {
    int alu1 = (ridx0*1024);
    for (int ridx1 = 0; ridx1 < 1024; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 1024; ridx2++) {
        float val2 = acc0;
        float val3 = data0[(alu1+ridx2)];
        float val4 = data1[((ridx1*1024)+ridx2)];
        acc0 = (val2+(val3*val4));
      }
      float val5 = acc0;
      data2[(alu1+ridx1)] = val5;
    }
  }
}
```

Is this correct? Every iteration of the parallel loop writes a different row of `C`, so there are no
races on the output. The accumulator `acc0` is the subtle part: it must be *private* to each thread. It
is, because the linearizer declares an ALLOC in the deepest scope common to its users (NOTES 9), which is
inside the `ridx1` loop, inside the parallel region, so each thread gets its own. Had it been declared
at the top of the function, all threads would share one `acc0` and the result would be garbage. On the
contiguous layout, threads take the naive kernel from 28.0 to 149.1 GFLOP/s: about 5.3x on a machine with 6
fast and 4 slow cores. gcc's OpenMP runtime splits the iterations statically into equal chunks, so on a
hybrid CPU the performance cores probably finish early and wait for the efficiency cores (exercise 7
measures this).

Threads have a fixed cost: waking the thread pool and joining it at the end costs microseconds, which is
a lot for a 128-element kernel. `hand_opt_choice` (below) only threads kernels with at least `2^16`
units of work (output elements times reduce size).

## 10.8 Fix 3: register tiling

The naive kernel computes one output element at a time: for each `k`, load `A[i, k]`, load `B[j, k]`,
one FMA. Two loads per FMA. Every `A[i, k]` gets loaded again for the next `j`, and every `B[j, k]` again
for the next `i`. The roofline at the register level says: the core can do two FMAs per cycle but only
two or three loads per cycle, so a kernel with two loads per FMA can never exceed about one FMA per cycle,
half of peak even with perfect caches.

**Register tiling** computes a small block of outputs at once. With a 4x4 tile, each iteration of `k`
loads four values of `A` (rows `i..i+3`) and four of `B` (rows `j..j+3`) and updates all sixteen
products, keeping the sixteen running sums in registers for the whole `k` loop:

@diagram 10-register-tile

Eight loads now feed sixteen FMAs: 2 FMAs per load instead of 0.5, a fourfold drop in load traffic per
FLOP. Larger tiles reuse more (4x8 gives 2.7, 8x8 gives 4), until the accumulators no longer fit in the
16 vector registers of AVX2. And there is a second, quieter benefit: the sixteen accumulators are sixteen
*independent* dependency chains, so the FMA pipelines can overlap them, which removes the single-chain
latency bound we saw in 10.6.

### How rangeify builds a tile

In the syllabus vocabulary this is UPCAST: an axis that is "fully unrolled, becomes straight-line code".
`uopc` implements it without a new kind of loop. An output axis of size `s` with upcast factor `f`
becomes a RANGE of size `s/f` plus a constant offset `c in 0..f-1`, and rangeify is simply called once
per combination of constants:

@include uopc/rangeify.py#_axis_parts

@include uopc/rangeify.py#_combos

`lower` computes the index tuples for the whole tile and rangeifies the output once per tuple, producing
`f0 * f1` separate stores:

@include uopc/rangeify.py#lower

That alone would produce sixteen *separate* reduce loops, one per tile element, each with its own
accumulator: correct, and no faster than before, because nothing is shared. The sharing comes from
`_reduce`. It memoizes the reduce RANGEs by the *set of loop ranges* in the output index, ignoring the
constants:

@include uopc/rangeify.py#_reduce

All sixteen tile elements have indices like `(4*ridx0 + c0, 4*ridx1 + c1)`, which mention the same two
RANGEs, so `rkey` is the same for all of them and they reuse one set of reduce RANGEs. Each still gets
its own accumulator ALLOC and its own update store, but all of them hang off the same loop. (The `unroll`
factor for the innermost reduce axis is applied here too; see 10.10.)

There is one more step. Each of the sixteen update stores is closed by its own END over the shared
reduce range; the linearizer would see sixteen ENDs of one loop. `merge_ends` groups ENDs that close the
same ranges and replaces them with one END over a SINK of their bodies:

@include uopc/rangeify.py#merge_ends

The result, for a small 8x8 GEMM with a 2x2 tile, is one shared `k` loop updating four accumulators:

```c
void gemm(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*16);
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu2 = (ridx1*2);
      int alu3 = (alu1+alu2);
      float acc0;
      acc0 = 0.0f;
      float acc1;
      acc1 = 0.0f;
      float acc2;
      acc2 = 0.0f;
      float acc3;
      acc3 = 0.0f;
      int alu4 = (alu2+alu1);
      for (int ridx2 = 0; ridx2 < 8; ridx2++) {
        float val5 = acc0;
        int alu6 = (alu1+ridx2);
        float val7 = data0[alu6];
        int alu8 = ((ridx1*16)+ridx2);
        float val9 = data1[alu8];
        acc0 = (val5+(val7*val9));
        float val10 = acc1;
        float val11 = data1[(alu8+8)];
        acc1 = (val10+(val7*val11));
        float val12 = acc2;
        float val13 = data0[(alu6+8)];
        acc2 = (val12+(val13*val9));
        float val14 = acc3;
        acc3 = (val14+(val13*val11));
      }
      float val15 = acc0;
      data2[alu3] = val15;
      float val16 = acc1;
      data2[(alu4+1)] = val16;
      float val17 = acc2;
      data2[(alu3+8)] = val17;
      float val18 = acc3;
      data2[(alu4+9)] = val18;
    }
  }
}
```

Two `A` loads (`val7`, `val13`) and two `B` loads (`val9`, `val11`) feed four FMAs. Each value is loaded
once and used twice, exactly the outer product of the diagram. After `merge_ends`, the program has
exactly two ENDs: one over a SINK of the four update stores closing the reduce range, one over a SINK of
the four output stores closing the two output ranges:

```text
>>> [(len(e.src), e.src[0].op.name, [r.arg.idx for r in e.src[1:]]) for e in ends]
[(2, 'SINK', [2]), (3, 'SINK', [0, 1])]
```

The test pins the structure down: two output loops plus one shared reduce loop, at least four
accumulators:

@include tests/test_w06_opt.py#test_upcast_shares_reduce_loop

Notice that `alu3 = (alu1+alu2)` and `alu4 = (alu2+alu1)` are the same value computed twice: the
simplifier does not put commutative operands in a canonical order, so CSE by hash-consing misses it.
gcc removes the duplicate, so it costs nothing at runtime, but it is a small blemish (exercise 9).

### What the hardware sees

For the 4x4 tile at N=1024, gcc vectorizes the shared `k` loop: each of the sixteen scalar accumulators
becomes an 8-lane vector of partial sums over `k`, and the inner loop is sixteen `vfmadd` instructions fed
by eight vector loads:

```text
.L6:
	movq	24(%rsp), %rsi
	vmovups	(%rcx,%rax), %ymm1
	vmovups	(%r14,%rax), %ymm6
	vmovups	(%r15,%rax), %ymm4
	vmovups	(%rsi,%rax), %ymm2
	vmovups	(%rdx,%rax), %ymm0
	vfmadd231ps	%ymm6, %ymm1, %ymm10
	vfmadd231ps	%ymm4, %ymm1, %ymm11
	vfmadd231ps	%ymm2, %ymm1, %ymm15
	vfmadd213ps	160(%rsp), %ymm0, %ymm1
	vmovaps	%ymm1, 160(%rsp)
	vmovups	(%r9,%rax), %ymm1
	vfmadd231ps	%ymm6, %ymm1, %ymm12
	vfmadd231ps	%ymm4, %ymm1, %ymm3
	vfmadd231ps	%ymm2, %ymm1, %ymm5
	vfmadd213ps	128(%rsp), %ymm0, %ymm1
	vmovaps	%ymm1, 128(%rsp)
	vmovups	(%r10,%rax), %ymm1
	vfmadd231ps	%ymm6, %ymm1, %ymm7
	vfmadd231ps	%ymm4, %ymm1, %ymm8
	vfmadd231ps	%ymm2, %ymm1, %ymm9
	vfmadd231ps	%ymm0, %ymm1, %ymm14
	vmovups	(%r11,%rax), %ymm1
	addq	$32, %rax
	vfmadd213ps	96(%rsp), %ymm1, %ymm6
	vfmadd213ps	64(%rsp), %ymm1, %ymm4
	cmpq	$4096, %rax
	vfmadd213ps	32(%rsp), %ymm1, %ymm2
	vfmadd231ps	%ymm0, %ymm1, %ymm13
	vmovaps	%ymm6, 96(%rsp)
	vmovaps	%ymm4, 64(%rsp)
	vmovaps	%ymm2, 32(%rsp)
	jne	.L6
```

Count the registers: sixteen accumulators plus eight loaded vectors is 24, and AVX2 has 16 `ymm`
registers. gcc keeps eleven accumulators in registers and *spills* five to the stack: the
`vfmadd213ps 160(%rsp)` / `vmovaps %ymm1, 160(%rsp)` pairs read an accumulator from memory and write it
back every iteration. (The first version of `hand_opt_choice` carried a comment claiming that sixteen
vector accumulators fit in AVX2's sixteen registers; this assembly showed they do not once the operands
need registers too, and the comment now says so.) Spills to L1 are cheap, and with threads this kernel
has measured between 114 and 517 GFLOP/s depending on the run, but it is a reminder to check the
assembly rather than trust the arithmetic. In one session the autotuner of 10.13 found that, at N=512,
an 8x1 tile (eight accumulators, nine operands) beat the 4x4; in another the 4x4 won.

Tiling also explains the curious ordering in the benchmark. Tiling alone (85.3 GFLOP/s) is slower than
threads alone (149.1), because one core cannot beat sixteen hardware threads however well it uses its
registers. But threads alone are limited by each thread's single dependency chain and two loads per FMA,
and tiling alone is limited to one core. Together, each of sixteen threads runs a latency-hiding,
load-efficient inner loop: 237.8 GFLOP/s in the captured run. On the strided layout, tiling rescues even
the column walk (7.1 to 65.1) because the four `j` of a tile share each fetched cache line.

## 10.9 The flag that matters most: `-fassociative-math`

Look back at the B^T naive inner loop: `vmulps` then `vaddps` into a vector of partial sums. That loop
adds `A[i, 0..7] * B[j, 0..7]` lane by lane into eight separate running sums and combines them at the
end. In exact arithmetic that is the same as adding the products left to right. In floating point it is
not: `(a + b) + c` and `a + (b + c)` can differ in the last bits. The C standard requires gcc to evaluate
`acc0 = acc0 + x` in program order, so without permission it may not vectorize a float reduction at all.

The permission is `-fassociative-math`, which the runtime passes by default:

@include uopc/runtime.py#CFLAGS

Here is the scratch experiment that measured it (same kernels, only `CFLAGS` changed, run on AC power
in the same session as the peak measurement in 10.3):

```text
CFLAGS: -O3 -march=native -fopenmp -ffp-contract=fast -fassociative-math -fno-signed-zeros -fno-trapping-math
  B^T naive                   24.6 GFLOP/s
  B^T 4x4 tile + threads     452.2 GFLOP/s
CFLAGS: -O3 -march=native -fopenmp -ffp-contract=fast
  B^T naive                    4.1 GFLOP/s
  B^T 4x4 tile + threads      53.3 GFLOP/s
```

A factor of six to eight from one flag, on a kernel that is otherwise identical; NOTES 20 records 17
versus 170 in an earlier run, on battery. Every ML compiler makes this trade: reductions are reordered for speed, so a sum in `f32`
is reproducible run to run (our kernels are deterministic) but not bit-identical to a sequential sum.
We stop short of `-ffast-math`, which would also let gcc assume there are no NaNs or infinities; that
would break `softmax`'s `-INFINITY` initial value and every test that checks NaN propagation.
`-fno-signed-zeros` and `-fno-trapping-math` are the small extra permissions `-fassociative-math` needs
to be effective; `-ffp-contract=fast` allows fusing `a*b + c` into one FMA.

## 10.10 Reduce unrolling

The third knob unrolls the innermost reduce axis by a factor `u`: the reduce RANGE shrinks by `u`, and
the `u` values are combined in a balanced tree before being added to the accumulator. It is the same
`_axis_parts` machinery, applied to reduce axes in `_reduce`. With `Opts(unroll=4)`:

```c
      for (int ridx2 = 0; ridx2 < 256; ridx2++) {
        float val3 = acc0;
        int alu4 = (ridx2*4);
        float val5 = data0[(alu1+alu4)];
        float val6 = data1[(alu2+alu4)];
        int alu7 = (alu4+alu1);
        float val8 = data0[(alu7+1)];
        int alu9 = (alu4+alu2);
        float val10 = data1[(alu9+1)];
        float val11 = data0[(alu7+2)];
        float val12 = data1[(alu9+2)];
        float val13 = data0[(alu7+3)];
        float val14 = data1[(alu9+3)];
        acc0 = (val3+(((val5*val6)+(val8*val10))+((val11*val12)+(val13*val14))));
      }
```

The tree `((p0 + p1) + (p2 + p3))` shortens the dependency chain through `acc0`, the classic reason to
unroll a reduction by hand. With an auto-vectorizer behind us, it mostly gets in the way. Measured at
N=1024, single-threaded, in the same session:

```text
  upcast={} thread=None unroll=1:   24.1 GFLOP/s
  upcast={} thread=None unroll=4:   24.7 GFLOP/s
  upcast={0: 4, 1: 4} thread=None unroll=1:   82.4 GFLOP/s
  upcast={0: 4, 1: 4} thread=None unroll=4:   12.6 GFLOP/s
  upcast={0: 2, 1: 4} thread=None unroll=1:   64.9 GFLOP/s
```

On its own, unrolling changes nothing (gcc vectorized the plain loop already). Combined with the 4x4
tile it is a disaster: the assembly of the unrolled tile contains only scalar instructions
(`vfmadd231ss`, `vmulss`, `vaddss`) and not a single packed one. Sixty-four unrolled products feeding
sixteen trees were more than gcc's vectorizer would untangle, so it gave up. `hand_opt` uses `unroll`
only for short reductions with no tile, where it removes the loop entirely (the
`test_reduce_unroll_has_no_inner_loop` test unrolls a 16-element sum into straight-line code). The general
lesson: **when you generate C, gcc is part of your backend, and its heuristics are part of your cost
model.**

## 10.11 Cache blocking the output axes

A register tile reuses each loaded value from a register. Cache blocking (also called loop tiling) does
the same one level up: it reorders the output loops so that the operand rows a group of outputs needs are
still in L1 or L2 when the next output wants them.

### The working set

The caches of this machine, as `lscpu -C` reports them:

```text
NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL  SETS PHY-LINE COHERENCY-SIZE
L1d       48K     416K   12 Data            1    64        1             64
L1i       32K     448K    8 Instruction     1    64        1             64
L2       1.3M     9.5M   10 Unified         2  2048        1             64
L3        24M      24M   12 Unified         3 32768        1             64
```

Each P-core has 48 KB of L1d and 1.25 MB of L2 (`/sys/devices/system/cpu/cpu0/cache/index2/size`
says `1280K`), shared by its two hardware threads. The four E-cores share one 2 MB L2 (9.5 MB in all is
6 x 1.25 MB + 2 MB). The 24 MB L3 is shared by every core.

Now follow the data through the `4x4 tile + threads` kernel at N = K = 1024. The THREAD loop hands each
thread strips of four rows of `A`. For one strip, the loop over `j` visits all 256 groups of four `B^T`
rows, and the `k` loop of each tile reads:

- four rows of `A`, 4 x 4 KB = 16 KB. They are the same for every `j` of the strip, so they stay in L1
  and are reused 256 times;
- four rows of `B^T`, another 16 KB. Each group is used once per strip. By the time the next strip needs
  it again, the thread has read all of `B^T`, 4 MB, which does not fit in L2. It comes back from L3.

Per strip that is 4 MB of `B^T` from L3 for 4 x 1024 x 1024 x 2 = 8.4 million FLOPs: 2 FLOPs per byte of
L3 traffic.

Blocking the output into 64x64 squares changes the order. All the tiles of one block are computed
before moving on, so the block needs 64 rows of `A` (256 KB) and 64 rows of `B^T` (256 KB). That 512 KB
fits in a P-core's L2, even with the second hardware thread working on another block. Each `B^T` row is
now read from L3 once per block and then reused from L2 by the block's sixteen row strips. L3 traffic
per block is 512 KB for 64 x 64 x 1024 x 2 = 8.4 million FLOPs, 16 FLOPs per byte: eight times less
traffic to L3 than without blocks.

### The implementation

`Opts.tile` maps an output axis to a block size. `lower` splits each tiled axis into a block loop and a
loop inside the block, and creates all the block loops first, so they get the lowest range indices and
the linearizer places them outermost:

@include uopc/rangeify.py:180-194

A tiled axis of size `s` with block `t` and upcast `f` becomes `outer * (t/f) + inner`, with `outer` of
size `s/t` and `inner` of size `t/f`. The upcast constant is added later by `_combos`, as for an
untiled axis, so the register tile still works inside a block. If the THREAD axis is tiled, its block
loop gets the THREAD type: OpenMP then divides whole blocks among the threads. A tile size that does
not divide the axis, is not a multiple of the upcast, or covers the whole axis is ignored.

For a 128x128 GEMM with `Ctx(opts=Opts(upcast={0: 4, 1: 4}, thread=0, tile={0: 64, 1: 64}))`, `render_c`
produces the following (abridged: the `...` lines stand for the same pattern repeated for the other
accumulators and stores):

```c
void gemm(float* restrict data0, float* restrict data1, float* restrict data2) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      for (int ridx2 = 0; ridx2 < 16; ridx2++) {
        int alu1 = ((ridx0*8192)+(ridx2*512));
        for (int ridx3 = 0; ridx3 < 16; ridx3++) {
          int alu2 = ((ridx1*64)+(ridx3*4));
          int alu3 = (alu1+alu2);
          float acc0;
          acc0 = 0.0f;
          ...
          float acc15;
          acc15 = 0.0f;
          int alu4 = (alu2+alu1);
          for (int ridx4 = 0; ridx4 < 128; ridx4++) {
            float val5 = acc0;
            int alu6 = (alu1+ridx4);
            float val7 = data0[alu6];
            int alu8 = (((ridx1*8192)+(ridx3*512))+ridx4);
            float val9 = data1[alu8];
            acc0 = (val5+(val7*val9));
            ...
          }
          float val31 = acc0;
          data2[alu3] = val31;
          ...
        }
      }
    }
  }
}
```

`ridx0` and `ridx1` are the block loops (two blocks of 64 per axis), `ridx2` and `ridx3` walk the
sixteen 4x4 tiles of a block along each axis, and `ridx4` is the shared reduce loop of 10.8. The row of
`A` is `ridx0*64 + ridx2*4` (hence `ridx0*8192 + ridx2*512` in elements), and the row of `B^T` is
`ridx1*64 + ridx3*4`. The test checks the results against NumPy for three tilings and counts one extra
loop per tiled axis:

@include tests/test_w06_opt.py#test_cache_blocking_correct

### The measurement

The captured runs show the blocked kernel ahead at N=1024 and level at N=2048
(`book/outputs/bench_gemm_2048.txt`):

```text
N=2048  (17.18 GFLOP per GEMM)
  ...
  B^T        4x4 tile + threads      334.7 GFLOP/s   max err 1.2e-04
  B^T        + 64x64 cache blocks    337.9 GFLOP/s   max err 1.2e-04
```

The 1024 row (288.5 against 237.8) looks like a 20% gain, but an earlier run on a quieter machine had
it the other way round: 430 with 64x64 blocks against 517 without. To separate the effect from the noise,
this scratch script compiles four variants and times them in turn, three rounds, so that each round sees
roughly the same machine state:

```python
# tiles.py: 4x4 tile + threads, with and without cache blocks, interleaved over three rounds
import sys; sys.path.insert(0, "examples")
import numpy as np
from bench_gemm import gemm_graph, bench
from uopc.uop import UOp, Ops, ParamArg
from uopc.dtype import dtypes
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.runtime import Program
for N in (1024, 2048):
  a, bt, c = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32), np.zeros(N * N, np.float32)
  out = UOp(Ops.BUFFER, (), ParamArg(2, dtypes.f32, (N * N,)))
  variants = [("no blocks", {})] + [(f"{t}x{t} blocks", {0: t, 1: t}) for t in (32, 64, 128)]
  prgs = [(n, Program(render_c(lower(gemm_graph(N, True), out, Ctx(opts=Opts(upcast={0: 4, 1: 4}, thread=0, tile=t))), "gemm"), "gemm")) for n, t in variants]
  res = {n: [] for n, _ in prgs}
  for _ in range(3):
    for n, p in prgs: res[n].append(2 * N**3 / bench(lambda: p(a, bt, c)) / 1e9)
  print(f"N={N}")
  for n, v in res.items(): print(f"  {n:14s}" + "".join(f"{x:8.1f}" for x in v) + " GFLOP/s")
```

```text
N=1024
  no blocks        237.9   433.4   429.9 GFLOP/s
  32x32 blocks     327.2   426.4   399.9 GFLOP/s
  64x64 blocks     496.8   403.6   382.0 GFLOP/s
  128x128 blocks   312.1   280.4   296.0 GFLOP/s
N=2048
  no blocks        385.0   379.3   313.3 GFLOP/s
  32x32 blocks     379.4   337.2   317.9 GFLOP/s
  64x64 blocks     330.8   366.9   321.2 GFLOP/s
  128x128 blocks   333.3   377.8   363.6 GFLOP/s
```

Read the columns. In the first round at N=1024 the unblocked kernel, timed first, measured 237.9 and
64x64 blocks 496.8; in the second and third rounds the blocks were slower. At N=2048 no variant wins consistently. The one stable result
is that 128x128 blocks are the slowest at N=1024. On this machine, once the layout pass has made the
reduce axis contiguous, cache blocking of the output axes did not help reliably. Four reasons:

1. **The traffic it removes was not the bottleneck.** Without blocks, the kernel reads `B^T` from L3 at
   2 FLOPs per byte. At 240 to 500 GFLOP/s that is 120 to 250 GB/s across the chip, and the
   measurements say L3 delivers it: cutting that traffic by 8x did not make the kernel faster. Each `k`
   loop reads eight rows sequentially, 4 KB each, which is the easiest possible pattern for the hardware
   prefetchers. They bring the lines of `B^T` into L2 ahead of use, whether or not a block keeps them
   there.
2. **The register tile already provides the reuse that matters most.** The rows of `A` for a strip
   already stay in L1 (16 KB of 48), and every loaded vector feeds four FMAs. What limits the inner loop
   now is in the core: the five spilled accumulators of 10.8, and the horizontal sum at the end of every
   1024-long dot product.
3. **Blocks are coarser units of parallel work.** With a tiled THREAD axis, OpenMP divides whole blocks
   among the threads. At N=1024 there are 1024/64 = 16 block rows for 16 hardware threads, one each,
   so a thread that lands on an E-core or shares its core with another process holds up the rest. With
   128x128 blocks there are only 8 block rows, and half of the hardware threads have nothing to do; that
   is why the 128x128 row is consistently the slowest at N=1024.
4. **The block's working set grows with `K`.** A 64x64 block needs 2 x 64 x K x 4 bytes of operand rows:
   512 KB at K=1024, 1 MB at K=2048 (two hardware threads on one core then need 2 MB, more than the
   1.25 MB L2), and 2 MB at K=4096.

Blocking does help when the unblocked kernel really does run out of cache. At N=4096, `B^T` is 64 MB,
larger than the 24 MB L3, so without blocks every strip streams `B^T` from DRAM at 2 FLOPs per byte. With
55.9 GB/s that alone would cap the kernel near 112 GFLOP/s, except that threads working on neighbouring
strips at the same time share some of what is in L3. With blocks, the operand rows of a block (2 MB for
64x64 at K=4096, 1 MB for 32x32) are reused from L3 or L2 instead. The same script at N=4096:

```text
N=4096
  no blocks        253.0   186.7   180.6 GFLOP/s
  32x32 blocks     299.4   250.7   238.5 GFLOP/s
  64x64 blocks     296.4   234.0   231.1 GFLOP/s
  128x128 blocks   294.7   226.0   238.9 GFLOP/s
```

Every blocked variant beats the unblocked one in every round, by 15 to 35%. (An earlier run of the same
script had the blocked variants ahead in eight of nine comparisons.)

So blocking the output axes helps when the operands do not fit in L3. When `K` is large, a block's rows no longer fit in L2 however small the
block is along `i` and `j`, and the fix is to block the reduce axis as well: split `k` into chunks of,
say, 256, and compute every tile of the block for one chunk before moving to the next, so that the block
needs 2 x 64 x 256 x 4 = 128 KB of operand rows for any `K`. That changes the accumulators. Today each
tile's sixteen sums live in registers from the start of the `k` loop to the store, and the ALLOC sits
inside the output loops. With `k` split and the chunk loop outside the tile loops, each tile is visited
once per chunk, so its partial sums must survive between visits: the block needs a persistent
accumulator buffer of 64 x 64 floats (16 KB) per thread, zeroed before the first chunk and written to `C`
after the last, or `C` itself must be zeroed and accumulated into. That is exercise 6. It is also the
loop structure of OpenBLAS and BLIS, where the `k` block is chosen so that a packed panel of `B` fits in
L2 (Goto and van de Geijn, 2008).

## 10.12 SIMD: what gcc already does

Every fast row of the benchmark depends on 8-wide AVX2 instructions, yet `uopc` never emits a vector
type. gcc's loop vectorizer does it. You can check this for any kernel by compiling the generated C to
assembly with the runtime's own flags and `-S -o -`:

```python
# simd.py: compile generated GEMM kernels to assembly with the runtime's flags and count FMAs
import subprocess, sys
sys.path.insert(0, "examples")
from bench_gemm import gemm_graph
from uopc.uop import UOp, Ops, ParamArg
from uopc.dtype import dtypes
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.runtime import CFLAGS
N = 1024
out = UOp(Ops.BUFFER, (), ParamArg(2, dtypes.f32, (N * N,)))
for o in [Opts(upcast={0: 4, 1: 4}, thread=0), Opts(upcast={0: 4, 1: 4}, thread=0, unroll=4)]:
  src = render_c(lower(gemm_graph(N, True), out, Ctx(opts=o)), "gemm")
  asm = subprocess.run(["gcc", *CFLAGS, "-x", "c", "-", "-S", "-o", "-"], input=src, capture_output=True, text=True).stdout
  fma = [l.strip() for l in asm.splitlines() if l.strip().startswith("vfmadd")]
  print(f"unroll={o.unroll}: {sum(f.split()[0].endswith('ps') for f in fma)} packed (ps), "
        f"{sum(f.split()[0].endswith('ss') for f in fma)} scalar (ss) FMA instructions")
  for f in fma[:3]: print("   ", f)
```

```text
unroll=1: 16 packed (ps), 0 scalar (ss) FMA instructions
    vfmadd231ps	%ymm6, %ymm1, %ymm10
    vfmadd231ps	%ymm4, %ymm1, %ymm11
    vfmadd231ps	%ymm2, %ymm1, %ymm15
unroll=4: 0 packed (ps), 32 scalar (ss) FMA instructions
    vfmadd231ss	-16(%r14), %xmm13, %xmm1
    vfmadd231ss	-8(%r14), %xmm7, %xmm0
    vfmadd231ss	4080(%r14), %xmm13, %xmm1
```

`vfmadd231ps` on `ymm` registers is a packed single-precision FMA on eight lanes; `vfmadd231ss` on `xmm`
registers works on one lane. The tiled kernel has sixteen packed FMAs, one per accumulator, exactly the
loop listed in 10.8. gcc vectorized it along `k`: each scalar accumulator became a vector of eight
partial sums, each `vmovups` loads eight consecutive `k` of one row, and after the loop a short sequence
(`vextractf128`, `vaddps`, `vmovhlps`) adds the eight lanes of each accumulator together. That
reordering of the sum is what `-fassociative-math` (10.9) permits. With `unroll=4`, the same tile gets
no packed FMAs at all, which is the 6.5x slowdown measured in 10.10.

So `uopc` gets SIMD for free, but only as long as gcc's heuristics agree, and only along the axis gcc
chooses. Explicit vectorization in the IR would take that decision into the compiler. It needs three
things that `uopc` does not have:

- **A vector dtype**, for example `float8` (eight `f32` lanes), rendered in C as
  `float __attribute__((vector_size(32)))` or as an `__m256`.
- **A VECTORIZE op** that groups the copies an upcast axis produces into one vector value, and its
  inverse, an op that extracts one lane. When an upcast axis has factor 8 and its index is contiguous in
  memory, the eight scalar loads become one vector load, the eight FMAs one vector FMA, and the eight
  stores one vector store. In a GEMM the natural choice is the `j` axis of the output tile: broadcast
  one `A[i, k]`, multiply it by eight consecutive values along `j`, and the result is eight finished
  outputs, with no horizontal sum at the end.
- **Aligned loads.** `vmovaps` requires an address that is a multiple of 32 bytes. Using it needs a
  guarantee that the buffer is aligned and a proof from the symbolic layer that the index is a multiple
  of 8. Without that proof, the renderer must emit unaligned loads (`vmovups`), which current cores
  execute at the same speed as aligned ones when the data happens to be aligned.

tinygrad has all three (its `dtype.vec`, `Ops.VECTORIZE` and `Ops.GEP`). Adding them to `uopc` is
exercise 10.

## 10.13 Choosing the opts

### A hand-written heuristic

`hand_opt_choice` encodes the lessons of this chapter in a dozen lines:

@include uopc/opt.py#hand_opt_choice

If the kernel reduces, tile the last two output axes 4x4 (falling back to 2 if the size is not
divisible); if it is a short reduction with no tile, unroll it; and if there is at least `2^16` units of
work, thread the first axis that still has at least two iterations after tiling. Applied to a few
shapes:

```text
GEMM 1024^3            Opts(upcast={1: 4, 0: 4}, thread=0, unroll=1, tile={}, gpu=False, local={})
row max (128,10)       Opts(upcast={0: 4}, thread=None, unroll=1, tile={}, gpu=False, local={})
elementwise (784,128)  Opts(upcast={}, thread=0, unroll=1, tile={}, gpu=False, local={})
elementwise (128,)     Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=False, local={})
sum (16,)              Opts(upcast={}, thread=None, unroll=4, tile={}, gpu=False, local={})
```

The `row max (128, 10)` case is the `log_softmax` kernel from the MNIST step: four rows per iteration, no
threads (only 1280 elements). The heuristic never sets `tile`, for the reasons measured in 10.11.
`HandOpt` packages the layout pass and this heuristic for the kernelizer;
it is the default (`OPT=1`), and `OPT=0` turns both off:

@include uopc/opt.py#HandOpt

### Search

A heuristic encodes what you know. Search finds what you do not. `search_opt` compiles a list of
candidate `Opts` for the kernel, times each on random data, and keeps the fastest:

@include uopc/opt.py#candidate_opts

@include uopc/opt.py#search_opt

It is a one-step version of tinygrad's `BEAM` search: no beam, no iteration, just "try the plausible
register tiles on the last two axes, with and without unrolling, plus 32, 64 and 128 cache blocks on
those axes, and measure". Enable it with `OPT=2` (`SearchOpt`). Here is what it sees for a 512x512 GEMM
(a scratch copy of `search_opt` that prints every candidate's time instead of only keeping the best;
AC power, same session as 10.3):

```text
29 distinct candidates; hand_opt picks {1: 4, 0: 4} thread=0
   454.2 GFLOP/s  upcast={0: 4, 1: 4} thread=0 unroll=1
   391.5 GFLOP/s  upcast={0: 8} thread=0 unroll=1
   347.5 GFLOP/s  upcast={0: 4, 1: 8} thread=0 unroll=1
   345.7 GFLOP/s  upcast={0: 4} thread=0 unroll=1
   292.2 GFLOP/s  upcast={0: 8, 1: 4} thread=0 unroll=1
   253.1 GFLOP/s  upcast={0: 8, 1: 8} thread=0 unroll=1
  ...
    39.9 GFLOP/s  upcast={1: 16} thread=0 unroll=4
    38.6 GFLOP/s  upcast={0: 4, 1: 8} thread=0 unroll=4
    28.8 GFLOP/s  upcast={} thread=None unroll=1
```

This time the heuristic's 4x4 tile won. In an earlier session on battery the same search put `{0: 8}`
first (117.8 GFLOP/s) and the 4x4 second (102.8), so the top two swap places with the machine's state.
Every candidate with `unroll=4` is near the bottom in both sessions, as 10.10 predicted. The count is 29
although `candidate_opts` returns 53 entries: most are duplicates. This run was made with the first version
of `search_opt`, whose deduplication key left out `tile`, so the three cache-blocked candidates were
silently skipped as well (exercise 11). With `tile` in the key there are 32 candidates; in a later run on the
same machine none of the blocked ones won at N=512. Search pays for its answers: compiling and timing the 29 variants took 3.6
seconds of wall time with an empty compile cache, and 0.56 seconds once the runtime had cached the
compiled kernels on disk. Timing still runs every time, which is why search is opt-in and why production
autotuners also cache their *choices* on disk, keyed by the kernel (exercise 8).

## 10.14 Results

Putting it together, the captured table from 10.1, with the reason for each step:

| version | GFLOP/s | what changed |
|---|---|---|
| B strided, naive | 5.2 | column walk: new line and new page every `k` |
| B strided, threads | 7.1 | sixteen threads fighting over caches and TLBs |
| B strided, 4x4 tile + threads | 65.1 | four `j` share each line; still strided |
| B^T, naive | 28.0 | contiguous and vectorized, but one dependency chain |
| B^T, threads | 149.1 | 16 threads x the naive inner loop |
| B^T, 4x4 tile | 85.3 | 2 FMAs per load, 16 independent chains, one core |
| B^T, 4x4 tile + threads | 237.8 | both |
| B^T, + 64x64 cache blocks | 288.5 | within run-to-run noise at this size (10.11) |
| Tensor `A @ B` (layout pass + `hand_opt`) | 321.4 | the compiler does all of the above, transpose included |
| numpy (OpenBLAS) | 449.4 | cache blocking with packing, hand-written microkernels |

The last four rows are close enough that their order changes between runs; on a quiet machine the
`4x4 tile + threads` kernel reached 517 GFLOP/s against NumPy's 542 at N=1024, and at N=2048 392 against
243. What does OpenBLAS do that we do not? Three things, each a good final project. **Cache blocking of
all three loops**: it blocks `k` as well as `i` and `j`, so that a block of `B` stays in L2 for any
matrix size (10.11 blocks only the output axes). **Packing**: it copies blocks of both operands into
contiguous, tile-ordered buffers, so the microkernel's loads are perfectly sequential.
**A hand-scheduled microkernel**: a register tile shaped for the register file (for example 6x16 on
AVX2: twelve vector accumulators, a few operand registers, no spills), with explicit broadcast loads and
prefetches. Each one is a few dozen lines on top of what you have. They matter most where our kernel is
weakest: large `K` and matrices that do not fit in L3, where 10.11 shows the output-only blocking start
to help.

## Pitfalls

**Threads made it slower (NOTES 19).** Adding `#pragma omp parallel for` to the strided GEMM took it from
1.9 to 1.1 GFLOP/s on battery, and on AC power sixteen threads gained only 1.4x (5.2 to 7.1). Parallelism multiplies memory pressure along with compute. Check locality
(strides along the innermost loop) before threading, and expect threads to help only once each thread's
inner loop is efficient.

**The missing flag (NOTES 20).** Without `-fassociative-math` gcc will not vectorize a float reduction:
the same tiled kernel ran at 53.3 instead of 452.2 GFLOP/s. If a reduce kernel's assembly shows only `ss`
(scalar single) instructions, check the flags first.

**The layout pass that copied too much (NOTES 21).** A layout pass that stages any strided view will
eventually stage a broadcast product larger than the computation it serves (a `(128, 784, 128)` tensor
in MNIST's backward pass, a 21 ms kernel). Copy operands *before* their EXPAND, and only when the copy
is strictly smaller than the reduce input.

**Spilled accumulators.** A 4x4 tile has sixteen vector accumulators, and with eight operand vectors
that is more than AVX2's sixteen registers; gcc spills five of them to the stack every iteration. The
kernel is still fast, but "it fits in registers" is a claim to verify with `gcc -S`, not to assume.

**Unrolling can defeat the vectorizer.** `Opts(upcast={0: 4, 1: 4}, unroll=4)` compiled to purely scalar
code and ran 6.5x slower than the same tile without unrolling. A transformation that helps a naive
compiler can hurt a smart one.

**The microbenchmark the compiler optimized away.** The first version of `peak.c` gave all FMA chains the
same initial values; gcc merged them and the measured "peak" was six times too high per instruction.
Read the assembly of every microbenchmark.

**Fixed costs dominate small kernels.** A fresh output buffer is page-faulted on first touch inside the
kernel (`DEBUG=3`'s "again" column runs each kernel a second time to show the difference: `0.603 ms`
versus `0.359 ms` for an AdamW update), and OpenMP's fork/join costs microseconds. Do not tune a
50-microsecond kernel by staring at its inner loop.

**Noisy benchmarks.** Laptop clocks depend on power source and temperature, the hybrid P/E cores make
thread placement matter, and other processes steal cores. The tiled GEMM at N=1024 measured anywhere
from 114 to 517 GFLOP/s in this chapter, and single rows of one table can even swap order. Take the best
of several runs, interleave the variants you compare (as `tiles.py` in 10.11 does), and report the
machine state with the numbers.

**A gain that was noise.** The captured benchmark showed 64x64 cache blocks 21% faster than the plain
tiled kernel at N=1024; the interleaved measurement in 10.11 showed the sign of that difference changing
from round to round. Before you keep an optimization, check that its effect is larger than the spread
between repeated runs.

## Exercises

1. ★ **LayerNorm on the roofline.** Compute the arithmetic intensity of a LayerNorm over the last axis of a `(B, D)` tensor
   (read `x`, `weight`, `bias`, write the output). Where does it sit on this machine's roofline, and what is
   its best possible GFLOP/s?

2. ★ **Predict a register tile.** Predict the generated C for `Opts(upcast={1: 2})` on an 8x8 GEMM: how many accumulators, how
   many loads of `A` and of `B` per `k` iteration? Check with `render_c(lower(...))`.

3. ★ **Latency-bound adds.** Explain from the assembly in 10.6 why the B^T naive kernel cannot exceed one vector add per
   add latency. Estimate its GFLOP/s at 3.7 GHz (the clock measured in 10.3) for an add latency of 4 cycles,
   then compare with the measured 28.0 GFLOP/s. What latency does the measurement imply?

4. ★★ **The largest tile that fits.** Add a `(4, 3)` and a `(2, 8)` tile to `candidate_opts`. Using `gcc -S`, find the largest tile
   that does not spill on AVX2 and compare its speed with 4x4 at N=1024.

5. ★★ **Split-K.** Implement split-K: for a GEMM with a small output (say `(16, 16) @ (16, 65536)^T`), threads on
   the output axes have nothing to do. Split the reduce axis into `T` chunks computed in parallel into a
   temporary `(T, M, N)`, then sum. Where in the pipeline does this belong: `opt.py`, rangeify, or the
   kernelizer?

6. ★★★ **Block the reduce axis.** Block the reduce axis. Add an `Opts` field `ktile` that splits the innermost reduce axis
   into an outer loop over chunks of, say, 256 and an inner loop, and place the chunk loop outside the tile
   loops of a cache block (10.11), so the loop order is block `i`, block `j`, chunk `k`, tile `i`, tile `j`,
   inner `k`. The sixteen accumulators of a tile can no longer live in registers across the whole reduction:
   allocate a persistent accumulator buffer of `tile_i x tile_j` floats per block (`Ctx.alloc` takes a
   size), zero it before the first chunk, add each chunk's partial sums into it, and store it to the output
   after the last chunk. Check correctness against NumPy for a `K` that is not a power of two, then measure
   at N=2048 and N=4096 against output blocking alone. *Hint:* where the ALLOC is placed decides whether it
   is per-thread; reread the scope rule in 10.7.

7. ★★ **OpenMP scheduling on a hybrid CPU.** Make OpenMP use `schedule(dynamic)` or `schedule(guided)` for THREAD loops and measure the
   B^T threads row on this hybrid (P-core plus E-core) CPU. Explain the result.

8. ★★ **Cache search results.** Give `search_opt` a persistent cache: key it by the kernel's rendered C *before* opts (or by
   `_structure_key`), store the winning `Opts` in `~/.cache/uopc`, and verify a second run of a benchmark
   does no timing.

9. ★★ **Duplicate index terms.** Find the cause of the duplicated `alu3 = (alu1+alu2)` / `alu4 = (alu2+alu1)` in the tiled
   kernels and fix it in the symbolic layer, without breaking `tests/test_w01_uops.py`.

10. ★★★ **Explicit float8 code generation.** Explicit `float8` code generation (10.12). Add a vector dtype (`dtypes.f32.vec(8)`), a
   VECTORIZE op that packs eight scalar UOps into one vector, and an op that extracts one lane. After
   rangeify, when an upcast axis has factor 8 and its index is contiguous in memory, rewrite the eight
   copies of each load, ALU op and store into one vector op. Render vectors as
   `float __attribute__((vector_size(32)))`, and use an unaligned load (`memcpy` into the vector, or
   `_mm256_loadu_ps`) unless you can prove the index is a multiple of 8. Vectorize the GEMM tile along `j`
   (broadcast `A[i, k]`, eight outputs per vector, no horizontal sum), check with `gcc -S` that the result
   uses packed FMAs even with `unroll=4`, and compare its speed with the best tile from exercise 4.

11. ★ **A key that forgot a field.** The first version of `search_opt` deduplicated candidates by
   `(upcast, thread, unroll)`. Explain why that silently dropped all three cache-blocked candidates, and why
   no test failed. Write a test that would have caught it.

12. ★ **Work per thread.** For the tiled GEMM at N=1024 with `thread=0`, how many iterations does the parallel loop have
   with no cache blocks, with 64x64 blocks and with 128x128 blocks? Use the answer to explain the slowest
   row of the N=1024 table in 10.11.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** Per element: subtract mean, square, multiply by the reciprocal standard deviation, multiply by
the weight, add the bias, plus the two reductions: roughly 8 FLOPs. Bytes: read `x` (4) and write the
output (4); `weight` and `bias` are `D` floats reused across all rows, and if the mean and variance passes
re-read `x` from cache rather than DRAM, the compulsory traffic is about 8 bytes per element. AI is
about 1 FLOP/byte, far left of the ridge (14): memory bound, at most about 56 GFLOP/s on this machine
with the 55.9 GB/s roof of 10.3.

**2.** Two accumulators (the 2 tile columns), one shared `k` loop; per iteration one load of `A`
(`A[i, k]`, shared by both) and two of `B` (`B^T[2j, k]`, `B^T[2j+1, k]`): three loads for two FMAs.

**3.** Every `vaddps` writes `%ymm1` and the next one reads it, so at most one vector add completes per
add latency. Each add finishes 8 lanes x 2 FLOPs (the multiply is independent and overlaps), so with a
4-cycle latency the bound is 4 FLOP/cycle, about 15 GFLOP/s at 3.7 GHz. The measured 28.0 is nearly twice
that, which implies about 2 cycles per iteration: 28.0 / 3.7 GHz is 7.6 FLOP/cycle, close to 16 FLOPs
every 2 cycles. The P-cores of this CPU (Raptor Cove, derived from Golden Cove) have dedicated
floating-point adders with a 2-cycle latency, half that of the FMA unit, so a chain of plain adds runs
twice as fast as a chain of FMAs.

**11.** The cache-blocked candidates copy `upcast`, `thread` and `unroll` from `hand_opt_choice`, so with
the old key they looked identical to the heuristic's own candidate, which was seen first, and were skipped.
Every test still passed because search only ever picks *some* valid candidate: dropping candidates changes
speed, never results. A test that checks the set of timed candidates (for example, that every `tile` from
`candidate_opts` reaches the timing loop) catches it. The fix adds `tuple(sorted(o.tile.items()))` to the key.

**12.** The THREAD loop is the first axis. Without blocks it runs over tiles of 4 rows: 1024 / 4 = 256
iterations, 16 per hardware thread. With blocks it runs over block rows: 1024 / 64 = 16 (one per
hardware thread) and 1024 / 128 = 8. With 8 iterations, half of the 16 hardware threads have no work,
so the 128x128 variant is the slowest in every round.

</details>

## Quiz

1. What is the arithmetic intensity of a 1024x1024x1024 `f32` GEMM, counting only compulsory DRAM
   traffic? Is it memory- or compute-bound on a machine with 777 GFLOP/s and 55.9 GB/s?
2. Why does reading a column of a row-major 1024x1024 `f32` matrix touch a new page on every element?
3. A 4x4 register tile: how many loads and FMAs per `k` step, and why does the tile also help with FMA
   latency?
4. Why must the accumulator of a parallel GEMM be declared inside the parallel loop?
5. What does `-fassociative-math` allow gcc to do, and why is it acceptable for ML kernels?
6. Why does the layout pass stage the operand *before* its EXPAND rather than the expanded view?
7. Why might adding threads make a kernel slower?
8. In one session `search_opt` measured `{0: 8}` faster than the heuristic's `{0: 4, 1: 4}`. Name one
   reason a heuristic based on "reuse per load" can pick the wrong tile.
9. 64x64 cache blocks cut the L3 traffic of the 1024 GEMM by 8x, yet did not make it reliably faster.
   Give two reasons, and name a case where blocking does help.

<details>
<summary>Answers</summary>

1. `2N^3 / (12N^2) = N/6 = 171` FLOP/byte. The ridge is `777 / 55.9 = 14`, so it is compute-bound (by a
   factor of twelve).
2. Consecutive elements of a column are one row apart: 1024 x 4 bytes = 4096 bytes, exactly one 4 KB
   page, and also a different 64-byte cache line.
3. 8 loads (4 of `A`, 4 of `B`) and 16 FMAs. The 16 accumulators are independent dependency chains, so
   the pipelined FMA units can overlap them instead of waiting on one chain's latency.
4. Each thread computes different outputs and needs its own running sum; a shared accumulator would be a
   data race. The linearizer places ALLOCs in the deepest common scope, inside the THREAD loop.
5. Reorder floating-point additions (for example into 8 lane-wise partial sums), which is what
   vectorizing a reduction requires. ML results already tolerate last-bit differences in sums; the
   kernels stay deterministic, and NaN/inf semantics are kept.
6. The unexpanded operand is `N*K` floats; the expanded view is `M*N*K`. Copying after the expand would
   materialize the broadcast and cost more than the GEMM.
7. Fork/join overhead on small kernels, and memory contention: threads multiply cache, TLB and bandwidth
   pressure, which can dominate when each thread's access pattern is poor.
8. Register pressure: a tile whose accumulators and operands exceed the register file spills; also gcc's
   vectorization choices, cache-line alignment of the tile, and the machine's load ports are not in the
   model. (And run-to-run noise: the two tiles swapped places between sessions.)
9. The removed traffic was not the bottleneck (L3 and the prefetchers already kept up with sequential
   row reads, and the register tile already keeps the `A` rows in L1); blocks are coarse units of parallel
   work (16 or 8 per parallel loop at N=1024); and the block's working set grows with `K`. Blocking helps
   when the operands do not fit in L3: at N=4096 every blocked variant beat the unblocked one.

</details>

## Further reading

- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore
  Architectures", *Communications of the ACM*, 2009.
- K. Goto and R. van de Geijn, "Anatomy of High-Performance Matrix Multiplication", *ACM TOMS*, 2008:
  the blocking and packing scheme behind OpenBLAS and BLIS.
- R. Bryant and D. O'Hallaron, *Computer Systems: A Programmer's Perspective*, chapters 5 (optimizing
  program performance) and 6 (the memory hierarchy).
- U. Drepper, "What Every Programmer Should Know About Memory", 2007.
- Intel, *Intel 64 and IA-32 Architectures Optimization Reference Manual*: latencies and throughputs of
  FMA and load units.
- The BLIS project's documentation of its microkernel ("BLIS: A Framework for Rapidly Instantiating BLAS
  Functionality", Van Zee and van de Geijn, *ACM TOMS*, 2015).
- tinygrad's `Opt`/`OptOps` kernel optimizations and its `BEAM` search (`beam_search`, in
  `tinygrad/engine/search.py` in recent versions): the production version of `Opts` and `search_opt`.
