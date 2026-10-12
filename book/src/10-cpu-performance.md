# 10. Fast CPU kernels

At the end of [Chapter 9](09-tensor-mnist.md) the compiler was correct and general. It was not fast.
Here is the opening position for this chapter, a 1024x1024x1024 matrix multiply in `f32`, compiled
straight from the shape graph, against NumPy (which calls OpenBLAS, decades of hand-tuned assembly):

```text
  B strided  naive                     1.9 GFLOP/s
  numpy                              131.6 GFLOP/s
```

Seventy times slower, for the same 2.15 billion floating-point operations. By the end of the chapter the
same compiler, with a couple of hundred lines of new code, produces a kernel that runs at 114 GFLOP/s in the same
measurement session, and on a quiet machine it has been seen to edge past NumPy. Nothing about the
math changes. What changes is *where the data is* when the arithmetic needs it: which cache level,
which register, which core. This chapter teaches you to reason about that quantitatively (FLOP counts,
bytes, arithmetic intensity, the roofline model), then implements the three transformations that
close the gap: a layout pass, multithreading, and register tiling.

**What you'll build**

- `Opts` in `uopc/rangeify.py`: `upcast` (register tiles), `unroll` (reduce unrolling) and `thread`
  (OpenMP), applied *during* rangeify, plus `merge_ends` to let unrolled copies share one loop.
- `uopc/opt.py`: the layout pass (`push_views`, `_last_axis_stride`, `layout_pass`), the hand-written
  heuristic `hand_opt_choice`, and a tiny autotuner, `search_opt`.
- `examples/bench_gemm.py`: a benchmark that walks the GEMM from 1.9 to 114 GFLOP/s step by step.

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
which is the same math with both operands contiguous along `k`. The captured run
(`book/outputs/bench_gemm_1024.txt`) is the table this chapter explains, row by row:

```text
N=1024  (2.15 GFLOP per GEMM)
  B strided  naive                     1.9 GFLOP/s   max err 2.1e-04
  B strided  threads                   1.1 GFLOP/s   max err 2.1e-04
  B strided  4x4 tile + threads       12.9 GFLOP/s   max err 2.1e-04
  B^T        naive                     8.0 GFLOP/s   max err 7.9e-05
  B^T        threads                  35.3 GFLOP/s   max err 7.9e-05
  B^T        4x4 tile                 27.1 GFLOP/s   max err 8.0e-05
  B^T        4x4 tile + threads      113.6 GFLOP/s   max err 8.0e-05
  Tensor A@B (layout pass+opt)       106.9 GFLOP/s   (end to end, includes the transpose copy)
  numpy                              131.6 GFLOP/s
```

Three things should surprise you. Adding threads to the strided version made it *slower*. Transposing
`B`, which does not change a single FLOP, made the naive kernel four times faster. And register tiling
without threads (27.1) is slower than threads without tiling (35.3), but the two together are worth
more than the product of their parts. By the end of the chapter each of these will be obvious.

**A word on measurement.** These numbers come from a laptop (Intel i7-13620H, gcc 15, see
`book/outputs/machine.txt`) that was on battery while this chapter was written, with other jobs running
at times. The same benchmark run again later, on a quieter machine, printed:

```text
  B strided  naive                     2.2 GFLOP/s   max err 2.1e-04
  B strided  threads                   1.4 GFLOP/s   max err 2.1e-04
  B strided  4x4 tile + threads       18.0 GFLOP/s   max err 2.1e-04
  B^T        naive                    12.2 GFLOP/s   max err 8.0e-05
  B^T        threads                  47.6 GFLOP/s   max err 8.0e-05
  B^T        4x4 tile                 32.2 GFLOP/s   max err 8.0e-05
  B^T        4x4 tile + threads      169.4 GFLOP/s   max err 8.0e-05
  Tensor A@B (layout pass+opt)       136.0 GFLOP/s   (end to end, includes the transpose copy)
  numpy                              162.7 GFLOP/s
```

Every row moved, some by 50%, and in this run the tiled kernel beat NumPy. The *ratios* between rows
are what this chapter explains, and they are stable; absolute numbers on a laptop are not. `bench`
in the script takes the best of five runs after a warm-up for exactly this reason. Always compare
variants within one session, and never trust a single run.

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

To know whether 1.9 GFLOP/s is bad, we need the machine's limits. The i7-13620H has 6 performance cores
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

Compiled with `gcc -O2 -march=native -fopenmp`, in the same session as the second benchmark table:

```text
 1 threads:    59.4 GFLOP/s
 6 threads:   174.8 GFLOP/s
10 threads:   220.9 GFLOP/s
16 threads:   293.6 GFLOP/s
```

Two lessons hide in this tiny program. The first draft initialized every chain to the same two
values, and gcc noticed that `a0`, `a2`, `a4`... computed identical things and merged them: the
disassembly had two FMAs per iteration, not twelve, and the "measurement" was off by a factor of six.
Always read the assembly of a microbenchmark. Second, 59 GFLOP/s on one core is well below the 157 of the
spec sheet: on battery this laptop holds its cores at much lower clocks (59.4 / 32 FLOP per cycle is
about 1.9 GHz). That is the real machine we are optimizing for, so we use the measured numbers.

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
scale  out=a*2  1 thread    11.8 GB/s
scale  out=a*2  threads     16.3 GB/s
rowsum sum(a)   1 thread    11.5 GB/s
rowsum sum(a)   threads     17.7 GB/s
```

About 17.7 GB/s when every core pulls (a plain C/OpenMP sum loop measured the same, 16.7 to 17.5 GB/s,
so the compiler is not the bottleneck here). One core alone gets about 11.5 GB/s: a single core cannot
keep enough cache misses in flight to saturate the memory system. The scale kernel's 16.3 GB/s
under-counts: writing a cache line the CPU does not have first *reads* it (write-allocate), so the true
traffic is 12 bytes per element, not 8.

## 10.4 The roofline model

The roofline model (Williams, Waterman and Patterson, 2009) combines the two limits into one picture. A
kernel with arithmetic intensity `AI` cannot run faster than either limit:

`attainable GFLOP/s = min(peak GFLOP/s, AI * peak GB/s)`

Plotted on log-log axes this is a slanted line (memory-bound region) meeting a flat line (compute-bound
region) at the **ridge point**, `AI = 294 / 17.7 = 17` FLOP per byte on this machine. A kernel to the
left of the ridge can only go faster by moving fewer bytes; a kernel to the right can only go faster by
doing its arithmetic faster. Here is the machine with the measured kernels:

@diagram 10-roofline

(The roofs and the GEMM points come from the second, quieter session above, so that all numbers in the
chart were measured within a few minutes of each other.)

The two elementwise kernels sit on the slanted roof: they are as fast as memory allows, and no
transformation in this chapter will help them. (Fusing them into their neighbours, so the data is not
written and re-read, is the only lever, and that is [Chapter 7](07-kernelize.md)'s business.) The GEMM
is far to the right of the ridge: with 171 FLOPs per byte it could, in principle, run at the full 294
GFLOP/s. The naive version runs 130 times below its roof.

> **Key idea:** the DRAM roofline says the naive GEMM is not limited by memory bandwidth. Its problem is
> lower in the hierarchy: how often it touches the caches, how often registers are reloaded, and how
> much of the core's parallelism (SIMD lanes, FMA pipelines, cores) it uses. The same roofline idea,
> applied per level (L1, L2, registers), explains every row of the benchmark.

The roofline also explains the MNIST kernels from [Chapter 9](09-tensor-mnist.md). The AdamW update of
`W1` touches about 1.6 MB with an intensity near 0.6, so its roof is about 10 GFLOP/s; it ran at
1.5 GFLOP/s. The gap is not the kernel's fault. Its output is a new parameter buffer, freshly allocated
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

Why did threads make this *slower* (1.9 to 1.1 GFLOP/s)? Each thread runs the same column walk on
different rows of `A`, so sixteen hardware threads now each stream a 4 KB-stride pattern, and pairs of
them share a core's L1, L2 and TLB. More threads means more simultaneous page walks and more cache lines
competing for the same small structures, while the arithmetic was never the bottleneck. This is NOTES 19
in `book/NOTES.md`, and its lesson generalizes: **parallelism multiplies whatever your kernel does,
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
thread goes from 1.9 to 8.0 GFLOP/s. Why not more? Look at `%ymm1`: there is *one* accumulator, and every
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

At 1024x1024 the copy costs about 2.5 ms against about 14 ms for the GEMM (`DEBUG=2`, steady state):

```text
  e_1_1024_1024                     2.504 ms     0.00 GFLOP/s
  r_1024_1024                      13.942 ms   154.03 GFLOP/s
realize: 2 kernels, schedule 0.1 ms, run 16.8 ms
```

An `O(N^2)` copy buys an `O(N^3)` kernel a factor of several; the larger the matrix, the better the deal.
This is the "Tensor A@B" row of the benchmark: 106.9 GFLOP/s end to end, copy included.

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
contiguous layout, threads take the naive kernel from 8.0 to 35.3 GFLOP/s: about 4.4x on a machine with 6
fast and 4 slow cores. gcc's OpenMP runtime splits the iterations statically into equal chunks, so on a
hybrid CPU the performance cores probably finish early and wait for the efficiency cores (exercise 10.7
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
gcc removes the duplicate, so it costs nothing at runtime, but it is a small blemish (exercise 10.9).

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
need registers too, and the comment now says so.) Spills to L1 are cheap, and this kernel still runs at 114 to 169 GFLOP/s with threads, but it is a reminder to
check the assembly rather than trust the arithmetic. The autotuner in 10.11 found that, at N=512, an
8x1 tile (eight accumulators, nine operands) beat the 4x4.

Tiling also explains the curious ordering in the benchmark. Tiling alone (27.1 GFLOP/s) is slower than
threads alone (35.3), because one core cannot beat sixteen hardware threads however well it uses its
registers. But threads alone are limited by each thread's single dependency chain and two loads per FMA,
and tiling alone is limited to one core. Together, each of sixteen threads runs a latency-hiding,
load-efficient inner loop: 113.6 GFLOP/s. On the strided layout, tiling rescues even the column walk
(1.1 to 12.9) because the four `j` of a tile share each fetched cache line.

## 10.9 The flag that matters most: `-fassociative-math`

Look back at the B^T naive inner loop: `vmulps` then `vaddps` into a vector of partial sums. That loop
adds `A[i, 0..7] * B[j, 0..7]` lane by lane into eight separate running sums and combines them at the
end. In exact arithmetic that is the same as adding the products left to right. In floating point it is
not: `(a + b) + c` and `a + (b + c)` can differ in the last bits. The C standard requires gcc to evaluate
`acc0 = acc0 + x` in program order, so without permission it may not vectorize a float reduction at all.

The permission is `-fassociative-math`, which the runtime passes by default:

@include uopc/runtime.py#CFLAGS

Here is the scratch experiment that measured it (same kernels, only `CFLAGS` changed):

```text
CFLAGS: -O3 -march=native -fopenmp -ffp-contract=fast -fassociative-math -fno-signed-zeros -fno-trapping-math
  B^T naive                    6.0 GFLOP/s
  B^T 4x4 tile + threads     105.0 GFLOP/s
CFLAGS: -O3 -march=native -fopenmp -ffp-contract=fast
  B^T naive                    1.4 GFLOP/s
  B^T 4x4 tile + threads      17.5 GFLOP/s
```

A factor of six from one flag, on a kernel that is otherwise identical; NOTES 20 records 17 versus 170 in
an earlier run. Every ML compiler makes this trade: reductions are reordered for speed, so a sum in `f32`
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
N=1024, single-threaded:

```text
  upcast={} thread=None unroll=1:    8.4 GFLOP/s
  upcast={} thread=None unroll=4:    8.3 GFLOP/s
  upcast={0: 4, 1: 4} thread=None unroll=1:   31.9 GFLOP/s
  upcast={0: 4, 1: 4} thread=None unroll=4:    4.5 GFLOP/s
  upcast={0: 2, 1: 4} thread=None unroll=1:   17.5 GFLOP/s
```

On its own, unrolling changes nothing (gcc vectorized the plain loop already). Combined with the 4x4
tile it is a disaster: the assembly of the unrolled tile contains only scalar instructions
(`vfmadd231ss`, `vmulss`, `vaddss`) and not a single packed one. Sixty-four unrolled products feeding
sixteen trees were more than gcc's vectorizer would untangle, so it gave up. `hand_opt` uses `unroll`
only for short reductions with no tile, where it removes the loop entirely (the
`test_reduce_unroll_has_no_inner_loop` test unrolls a 16-element sum into straight-line code). The general
lesson: **when you generate C, gcc is part of your backend, and its heuristics are part of your cost
model.**

## 10.11 Choosing the opts

### A hand-written heuristic

`hand_opt_choice` encodes the lessons of this chapter in a dozen lines:

@include uopc/opt.py#hand_opt_choice

If the kernel reduces, tile the last two output axes 4x4 (falling back to 2 if the size is not
divisible); if it is a short reduction with no tile, unroll it; and if there is at least `2^16` units of
work, thread the first axis that still has at least two iterations after tiling. Applied to a few
shapes:

```text
GEMM 1024^3            Opts(upcast={1: 4, 0: 4}, thread=0, unroll=1, gpu=False, local={})
row max (128,10)       Opts(upcast={0: 4}, thread=None, unroll=1, gpu=False, local={})
elementwise (784,128)  Opts(upcast={}, thread=0, unroll=1, gpu=False, local={})
elementwise (128,)     Opts(upcast={}, thread=None, unroll=1, gpu=False, local={})
sum (16,)              Opts(upcast={}, thread=None, unroll=4, gpu=False, local={})
```

The `row max (128, 10)` case is the `log_softmax` kernel from the MNIST step: four rows per iteration, no
threads (only 1280 elements). `HandOpt` packages the layout pass and this heuristic for the kernelizer;
it is the default (`OPT=1`), and `OPT=0` turns both off:

@include uopc/opt.py#HandOpt

### Search

A heuristic encodes what you know. Search finds what you do not. `search_opt` compiles a list of
candidate `Opts` for the kernel, times each on random data, and keeps the fastest:

@include uopc/opt.py#candidate_opts

@include uopc/opt.py#search_opt

It is a one-step version of tinygrad's `BEAM` search: no beam, no iteration, just "try the plausible
tiles on the last two axes, with and without unrolling, and measure". Enable it with `OPT=2`
(`SearchOpt`). Here is what it sees for a 512x512 GEMM (a scratch copy of `search_opt` that prints every
candidate's time instead of only keeping the best):

```text
29 distinct candidates; hand_opt picks {1: 4, 0: 4} thread=0
   117.8 GFLOP/s  upcast={0: 8} thread=0 unroll=1
   102.8 GFLOP/s  upcast={0: 4, 1: 4} thread=0 unroll=1
   102.5 GFLOP/s  upcast={0: 8, 1: 4} thread=0 unroll=1
    97.8 GFLOP/s  upcast={0: 8, 1: 8} thread=0 unroll=1
    91.7 GFLOP/s  upcast={1: 16} thread=0 unroll=1
    91.1 GFLOP/s  upcast={0: 16, 1: 2} thread=0 unroll=1
  ...
    15.4 GFLOP/s  upcast={1: 2} thread=0 unroll=4
    13.5 GFLOP/s  upcast={1: 16} thread=0 unroll=4
    11.7 GFLOP/s  upcast={} thread=None unroll=1
```

The heuristic's choice is second, within 15% of the winner, and every candidate with `unroll=4` is near
the bottom, as 10.10 predicted. Search pays for its answers: compiling and timing 29 variants took about
13 seconds of wall time for this one kernel. That is why it is opt-in, and why production autotuners
cache their results on disk keyed by the kernel (exercise 10.8).

## 10.12 Results

Putting it together, the captured table from 10.1, with the reason for each step:

| version | GFLOP/s | what changed |
|---|---|---|
| B strided, naive | 1.9 | column walk: new line and new page every `k` |
| B strided, threads | 1.1 | sixteen threads fighting over caches and TLBs |
| B strided, 4x4 tile + threads | 12.9 | four `j` share each line; still strided |
| B^T, naive | 8.0 | contiguous and vectorized, but one dependency chain |
| B^T, threads | 35.3 | 16 threads x the naive inner loop |
| B^T, 4x4 tile | 27.1 | 2 FMAs per load, 16 independent chains, one core |
| B^T, 4x4 tile + threads | 113.6 | both |
| Tensor `A @ B` (layout pass + `hand_opt`) | 106.9 | the compiler does all of the above, transpose included |
| numpy (OpenBLAS) | 131.6 | cache blocking, packing, hand-written microkernels |

What does OpenBLAS do that we do not? Three things, each a good final project. **Cache blocking**: it
tiles the `k` and `j` loops so that a block of `B` stays in L2 while many rows of `A` stream past it;
our kernel re-reads each row of `B^T` from L3 for every tile of rows. **Packing**: it copies blocks of
both operands into contiguous, tile-ordered buffers, so the microkernel's loads are perfectly sequential.
**A hand-scheduled microkernel**: a register tile shaped for the register file (for example 6x16 on
AVX2: twelve vector accumulators, a few operand registers, no spills), with explicit broadcast loads and
prefetches. Each one is a few dozen lines on top of what you have, and the last 15% is where they live.

## Pitfalls

**Threads made it slower (NOTES 19).** Adding `#pragma omp parallel for` to the strided GEMM took it from
1.9 to 1.1 GFLOP/s. Parallelism multiplies memory pressure along with compute. Check locality
(strides along the innermost loop) before threading, and expect threads to help only once each thread's
inner loop is efficient.

**The missing flag (NOTES 20).** Without `-fassociative-math` gcc will not vectorize a float reduction:
the same tiled kernel ran at 17.5 instead of 105 GFLOP/s. If a reduce kernel's assembly shows only `ss`
(scalar single) instructions, check the flags first.

**The layout pass that copied too much (NOTES 21).** A layout pass that stages any strided view will
eventually stage a broadcast product larger than the computation it serves (a `(128, 784, 128)` tensor
in MNIST's backward pass, a 21 ms kernel). Copy operands *before* their EXPAND, and only when the copy
is strictly smaller than the reduce input.

**Spilled accumulators.** A 4x4 tile has sixteen vector accumulators, and with eight operand vectors
that is more than AVX2's sixteen registers; gcc spills five of them to the stack every iteration. The
kernel is still fast, but "it fits in registers" is a claim to verify with `gcc -S`, not to assume.

**Unrolling can defeat the vectorizer.** `Opts(upcast={0: 4, 1: 4}, unroll=4)` compiled to purely scalar
code and ran 7x slower than the same tile without unrolling. A transformation that helps a naive
compiler can hurt a smart one.

**The microbenchmark the compiler optimized away.** The first version of `peak.c` gave all FMA chains the
same initial values; gcc merged them and the measured "peak" was six times too high per instruction.
Read the assembly of every microbenchmark.

**Fixed costs dominate small kernels.** A fresh output buffer is page-faulted on first touch inside the
kernel (`DEBUG=3`'s "again" column runs each kernel a second time to show the difference: `0.603 ms`
versus `0.359 ms` for an AdamW update), and OpenMP's fork/join costs microseconds. Do not tune a
50-microsecond kernel by staring at its inner loop.

**Noisy benchmarks.** Laptop clocks depend on power source and temperature; other processes steal
cores. The same benchmark moved by up to 50% between two sessions in this chapter. Take the best of
several runs, compare variants within one run, and report the machine state with the numbers.

## Exercises

**10.1** ★ Compute the arithmetic intensity of a LayerNorm over the last axis of a `(B, D)` tensor
(read `x`, `weight`, `bias`, write the output). Where does it sit on this machine's roofline, and what is
its best possible GFLOP/s?

**10.2** ★ Predict the generated C for `Opts(upcast={1: 2})` on an 8x8 GEMM: how many accumulators, how
many loads of `A` and of `B` per `k` iteration? Check with `render_c(lower(...))`.

**10.3** ★ Explain from the assembly in 10.6 why the B^T naive kernel cannot exceed one vector add per
add-latency, and estimate its GFLOP/s at 1.9 GHz with a 4-cycle add latency.

**10.4** ★★ Add a `(4, 3)` and a `(2, 8)` tile to `candidate_opts`. Using `gcc -S`, find the largest tile
that does not spill on AVX2 and compare its speed with 4x4 at N=1024.

**10.5** ★★ Implement split-K: for a GEMM with a small output (say `(16, 16) @ (16, 65536)^T`), threads on
the output axes have nothing to do. Split the reduce axis into `T` chunks computed in parallel into a
temporary `(T, M, N)`, then sum. Where in the pipeline does this belong: `opt.py`, rangeify, or the
kernelizer?

**10.6** ★★ Implement cache blocking of the reduce axis: split `k` into an outer loop over blocks of 256
and an inner loop, with the output tile loops between them. Measure at N=2048. *Hint:* the accumulator
must now live across the outer `k` loop.

**10.7** ★★ Make OpenMP use `schedule(dynamic)` or `schedule(guided)` for THREAD loops and measure the
B^T threads row on this hybrid (P-core plus E-core) CPU. Explain the result.

**10.8** ★★ Give `search_opt` a persistent cache: key it by the kernel's rendered C *before* opts (or by
`_structure_key`), store the winning `Opts` in `~/.cache/uopc`, and verify a second run of a benchmark
does no timing.

**10.9** ★★ Find the cause of the duplicated `alu3 = (alu1+alu2)` / `alu4 = (alu2+alu1)` in the tiled
kernels and fix it in the symbolic layer, without breaking `tests/test_w01_uops.py`.

**10.10** ★★★ Write a vectorizing renderer: when an upcast axis has factor 8 and is contiguous in memory,
emit `float __attribute__((vector_size(32)))` loads and arithmetic instead of eight scalar copies, so the
tile no longer depends on gcc's auto-vectorizer. Compare against 10.4's best tile.

<details>
<summary>Solutions to the ★ exercises</summary>

**10.1** Per element: subtract mean, square, multiply by the reciprocal standard deviation, multiply by
the weight, add the bias, plus the two reductions: roughly 8 FLOPs. Bytes: read `x` (4) and write the
output (4); `weight` and `bias` are `D` floats reused across all rows, and if the mean and variance passes
re-read `x` from cache rather than DRAM, the compulsory traffic is about 8 bytes per element. AI is
about 1 FLOP/byte, far left of the ridge (17): memory bound, at most about 17.7 GFLOP/s on this machine.

**10.2** Two accumulators (the 2 tile columns), one shared `k` loop; per iteration one load of `A`
(`A[i, k]`, shared by both) and two of `B` (`B^T[2j, k]`, `B^T[2j+1, k]`): three loads for two FMAs.

**10.3** Every `vaddps` writes `%ymm1` and the next one reads it, so at most one vector add completes per
add latency: 8 lanes x 2 FLOPs (the multiply is independent and overlaps) per 4 cycles is 4 FLOP/cycle,
about 7.6 GFLOP/s at 1.9 GHz, before the horizontal reduction at the end of each dot product. The
measured 8.0 (and 12.2 in the quieter session, presumably at a higher clock) are close to that bound.

</details>

## Quiz

1. What is the arithmetic intensity of a 1024x1024x1024 `f32` GEMM, counting only compulsory DRAM
   traffic? Is it memory- or compute-bound on a machine with 294 GFLOP/s and 17.7 GB/s?
2. Why does reading a column of a row-major 1024x1024 `f32` matrix touch a new page on every element?
3. A 4x4 register tile: how many loads and FMAs per `k` step, and why does the tile also help with FMA
   latency?
4. Why must the accumulator of a parallel GEMM be declared inside the parallel loop?
5. What does `-fassociative-math` allow gcc to do, and why is it acceptable for ML kernels?
6. Why does the layout pass stage the operand *before* its EXPAND rather than the expanded view?
7. Why might adding threads make a kernel slower?
8. `search_opt` measured `{0: 8}` faster than the heuristic's `{0: 4, 1: 4}`. Name one reason a
   heuristic based on "reuse per load" can pick the wrong tile.

<details>
<summary>Answers</summary>

1. `2N^3 / (12N^2) = N/6 = 171` FLOP/byte. The ridge is `294 / 17.7 = 17`, so it is compute-bound (by a
   factor of ten).
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
   model.

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
