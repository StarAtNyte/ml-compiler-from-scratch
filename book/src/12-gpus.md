# 12. GPU architecture and mapping kernels

> **Later hardware validation:** the original chapter's emulation and unavailable-driver notes
> describe its original runs. [Chapter 35](35-register-tiled-gpu.md) now adds a separate real-device
> GEMM implementation and recorded RTX 3050 results, with its own correctness and timing scope.

Every kernel you have generated so far is a loop nest that one CPU core, or a dozen of them under OpenMP, walks
through in order. A GPU offers a different deal. You don't write the outer loops. You describe what *one
iteration* does, say how many iterations you want, and the hardware runs tens of thousands of them at once
on a chip built for throughput rather than latency. This chapter teaches enough GPU architecture to see why that
deal is good and where it bites: SIMT execution, warps, the memory hierarchy, occupancy, and coalescing. Then it
shows how small the compiler change is. In `uopc` a GPU kernel is the same linearized loop nest as a CPU kernel,
except that some ranges are tagged `GLOBAL` or `LOCAL` and render as `blockIdx` and `threadIdx` where a CPU
kernel would have a `for`.

**What you'll build**

- Two new axis types, `GLOBAL` (one per thread block) and `LOCAL` (one per thread in a block), and the
  `Opts.gpu`/`Opts.local` knobs that rangeify uses to split an output axis into `(block, thread, upcast)` parts.
- `CUDARenderer`, a 30-line subclass of the C renderer that turns those ranges into `blockIdx.x` and
  `threadIdx.x`, plus `launch_dims`, which reads the grid and block sizes back off the IR.
- `gpu_opt_choice`, a heuristic that picks a launch configuration for any kernel.
- `emulate`, which tests every GPU kernel on the CPU by rendering it as plain C where the GPU ranges are loops.
- A CUDA runtime in about 90 lines of `ctypes`: NVRTC compiles the source, the driver API launches it.

**Where this fits.** [Chapter 10](10-cpu-performance.md) made CPU kernels fast with register tiles (UPCAST)
and OpenMP threads (THREAD). This chapter reuses that machinery and moves it to a GPU.
[Chapter 13](13-shared-memory-tensor-cores.md) adds shared memory, barriers and tensor cores, which are the
features that make GPU GEMMs fast.

> **A note on what was measured.** The reference machine for this book (an Intel i7-13620H laptop, see
> `book/outputs/machine.txt`) has no working NVIDIA driver or CUDA toolkit. Every GPU kernel in this chapter
> is therefore checked in two ways. The *rendered CUDA source* is shown exactly as `render_cuda` prints it. The
> *numerical results* come from the CPU emulator, which runs the same index arithmetic. Hardware numbers
> (SM counts, bandwidths, latencies) are general knowledge taken from vendor documentation and are labeled
> that way. Nothing in this chapter is a GPU timing measured by the author. Section 12.11 shows how a reader
> who has a GPU can run the real thing.

## 12.1 Why GPUs: latency machines and throughput machines

A CPU core is a *latency machine*. Out-of-order execution, branch prediction, big caches and prefetchers all
exist so that one thread's next instruction can run as soon as possible. In [Chapter 10](10-cpu-performance.md)
we pushed one such chip to 114 GFLOP/s on a 1024³ GEMM, using every core and AVX2 FMAs, against OpenBLAS's 132
(`book/outputs/bench_gemm_1024.txt`).

A GPU is a *throughput machine*. It spends its transistors on arithmetic units and register files instead of
on making one thread fast. Here are the numbers for an NVIDIA A100 (SXM, 40 GB), from NVIDIA's A100
whitepaper. These are general knowledge, not measured here:

| | i7-13620H (this book's laptop) | A100 |
|---|---|---|
| cores / SMs | 6 P + 4 E cores | 108 streaming multiprocessors |
| peak FP32 | under 1 TFLOP/s (AVX2 FMA; see Chapter 10's roofline) | 19.5 TFLOP/s |
| peak FP16 tensor core | none | 312 TFLOP/s (dense) |
| DRAM bandwidth | tens of GB/s (DDR5) | 1555 GB/s (HBM2) |
| threads resident at once | 16 hardware threads | 108 × 2048 = 221,184 |

The last row is the most important one. A GPU thread is slow: it issues in order, it has no branch predictor,
and a load from DRAM takes on the order of 500 cycles to return. The GPU gets its speed by keeping *hundreds
of thousands* of threads resident. When one warp stalls on a load, the scheduler issues an instruction from
another warp in the next cycle, at no cost. Latency is hidden by having plenty of other work, not by being
clever about any single thread.

How much other work is needed? **Little's law**: the bytes in flight equal the bandwidth times the latency. At
1555 GB/s and roughly 600 ns of memory latency, 1555·10⁹ × 600·10⁻⁹ ≈ 930 KB must be in flight at all times
to saturate DRAM. That is about 8.6 KB per SM, or about 2,200 outstanding 4-byte loads per SM, and an SM can
hold at most 2,048 threads. So even a full SM barely keeps memory busy if every thread has one scalar load
outstanding. Fast kernels also need several independent loads per thread (instruction-level parallelism,
which is what UPCAST gives us) or wide vector loads. Keep this calculation in mind. It explains occupancy,
upcasting, and much of the folklore about GPU performance.

The roofline from [Chapter 10](10-cpu-performance.md) still applies, with a different ridge point:
19.5 TFLOP/s ÷ 1.555 TB/s ≈ **12.5 FLOP per byte** for FP32. A kernel that does less arithmetic than that
per byte of DRAM traffic is memory-bound on an A100. An elementwise add does 1 FLOP per 12 bytes. A naive GEMM
that reloads both operands for every multiply-add does 2 FLOPs per 8 bytes. Both sit far below the ridge.
Chapter 13 is about pushing GEMM above it.

## 12.2 The execution model: grids, blocks, warps, threads

CUDA's model has four levels. Learn them now. The rest of the chapter maps the compiler onto them.

- A **kernel** is a function marked `__global__`. The host *launches* it with a grid size and a block size.
- The **grid** is a 1-, 2- or 3-D array of **thread blocks**. `blockIdx.{x,y,z}` says which block you are,
  and `gridDim` says how many blocks there are.
- A **block** is a 1-, 2- or 3-D array of **threads**, at most 1024 in total. `threadIdx.{x,y,z}` says which
  thread you are inside the block. Threads in one block can share fast on-chip memory (`__shared__`) and wait
  for each other (`__syncthreads()`). Threads in *different* blocks can do neither.
- The hardware runs the threads of a block in groups of 32 called **warps**. A warp has one instruction
  stream. Each cycle the SM's scheduler picks a ready warp and issues one instruction for all of its 32 lanes.
  The model is SIMT, "single instruction, multiple threads": you write scalar code for one thread, and the
  hardware runs 32 of them in lockstep, much as one AVX instruction works on 8 floats.

@diagram 12-gpu-hierarchy

Blocks are scheduled onto **streaming multiprocessors** (SMs). A block is assigned to one SM when it starts
and stays there until it finishes. An SM holds as many blocks at once as its resources allow (Section 12.4),
and the blocks in a grid run in an order that is not specified. That is why blocks cannot synchronize with
each other. Block 900 may not have started when block 0 finishes, so a barrier between them would deadlock.
The only communication between blocks is through global memory, normally across a kernel boundary.

The thread index has a fixed linearization: `tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y *
threadIdx.z)`, and warp *w* is threads `32w … 32w+31`, so **`.x` varies fastest**. Consecutive
`threadIdx.x` values sit in the same warp and execute the same load instruction in the same cycle. That single
fact decides which loop axis we map to `.x` (Section 12.5).

**Divergence.** All 32 lanes share one instruction stream. When lanes take different sides of a branch, the
warp runs both sides one after the other, with the inactive lanes masked off. A `WHERE` in our IR renders as a
`?:` select, which compiles to a predicated select instruction rather than a branch, so `uopc` kernels rarely
diverge. Since Volta each thread has its own program counter ("independent thread scheduling"), but
divergent paths are still serialized, so the cost model hasn't changed.

## 12.3 The memory hierarchy

Performance on a GPU is mostly a question of *where the data is*. The levels, with A100 numbers as a reference
point (general knowledge; latencies are rough orders of magnitude):

| level | scope | size (A100) | latency, roughly | in `uopc` |
|---|---|---|---|---|
| registers | one thread | 64K × 32-bit per SM, ≤ 255 per thread | ~1 cycle | `acc0`, `val3`, `alu1` |
| shared memory | one block | up to 164 KB per SM (carved from 192 KB L1) | ~20–30 cycles | `ALLOC` with `AddrSpace.SHARED` (Ch. 13) |
| L1 cache | one SM | the rest of the 192 KB | ~30 cycles | automatic |
| L2 cache | whole GPU | 40 MB | ~200 cycles | automatic |
| global (HBM) | whole GPU and host-visible | 40–80 GB | ~500 cycles | `BUFFER` (`data0`, …) |

Three consequences matter for the compiler.

1. **Registers are the fastest memory and they are allocated statically.** Every `float acc0;` in a kernel is a
   register. A 4×4 register tile costs 16 registers per thread *for every thread*. Use too many and fewer
   threads fit on the SM (next section). Use more than the hardware has and the compiler *spills* to "local
   memory", which in spite of the name is DRAM-backed and slow.
2. **Shared memory is a scratchpad managed by software.** Unlike a cache it is explicit: you load into it, you
   synchronize, you read from it. In exchange you get guaranteed on-chip reuse. Chapter 13 is about using it.
3. **Global memory is fast only in bulk.** DRAM moves data in 32-byte *sectors*, and a warp's 32 loads are
   combined into as few sectors as possible (Section 12.5).

## 12.4 Occupancy

**Occupancy** is the ratio of warps resident on an SM to the maximum it could hold (64 warps = 2048 threads on
an A100). It matters because of the Little's-law calculation in 12.1: resident warps are what the scheduler
can switch to while other warps wait on memory.

Three resources limit how many blocks fit on an SM. Here they are worked through for a block of 256 threads
(8 warps), using A100 limits: 65,536 registers, 164 KB of shared memory, 2,048 threads and 32 blocks per SM.

| per-thread registers | shared memory per block | blocks limited by regs | by smem | by threads | resident blocks | occupancy |
|---|---|---|---|---|---|---|
| 32 | 0 | 65536 / (256·32) = 8 | – | 2048 / 256 = 8 | 8 | 64/64 = 100% |
| 64 | 0 | 65536 / (256·64) = 4 | – | 8 | 4 | 32/64 = 50% |
| 32 | 48 KB | 8 | ⌊164 / 48⌋ = 3 | 8 | 3 | 24/64 = 37.5% |
| 128 | 16 KB | 65536 / (256·128) = 2 | 10 | 8 | 2 | 16/64 = 25% |

(Real allocators round registers up to granules of 256 per warp, so treat these numbers as idealized. The
CUDA occupancy calculator, or `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, gives exact figures.)

Each `uopc` kernel is declared `__launch_bounds__(N)`, where N is the block size. That is a promise to the
compiler that the kernel never launches with more than N threads per block, so it can budget registers for
exactly N. Without it, `nvcc`/NVRTC must assume 1024 threads and may restrict registers too much.

Occupancy is a means to an end. Vasily Volkov's well-known GTC 2010 talk, "Better Performance at Lower
Occupancy", shows kernels that run faster at 25% occupancy because each thread keeps many independent loads
and FMAs in flight. Little's law counts *bytes in flight*. Threads are one way to get them and ILP is another.
This is the GPU version of the register-tile argument from Chapter 10: an upcast kernel with 4 accumulators
per thread has 4 independent FMA chains, so it needs a quarter as many warps to hide the same latency.

## 12.5 Coalescing: how a warp talks to DRAM

When a warp executes a load, the hardware collects the 32 addresses and fetches the set of 32-byte sectors
that cover them. The cost is the number of distinct sectors, not the number of threads.

- 32 threads reading 32 consecutive floats (128 bytes, aligned) → **4 sectors**. Every byte fetched is used.
  This is a *coalesced* access.
- 32 threads reading floats 256 bytes apart → **32 sectors**, of which 4 bytes in 32 are used. The access
  costs 8× more bandwidth for the same work.
- 32 threads reading *the same* address → **1 sector**, broadcast to all lanes. That is free.

The rule for a code generator follows: **the loop axis that indexes memory contiguously must be driven by
`threadIdx.x`.** We can check this on real IR. The script below lowers our usual 64×64×64 GEMM with 16×16
blocks, takes warp 0 of block (0,0), and evaluates every global `INDEX` for its 32 lanes by substituting
concrete values for the ranges and calling `simplify`. That is constant folding (Chapter 3) used as an
address calculator. It does so twice: once with `.x` driving the last output axis *n*, as `CUDARenderer` does,
and once with `.x` driving the first axis *m*.

```python
# scratch/coalesce.py (scratch script, not part of the repo)
import sys; sys.path.insert(0, "tests")
from helpers import buf
from uopc.uop import UOp, Ops, AxisType
from uopc.rangeify import Opts
from uopc.symbolic import simplify
from uopc.gpu import gpu_lower

def warp_addresses(ast, x_axis_first_dim: bool):
  """For warp 0 of block (0,0): the byte addresses each global LOAD/STORE touches at reduce step 0."""
  rngs = sorted({u for u in ast.toposort() if u.op is Ops.RANGE}, key=lambda r: r.arg)
  loc = [r for r in rngs if r.arg.kind is AxisType.LOCAL]          # loc[0] from axis 0 (M), loc[1] from axis 1 (N)
  xs, ys = (loc[1], loc[0]) if not x_axis_first_dim else (loc[0], loc[1])
  out = {}
  for u in ast.toposort():
    if u.op is Ops.INDEX and u.src[0].op is Ops.BUFFER:
      addrs = []
      for t in range(32):                                            # threads 0..31 = one warp
        tx, ty = t % xs.src[0].arg, t // xs.src[0].arg
        sub = {r: UOp.const(0) for r in rngs}; sub[xs] = UOp.const(tx); sub[ys] = UOp.const(ty)
        addrs.append(simplify(u.src[1].substitute(sub)).arg * 4)
      out[f"data{u.src[0].arg.slot}"] = addrs
  return out

M = N = K = 64
A, B = buf(0, M * K), buf(1, K * N)
g = (A.reshape((M, 1, K)).expand((M, N, K)) * B.reshape((K, N)).permute((1, 0)).reshape((1, N, K)).expand((M, N, K))).reduce(Ops.ADD, 1)
ast = gpu_lower(g, buf(2, M * N), Opts(gpu=True, local={0: 16, 1: 16}))
for label, flag in [("threadIdx.x <- last axis (n)", False), ("threadIdx.x <- first axis (m)", True)]:
  print(label)
  for name, a in warp_addresses(ast, flag).items():
    sectors = len({x // 32 for x in a})
    print(f"  {name}: first addresses {a[:4]}...  distinct={len(set(a)):2d}  32B sectors={sectors:2d}")
```

```text
threadIdx.x <- last axis (n)
  data2: first addresses [0, 4, 8, 12]...  distinct=32  32B sectors= 4
  data0: first addresses [0, 0, 0, 0]...  distinct= 2  32B sectors= 2
  data1: first addresses [0, 4, 8, 12]...  distinct=16  32B sectors= 2
threadIdx.x <- first axis (m)
  data2: first addresses [0, 256, 512, 768]...  distinct=32  32B sectors=16
  data0: first addresses [0, 256, 512, 768]...  distinct=16  32B sectors=16
  data1: first addresses [0, 0, 0, 0]...  distinct= 2  32B sectors= 1
```

Read the store to `data2` (the output C) first. With the right mapping, warp 0 writes two rows of 16
consecutive floats into 4 sectors, the minimum possible. With the wrong mapping the same 32 stores land in 16
different sectors, a 4× increase in traffic for the store. The loads show the full picture. With the right
mapping, `A[m, k]` (`data0`) is the same address for all 16 lanes of a row, a broadcast, and `B[k, n]`
(`data1`) is 16 consecutive floats. With the wrong mapping these swap, and A is now read down a column, 16
sectors per warp.

@diagram 12-coalescing

This is NOTES 23 from the development log: "CUDA's `.x` dimension is the fastest-varying across a warp: map
the *last* (contiguous) axis to `.x` or loads/stores are uncoalesced." Our tensors are row-major, so the last
output axis is the one whose neighbours are adjacent in memory, at least in the output. `CUDARenderer` sorts
the ranges so that the highest-numbered `GLOBAL` and `LOCAL` ranges, which come from the last axis, get `.x`.

## 12.6 From loops to a grid: GLOBAL and LOCAL ranges

> **Key idea:** A GPU kernel is the body of a loop nest whose outer loops the hardware runs for you. In the IR
> nothing changes except the *kind* of some ranges. A `GLOBAL` range of size G is "for each of G blocks",
> and a `LOCAL` range of size L is "for each of L threads in the block". The renderer prints a hardware index
> where a `for` would have been.

Recall `AxisType` from Chapter 10. Weeks 7–8 add the last two members:

@include uopc/uop.py#AxisType

The choice of how to map a kernel is an `Opts`, the same dataclass as in Chapter 10, with two new fields.
`gpu=True` turns output axes into `GLOBAL` ranges, and `local` gives, per output axis, the number of threads
per block along that axis:

@include uopc/rangeify.py#Opts

### Splitting one axis three ways

An output axis of size `s` becomes three nested parts: `s = blocks × local × upcast`. The index is

```text
index = (gidx * local + lidx) * upcast + c        c = 0 .. upcast-1, unrolled
```

where `gidx` is a `GLOBAL` range of size `s / (local·upcast)`, `lidx` is a `LOCAL` range of size `local`, and
`c` is a constant (each of the `upcast` copies of the body gets its own). That is `_gpu_axis`:

@include uopc/rangeify.py#_gpu_axis

It reuses the machinery from Chapter 10. `_gpu_axis` returns the same `(base, factor)` pair that `_axis_parts`
returns for CPU axes, and `_combos` turns those pairs into the unrolled index tuples. Two details
matter:

- An axis that needs only one block gets `CONST 0` instead of a range, so a 16×32 kernel run as one block has
  no `blockIdx` at all (you will see this in the softmax kernel in 12.9).
- The assert demands exact divisibility. Production compilers pad the grid up and guard the stores with
  `if (idx < n)`. We kept the simpler contract and made `gpu_opt_choice` choose divisors. Exercise 12.5 asks
  you to add guards.

`lower` uses `_gpu_axis` when `opts.gpu` is set. Here is the whole function again, now that its GPU branch
makes sense:

@include uopc/rangeify.py#lower

`_gpu_axis` is called once per output axis, so the range numbers *interleave*: axis 0's `GLOBAL` is range 0, axis 0's
`LOCAL` is range 1, axis 1's `GLOBAL` is range 2, and so on. On a GPU the order is irrelevant, because neither
kind of range is a loop. It does decide the loop order in the emulator, which you will see in 12.10. The
emulated results are correct either way, since in a kernel without shared memory every order of the threads
gives the same answer.

@diagram 12-grid-mapping

## 12.7 The CUDA renderer

The C renderer from Chapter 4 was written with this chapter in mind. It has a few class attributes and hook
methods that a subclass can override:

@include uopc/render_c.py#CRenderer.render_range

@include uopc/render_c.py#CRenderer.is_loop

`is_loop` returning `True` for *every* range, GPU ranges included, is what will make the emulator free. The
CUDA renderer overrides it, along with the prefixes, the range naming and the header:

@include uopc/gpu.py#CUDARenderer

Line by line:

- `buffer_prefix`, `shared_prefix`, `barrier`, `restrict` are the spelling differences. `__restrict__` is
  CUDA's `restrict`, and it tells the compiler that `data0` and `data2` never alias, so loads can be
  reordered and cached in the read-only path. `shared_prefix` and `barrier` are used in Chapter 13.
- `render` sorts the GPU ranges by `RangeArg` in *reverse* order, so the range created last, which comes from
  the last output axis, becomes `gidx0`/`lidx0` and is bound to `.x`. That is the coalescing rule from 12.5,
  implemented in one `sorted(..., reverse=True)`.
- The kernel prefix is `extern "C" __global__ void __launch_bounds__(threads)`. `extern "C"` turns off C++
  name mangling, so `cuModuleGetFunction` can find the kernel by its plain name.
- `is_loop` is `False` for `GLOBAL`/`LOCAL` ranges. The base renderer then neither indents nor emits a closing
  `}` for them (look at the `END` case in `CRenderer.render`).
- `render_range` declares `int gidx0 = blockIdx.x;` and keeps the range size as a comment, because the
  hardware, not the code, enforces the bound.

The prelude replaces `<math.h>`. NVRTC compiles without the host's system headers, so `INFINITY` and `NAN`
have to be defined from bit patterns, and the floor-division helpers from Chapter 4 become `__device__`
functions:

@include uopc/gpu.py#CUDA_PRELUDE

Finally, `launch_dims` reads the grid and block sizes back off the IR, in the same reversed order the renderer
used:

@include uopc/gpu.py#launch_dims

### First kernel: an elementwise op

Let's put a `(64, 128)` elementwise kernel, `c = a*2 + b`, through the whole path:

```python
# scratch/e1.py
import numpy as np
from uopc.uop import UOp, Ops, ParamArg
from uopc.dtype import dtypes
from uopc.rangeify import Opts
from uopc.gpu import gpu_lower, gpu_opt_choice, emulate, render_cuda, launch_dims
from uopc.runtime import Program
def buf(slot, n, d=dtypes.f32): return UOp(Ops.BUFFER, (), ParamArg(slot, d, (n,)))
# elementwise: c = a*2 + b over (64, 128)
A = buf(0, 64*128).reshape((64,128)); B = buf(1, 64*128).reshape((64,128))
body = A * 2.0 + B
o = gpu_opt_choice(body); print("opts:", o)
ast = gpu_lower(body, buf(2, 64*128), o)
print("launch:", launch_dims(ast))
print(render_cuda(ast, "add2"))
print(emulate(ast, "add2"))
a = np.random.randn(64*128).astype(np.float32); b = np.random.randn(64*128).astype(np.float32); c = np.zeros_like(a)
Program(emulate(ast, "add2"), "add2")(a, b, c)
print("max error:", np.abs(c - (a*2+b)).max())
```

```text
opts: Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={1: 16, 0: 16})
launch: ((8, 4, 1), (16, 16, 1))
```

```c
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
__device__ inline int floordiv_i32(int a, int b) { int q = a / b; return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q; }
__device__ inline int floormod_i32(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
extern "C" __global__ void __launch_bounds__(256) add2(float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2) {
  int gidx1 = blockIdx.y; /* 4 */
  int lidx1 = threadIdx.y; /* 16 */
  int gidx0 = blockIdx.x; /* 8 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu1 = (((gidx1*2048)+(lidx1*128))+((gidx0*16)+lidx0));
  float val2 = data0[alu1];
  float val3 = data1[alu1];
  data2[alu1] = ((val2*2.0f)+val3);
}
```

There are 8×4 = 32 blocks of 16×16 = 256 threads, one thread per output element. The row is
`gidx1*16 + lidx1` and the column is `gidx0*16 + lidx0`. The flat address `row*128 + col` has been simplified
into the single sum `alu1`. Nothing in this kernel loops. Its "loop nest" is the launch.

Compare the emulator's version of the same kernel, printed by the same script:

```c
void add2(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 16; ridx1++) {
      for (int ridx2 = 0; ridx2 < 8; ridx2++) {
        for (int ridx3 = 0; ridx3 < 16; ridx3++) {
          int alu1 = (((ridx0*2048)+(ridx1*128))+((ridx2*16)+ridx3));
          float val2 = data0[alu1];
          float val3 = data1[alu1];
          data2[alu1] = ((val2*2.0f)+val3);
        }
      }
    }
  }
}
```

```text
max error: 0.0
```

It is the same linearized program with four `for` loops added. Range 0 is axis 0's `GLOBAL`, range 1 is axis
0's `LOCAL`, then axis 1's two ranges. That is the interleaving mentioned in 12.6. The body and its index
arithmetic are identical, character for character, apart from the names.

## 12.8 GEMM on the grid

The elementwise case is easy because every thread is independent and touches one element. A GEMM adds a reduce
loop and makes the choices matter. Here are three configurations of the 64×64×64 GEMM from
`tests/test_w07_gpu.py`, each rendered, emulated, and compared against numpy:

```python
# scratch/e2.py
import sys; sys.path.insert(0, "tests")
import numpy as np
from helpers import buf
from uopc.uop import Ops
from uopc.rangeify import Opts
from uopc.gpu import gpu_lower, gpu_opt_choice, emulate, render_cuda, launch_dims
from uopc.runtime import Program
def gemm_graph(M, N, K):
  A, B = buf(0, M * K), buf(1, K * N)
  return (A.reshape((M, 1, K)).expand((M, N, K)) * B.reshape((K, N)).permute((1, 0)).reshape((1, N, K)).expand((M, N, K))).reduce(Ops.ADD, 1)
M, N, K = 64, 64, 64
g = gemm_graph(M, N, K)
for name, o in [("plain", Opts(gpu=True)), ("local", Opts(gpu=True, local={0: 8, 1: 16})), ("auto", gpu_opt_choice(g))]:
  ast = gpu_lower(g, buf(2, M*N), o)
  print(f"=== {name}: {o}\nlaunch (grid, block): {launch_dims(ast)}")
  print(render_cuda(ast, "gemm").split("floormod_i32(int a, int b)")[1].split("\n",1)[1])
  a, b = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32); c = np.zeros(M*N, np.float32)
  Program(emulate(ast, "gemm"), "gemm")(a, b, c)
  print("emulated max error vs numpy:", np.abs(c.reshape(M,N) - a@b).max())
```

**Configuration 1: one thread per block.**

```text
=== plain: Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={})
launch (grid, block): ((64, 64, 1), (1, 1, 1))
```

```c
extern "C" __global__ void __launch_bounds__(1) gemm(float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2) {
  int gidx1 = blockIdx.y; /* 64 */
  int alu1 = (gidx1*64);
  int gidx0 = blockIdx.x; /* 64 */
  float acc0;
  acc0 = 0.0f;
  for (int ridx2 = 0; ridx2 < 64; ridx2++) {
    float val2 = acc0;
    float val3 = data0[(alu1+ridx2)];
    float val4 = data1[((ridx2*64)+gidx0)];
    acc0 = (val2+(val3*val4));
  }
  float val5 = acc0;
  data2[(alu1+gidx0)] = val5;
}
```

```text
emulated max error vs numpy: 5.722046e-06
```

This is correct and wasteful. Every block is a single thread, and a warp is still 32 lanes wide, so 31 of every
32 lanes do nothing. You pay for 32 lanes and use one. Notice the reduce loop: it is the ordinary `RANGE` of
kind `REDUCE` from Chapter 6, and the accumulator `acc0` is a register. The linearizer placed it inside the
`gidx` scope exactly as it would place a CPU accumulator inside an output loop. None of that code changed.

**Configuration 2: 16×8 threads per block.**

```text
=== local: Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={0: 8, 1: 16})
launch (grid, block): ((4, 8, 1), (16, 8, 1))
```

```c
extern "C" __global__ void __launch_bounds__(128) gemm(float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2) {
  int gidx1 = blockIdx.y; /* 8 */
  int lidx1 = threadIdx.y; /* 8 */
  int alu1 = ((gidx1*512)+(lidx1*64));
  int gidx0 = blockIdx.x; /* 4 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu2 = ((gidx0*16)+lidx0);
  float acc0;
  acc0 = 0.0f;
  for (int ridx4 = 0; ridx4 < 64; ridx4++) {
    float val3 = acc0;
    float val4 = data0[(alu1+ridx4)];
    float val5 = data1[((ridx4*64)+alu2)];
    acc0 = (val3+(val4*val5));
  }
  float val6 = acc0;
  data2[(alu1+alu2)] = val6;
}
```

```text
emulated max error vs numpy: 5.722046e-06
```

Now 128 threads cooperate in each block. Look at the loads inside the loop with a warp in mind. A warp is two
values of `lidx1` × sixteen of `lidx0`. `data0[alu1 + ridx4]` does not depend on `lidx0`, so the 16 lanes
in a row read the same word, a broadcast. `data1[ridx4*64 + alu2]` reads 16 consecutive floats per row. Both
are coalesced, which is the analysis from 12.5 seen in source.

**Configuration 3: what `gpu_opt_choice` picks.**

```text
=== auto: Opts(upcast={1: 2, 0: 2}, thread=None, unroll=1, tile={}, gpu=True, local={1: 16, 0: 16})
launch (grid, block): ((2, 2, 1), (16, 16, 1))
```

```c
extern "C" __global__ void __launch_bounds__(256) gemm(float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2) {
  int gidx1 = blockIdx.y; /* 2 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu1 = ((gidx1*2048)+(lidx1*128));
  int gidx0 = blockIdx.x; /* 2 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu2 = ((gidx0*32)+(lidx0*2));
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
  for (int ridx4 = 0; ridx4 < 64; ridx4++) {
    float val5 = acc0;
    int alu6 = (alu1+ridx4);
    float val7 = data0[alu6];
    int alu8 = (ridx4*64);
    float val9 = data1[(alu8+alu2)];
    acc0 = (val5+(val7*val9));
    float val10 = acc1;
    float val11 = data1[((alu2+alu8)+1)];
    acc1 = (val10+(val7*val11));
    float val12 = acc2;
    float val13 = data0[(alu6+64)];
    acc2 = (val12+(val13*val9));
    float val14 = acc3;
    acc3 = (val14+(val13*val11));
  }
  float val15 = acc0;
  data2[alu3] = val15;
  float val16 = acc1;
  data2[(alu4+1)] = val16;
  float val17 = acc2;
  data2[(alu3+64)] = val17;
  float val18 = acc3;
  data2[(alu4+65)] = val18;
}
```

```text
emulated max error vs numpy: 7.6293945e-06
```

This is Chapter 10's register tile running on a GPU. Each thread owns a 2×2 tile of C: four accumulators, two
loads of A and two of B per iteration, four FMAs. That doubles the FLOPs per load and gives four independent
FMA chains, which is the ILP that Volkov's argument asks for. Only 4 blocks are launched, though, which would
leave 104 of an A100's 108 SMs idle. At 64×64 the problem is simply too small for a GPU. A real launch
configuration has to balance work per thread against enough blocks to fill the machine. Exercise 12.8 asks
you to add that constraint to the heuristic.

The 2×2 upcast has a cost that is easy to miss. The column index is `alu2 = gidx0*32 + lidx0*2`, so
neighbouring lanes are now *two* floats apart. Each B load (`data1[alu8+alu2]` and the `+1` one) touches every
other word across 16 lanes. The pair of loads together covers a contiguous range, and the second one will
usually hit in L1, but each instruction alone is only half-coalesced. The fix is to make the unrolled copies
the *slow* part of the index, `(gidx*upcast + c)*local + lidx` rather than `(gidx*local + lidx)*upcast + c`.
That is Exercise 12.4. CPU compilers want the unrolled elements adjacent (one SIMD vector per thread). GPU
compilers want the *threads* adjacent (one vector per warp). The same UPCAST needs a different index layout on
each target.

One more detail appears in the source. `alu3 = (alu1+alu2)` and `alu4 = (alu2+alu1)` are the same value
written twice. The symbolic engine does not put the operands of commutative ops in a canonical order, so hash
consing sees two different nodes. NVRTC will merge them, so the cost is cosmetic, but it shows how much
generated code quality depends on canonicalization (Chapter 3).

## 12.9 Choosing a launch configuration

`gpu_opt_choice` is the GPU counterpart of `hand_opt_choice` from Chapter 10. It is a dozen lines of
heuristics:

@include uopc/gpu.py#gpu_opt_choice

The rules, and the reasons for them:

1. **At most three non-trivial output axes go on the grid**, the last three, because CUDA has three grid
   dimensions and three block dimensions. A kernel with four (GPT's attention scores are `(B, H, T, T)`)
   raises an error. The fix is to fold two axes into one with a reshape before lowering (Exercise 12.7), the
   same trick `tinygrad` uses.
2. **Threads go on the last axes first**, so `.x` gets the contiguous axis (12.5).
3. **Power-of-two thread counts that divide the axis**, at most 16 per axis when there are several axes (16×16
   = 256) and up to 256 for a 1-D kernel. 256 threads is a common default: 8 warps, enough to hide latency,
   and small enough to leave registers for a tile.
4. **A 2×2 upcast if the kernel reduces**, because a reduce loop reloads its inputs on every iteration and a
   register tile cuts those loads in half (Chapter 10).

Its choices for some shapes, from a scratch script:

```python
# scratch/e3.py (excerpt)
for shape in [(1024,), (4096,), (64, 64), (3, 100), (8, 64, 64)]:
  b = buf(0, int(np.prod(shape))).reshape(shape)
  print(shape, "->", gpu_opt_choice(b + 1.0))
s = (8, 4, 64, 64)
try: gpu_opt_choice(buf(0, int(np.prod(s))).reshape(s) + 1.0)
except NotImplementedError as e: print(s, "-> NotImplementedError:", e)
```

```text
(1024,) -> Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={0: 256})
(4096,) -> Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={0: 256})
(64, 64) -> Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={1: 16, 0: 16})
(3, 100) -> Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={1: 4})
(8, 64, 64) -> Opts(upcast={}, thread=None, unroll=1, tile={}, gpu=True, local={2: 16, 1: 16})
(8, 4, 64, 64) -> NotImplementedError: more than 3 non-trivial output axes: reshape first
```

`(3, 100)` shows the weakness of the divisibility rule. 100 = 4 × 25, so the best power of two is 4 threads per
block, and three is odd, so it gets none. The result is 75 blocks of 4 threads, one eighth of a warp each.
Padding to 104 or 128 with guarded stores would give full warps. This is the same lesson as Configuration 1:
a block size smaller than 32 wastes lanes.

### A kernel with a reduction inside

The second test in `tests/test_w07_gpu.py` is the numerator of a softmax, `exp2(x - max(x))` over rows of a
16×32 matrix. The `max` is a `REDUCE` whose result is broadcast back over the row:

```text
Opts(upcast={1: 2}, thread=None, unroll=1, tile={}, gpu=True, local={1: 16, 0: 16})
((1, 1, 1), (16, 16, 1))
```

```c
extern "C" __global__ void __launch_bounds__(256) softmax_num(float* __restrict__ data0, float* __restrict__ data1) {
  int lidx1 = threadIdx.y; /* 16 */
  int alu1 = (lidx1*32);
  float acc0;
  acc0 = -INFINITY;
  for (int ridx2 = 0; ridx2 < 32; ridx2++) {
    float val2 = acc0;
    float val3 = data0[(alu1+ridx2)];
    acc0 = ((val2>val3)?val2:val3);
  }
  float val4 = acc0;
  float alu5 = (val4*-1.0f);
  int lidx0 = threadIdx.x; /* 16 */
  int alu6 = (lidx0*2);
  int alu7 = (alu1+alu6);
  float val8 = data0[alu7];
  data1[alu7] = exp2f((val8+alu5));
  int alu9 = ((alu6+alu1)+1);
  float val10 = data0[alu9];
  data1[alu9] = exp2f((val10+alu5));
}
```

```text
max rel error: 0.0
```

The whole problem fits in one block, so both axes became `CONST 0` + `LOCAL` and no `blockIdx` appears. The
`exp2` axis has `upcast={1: 2}`, so each of the 16 `lidx0` threads writes two adjacent elements. Notice where
the linearizer put the reduce loop: *before* `int lidx0 = threadIdx.x;`. The max depends only on the row
(`lidx1`), so its scope does not include `lidx0`, and the linearizer emits it as early as it can. On a CPU
that ordering would be loop-invariant code motion, computing the max once per row instead of once per
element. On a GPU every thread still runs the whole kernel, so all 16 threads of a row compute the same max,
each reading the full row. The work is 16× redundant, and it's cheap here only because the row is in cache. A
real GPU compiler computes the reduction *cooperatively*: each thread reduces a slice, and the partial results
are combined through shared memory or warp shuffles (`__shfl_xor_sync`). Exercise 12.6 sketches it. This is
the most important difference between CPU and GPU code generation in this chapter. **Loop-invariant on a CPU
means "hoist it". Thread-invariant on a GPU means "everybody does it".**

## 12.10 Testing without a GPU: the emulator

`emulate` is the shortest function in the chapter:

@include uopc/gpu.py#emulate

It is the *C* renderer, applied to the GPU-lowered IR. `CRenderer.is_loop` returns `True` for every range,
`GLOBAL` and `LOCAL` included, so each one becomes a `for` loop, and the resulting C runs through the ordinary
gcc runtime from Chapter 4. Why is that a valid test?

A CUDA kernel without shared memory or barriers specifies one independent program per thread. The program for
thread `(b, t)` reads its inputs, computes, and writes *its own* output elements. If no two threads write the
same element and no thread reads what another writes (true of everything rangeify produces: each output
element is written by exactly one combination of indices), then every interleaving of the threads gives the
same result, and that includes running them one after another in nested loops. The emulator runs the same
`alu` expressions, the same accumulator placement and the same unrolled copies as the CUDA kernel. If the
emulated kernel computes `a @ b`, then the index math, the reduce lowering and the upcast are all right. Only
the spelling of `blockIdx` is left untested, and the rendering tests cover that.

This is how all of `tests/test_w07_gpu.py` runs on the reference machine:

@include tests/test_w07_gpu.py#test_gpu_gemm_emulated

Each parameterization lowers the GEMM with different GPU opts, runs the emulated C, compares with numpy, and
then checks that the CUDA rendering mentions `threadIdx` exactly when there are `LOCAL` axes. The run on this
machine:

```text
tests/test_w07_gpu.py::test_gpu_gemm_emulated[opts0] PASSED              [  9%]
tests/test_w07_gpu.py::test_gpu_gemm_emulated[opts1] PASSED              [ 18%]
tests/test_w07_gpu.py::test_gpu_gemm_emulated[opts2] PASSED              [ 27%]
tests/test_w07_gpu.py::test_gpu_softmax_like_emulated PASSED             [ 36%]
tests/test_w07_gpu.py::test_launch_dims_and_auto_choice PASSED           [ 45%]
tests/test_w07_gpu.py::test_smem_gemm_renders PASSED                     [ 54%]
tests/test_w07_gpu.py::test_wmma PASSED                                  [ 63%]
tests/test_w07_gpu.py::test_cuda_smem_gemm_on_gpu SKIPPED (no CUDA d...) [ 72%]
```

What the emulator *cannot* tell you:

- **Anything involving barriers.** If threads communicate through shared memory, running them one after
  another is wrong: thread 0 would read the tile before thread 255 has written its part. Chapter 13 shows the
  failure and works around it. (NOTES 25: "Barriers can't be emulated by running threads one after
  another".)
- **Races** that only a parallel execution exposes, such as two threads accumulating into the same output.
- **CUDA-specific compile errors.** The emulator compiles C with gcc, not CUDA C++ with NVRTC.
- **Performance.** An emulated kernel's speed says nothing about the GPU's.

The emulator gives up all of that in exchange for a property that matters a lot in a course: every GPU code
path in the compiler is tested on every student's laptop, in milliseconds, with an oracle (numpy).

## 12.11 Running on a real GPU: `cuda.py`

To run on an NVIDIA GPU we need three things: a compiler from CUDA C++ to something the GPU can load, a way to
move memory to and from the device, and a way to launch. `uopc` does all three with `ctypes` and two shared
libraries, so it needs no `nvcc` and no PyTorch:

- **NVRTC** (`libnvrtc.so`, part of the CUDA toolkit) is a runtime compiler. It takes CUDA source as a string
  and produces **PTX**, NVIDIA's virtual assembly.
- The **driver API** (`libcuda.so`, installed with the NVIDIA driver) loads PTX, which the driver JIT-compiles
  to the GPU's machine code (SASS), and manages contexts, memory and launches.

The whole runtime:

@include uopc/cuda.py#CUDA

The calls, in order:

1. `cuInit(0)`, `cuDeviceGet`, `cuCtxCreate_v2`. A *context* is the GPU equivalent of a process address space.
   All allocations and modules belong to one. `cuDeviceComputeCapability` returns e.g. `(8, 0)` for an A100,
   giving `sm_80`.
2. `compile`: `nvrtcCreateProgram` → `nvrtcCompileProgram` with `--gpu-architecture=compute_80` → fetch the
   PTX. On failure it fetches the compiler log and raises it with the source attached. You will want that the
   first time a renderer change produces invalid CUDA.
3. `function`: `cuModuleLoadData` (PTX → SASS, inside the driver) and `cuModuleGetFunction` by name, which is
   why the kernel is `extern "C"`.
4. `alloc`/`to_device`/`to_host`: `cuMemAlloc_v2` returns a 64-bit device pointer, a `CUdeviceptr`, which is
   just an integer. `cuMemcpyHtoD_v2`/`DtoH_v2` copy to and from numpy arrays.
5. `launch`: `cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedBytes, stream, kernelParams, extra)`.
   `kernelParams` is an array of *pointers to* the arguments. Each argument is a `CUdeviceptr`, so we pass the
   address of each `c_uint64`. That double indirection is the classic first-time bug, and the reason for the
   `ctypes.cast(ctypes.byref(p), c_void_p)`. Then `cuCtxSynchronize` waits for the kernel, because launches
   are asynchronous.

`run_cuda` packages the round trip:

@include uopc/cuda.py#run_cuda

**On this machine** `find_library("cuda")` succeeds, since driver libraries are installed but no working driver is loaded, while NVRTC is missing altogether. So the constructor fails before any CUDA call:

```python
from uopc.cuda import CUDA
try: CUDA()
except Exception as e: print(type(e).__name__+':', e)
```

```text
OSError: none of ('nvrtc',) found
```

So `cuda.py` has not been executed on the reference machine (NOTES 25). It follows the documented driver and
NVRTC APIs, and the test below is the way to check it. Treat it as your week-7 starting point, not as proven
code.

### How to run the GPU test if you have a GPU

@include tests/test_w07_gpu.py#test_cuda_smem_gemm_on_gpu

The test is skipped unless `CUDA()` constructs successfully, so it runs automatically on a machine with a
working setup:

1. Install an NVIDIA driver (`nvidia-smi` should list your GPU) and the CUDA toolkit, or just NVRTC, from the
   CUDA redistributables.
2. Make sure `ctypes.util.find_library("cuda")` and `find_library("nvrtc")` find the libraries. On Linux they
   look in the `ldconfig` cache, so either run `ldconfig` after installing, or add the toolkit's `lib64`
   directory to `LD_LIBRARY_PATH`. Check with
   `python -c "import ctypes.util as u; print(u.find_library('cuda'), u.find_library('nvrtc'))"`.
3. Run `.venv/bin/python -m pytest tests/test_w07_gpu.py::test_cuda_smem_gemm_on_gpu -v`. It should report
   `PASSED` rather than `SKIPPED`.

If it fails, the most likely culprits are the ones `ctypes` can't check for you. Functions return `int` by
default, which is right for `CUresult` and `nvrtcResult`, but wrong if you add any API that returns a
pointer. Check the size types (`size_t` versus `int`), and the `_v2` suffixes, which are the real exported
symbol names. To time kernels, add `cuEventCreate`/`cuEventRecord`/`cuEventElapsedTime` around the launch.
Wall-clock timing in Python also includes the launch overhead, typically several microseconds.

## 12.12 Beyond CUDA: HIP, Metal, OpenCL

All GPU programming models share the structure of 12.2 and differ in spelling. Since `CUDARenderer` is a set of
spellings on top of `CRenderer`, porting to another API means writing another subclass. The mapping:

| concept | CUDA | HIP (AMD) | Metal (Apple) | OpenCL |
|---|---|---|---|---|
| kernel qualifier | `__global__` | `__global__` | `kernel` | `__kernel` |
| block / thread index | `blockIdx.x`, `threadIdx.x` | same as CUDA | `[[threadgroup_position_in_grid]]`, `[[thread_position_in_threadgroup]]` arguments | `get_group_id(0)`, `get_local_id(0)` |
| block is called | thread block | workgroup / block | threadgroup | work-group |
| SIMD width | warp = 32 | wavefront = 64 (CDNA), 32 or 64 (RDNA) | SIMD-group = 32 | sub-group, device-dependent |
| shared memory | `__shared__` | `__shared__` (LDS) | `threadgroup` | `__local` |
| barrier | `__syncthreads()` | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | `barrier(CLK_LOCAL_MEM_FENCE)` |
| runtime compiler | NVRTC | hipRTC | `newLibraryWithSource` | `clBuildProgram` |
| launch | `cuLaunchKernel` | `hipModuleLaunchKernel` | `dispatchThreadgroups` | `clEnqueueNDRangeKernel` |

HIP is close to source-compatible with CUDA. A `HIPRenderer` is mostly `CUDARenderer` with a different
prelude, and `hipRTC`/`hipModuleLaunchKernel` mirror NVRTC and the driver API call for call. One thing does
change on AMD: wavefronts on data-center GPUs are 64 wide, so blocks of 64 or 256 threads, and coalescing
across 64 lanes, are the units to plan around. Metal and OpenCL pass the indices as kernel arguments or
built-in calls rather than as global variables, which means overriding `render_range` and the parameter list.
That is still a subclass.

## Pitfalls

**1. The wrong axis on `.x` (NOTES 23).** The obvious way to bind ranges is in creation order, `x, y, z`. That
gives the *first* output axis `.x`. Every kernel still produces correct results, in the emulator and on the GPU,
because coalescing is a performance property, not a correctness property. Section 12.5 shows the cost: four
times the store traffic for a GEMM output, and A read down a column. The fix is one `reverse=True` in
`CUDARenderer.render` and `launch_dims`. The lesson: **a test that only checks numbers will never catch this.**
Check the access pattern instead, by evaluating the index expressions for one warp, as the script in 12.5 does.

**2. Blocks smaller than a warp.** `Opts(gpu=True)` with no `local` launches one thread per block (12.8), and
`gpu_opt_choice` on a `(3, 100)` tensor launches blocks of 4. Both are correct and both waste 7/8 or more of
the machine. Hardware allocates warps, not threads.

**3. Indivisible sizes.** `_gpu_axis` asserts `size % (local * factor) == 0`. Without guarded stores, any other
choice would write out of bounds. On a GPU an out-of-bounds write may silently corrupt a neighbouring buffer,
and the emulator would not catch it either (it is C, without sanitizers). If you add padding (Exercise 12.5),
also add an emulator test with a size like 100 that doesn't divide.

**4. Four or more axes.** `gpu_opt_choice` raises `NotImplementedError` for `(8, 4, 64, 64)`. The attention
kernels of [Chapter 14](14-gpt.md) have exactly this shape, so moving GPT to the GPU requires folding axes
first.

**5. Thinking the emulator covers barriers.** Run `emulate` on the shared-memory GEMM of Chapter 13 and
it refuses (the first version of `emulate` crashed with an opaque `TypeError` from the renderer's
`barrier = None`; it now checks up front):

```text
NotImplementedError: emulate: kernels with BARRIER need all threads to run in lockstep; running threads one after another (what loops do) would be wrong
```

Supplying a no-op barrier spelling to force it through makes it render, and then it computes garbage. Chapter 13 shows the exact
numbers. The emulator is sound only for kernels whose threads don't communicate.

**6. Hoisting looks free and isn't.** The softmax kernel in 12.9 computes each row's max once per *thread*,
16 times per row. Placement decisions that are optimal for a CPU, such as loop-invariant hoisting, become
redundant work when the "loop" is a set of parallel threads.

**7. `INFINITY` doesn't exist in NVRTC.** NVRTC does not see the host's `<math.h>`, so the C prelude's
`INFINITY` (used to initialize max-reductions) would fail to compile. `CUDA_PRELUDE` defines it and `NAN`
from bit patterns with `__int_as_float`. Every spelling the renderer emits has to exist in the target's
dialect, and the emulator can't check that for you, since it compiles with gcc.

## Exercises

1. ★ **Predict the launch.** For the 32×64×48 GEMM of `test_gpu_gemm_emulated`, predict `launch_dims` for
   `Opts(gpu=True, local={0: 4, 1: 8})` and for `Opts(gpu=True, local={1: 16}, upcast={0: 2, 1: 2})`. Then
   check.
2. ★ **Count the lanes.** For each configuration in 12.8, compute the number of warps launched and the
   fraction of lanes that do useful work.
3. ★★ **Find the bug.** A student's renderer binds ranges in creation order (`sorted(rngs, key=lambda r:
   r.arg)`, with no `reverse`). All of their tests pass. Write a test that fails for their renderer and passes
   for `CUDARenderer`. *Hint:* don't run anything. Inspect `render_cuda(ast)` or evaluate the index of a
   store for lanes 0 and 1, as in 12.5.
4. ★★ **Coalesced upcasts.** Change the index layout for GPU axes from `(g*local + l)*f + c` to
   `(g*f + c)*local + l`, so that neighbouring lanes stay adjacent in every unrolled copy. Use the warp-address
   script to show that every B load in Configuration 3 now touches 2 sectors per instruction, not 4, and that
   the emulator tests still pass. *Hint:* only `_gpu_axis` and `_combos` need to know the layout.
5. ★★ **Guarded kernels.** Support sizes that local × upcast does not divide. Round the grid up, and wrap each
   store in a condition `idx < size`. You can express the condition in the IR as a `WHERE` around an `INDEX`
   to a dummy slot, or add a `gate` source to `STORE`, rendered as `if (...)`. Test `(3, 100)` with 32 threads
   per block in the emulator.
6. ★★ **Cooperative row max.** Write, by hand in the loop IR (like `smem_gemm_ir` in Chapter 13), a kernel with
   one block per row and 32 threads per block, where each thread takes the max over a strided slice of the row
   and lane 0 combines the 32 partials from a `SHARED` array after a `BARRIER`. Render it with
   `render_cuda`. Explain why the emulator cannot check it.
7. ★★ **Fold axes.** Make `gpu_lower` accept kernels with more than three non-trivial output axes by
   reshaping the leading axes into one before lowering, and un-reshaping the store index. Test on
   `(8, 4, 64, 64)` in the emulator.
8. ★★★ **Fill the machine.** Extend `gpu_opt_choice` with a `min_blocks` parameter (say 2 × the SM count). It
   should drop the upcast, and then shrink `local`, until the grid has at least that many blocks. Write down
   the trade-off you are making in each step.
9. ★★★ **HIP backend.** Write `HIPRenderer` and a `hip.py` runtime mirroring `cuda.py` (`hiprtc`,
   `hipModuleLoadData`, `hipModuleLaunchKernel`). Without AMD hardware, test only rendering. With it, port
   `test_cuda_smem_gemm_on_gpu`.
10. ★★★ **Occupancy calculator.** Write `occupancy(threads_per_block, regs_per_thread, smem_per_block,
    limits)` returning resident blocks and occupancy, reproduce the table in 12.4, and add the 256-register
    allocation granularity per warp from the CUDA occupancy calculator.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** M=32 with 4 threads per block gives 8 blocks along y. N=64 with 8 threads gives 8 blocks along x. So the
grid is `(8, 8, 1)` and the block is `(8, 4, 1)`, with `.x` first and coming from axis 1. With `local={1: 16}`
and a 2×2 upcast: axis 0 has 32/2 = 16 blocks and no threads, and axis 1 has 64/(16·2) = 2 blocks of 16
threads. That gives grid `(2, 16, 1)` and block `(16, 1, 1)`. The run:

```text
{0: 4, 1: 8} {} ((8, 8, 1), (8, 4, 1))
{1: 16} {0: 2, 1: 2} ((2, 16, 1), (16, 1, 1))
```

**2.** Plain: 4096 blocks of 1 thread is 4096 warps, with 1/32 = 3.1% of lanes busy. Local: 32 blocks of 128
threads is 128 warps, 100% busy. Auto: 4 blocks of 256 threads is 32 warps, 100% busy, but only 4 SMs used.

</details>

## Quiz

1. Why is `threadIdx.x`, and not `.y`, bound to the last axis of a row-major output?
2. A block of 256 threads uses 80 registers per thread on an SM with 65,536 registers and a 2,048-thread
   limit. How many blocks fit, and what is the occupancy?
3. What does `__launch_bounds__(256)` tell the compiler, and why does `uopc` always emit it?
4. Why can't two thread blocks synchronize with each other inside a kernel?
5. Why is it valid to test a GPU kernel by running it as nested CPU loops, and when does it stop being valid?
6. In the softmax kernel of 12.9, how many times is each row's max computed, and why?
7. What does NVRTC produce, and who turns that into machine code?
8. What is a warp called on AMD's data-center GPUs, and how wide is it?

<details>
<summary>Answers</summary>

1. Consecutive `.x` values are consecutive lanes of a warp, so they issue the same load together. Binding them
   to the contiguous axis puts the warp's 32 addresses into the fewest 32-byte sectors (coalescing).
2. 65536 / (256·80) = 3.2, so 3 blocks (768 threads = 24 warps), and occupancy 24/64 = 37.5%. The thread
   limit, 8 blocks, doesn't bind.
3. That the kernel is never launched with more than 256 threads per block. The register allocator can then
   give each thread up to 65536/256 = 256 registers (capped at the 255-register limit) and still fit one
   block. Without the bound it must assume 1024 threads, or 64 registers each. `uopc` knows the exact block
   size, so the bound is always tight.
4. Blocks are scheduled in an unspecified order and may not all be resident at once. A barrier could wait
   forever for a block that cannot start until the waiting ones finish.
5. Without shared memory or communication, each thread computes and writes its own output elements, so every
   execution order, including a serial one, gives the same result. It stops being valid with barriers or
   shared memory, and it can't detect races.
6. 16 times, once per thread in `x`. The reduce depends only on `lidx1`, so on a CPU it would be hoisted out of
   the inner loop. On a GPU every thread runs the hoisted code.
7. PTX, a virtual ISA. The driver (`cuModuleLoadData`) JIT-compiles PTX into SASS for the specific GPU.
8. A wavefront, 64 lanes on CDNA (RDNA supports 32 and 64).

</details>

## Further reading

- NVIDIA, *CUDA C++ Programming Guide*: the programming model, memory hierarchy, occupancy, and the
  `__launch_bounds__` and WMMA sections. Also the *CUDA C++ Best Practices Guide* on coalescing.
- NVIDIA, *NVRTC* and *CUDA Driver API* reference manuals: every function `cuda.py` calls.
- NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture* whitepaper (2020): the numbers in 12.1 and 12.3.
- V. Volkov, "Better Performance at Lower Occupancy", GTC 2010.
- D. Kirk and W. Hwu, *Programming Massively Parallel Processors*: the textbook treatment of everything here.
- J. Hennessy and D. Patterson, *Computer Architecture: A Quantitative Approach*, chapter 4 (data-level
  parallelism, GPUs).
- tinygrad: `tinygrad/renderer/cstyle.py` (the CUDA, Metal, OpenCL and HIP renderers are subclasses, the same
  design as `CUDARenderer`) and its GPU dimension handling, which folds and limits axes to the hardware's
  grid.
- AMD, *HIP Programming Guide*; Apple, *Metal Shading Language Specification*; Khronos, *OpenCL C
  Specification*.
