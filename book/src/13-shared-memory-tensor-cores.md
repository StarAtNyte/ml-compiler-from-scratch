# 13. Shared memory and tensor cores

The GPU GEMMs of [Chapter 12](12-gpus.md) are correct, coalesced, and still slow. Each thread computes its
output element by streaming a full row of A and a full column of B out of global memory. Across the whole
grid every element of A is fetched N times and every element of B is fetched N times. That is 2 FLOPs per 8
bytes, a fiftieth of what an A100 needs to be compute-bound. This chapter fixes it the way every fast GPU GEMM
does. Threads in a block *cooperate*: they stage tiles of A and B in on-chip **shared memory**, wait for each
other at **barriers**, and reuse each tile T times from there. Then we go one level further, to **tensor
cores**, the matrix units that do a 16×16×16 multiply-accumulate as a single warp-wide operation, and to the
pattern-matching lowering that `uopc` uses to reach them.

**What you'll build**

- `AddrSpace.SHARED` allocations, pinned by the linearizer to the top of the kernel (one copy per block).
- The `BARRIER` op, rendered as `__syncthreads()`.
- `smem_gemm_ir`, a tiled GEMM written by hand in the loop IR with two shared tiles and two barriers, and its
  rendered CUDA.
- A way to check a barrier kernel without a GPU (a lockstep simulation), and an understanding of why the
  Chapter 12 emulator can't.
- `match_gemm` and `wmma_gemm`: recognize a GEMM in the shaped graph and emit a WMMA tensor-core kernel.

**Where this fits.** This is the second half of weeks 7–8. Chapter 12 mapped loops to the grid. This chapter
adds the memory level between registers and DRAM. [Chapter 14](14-gpt.md) puts the whole compiler to work on a
real model. As in Chapter 12, the reference machine has no NVIDIA GPU. CUDA source here is real `render_cuda`
output, numerical checks are CPU simulations of the same index arithmetic, and GPU performance figures are
general knowledge from vendor documentation, not measurements.

## 13.1 The traffic problem, in numbers

Take a square GEMM, `C = A @ B` with N = 4096. It performs 2N³ ≈ 137 GFLOP. How many bytes must cross the DRAM
bus?

- **The compulsory minimum** is to read A and B once and write C once: 3 · N² · 4 bytes ≈ 200 MB, of which 134
  MB are reads. Against 137 GFLOP that would be about 1000 FLOP per byte, so in principle GEMM is as
  compute-bound as a kernel can be.
- **The naive kernel** (one thread per output, Chapter 12) loads one A element and one B element per FMA:
  2N³ loads.
- **A block tile** of BM × BN outputs, computed by one thread block that loads a BM × K panel of A and a
  K × BN panel of B *once*, reads (BM + BN)·N floats per block. There are (N/BM)(N/BN) blocks.

A scratch script makes the table:

```python
# scratch/traffic.py
N = 4096
flops = 2 * N**3
print(f"N = {N}: {flops/1e9:.0f} GFLOP per GEMM")
print(f"{'scheme':34s} {'DRAM reads':>12s} {'FLOP/byte':>10s}")
def row(name, floats_loaded):
  b = 4 * floats_loaded
  print(f"{name:34s} {b/1e9:9.1f} GB {flops/b:10.1f}")
row("naive: 2 loads per FMA", 2 * N**3)
for BM, BN in [(16, 16), (32, 32), (64, 64), (128, 128)]:
  # each block loads a BM x K panel of A and a K x BN panel of B, (N/BM)*(N/BN) blocks
  row(f"block tile {BM}x{BN} in shared memory", (N // BM) * (N // BN) * (BM + BN) * N)
print(f"compulsory (each input read once): {2*N*N*4/1e6:.0f} MB")
```

```text
N = 4096: 137 GFLOP per GEMM
scheme                               DRAM reads  FLOP/byte
naive: 2 loads per FMA                 549.8 GB        0.2
block tile 16x16 in shared memory       34.4 GB        4.0
block tile 32x32 in shared memory       17.2 GB        8.0
block tile 64x64 in shared memory        8.6 GB       16.0
block tile 128x128 in shared memory       4.3 GB       32.0
compulsory (each input read once): 134 MB
```

The closed form is intensity = BM·BN / (2(BM + BN)) FLOP per byte, which is **T/4** for a square T × T tile.
The A100's FP32 ridge point is about 12.5 FLOP/byte (Chapter 12). So a 16×16 tile is still memory-bound, and
64×64 or 128×128 tiles clear the ridge comfortably. Two caveats apply. First, the naive row
overstates real traffic, because L1 and L2 catch many of those repeated loads, so the measured gap is smaller
than 140×. Second, caches are not *guaranteed* to catch them, and a 4096² matrix (64 MB) does not fit in a 40
MB L2. Shared memory turns "hopefully cached" into "certainly on chip".

> **Key idea:** Tiling is a change of loop order that shortens reuse distance. The block computes the same
> sums in a different order, so that each value fetched from DRAM is used T times while it is still on chip,
> and *the programmer*, not a cache policy, decides what stays.

## 13.2 Shared memory and barriers in the IR

Two additions to the IR make tiling expressible.

**Shared allocations.** `ALLOC` already took an address space in its `ParamArg` (Chapter 4 used `LOCAL` for
accumulators). Weeks 7–8 add one more:

@include uopc/dtype.py#AddrSpace

A `SHARED` allocation is one array per *block*, visible to every thread in it. Its placement matters. Recall
from Chapter 5 that the linearizer declares an `ALLOC` in the deepest scope shared by all its users, so that
each OpenMP thread gets a private accumulator (NOTES 9). For shared memory that rule is wrong. The users of a
tile are inside the `LOCAL` (thread) ranges, so the "deepest common scope" would be per thread, which is the
opposite of shared. The fix is one line (NOTES 24):

@include uopc/linearize.py:71-77

A shared allocation always goes in the empty scope, at the top of the kernel. The renderer gives it the
`__shared__` prefix (`CRenderer.shared_prefix` is `""` and `CUDARenderer.shared_prefix` is `"__shared__ "`).
That is also what CUDA requires in practice: a `__shared__` array has block lifetime wherever it is declared,
so declaring it at the top states its real scope.

**Barriers.** `BARRIER` is an effect, like `STORE`. It has no value (`dtypes.void`) and its sources are the
effects it must come after:

```python
BARRIER = auto()    # (*deps): __syncthreads()
```

Its semantics: *no thread in the block proceeds past the barrier until every thread in the block has
reached it, and all shared-memory writes made before it are visible to all threads after it.* In SSA terms
(Chapter 4), a barrier is an `AFTER` that waits on *other threads'* stores as well as this thread's. The
renderers spell it with a class attribute: `CUDARenderer.barrier = "__syncthreads()"`, and the base
`CRenderer.barrier = None`. The `None` is deliberate: the C renderer has no meaning for a barrier, and
`emulate` refuses barrier kernels outright. That is how NOTES 25 shows up in Section 13.5.

## 13.3 `smem_gemm_ir`, line by line

There is no automatic pass that introduces shared memory in `uopc`. Like the flash-attention kernel of
[Chapter 11](11-fusion-flash.md), the tiled GEMM is written by hand in the loop IR, from the same ops the rest
of the compiler uses. Writing it by hand is the assignment for week 8, and it is the best way to learn what an
automatic pass would have to produce.

@include uopc/gpu.py#smem_gemm_ir

@diagram 13-smem-tiling

Line by line, using N = 64 and T = 16:

- **Buffers and allocations.** `A`, `B`, `C` are global (`BUFFER`). `sA` and `sB` are `T*T = 256` floats each
  in `SHARED`. `acc` is a 1-element `LOCAL` allocation, which the renderer turns into a scalar (a register).
- **The grid.** `by, bx` are `GLOBAL` ranges of size N/T = 4, and `ty, tx` are `LOCAL` ranges of size T = 16.
  The range indices 0–3 put `bx` and `tx` last, so the renderer binds them to `.x` (Chapter 12's coalescing
  rule). Each thread owns exactly one C element: `row = by*T + ty`, `col = bx*T + tx`.
- **`acc0 = acc.after(by, bx, ty, tx)`** names the accumulator *for this thread*. The `AFTER` on the ranges
  puts the `init` store inside the thread's scope (Chapter 6's reduce lowering uses the same idiom).
- **`kt = range_(N // T, 4)`** is the outer reduction loop, one iteration per tile along K.
- **`ld_a`, `ld_b`** are the cooperative copy. Thread `(ty, tx)` copies *one* element of each tile:
  `A[row, kt*T + tx]` into `sA[ty*T + tx]`, and `B[kt*T + ty, col]` into `sB[ty*T + tx]`. Together the 256
  threads copy two whole 16×16 tiles. Look at which index `tx` drives: the *column* of A and the column of B,
  so both global loads are coalesced. `sA.after(kt)` says "the version of `sA` in iteration `kt`", which puts
  the store inside the `kt` loop.
- **`sync1 = BARRIER(ld_a, ld_b)`.** Every thread waits until all 512 copies are done.
- **`k = range_(T, 5)` and `inner`** are the ordinary reduce body, reading both operands out of shared memory:
  `acc += sA[ty*T + k] * sB[k*T + tx]`. The loads go through `sA.after(sync1)`, so they are ordered after the
  barrier. That data dependency is what places them below the `__syncthreads()` in the linear order.
- **`sync2 = BARRIER(inner)`.** Every thread waits until all threads have finished reading the tiles, and
  `loop = sync2.end(kt)` closes the K loop.
- **`out`** writes the accumulator to `C[row*N + col]` after the loop (`acc0.after(loop)`), and the `SINK`
  ends the four grid ranges.

Here is the rendering. It is short enough to read in full:

```python
from uopc.gpu import smem_gemm_ir, render_cuda, launch_dims
ast = smem_gemm_ir(64, 16)
print(launch_dims(ast))
print(render_cuda(ast, 'smem_gemm'))
```

```text
((4, 4, 1), (16, 16, 1))
```

```c
extern "C" __global__ void __launch_bounds__(256) smem_gemm(float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2) {
  __shared__ float smem0[256];
  __shared__ float smem1[256];
  int gidx1 = blockIdx.y; /* 4 */
  int gidx0 = blockIdx.x; /* 4 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu1 = (((gidx1*16)+lidx1)*64);
  int alu2 = (lidx1*16);
  int lidx0 = threadIdx.x; /* 16 */
  int alu3 = ((gidx0*16)+lidx0);
  float acc2;
  acc2 = 0.0f;
  int alu4 = (alu2+lidx0);
  for (int ridx4 = 0; ridx4 < 4; ridx4++) {
    int alu5 = (ridx4*16);
    float val6 = data0[((alu1+alu5)+lidx0)];
    smem0[alu4] = val6;
    float val7 = data1[(((alu5+lidx1)*64)+alu3)];
    smem1[alu4] = val7;
    __syncthreads();
    for (int ridx5 = 0; ridx5 < 16; ridx5++) {
      float val8 = acc2;
      float val9 = smem0[(alu2+ridx5)];
      float val10 = smem1[((ridx5*16)+lidx0)];
      acc2 = (val8+(val9*val10));
    }
    __syncthreads();
  }
  float val11 = acc2;
  data2[(alu1+alu3)] = val11;
}
```

(The `#define` prelude is the same as in Chapter 12 and is omitted.) Every piece of the IR is visible: the two
`__shared__` arrays at the top (NOTES 24), the thread's row offset `alu1 = row*64`, its column `alu3`, the
`kt` loop (`ridx4`), the copies, the barriers in the right places, and the inner product out of shared memory.
The test pins the important properties:

@include tests/test_w07_gpu.py#test_smem_gemm_renders

Count the traffic for this kernel. Per block and per `kt`, 2 × 256 floats come from global memory and the
inner loop performs 256 × 16 FMAs. That is 4096 FMAs per 512 loads, 8 FMAs per load, against 1 per 2 loads in
the naive kernel: a factor of T = 16, as the table predicted.

## 13.4 Why two barriers

Barriers are the part of GPU programming that is easiest to get wrong, so let's be precise about what each
one prevents. The warps of a block do not run in lockstep with *each other*. The scheduler interleaves them,
and one warp can get well ahead of another.

- **Without `sync1`** a fast warp finishes its copy and starts the inner loop while a slow warp hasn't yet
  written its part of the tile. The fast warp reads stale shared memory: whatever the previous `kt` left
  there, or uninitialized garbage on the first iteration. This is a **read-after-write** hazard.
- **Without `sync2`** a fast warp finishes its inner loop, loops around to `kt + 1`, and *overwrites* `sA` and
  `sB` with the next tiles while a slow warp is still reading the current ones. This is a
  **write-after-read** hazard.

@diagram 13-barriers

The second barrier is the one people forget, because the code "obviously" reads the tile before the loop
comes around again, and in a single thread it does. The comment in `smem_gemm_ir`, *"don't overwrite tiles
others still read"*, is there because the hazard can't be seen by reading one thread's code.

Two more rules complete the picture:

1. **Every thread of the block must reach the same barrier.** Putting `__syncthreads()` inside a branch that
   some threads skip is undefined behaviour, and in practice the kernel often deadlocks. Our IR can't express
   that mistake, since `BARRIER` has no condition. Exercise 13.7 asks what a guarded store (Chapter 12,
   Exercise 5) would do near a barrier.
2. **A barrier is per block.** There is no `__syncthreads()` across blocks. Cross-block communication goes
   through a kernel boundary (or through cooperative groups, which this book does not cover).

The cost of a barrier is the time the fastest warp waits for the slowest. Double buffering (Section 13.9)
uses two pairs of tiles so that the copy of tile `kt + 1` overlaps the computation on tile `kt`, and that
removes one of the two waits.

## 13.5 Testing a barrier kernel without a GPU

The Chapter 12 emulator runs threads one at a time, in nested loops. For this kernel the loop order is `by,
bx, ty, tx, kt, k`, so thread (0, 0) runs its *whole* `kt` loop before thread (0, 1) starts. When thread (0,0)
reads row 0 of `sA` it finds only its own element written. That is exactly the read-after-write hazard above,
made deterministic.

First, `emulate` refuses the kernel: it checks for `BARRIER` before rendering anything.

```python
from uopc.gpu import smem_gemm_ir, emulate
try: print(emulate(smem_gemm_ir(64,16)))
except Exception as e: print(type(e).__name__+':', e)
```

```text
NotImplementedError: emulate: kernels with BARRIER need all threads to run in lockstep; running threads one after another (what loops do) would be wrong
```

(The first version crashed here with `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'`,
from `self.barrier + ";"` in the renderer. An error should say *what* is unsupported and *why*; this one now
does.) Suppose we ignore the warning and force it through by giving the C renderer a no-op barrier:

```python
# scratch/nobarrier.py
import numpy as np
from uopc.render_c import CRenderer
from uopc.runtime import Program
from uopc.gpu import smem_gemm_ir

class SerialThreads(CRenderer):
  barrier = "/* __syncthreads() */"      # threads run one after another: a barrier is a no-op

N = 64
src = SerialThreads().render(smem_gemm_ir(N, 16), "smem_gemm")
a, b = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
c = np.zeros(N * N, np.float32)
Program(src, "smem_gemm")(a, b, c)
err = np.abs(c.reshape(N, N) - a @ b)
print(f"max error {err.max():.2f}, wrong elements {np.mean(err > 1e-3) * 100:.1f}%")
```

```text
max error nan, wrong elements 99.3%
```

Almost every element is wrong, and some are NaN, because they were computed from shared-memory slots that no
thread had written yet (uninitialized stack memory in the C version). This is NOTES 25: *barriers cannot be
emulated by running threads one after another.*

To check the *algorithm* we need an execution model that respects barriers. The simplest one runs the threads
of a block in **lockstep**: every phase between barriers completes for all threads before the next begins.
With numpy that comes almost for free, because one array operation over a `(T, T)` grid of thread indices is
one phase for all threads:

```python
# scratch/lockstep.py
import numpy as np

def smem_gemm_lockstep(A, B, T=16):
  """Run the smem_gemm schedule with every thread of a block in lockstep: each phase
  between barriers finishes for all T*T threads before the next one starts."""
  N = A.shape[0]
  C = np.zeros((N, N), np.float32)
  ty, tx = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")   # all threads of a block at once
  for by in range(N // T):
    for bx in range(N // T):
      row, col = by * T + ty, bx * T + tx
      acc = np.zeros((T, T), np.float32)                              # one register per thread
      sA, sB = np.empty(T * T, np.float32), np.empty(T * T, np.float32)
      for kt in range(N // T):
        sA[ty * T + tx] = A[row, kt * T + tx]                         # phase 1: cooperative copy
        sB[ty * T + tx] = B[kt * T + ty, col]
        # __syncthreads()
        for k in range(T):                                            # phase 2: multiply from smem
          acc += sA[ty * T + k] * sB[k * T + tx]
        # __syncthreads()
      C[row, col] = acc
  return C

N = 64
A, B = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
print("max error:", np.abs(smem_gemm_lockstep(A, B) - A @ B).max())
```

```text
max error: 7.6293945e-06
```

The index expressions are copied from the IR (`row*N + kt*T + tx` and so on). This checks the tiling algorithm
and its index math, not the generated code. The generated code is checked by rendering tests, and by
`test_cuda_smem_gemm_on_gpu` on a machine with a GPU (Chapter 12, Section 12.11).

A lockstep simulation is a sound model for kernels whose barriers are all reached by all threads, which our
IR guarantees. It can be made fully automatic. Split the linearized kernel at its barriers into phases, then
emulate each phase as a loop over all threads, with per-thread registers kept in arrays indexed by thread. That
is Exercise 13.6. It is also the core idea of MCUDA (Stratton, Stone and Hwu, 2008), which compiled CUDA for
CPUs by wrapping each barrier-free region of a kernel in a "thread loop".

## 13.6 Bank conflicts

Shared memory is not a single array. It is **32 banks**, each 4 bytes wide, with word *w* in bank *w* mod 32.
Each bank can serve one word per cycle. When the 32 lanes of a warp access shared memory:

- distinct words in distinct banks are served in one pass;
- lanes reading *the same word* get a broadcast, also one pass;
- *distinct* words in the *same* bank are serialized, one pass per word. This is a **bank conflict**, and an
  *n*-way conflict makes the access *n* times slower.

These rules are general knowledge (CUDA Programming Guide, "Shared Memory"). Let's apply them to our kernel's
three shared-memory accesses, and to the classic bad case:

```python
# scratch/banks.py
def conflict(addr_of_thread, nthreads=32):
  """Shared memory: 32 banks of 4 bytes. Threads reading the *same* word are a broadcast (free);
  distinct words in the same bank are serialized. Returns the worst-case number of passes."""
  words = {addr_of_thread(t) for t in range(nthreads)}
  per_bank = {}
  for w in words: per_bank.setdefault(w % 32, set()).add(w)
  return max(len(s) for s in per_bank.values())

T, k = 16, 3
tid = lambda t: (t // T, t % T)                     # (ty, tx) of thread t in a 16x16 block
print("smem_gemm, T=16 (warp = 2 rows of 16 threads)")
print("  store sA[ty*T+tx]  :", conflict(lambda t: tid(t)[0] * T + tid(t)[1]), "pass(es)")
print("  load  sA[ty*T+k]   :", conflict(lambda t: tid(t)[0] * T + k), "pass(es)")
print("  load  sB[k*T+tx]   :", conflict(lambda t: k * T + tid(t)[1]), "pass(es)")
T = 32
tid = lambda t: (t // T, t % T)                     # T=32: a warp is one row of threads
print("column walk, T=32")
print("  load  s[tx*32+k]   :", conflict(lambda t: tid(t)[1] * 32 + k), "pass(es)")
print("  load  s[tx*33+k]   :", conflict(lambda t: tid(t)[1] * 33 + k), "pass(es)  (padded row)")
```

```text
smem_gemm, T=16 (warp = 2 rows of 16 threads)
  store sA[ty*T+tx]  : 1 pass(es)
  load  sA[ty*T+k]   : 1 pass(es)
  load  sB[k*T+tx]   : 1 pass(es)
column walk, T=32
  load  s[tx*32+k]   : 32 pass(es)
  load  s[tx*33+k]   : 1 pass(es)  (padded row)
```

Our kernel is conflict-free. The copy writes 32 consecutive words. The `sA` read is two distinct words, one
per row of the warp, so two broadcasts in different banks. The `sB` read is 16 consecutive words, the same
16 for both halves of the warp.

The bad case appears as soon as a thread walks *down a column* of a tile whose row length is a multiple of
32. That happens, for example, if you store a tile transposed so that the inner loop reads `s[tx*32 + k]`.
Every lane hits bank `k mod 32`: a 32-way conflict, and shared memory becomes slower than L2. There are two
standard fixes:

- **Padding.** Allocate rows of 33 floats instead of 32. Element `(r, c)` lives at `r*33 + c`, so a column walk
  moves one bank per row. One wasted word per row buys conflict-free columns.
- **Swizzling.** Store element `(r, c)` at column `c XOR (r mod 32)` (or a similar permutation). No memory is
  wasted, and the layout composes with vectorized 16-byte accesses. CUTLASS and Triton swizzle. It's a
  natural fit for our IR, since the swizzle is just an index expression on the `INDEX` into a `SHARED`
  allocation (Exercise 13.5).

## 13.7 Register blocking on top of shared memory

Shared memory removed most of the DRAM traffic. Now the shared-memory traffic is the bottleneck. In
`smem_gemm_ir` each FMA needs two shared-memory loads. An SM can serve 32 banks × 4 bytes = 128 bytes of shared
memory per cycle, and an A100 SM can issue 64 FP32 FMAs per cycle (general knowledge, whitepaper numbers). At 2
loads × 4 bytes per FMA, keeping 64 FMAs per cycle busy needs 512 bytes per cycle, four times what shared
memory delivers.

The fix is the *same* trick as Chapter 10's register tile, one level down the hierarchy. Let each thread
compute a TM × TN micro-tile of C. Per step of `k` it loads TM values of `sA` and TN values of `sB` into
registers and does TM·TN FMAs:

| micro-tile per thread | shared loads per FMA | bytes/cycle to feed 64 FMA/cycle |
|---|---|---|
| 1 × 1 (`smem_gemm_ir`) | 2 | 512 |
| 2 × 2 | 1 | 256 |
| 4 × 4 | 0.5 | 128 |
| 8 × 8 | 0.25 | 64 |

So a 4×4 micro-tile is the break-even point and 8×8 leaves headroom. The full hierarchy of a classic FP32 GEMM
kernel (the structure of CUTLASS's SIMT kernels and of many "how to write a fast matmul" tutorials) looks like
this:

```text
block tile    128 x 128 of C       global -> shared, per k-step of 8..32
  warp tile    64 x 32 of C        (8 warps per block)
    thread tile  8 x 8 of C        shared -> registers, 64 accumulators per thread
```

256 threads × 64 outputs = 16,384 = 128 × 128. Every level multiplies reuse: DRAM bytes are reused 128×
through shared memory, and shared bytes are reused 8× through registers.

In `uopc` the register level already exists. Chapter 12's `gpu_opt_choice` emits a 2×2 `UPCAST` on GPU
kernels, and `Opts(gpu=True, local=..., upcast=...)` can request any size. What the compiler cannot yet do is
combine an upcast with a shared-memory stage automatically. Writing `smem_gemm_ir` with a 2×2 or 4×4 thread
tile by hand (Exercise 13.3) is the most instructive thing you can do this week. The pieces are all there:
four accumulator allocations, a cooperative copy where each thread copies several elements, and the unrolled
inner product.

## 13.8 Tensor cores

Since Volta (V100, 2017), NVIDIA GPUs have **tensor cores**: units that perform a small matrix
multiply-accumulate, `D = A·B + C`, as a single operation, with low-precision inputs and higher-precision
accumulation. The throughput difference is large. On an A100, dense FP16 or BF16 tensor-core math peaks at
312 TFLOP/s, against 19.5 TFLOP/s for ordinary FP32 FMAs (whitepaper numbers). Any serious GEMM or attention
kernel on such hardware runs on tensor cores, and so does essentially all LLM training.

Three facts define the programming model:

1. **The unit of work is a warp, not a thread.** All 32 lanes jointly issue a matrix operation on tiles that
   are spread across their registers.
2. **The shapes are fixed.** The C++ WMMA API (`nvcuda::wmma`, Volta and later) offers m16n16k16 (plus
   m32n8k16 and m8n32k16) for FP16 inputs. The lower-level PTX `mma.sync` instructions expose shapes such as
   m16n8k16 on Ampere. Hopper's `wgmma` operates on m64nNk16 tiles across a warp *group* of four warps.
3. **Precision is mixed by design.** Inputs are FP16 (5 exponent bits, 10 mantissa bits, max 65504) or BF16
   (8 exponent bits like FP32, 7 mantissa bits). Products are formed exactly and accumulated in FP32. On A100,
   TF32 (FP32 range, 10-bit mantissa) is a third option for FP32 inputs.

**Fragments.** The WMMA API hides the register layout behind a type, `wmma::fragment<use, M, N, K, T,
layout>`. A `matrix_a` fragment holds a 16×16 tile of A, and an `accumulator` fragment holds a 16×16 tile of C
as 256 floats across 32 lanes, 8 per lane. *Which* lane holds *which* element is deliberately unspecified and
differs between architectures. Your code may only move whole fragments: `load_matrix_sync` (from memory, with
a leading dimension), `mma_sync` (multiply-accumulate), `store_matrix_sync`, `fill_fragment`. The `_sync`
suffix is a reminder that every lane of the warp must call them together.

@diagram 13-wmma-fragments

### Why FP32 accumulation matters

The precision claim is easy to check without a tensor core. Here is a numpy model of the kernel's tile loop,
with FP16 inputs, products formed exactly, and FP32 accumulation, compared against an FP16 accumulator:

```python
# scratch/wmma_sim.py
import numpy as np
def wmma_sim(A, B):
  """The tile loop of wmma_gemm in numpy: fp16 inputs, one 16x16x16 MMA with fp32 accumulate per step."""
  M, K = A.shape; N = B.shape[1]
  C = np.zeros((M, N), np.float32)
  for tile_m in range(M // 16):                      # blockIdx.y
    for tile_n in range(N // 16):                    # blockIdx.x
      c = np.zeros((16, 16), np.float32)             # fill_fragment(c, 0)
      for k in range(0, K, 16):
        a = A[tile_m*16:tile_m*16+16, k:k+16]        # load_matrix_sync(a, A + tile_m*16*K + k, K)
        b = B[k:k+16, tile_n*16:tile_n*16+16]        # load_matrix_sync(b, B + k*N + tile_n*16, N)
        c += a.astype(np.float32) @ b.astype(np.float32)   # mma_sync: products exact in f32, f32 sum
      C[tile_m*16:tile_m*16+16, tile_n*16:tile_n*16+16] = c
  return C
rng = np.random.default_rng(0)
M, N, K = 64, 32, 48
A32, B32 = rng.standard_normal((M, K)).astype(np.float32), rng.standard_normal((K, N)).astype(np.float32)
A, B = A32.astype(np.float16), B32.astype(np.float16)
ref = A32.astype(np.float64) @ B32.astype(np.float64)
print("wmma (f16 in, f32 acc) vs f64 of the f32 inputs: max abs err %.4f" % np.abs(wmma_sim(A, B) - ref).max())
print("same f16 inputs, exact f64 math               : max abs err %.6f" % np.abs(wmma_sim(A, B) - A.astype(np.float64) @ B.astype(np.float64)).max())
c16 = np.zeros((M, N), np.float16)
for k in range(K): c16 = (c16 + (A[:, k:k+1] * B[k:k+1, :])).astype(np.float16)
print("f16 inputs AND f16 accumulate                 : max abs err %.4f" % np.abs(c16 - ref).max())
```

```text
wmma (f16 in, f32 acc) vs f64 of the f32 inputs: max abs err 0.0076
same f16 inputs, exact f64 math               : max abs err 0.000003
f16 inputs AND f16 accumulate                 : max abs err 0.0330
```

Read the three lines together. With FP32 accumulation, essentially all the error comes from *rounding the
inputs* to FP16 (0.0076 against the FP32 data), and almost none from the arithmetic (0.000003 against exact
math on the same FP16 inputs). Accumulating in FP16 adds four times more error even at K = 48, and the error
grows with K. That is why tensor cores accumulate in FP32, and why mixed-precision training keeps master
weights and reductions in FP32 ([Chapter 15](15-beyond.md)).

## 13.9 Pattern-based lowering: `match_gemm` and `wmma_gemm`

How does a compiler get from `A @ B` in a Tensor program to `mma_sync`? There are two broad designs.

1. **Make tensor cores an IR operation.** Add a `WMMA` UOp whose sources are fragments, and teach the
   optimizer to *reshape* the loop nest so that a 16×16×16 sub-block of the GEMM becomes one `WMMA`, with the
   remaining axes as ordinary `GLOBAL`/`LOCAL`/`LOOP` ranges. This is what tinygrad does: a tensor-core
   "opt" splits the M, N and K axes by the MMA shape and the renderer emits the intrinsic. It composes with
   everything else (fusion, upcasting, other ops in the same kernel), and it is a lot of machinery.
2. **Recognize the pattern and call a template.** Look at the shaped graph *before* rangeify. If a kernel is
   exactly a GEMM, emit a hand-written kernel for it. This is how many production compilers dispatch to
   cuBLAS or CUTLASS, and how `uopc` does it.

`uopc` reserves the `WMMA` op (`Ops.WMMA`, with a shape rule in `_infer` and a `render_wmma` hook in the C
renderer that raises) but takes route 2 for the course:

@include uopc/gpu.py#match_gemm

A GEMM in our shaped IR has a canonical form, the one `Tensor.matmul` builds (Chapter 9): a `REDUCE(ADD, 1)`
over a `MUL` of two operands broadcast to `(M, N, K)`. `match_gemm` checks for that form and for sizes that are
multiples of 16:

```python
# scratch/wmma.py
import sys; sys.path.insert(0, "tests")
from helpers import buf
from uopc.uop import Ops
from uopc.gpu import match_gemm, wmma_gemm
def gemm_graph(M, N, K):
  A, B = buf(0, M * K), buf(1, K * N)
  return (A.reshape((M, 1, K)).expand((M, N, K)) * B.reshape((K, N)).permute((1, 0)).reshape((1, N, K)).expand((M, N, K))).reduce(Ops.ADD, 1)
print("gemm 64x32x48 :", match_gemm(gemm_graph(64, 32, 48)))
print("gemm 64x32x40 :", match_gemm(gemm_graph(64, 32, 40)))
X = buf(0, 64 * 64).reshape((64, 64))
print("row sum       :", match_gemm(X.reduce(Ops.ADD, 1)))
print("row max       :", match_gemm(X.reduce(Ops.MAX, 1)))
print(wmma_gemm(64, 32, 48))
```

```text
gemm 64x32x48 : (64, 32, 48)
gemm 64x32x40 : None
row sum       : None
row max       : None
```

When it matches, `wmma_gemm` produces the kernel. One warp computes one 16×16 tile of C by walking K in steps
of 16:

@include uopc/gpu.py#wmma_gemm

For M, N, K = 64, 32, 48 the rendered source is:

```c
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void wmma_gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C) {
  int tile_m = blockIdx.y, tile_n = blockIdx.x;          // one warp per 16x16 tile of C
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  for (int k = 0; k < 48; k += 16) {
    wmma::load_matrix_sync(a, A + tile_m * 16 * 48 + k, 48);
    wmma::load_matrix_sync(b, B + k * 32 + tile_n * 16, 32);
    wmma::mma_sync(c, a, b, c);
  }
  wmma::store_matrix_sync(C + tile_m * 16 * 32 + tile_n * 16, c, 32, wmma::mem_row_major);
}
/* launch: grid = (2, 4), block = (32, 1, 1) */
```

Read it against the diagram. Block `(tile_n, tile_m)` is one warp, `block = (32, 1, 1)`. The A fragment starts
at row `tile_m*16`, column `k`, with leading dimension K. The B fragment starts at row `k`, column
`tile_n*16`, with leading dimension N. Three `mma_sync`s accumulate K = 48. The `wmma_sim` function in
Section 13.8 is a line-by-line numpy transcription of this loop (its comments name the CUDA calls), and it
matches the FP64 reference to within FP16 input rounding. That checks the index arithmetic. The CUDA
itself is render-tested only (`test_wmma`), since the reference machine has no GPU.

This kernel is a teaching kernel, not a fast one. It uses one warp per block, so a block has 32 threads (the
"too-small block" problem of Chapter 12). It has no shared-memory staging, so each warp reloads its A and B
fragments straight from global memory and there is no reuse across warps. And it has one accumulator per warp,
so there is no ILP. A competitive tensor-core kernel combines everything in this chapter: a block of 4–8 warps,
a 128×128 block tile staged through (swizzled) shared memory, each warp owning a 64×64 warp tile of 4×4 or more
accumulator fragments, and double-buffered asynchronous copies.

**The pattern had a hole.** The first `match_gemm` checked only the *shape* of the graph, `REDUCE(ADD)` of a
3-D `MUL`, not what was being multiplied, so any 3-D product reduced over its last axis matched, including
`(X*X).reduce(Ops.ADD, 1)`, a sum of squares. Dispatching that to `wmma_gemm` would compute the wrong thing.
The current version also requires the two operands to be `EXPAND`s broadcast along *different* axes (A along
N, B along M), which is what makes a product of two tensors a matrix product:

```python
X = buf(0, 32*32*32).reshape((32,32,32))
print('sum of squares over last axis:', match_gemm((X*X).reduce(Ops.ADD, 1)))
```

```text
sum of squares over last axis: None
```

It is still not fully sound: a production matcher must record the operands' layouts (`wmma_gemm` assumes A
row-major M×K and B row-major K×N) and dtypes (it assumes FP16 inputs). That is Exercise 13.2. This is the general weakness of pattern-based lowering: the pattern is a *claim* about
semantics, and every claim needs a test that tries to break it.

## 13.10 What production kernels add

The gap between `smem_gemm_ir` and cuBLAS is mostly made of the following techniques, all general knowledge
and all expressible as more of what this chapter did:

- **Double buffering / multi-stage pipelines.** Allocate two (or 3–4) sets of shared tiles. While computing on
  set *i*, copy the next tile into set *i+1*. On Ampere and later, `cp.async` copies global memory to shared
  memory *asynchronously*, without passing through registers, so the copy and the math truly overlap.
  Hopper's TMA (Tensor Memory Accelerator) moves whole tiles with one instruction.
- **Vectorized global loads.** `float4` loads move 16 bytes per lane per instruction, so fewer instructions are
  needed to keep enough bytes in flight (the Little's-law argument of Chapter 12).
- **Swizzled shared layouts**, as in Section 13.6, chosen to match the fragment load pattern of `ldmatrix`.
- **Warp tiling** between the block and thread levels, so that each warp's fragments come from a compact
  region of shared memory.
- **Split-K and stream-K** for shapes with small M·N and large K, where there are too few output tiles to fill
  108 SMs. Several blocks share one output tile and reduce their partial sums at the end.
- **Epilogue fusion.** Apply bias, activation, residual add or a type cast to the accumulator before the store,
  saving a whole kernel and its memory round trip. This is Chapter 7's fusion at the kernel template level.
- **Autotuning** over tile sizes, stages and warp counts. Chapter 10's `search_opt` is the same idea.

FlashAttention-2 ([Chapter 11](11-fusion-flash.md)) is this chapter's GEMM structure applied twice per tile,
`S = Q·Kᵀ` and `O += P·V` on tensor cores, with the online softmax in registers between them.

## Pitfalls

**1. The forgotten second barrier.** Without `sync2`, the kernel produces correct results *most of the time*.
Whether a fast warp overwrites a tile before a slow warp has read it depends on scheduling, clock speeds and
what else the GPU is doing. Races like this pass a test 99 times and fail on the 100th, or only on a different
GPU. Reason about every shared buffer's write-after-read and read-after-write orderings explicitly (Section
13.4), because the emulator can't help you (Pitfall 3).

**2. Shared allocations in the wrong scope (NOTES 24).** The generic rule "declare an `ALLOC` in the deepest
scope common to its users" is right for accumulators and wrong for shared memory. Applied to `sA`, it would
place the declaration inside the thread ranges. In CUDA that would still be one array per block, but the IR
would claim per-thread storage, and every pass that reasons about scope (for example, the emulator, which
would re-declare it on every iteration) would get it wrong. The linearizer pins `SHARED` allocations to the
empty scope.

**3. The emulator lies about barriers (NOTES 25).** `emulate` now refuses barrier kernels, and if you
force one through with a no-op barrier you get the 99.3%-wrong result of Section 13.5. Test barrier kernels with a
lockstep model or on real hardware.

**4. A matcher that matched things that aren't GEMMs.** The first `match_gemm` matched the shape of the graph,
not the operands, and accepted a sum of squares (13.9). A pattern-based lowering is only as sound as its
matcher, and every matcher needs a test that tries to fool it.

**5. Bank conflicts from "harmless" layout changes.** Transposing a shared tile so that the inner loop walks
columns of a 32-wide tile makes every access a 32-way conflict (Section 13.6). Results are unchanged and the
kernel gets several times slower. Pad or swizzle.

**6. FP16 overflow and accumulation.** FP16's largest finite value is 65504, so activations or gradients can
overflow to `inf` where FP32 would not. And accumulating in FP16 loses accuracy quickly with K (Section 13.8).
Use FP32 accumulators (the WMMA `accumulator` fragment type above is `float`), BF16 when range matters, and
loss scaling when training in FP16.

**7. Fragment layouts are opaque.** It is tempting to index into `c.x[i]` and assume element `i` is
`C[lane/4][...]`. The mapping is unspecified and has changed between GPU generations. Element-wise operations
over all fragment elements (scaling, activation) are safe. Anything that depends on position is not.

## Exercises

1. ★ **Traffic by hand.** For `smem_gemm_ir(N=1024, T=16)`, compute (a) the number of global loads per block,
   (b) the total global loads, (c) the ratio to the naive kernel's 2N³. Then do it for T = 32. Is T = 32 even
   possible with one thread per output? *Hint:* count threads per block.
2. ★★ **Make `match_gemm` sound.** Strengthen it to check that the two operands of the `MUL` are an
   `(M, 1, K)` and a `(1, N, K)` broadcast, each from a single buffer, and return the operands' layouts
   (whether K is contiguous). Add a test with the sum-of-squares counterexample of 13.9 and with a transposed
   B. *Hint:* reuse `_last_axis_stride` from `uopc/opt.py` (Chapter 10) to compute strides.
3. ★★ **Register-blocked shared-memory GEMM.** Write `smem_gemm_ir2(N, T, R)` in which a block of
   (T/R)×(T/R) threads computes a T×T tile, each thread an R×R micro-tile with R² accumulators. Each thread now
   copies R² elements of each tile. Render it with `render_cuda`, check it with a lockstep simulation, and fill
   in the "shared loads per FMA" column of 13.7 for your kernel.
4. ★★ **Predict the conflicts.** For T = 32 and the original `smem_gemm_ir` indexing (`sA[ty*T+k]`,
   `sB[k*T+tx]`), how many passes does each shared load take? What changes if the block is (32, 8) threads
   with each thread computing 4 rows? Check your predictions with `banks.py`.
5. ★★ **Swizzle in the IR.** Change `smem_gemm_ir` to store and load `sB` at a swizzled index
   (`k*T + (tx ^ k)`, written with the IR's integer ops, or `(tx + k) % T` if you don't want to add XOR) and
   verify with the lockstep model that results are unchanged. Explain why a swizzle must be applied to both
   the store and the load.
6. ★★★ **A barrier-aware emulator.** Write `emulate_lockstep(sink)`: linearize the kernel, split the
   instruction list at `BARRIER`s, and generate C in which each phase is a loop over all threads of a block,
   with every per-thread value that is live across a barrier (accumulators, indices) stored in an array
   indexed by thread id. Shared allocations stay scalar arrays per block. Make `smem_gemm_ir(64, 16)` pass
   against numpy. *Hint:* barriers inside a loop (`kt`) mean the phase split must happen *inside* the loop
   body. Start by supporting only barriers at loop-body depth 1.
7. ★★ **Barriers and guards.** Suppose you implemented guarded stores (Chapter 12, Exercise 5) by wrapping
   whole statements in `if (idx < n)`. Construct a shared-memory kernel where that transformation would place
   a `__syncthreads()` inside the `if`. What goes wrong, and how should the guard be expressed instead?
8. ★★★ **A WMMA UOp.** Implement route 1 of Section 13.9 for one case. Lower a `match_gemm`-able kernel with
   `GLOBAL` ranges over 16×16 output tiles and a `LOOP` over K/16, and an `Ops.WMMA` node whose sources are
   the A and B tile addresses and the accumulator. Implement `CUDARenderer.render_wmma` to emit the fragment
   calls. Compare your output with `wmma_gemm`'s.
9. ★★★ **Double buffering.** Write `smem_gemm_ir_db` with two copies of each tile (allocation slots 0–3), in
   which iteration `kt` computes on buffer `kt % 2` while copying tile `kt+1` into the other. How many barriers
   per iteration do you need now? Check it with your Exercise 6 emulator if you built it, or else with a
   lockstep numpy model.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** (a) Per `kt` a block loads 2·T² = 512 floats, and there are N/T = 64 values of `kt`, so 32,768 loads per
block. (b) There are (N/T)² = 4096 blocks, so 4096 × 32,768 = 134,217,728 ≈ 1.34·10⁸ loads. (c) The naive
kernel does 2N³ = 2.15·10⁹, so the ratio is 1/16 = 1/T. For T = 32 the ratio is 1/32. But a 32×32 tile with one
thread per output needs 1024 threads per block, the hardware maximum, which leaves almost no registers per
thread (64K / 1024 = 64) and a single block per SM. This is why real kernels use smaller thread blocks with
register micro-tiles (Section 13.7) instead of one thread per output.

</details>

## Quiz

1. What is the arithmetic intensity, in FLOP per byte, of a GEMM computed with T×T shared-memory tiles and no
   register blocking?
2. Why does the linearizer put `SHARED` allocations in the empty scope rather than in the deepest scope of
   their users?
3. Name the hazard each of the two barriers in `smem_gemm_ir` prevents.
4. Why can't `emulate` check `smem_gemm_ir`, and what kind of simulation can?
5. Thirty-two lanes read `s[lane * 32]` from shared memory. How many passes does that take, and what is the
   cheapest fix?
6. In `wmma::fragment<wmma::accumulator, 16, 16, 16, float>`, how many floats does each lane hold?
7. Why do tensor cores accumulate in FP32 even when their inputs are FP16?
8. What does `match_gemm` fail to check?

<details>
<summary>Answers</summary>

1. T/4. Each block reads 2·T·N floats (8TN bytes) for 2·T²·N FLOPs.
2. Shared memory is one array per block, visible to all threads. Its users sit inside the thread (`LOCAL`)
   ranges, so the "deepest common scope" rule would make it look per-thread. Pinning it to the top states its
   real lifetime and is where CUDA code conventionally declares it (NOTES 24).
3. `sync1`: read-after-write (computing before the tile is fully copied). `sync2`: write-after-read (the next
   iteration's copy overwriting a tile others still read).
4. The emulator runs threads one after another, so no thread ever waits for the others at a barrier, and reads
   see unwritten shared memory. A lockstep (phase-by-phase) simulation, which runs every thread up to the
   barrier before any thread continues, models barriers correctly.
5. Every address is in bank 0 with a distinct word: a 32-way conflict, 32 passes. Pad the row to 33 words
   (or swizzle).
6. 256 / 32 = 8.
7. FP16 has a 10-bit mantissa. Summing many products in FP16 accumulates rounding error that grows with K,
   whereas the products themselves are exact in FP32. In Section 13.8, FP16 accumulation had about 4× the
   error at K = 48.
8. That the operands are really an `(M,1,K)` and a `(1,N,K)` broadcast of two matrices (any 3-D product
   matches), their memory layouts, and their dtypes.

</details>

## Further reading

- NVIDIA, *CUDA C++ Programming Guide*: "Shared Memory" (banks, `__syncthreads`) and "Warp Matrix Functions"
  (WMMA). *PTX ISA*: `mma.sync`, `ldmatrix`, `cp.async`.
- NVIDIA, *Volta* (V100) and *Ampere* (A100) architecture whitepapers: tensor core shapes and throughput.
- NVIDIA, *CUTLASS* documentation, especially "Efficient GEMM in CUDA" (the block/warp/thread tile hierarchy).
- S. Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" (2022), which builds
  a GEMM kernel step by step through the ideas in 13.1–13.7.
- V. Volkov and J. Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra", SC 2008.
- T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023).
- P. Micikevicius et al., "Mixed Precision Training", ICLR 2018: FP16 storage, FP32 master weights, loss
  scaling.
- tinygrad: the tensor-core support in its optimizer (`tensor_cores` in the renderer, and the TC opt that
  reshapes axes to the MMA shape), route 1 of Section 13.9.
- P. Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations",
  MAPL 2019: a compiler whose unit of programming is exactly the block tile of this chapter.
