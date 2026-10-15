# 35. Register-Tiled GEMM on a Real GPU

The earlier GPU chapters separate index reasoning from hardware validation. This chapter adds an
executable naive-versus-register-tiled GEMM experiment, numerical checks on a real device, and an
honest timing protocol. It is not an emulator result and does not claim tensor-core or cuBLAS parity.

```bash
.venv/bin/python -m labs.gpu_gemm --shape 256 256 256 --repeat 10 --batch 20 --output book/outputs/gpu-register-gemm.json
.venv/bin/python -m pytest tests/test_gpu_gemm_lab.py -q
```

A working NVIDIA driver and NVRTC must be discoverable by `uopc.cuda`. A CUDA toolkit configuration
satisfying the runtime's library search is sufficient; environments with different NVRTC packaging
may need their library paths configured. Hardware tests explicitly skip when this setup is absent.
A skip establishes no device correctness or performance claim.

## 35.1 Derive output ownership first

The tiled kernel computes a 64×64 output block with a `(32,8)` thread block: 256 threads. Each thread
owns eight rows and two columns, or sixteen scalar accumulators. For thread coordinates `(tx,ty)`,
its output coordinates within the block are:

```text
row = ty + i*8       for i in 0..7
col = tx + j*32      for j in 0..1
```

Every output element has one owner: dividing the row by 8 yields i and its remainder yields ty;
dividing the column by 32 yields j and its remainder yields tx. Thus all 4096 elements are covered
without duplicate writes. Prove this mapping before optimizing the loads.

Adjacent lanes differ in tx, so at a fixed i and j they write adjacent columns. An alternative mapping
`col=tx*2+j` would make each instruction's adjacent-lane addresses stride by two. Both mappings can
be correct, but their transaction behavior differs. Layout is a performance decision constrained by
ownership and hardware access rules.

## 35.2 Cooperative loads and input reuse

Process the reduction in chunks of 16. One chunk stages `64*16` A elements and `16*64` B elements,
2048 float32 values or 8192 bytes of shared memory. Flatten the thread index as `lane=ty*32+tx`.
Each thread loads linear positions `lane`, `lane+256`, `lane+512`, and `lane+768` for each tile.
These positions cover each 1024-element tile exactly once.

The idealized input arithmetic intensity is:

```text
2 * 64 * 64 * 16 / (4 * (64*16 + 16*64)) = 16 FLOPs/byte
```

This excludes output writes, edge waste, cache effects, and repeated loads between blocks. It is
not a prediction of measured bandwidth. Compare its assumptions with the actual launch and shape.
At a 1×1 problem, the kernel still allocates and executes much of a large tile, making the idealized
model particularly misleading.

For each inner k, a thread reads eight A values and two B values into local fragments. It updates
all sixteen pairwise products. Each loaded A fragment value is reused twice, and each B fragment
value eight times within that thread. Accumulators persist in registers across chunks.

## 35.3 Shared-memory access and register cost

Within a warp, ty is constant and tx runs from 0 to 31. A fragment load at a fixed i and k reads
the same shared address across the warp, using broadcast behavior. B fragment loads at a fixed j
read adjacent columns. This avoids the obvious bank-conflict patterns discussed in Chapter 13.
Actual instruction widths and generated accesses still need inspection before claiming zero conflicts.

Sixteen accumulators plus fragments and address values create register demand. The compiler can
reuse fragment registers or spill values; source-level array size is not an exact physical-register
count. Inspect compiler/profiler resource reports for the target. This example does not collect
register, occupancy, or stall metrics automatically, so its throughput result cannot establish which
resource limits performance.

A useful extension varies output tile size and per-thread tile shape while keeping the same operation
and correctness checks. Record registers, spills, active blocks, shared bytes, and device time. A
larger register tile may improve reuse and still lose by reducing residency or causing spills.

## 35.4 Why both barriers are required

The first block barrier separates cooperative writes from all consumers of the staged tiles. Without
it, one warp may read values another warp has not written. The second barrier ensures every consumer
has finished before the next chunk overwrites the shared arrays. The first barrier of the next chunk
is too late: overwrite can already race with a previous-chunk read.

Threads in partial edge blocks remain active through both barriers. Invalid global coordinates load
zero into their shared positions. Only final output stores are masked. An early return before the
barriers would violate the cooperative execution contract for a partial block.

@include labs/kernels/gemm.cuh

**Experiment:** use dimensions `(31,33,17)` and `(63,65,33)`, where both output blocks and the final
reduction chunk are partial. Then deliberately remove one mask or barrier in a scratch version and
run numerical checks plus an available memory/race/synchronization checker. Numerical agreement in
a few runs cannot prove absence of a synchronization race.

## 35.5 Make the benchmark capable of detecting missing stores

Before each kernel's correctness launch, fill the device output with NaNs. If an edge element is
never written, comparison fails. Reusing the correct naive output as the tiled kernel's initial
output would conceal some missing-store bugs, so the harness explicitly resets it between kernels.

The reference multiplies the already-quantized float32 inputs in double precision on the CPU. This
separates input conversion error from device accumulation error. Every output element is checked
with declared absolute and relative tolerances. Tiny and irregular shapes are tested separately from
the benchmark shape; one aligned square matrix is inadequate correctness coverage.

## 35.6 Device events and host-submission gaps

Compilation and transfers occur before timing. Warm-up launches run before event samples. A sample
records a CUDA start event, enqueues a batch of kernels, records a stop event, waits for it, and divides
the elapsed interval by batch count. Repeat to obtain a distribution instead of one favorable number.

The harness enqueues directly through the driver API so it does not accidentally synchronize after
each launch. The default stream provides launch ordering. Events are destroyed and device buffers
are freed after use. Context/module ownership remains with the existing teaching runtime; this is
not a complete device-resource lifecycle service.

Device events do not magically remove host submission gaps. For tiny kernels, Python/ctypes may
submit work slowly enough that the GPU becomes idle within the measured interval. Those gaps are
included. CUDA graph replay or another controlled submission method is a subsequent experiment.
Do not compare these intervals to another library's differently scoped measurements as if they were
identical hardware kernel times.

@include labs/gpu_gemm.py

## 35.7 Recorded result and its limits

A recorded run on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (`sm_86`), driver version 13020 and
NVRTC 12.9, used float32 inputs and accumulators and the teaching runtime's NVRTC fast-math policy.
It did not use tensor cores. The report stores the kernel source SHA-256, device, launch dimensions,
raw samples, compilation time, and numerical error.

| 256×256×256 kernel | Median event interval per launch | Useful GFLOP/s | Max absolute error versus double oracle |
|---|---|---|---|
| Naive | 0.11594 ms | 289.4 | 4.55e-5 |
| Shared and register tiled | 0.04426 ms | 758.1 | 4.55e-5 |

This is about 2.62× for this comparison. A preceding run measured different absolute times while
showing a similar ratio, consistent with the need to record power, thermal, and timing variation.
Useful FLOPs count the mathematical `2*M*N*K`, not arithmetic wasted on masked tile padding.

The report does not compare cuBLAS or framework matmul, does not demonstrate tensor-core performance,
and does not contain profiler or sanitizer captures. Those are remaining evidence requirements,
not results to infer from a speedup over naive code. The hardware tests passed four shapes, including
partial tiles; they do not prove correctness for every shape, compiler version, or GPU architecture.

## 35.8 The next optimization must have a hypothesis

The staged loop alternates transfer and compute rather than overlapping them. A double-buffered
asynchronous-copy pipeline could reduce exposed memory latency, but it adds shared storage and
synchronization constraints. Establish that transfer latency is material with profiling before
adding that complexity.

Other projects include vectorized loads under explicit alignment guards, different thread tiles,
architecture-supported matrix instructions, and a dispatch policy for small or irregular problems.
Keep the naive kernel as a fallback and correctness baseline. Compare the same float32 mathematical
work, record the precision policy, and expand the workload table before making a competitive claim.
