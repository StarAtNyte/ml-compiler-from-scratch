# 24. Memory Planning, Precision, and Distributed Execution

A fast kernel is one component of a fast model. Allocation, transfer, synchronization, saved
activations, and communication can dominate the rest. Extend the runtime in small, measurable steps.

## 24.1 Buffer liveness and reuse

For a linear kernel schedule, assign each temporary an interval from its first write through its
last read. Inputs, outputs, aliases, and externally retained values may outlive the schedule.
Two temporaries can share storage when their lifetimes do not overlap, their sizes and alignments
fit, and the execution order guarantees the earlier work has finished.

Example: t1 is produced by kernel 0 and last read by kernel 2. t2 is produced by kernel 3. Their
storage may be reused on a sequential stream. A temporary written by kernel 2 cannot blindly reuse
t1 while that kernel is still reading t1. In-place operation requires a separate legality proof.

**Lab:** implement a best-fit free-list planner with explicit alignment. Compare planned peak bytes
against allocating every temporary separately. Test a branched graph with a late consumer, retained
outputs, and unequal temporary sizes. For asynchronous multi-stream execution, replace simple list
positions with happens-before reasoning; otherwise keep one stream.

## 24.2 Gradient checkpointing

Reverse mode saves forward values needed by backward. Checkpointing discards selected intermediates
and recomputes them later. For a chain of layers, saving every s-th boundary trades fewer saved
activations for extra forward work during backward. The best placement depends on tensor sizes and
operation costs, not just layer count.

Recomputation must preserve random operations, mutation, and numerical policy. Dropout needs the
same random mask or an equivalent reproducible generator state. Replaying a stateful update twice
changes the program.

**Lab:** checkpoint a pure block of the small GPT. Compare outputs and gradients, peak allocated
memory, and step time. Add a random operation and demonstrate deterministic recomputation. Report
extra work rather than describing reduced memory as a free optimization.

## 24.3 Mixed precision and quantization

Storage precision, multiplication precision, accumulation precision, and optimizer-state precision
are independent choices. Low-precision input does not require low-precision accumulation. Keep an
explicit conversion policy, including rounding, saturation, subnormal handling, and exceptional values.

For loss scaling, compute gradients of `scale*loss`, unscale before the optimizer update, and check
for nonfinite values. On overflow skip the update and lower the scale. Growth after sustained finite
steps is a policy to test. bf16's wider exponent range reduces one motivation for scaling, but does
not eliminate numerical error or all overflow risks.

For a symmetric per-output-channel weight quantizer, one simple policy is:

```text
scale[j] = max(abs(W[:,j])) / 127
q[:,j] = clamp(round(W[:,j] / scale[j]), -127, 127)
W_approx[:,j] = q[:,j] * scale[j]
```

Handle an all-zero channel without division by zero. Define the rounding rule. Storage savings do
not imply a speedup: dequantization overhead and hardware support matter. Activation quantization
adds calibration and outlier concerns; weight-only quantization is a smaller first project.

**Lab:** test conversion boundaries, small values, NaN/infinity policy, zero channels, and outliers.
Compare mixed-precision training with f32 on the same seed and batches. For quantization report model
quality delta, peak memory, and latency, including conversions.

## 24.4 Data and tensor parallelism

With P equal-sized data-parallel batches and a mean loss on each worker, averaging worker gradients
matches the full-batch gradient apart from rounding. Unequal local batch sizes require weighting by
sample count. An all-reduce sums or averages gradients; every worker then applies the same update.

For row-vector inputs and `Y=XW`, splitting W by output columns produces independent pieces of Y
that can be concatenated. Splitting W by input rows also splits X and produces partial outputs that
must be summed. Track bias placement: adding the full bias to every partial sum overcounts it.

Communication cost is often approximated as latency plus bytes divided by effective bandwidth.
Many tiny collectives pay latency repeatedly; bucketing helps but can delay communication overlap.
An asynchronous collective cannot consume gradients before they are ready or release their buffers
early. A real runtime also needs a policy for failed workers and communication errors.

**Lab:** compare a two-process data-parallel update with one full-batch update, including an unequal
batch case. Then partition one linear layer both ways and verify outputs and gradients. Record actual
communication bytes. A CPU process simulation teaches algebra; complete a separate real-device
measurement before claiming distributed GPU scaling.

## 24.5 Inference scheduling

Prefill processes prompt tokens together; decode repeatedly handles small additions to existing
contexts. They have different matrix shapes and resource needs. A serving scheduler must manage
request lengths, cache capacity, cancellation, and admission without corrupting per-request state.
Paged KV storage introduces a logical-to-physical block mapping so a request need not reserve one
large contiguous allocation. Indirection and allocation policy become part of the execution system.

**Lab:** build a simulator with variable prompt/output lengths and a finite cache budget. Compare
fixed batches and continuous admission. Report throughput, time to first token, inter-token latency,
and latency distribution. Then implement one policy with the actual model. Synthetic scheduler
results alone do not establish end-to-end serving performance.

**Exit gate:** correct buffer reuse, checkpointing, one measured precision optimization, a verified
parallel update, and an inference scheduling experiment. Keep each artifact independently reproducible.

## 24.6 Worked lab: buffer reuse with observable lifetimes

Take three temporaries: a uses 1024 bytes during kernels 0–2, b uses 512 bytes during kernel 1,
and c uses 768 bytes during kernels 3–4. Place a at offset 0 and b at offset 1024. At kernel 3,
a is dead, so c reuses a's 1024-byte slot. The arena needs 1536 bytes instead of 2304 bytes of
independent payload storage. Padding and different allocation strategies can change the comparison.

Lifetimes are inclusive. If b were produced by kernel 2 instead, it could not share a's storage:
kernel 2 may need to read a while writing b. The strict condition is `old.last < new.first`.
An in-place kernel is a different contract that this planner does not infer.

```bash
.venv/bin/python -m labs.memory
```

@include labs/memory.py

The planner uses best fit among whole freed slots. It does not split or coalesce slots and is not
an optimal packing algorithm. Retaining capacity makes its reuse decision simple and auditable.
Alignment is a condition on offsets relative to an appropriately aligned arena base; allocating
an arena with an insufficiently aligned base would violate the complete runtime contract.

**Experiment:** generate random lifetimes, write a distinct byte pattern when each temporary is
produced, and check every live pattern at every step. This simulates execution independently of
the overlap verifier. The included tests do exactly that; a verifier that repeats the planner's
mistake would otherwise provide false confidence.

**Next implementation:** infer lifetimes from `Plan` references, exclude aliases and retained outputs,
allocate an aligned device arena, and bind kernel arguments to slices. Keep the initial integration
on one ordered stream. Multi-stream reuse needs event-based happens-before proofs, not these integer
schedule positions alone.


## 24.7 Run the planner inside an actual compiler runtime

[Chapter 33](33-dynamic-tensor-compiler.md) derives lifetimes from rendered kernel dependencies,
allocates an aligned arena, and passes planned buffer slices to native kernels. Its tests check
actual matrix-chain results, memory savings, and retained-output safety. This implements the
single-schedule integration described above in a separate float32 compiler, without modifying
`uopc`'s existing device runtime.
