# 28. Irregular Workloads and Heterogeneous Accelerators

Dense GEMM is central but cannot represent every workload a frontier lab encounters. Sparse data,
variable-length batches, mixture-of-experts routing, and multiple hardware targets expose assumptions
hidden by static contiguous tensors.

## 28.1 Gather, scatter, and atomic reductions

Gather reads values at data-dependent indices. Scatter writes to data-dependent destinations, so
duplicate indices require explicit semantics: overwrite, ordered update, or reduction. Parallel
floating-point scatter-add often uses atomics and may be nondeterministic because update order varies.
A deterministic alternative can sort or group updates, at additional cost.

**Lab:** implement embedding lookup and its gradient with repeated token IDs. Test out-of-range
indices and state the error policy. Compare atomic and grouped reductions for correctness, reproducibility,
and skewed index distributions. Dense random indices alone miss contention hotspots.

## 28.2 Sparse formats and ragged batches

CSR stores row offsets, column indices, and nonzero values. Row r spans offsets `[rowptr[r],rowptr[r+1])`.
Work per row is uneven, so assigning one fixed unit of work per row can leave hardware underused.
Block-sparse formats trade some wasted arithmetic for regular tiles and reduced metadata overhead.
Sparsity percentage alone cannot predict speed: distribution, block structure, index traffic, and
conversion cost matter.

Ragged sequences use lengths or offsets rather than padding every example to the maximum length.
This reduces wasted arithmetic but complicates bounds, scheduling, and memory ownership. Dynamic
batch composition can change the best schedule from one request to the next.

**Lab:** implement CSR matrix-vector multiply and a segmented reduction. Test empty rows, long rows,
and heavily skewed lengths. Compare against dense and padded baselines including format conversion.
Explain where sparse execution becomes worthwhile for your actual hardware.

## 28.3 Mixture-of-experts dispatch

A routed layer selects experts per token, groups tokens by destination, computes expert outputs,
and combines them with routing weights. Correctness includes token ordering, capacity policy,
multiple selected experts, and the reverse mapping in backward. Distributed routing adds communication
and load imbalance; average tokens per expert can hide an overloaded expert that sets step time.

**Lab:** build a CPU reference with explicit token-to-expert lists, then implement grouped dispatch
and combine. Compare uniform and skewed routing. Measure padding waste, dispatch cost, expert compute,
and communication separately. Test gradients with repeated destinations and a defined overflow policy.

## 28.4 Portability and hardware/software codesign

Identify which compiler layers express mathematical semantics and which encode hardware assumptions.
A portable operation can still need a target-specific schedule, layout, synchronization mechanism,
and runtime ABI. Capability queries should select supported implementations and provide clear errors
or fallbacks for unsupported features.

For a new accelerator, write a machine model: compute units, supported numeric formats and operations,
memory hierarchy, bandwidth, local storage, communication topology, execution ordering, and launch cost.
Estimate a roofline before implementing a backend. Compare the estimate with microbenchmarks and
revise it; a simulator is useful only within the fidelity it has demonstrated.

**Lab:** port elementwise work and a reduction to a second target, or build a documented toy-accelerator
simulator if hardware is unavailable. Lower one tiled matrix operation and explain each architecture
assumption. Label simulated performance separately from measured performance.

**Exit gate:** correct irregular operations, a skew-sensitive benchmark suite, a routed-layer
prototype, and a portability report grounded in an explicit target model.

## 28.5 Worked lab: duplicate indices change the operation

A CSR matrix may contain several entries for the same column in one row. Matrix-vector multiplication
must add all their contributions. For offsets `[0,3,3,4]`, the second row is empty; its sum is zero.
Neither empty rows nor duplicates should be treated as malformed when the format contract allows them.

Embedding backward has the same accumulation issue. If tokens are `[1,1,2]`, gradients for the first
two positions both contribute to vocabulary row 1. NumPy advanced-index assignment
`result[indices] += gradients` uses temporary indexing and does not provide the required accumulation
for repeated indices. `np.add.at` makes the intended repeated updates explicit in the CPU reference.

```bash
.venv/bin/python -m labs.irregular
```

@include labs/irregular.py

The CSR implementation checks offsets, array lengths, index types, and bounds before execution.
This validation belongs at an import or construction boundary for trusted repeated workloads, not
necessarily in every timed kernel launch. Measure validation separately if it is amortized.

**Experiment:** compare CSR output against a dense matrix assembled by adding entries, not overwriting
duplicates. For embeddings, send every token to one vocabulary row and verify its gradient equals
the sum of all token gradients. That distribution is also a useful GPU contention stress case.

**Next implementation:** implement a real GPU scatter-add, then a sorted segmented reduction. Compare
precision, determinism, scratch memory, and skew sensitivity. A deterministic CPU reference provides
semantics; it does not require every parallel implementation to have identical rounding order.
