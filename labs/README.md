# Executable advanced labs

Run from the repository root after installing `.[dev]`:

```bash
python -m labs.ssa
python -m labs.attention
python -m labs.memory
python -m labs.dynamic
python -m labs.numerics
python -m labs.irregular
python -m pytest tests/test_advanced_labs.py -q
```

`dynamic` needs gcc and the existing `uopc` runtime. The other labs use Python and NumPy.
These are teaching implementations, separate from the reference compiler's production path.
Attention executes on the CPU and does not demonstrate GPU performance. Memory planning assumes
a sequential schedule, disjoint logical temporaries, and inclusive lifetimes. It does not infer
aliases. The SSA lab verifies already constructed SSA. The separate [`minicc`](../minicc/README.md) compiler
now constructs SSA from source and provides optimizers and backends. The
[`tensorjit`](../tensorjit/README.md) compiler integrates dynamic shapes, fusion, and memory planning.

| Lab | Book | What its tests establish |
|---|---|---|
| `ssa` | 21 | Diamond/loop dominance, phi edge liveness, rejection of malformed SSA |
| `attention` | 22 | Tiled forward and backward against dense evaluation and finite differences |
| `dynamic` | 23 | Runtime length reuse and dtype/layout fallback dispatch |
| `memory` | 24 | Aligned storage reuse without overwriting live synthetic values |
| `numerics` | 26 | Mergeable variance and int8 reconstruction bounds including zero channels |
| `irregular` | 28 | CSR empty rows, duplicate entries, and repeated-token gradient accumulation |

Progress by modifying the lab, predicting the effect, and running its independent checks. Keep
algorithm correctness, compiler integration, real-device correctness, and speed measurements as
separate milestones. Each chapter specifies the next extension beyond the executable baseline.


The seventh lab, `python -m labs.gpu_gemm`, requires real CUDA hardware and discoverable NVRTC.
It compares naive and shared/register-tiled GEMM with tail guards, a double-precision CPU oracle,
and device-event samples. `tests/test_gpu_gemm_lab.py` runs four real-device correctness cases or
explicitly skips without hardware. Chapter 35 explains the ownership proof and timing limitations.
The CUDA source is included as package data in `labs/kernels/gemm.cuh`.
