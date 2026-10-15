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
aliases. The SSA lab verifies already constructed SSA; it does not yet build SSA from source.

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
