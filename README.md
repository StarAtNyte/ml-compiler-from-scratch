# Building an ML Compiler from Scratch

[![Test and publish the book](https://github.com/StarAtNyte/ml-compiler-from-scratch/actions/workflows/pages.yml/badge.svg)](https://github.com/StarAtNyte/ml-compiler-from-scratch/actions/workflows/pages.yml)

**Read the book: <https://staratnyte.github.io/ml-compiler-from-scratch/>** · [PDF](https://staratnyte.github.io/ml-compiler-from-scratch/book.pdf)

Course notes and a reference implementation for a 14-week **Compilers for Machine Learning** course
([syllabus](book/SYLLABUS.md)), extended with a foundations-to-frontier self-study track.

`uopc` compiles tensor programs to C and CUDA. It has a graph IR of hash-consed `UOp`s, a rewrite engine,
symbolic simplification, *rangeify* (which turns shaped operations into loops), kernel fusion, autodiff, and
a CPU/GPU optimizer, in a compact Python implementation. It depends only on numpy and a C compiler.

| What it does | Result on a laptop (i7-13620H, gcc 15) |
|---|---|
| Evaluate week-1 programs by rewriting | `STACK`/`CALL`/`INDEX` programs reduce to constants |
| Compile any shape graph to C | GEMM, conv2d, softmax, attention, … all checked against numpy |
| Train an MLP on MNIST (week 5) | 94% test accuracy in 600 steps, ~15 ms/step |
| Fast GEMM (week 6) | 5 → **321 GFLOP/s** (1024³, end to end), numpy/OpenBLAS: 449 |
| Flash attention, three ways (week 6) | unfused / fused-recompute / online softmax, checked against reference results |
| GPU kernels (weeks 7–8) | CUDA renderer, shared-memory tiled GEMM, WMMA; indexing emulated, hardware tests optional |
| Train a GPT on Tiny Shakespeare (weeks 9–10) | 112k params, val loss 3.83 → 2.11 in 600 steps, ~0.24 s/step |

The syllabus is the minimum scope. The [frontier-lab preparation map](book/src/29-frontier-lab.md)
adds specialization paths and project gates for compiler, kernel, training, inference, and codesign
roles. Chapters 26–29 develop numerical analysis, scheduling and backend construction, irregular
workloads, distributed reliability, and research practice. Real-device and scale claims require
measured evidence; the extended text does not imply those features are implemented in `uopc`.

The advanced track includes [six runnable labs](labs/README.md), with independent checks for SSA,
tiled attention gradients, memory reuse, dynamic dispatch, numerical methods, and sparse operations.
Run `.venv/bin/python -m pytest tests/test_advanced_labs.py -q` to validate them.

## Quick start

```bash
git clone https://github.com/StarAtNyte/ml-compiler-from-scratch.git && cd ml-compiler-from-scratch
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # or: python -m venv .venv && pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # compiler and advanced lab checks
.venv/bin/python examples/mnist_mlp.py              # downloads MNIST, trains
.venv/bin/python examples/bench_gemm.py 1024        # the week-6 optimization ladder
.venv/bin/python examples/flash_attention.py 1024 64
.venv/bin/python examples/gpt.py                    # trains a char-level GPT, then samples
DEBUG=4 .venv/bin/python examples/mnist_mlp.py      # print every generated C kernel
```

Requirements: Python ≥ 3.11, numpy, gcc with OpenMP. Optional: an NVIDIA GPU + driver + NVRTC for `uopc/cuda.py`.

## The book

For a path from prerequisites through advanced engineering, begin with
[Foundations](book/src/00-prerequisites.md), then follow the existing compiler chapters and
[Chapters 20–29](book/src/20-syllabus-contract.md). The extension covers CFG/SSA and LLVM/MLIR,
real GPU optimization and attention backward, dynamic shapes and capture, memory and distributed
runtime systems, and a research capstone. Each stage has implementation labs and completion gates.
These labs extend the curriculum; they are not all implemented in the reference compiler.
The [syllabus contract](book/src/20-syllabus-contract.md) tracks the supplied course requirements,
including dtype-promotion differences and hardware-dependent performance targets.


```bash
.venv/bin/python tools/build_book.py --pdf     # -> book/site/ (index.html, one page per chapter, book.html, book.pdf)
python3 -m http.server -d book/site 8000       # browse it at http://localhost:8000 (search needs http)
```

Code listings in the book are pulled from this repository at build time (`@include path#symbol`), so
they match the tested code. Outputs shown in the book were produced by running it.

| Part | Chapters | Course weeks |
|---|---|---|
| I: The IR | 1 UOps · 2 Rewriting · 3 Symbolic algebra | 1 |
| II: From graphs to machine code | 4 Memory & C · 5 Loops & the linearizer · 6 Rangeify | 2–4 |
| III: A real compiler | 7 Kernel fusion · 8 Autodiff · 9 Tensors & MNIST | 5 |
| IV: Making it fast | 10 Fast CPU kernels · 11 Fusion & flash attention | 6 |
| V: Accelerators | 12 GPUs · 13 Shared memory & tensor cores | 7–8 |
| VI: Real models | 14 Training a GPT · 15 Beyond | 9–14 |
| Advanced track | 20 Syllabus contract · 21 Compiler foundations · 22 GPU engineering · 23 Dynamic execution · 24 Runtime & scale · 25 Capstone | Self-paced |
| Frontier track | 26 Numerics · 27 Scheduling & backends · 28 Irregular workloads · 29 Frontier-lab preparation | Self-paced |
| Appendices | A Reference · B Testing & slop management · C Glossary & quiz bank | — |

## Repository map

```
uopc/            the compiler (one module per idea, in course order)
  uop.py dtype.py rewrite.py symbolic.py uir.py     week 1: IR, rewriting, algebra, text format
  render_c.py runtime.py linearize.py               weeks 2-3: memory, C, loops
  rangeify.py composite.py                          week 4: shapes -> loops
  kernelize.py autodiff.py tensor.py nn.py          week 5: fusion, gradients, Tensor, layers
  opt.py                                            week 6: upcast, threads, layout, search
  gpu.py cuda.py                                    weeks 7-8: GPU mapping, CUDA, tensor cores
tests/           test_wNN_*.py, one file per week
examples/        mnist_mlp.py bench_gemm.py flash_attention.py gpt.py
book/            src/ (chapters + diagrams), outputs/ (captured runs), NOTES.md (real bugs), site/ (built)
tools/           build_book.py, site/ (the website's CSS and JS)
.github/         CI: run the tests, build the site and PDF, deploy to GitHub Pages
```

## Limitations

- The reference machine has no working NVIDIA driver, so CUDA output is render-tested and GPU index
  math is verified by running the same kernels through a CPU emulator. `tests/test_w07_gpu.py` runs the
  shared-memory GEMM on a real GPU when one is available.
- Symbolic shapes and KV caching are design exercises (chapter 15), not implemented.
- Benchmarks are from a laptop and vary between runs by up to 2x (thermal limits, power source,
  background load). `book/outputs/` has the raw runs.
