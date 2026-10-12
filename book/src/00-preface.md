# Preface: How to Read This Book

You type `loss.backward()` and `optimizer.step()`, and a few milliseconds later your laptop has done
a few billion floating point operations in exactly the right order. Somewhere between the Python and
the silicon there is a program that turned a description of *what* to compute into instructions for
*how* to compute it: which loops, in what order, fused how, tiled how, on how many cores. That program
is an ML compiler. This book builds one from nothing: no starter code, no hidden library, about 2,700
lines of Python that generate C and CUDA. By the last chapter it trains a GPT on Shakespeare, and you
will understand every line of every kernel it ran.

## What you'll have built

The compiler is called `uopc` (after its one data structure, the *UOp*). By the end of the book
it does all of this, and so will yours:

- **Week 1–2:** take a small program written as a graph, simplify it with algebra, and compile it to a
  C function that `gcc` turns into a shared library you call from Python.
- **Week 3–4:** compile *any* tensor program, with loops, reductions, broadcasting, reshapes and
  transposes, to correct (if slow) C.
- **Week 5:** split programs into kernels, differentiate them automatically, and train an MLP on MNIST.
  The reference run reaches 94.06% test accuracy at 14.4 ms per step on a laptop CPU
  (`book/outputs/mnist.txt`).
- **Week 6:** make CPU kernels fast. A 1024×1024 matrix multiply goes from 1.9 GFLOP/s to 113.6 GFLOP/s,
  against 131.6 GFLOP/s for numpy's OpenBLAS (`book/outputs/bench_gemm_1024.txt`). Flash attention,
  three ways.
- **Week 7–8:** map kernels onto GPUs: blocks, threads, shared memory, tensor cores, rendered as CUDA.
- **Week 9–10:** train a 112.6k-parameter GPT from scratch at 235 ms per step, and sample from it:

```
step     0  train 4.266  val 3.827    8933 ms/step
step   100  train 2.616  val 2.559     288 ms/step
step   599  train 2.015  val 2.110     235 ms/step

ICGLOUTery man!

For apbeng mantound rear:
```

(Small model, 600 steps, CPU. It has learned that Shakespeare has speaker names in capitals, line
breaks, and colons. It has not learned English. Neither had you at that age.)

## A first taste

Here is the whole compiler in one breath. Six lines of tensor code:

```python
import numpy as np
from uopc import Tensor
x = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
y = (x * 2 + 1).relu().sum(1)
print(y.numpy())
```

Run it with `OPT=0 DEBUG=4` (no optimizations, print every generated kernel) and you get this,
verbatim:

```c
#include <math.h>
#include <stdbool.h>
static inline int floordiv_i32(int a, int b) { int q = a / b; return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q; }
static inline int floormod_i32(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
void r_2(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val1 = acc0;
      float val2 = data0[((ridx0*3)+ridx1)];
      float alu3 = ((val2*2.0f)+1.0f);
      acc0 = (val1+((0.0f<alu3)?alu3:0.0f));
    }
    float val4 = acc0;
    data1[ridx0] = val4;
  }
}

  r_2                               0.114 ms     0.00 GFLOP/s  (again: 0.022 ms)
realize: 1 kernels, schedule 120.1 ms, run 0.2 ms
[ 9. 27.]
```

Everything in this book is in that listing somewhere. The multiply, add and ReLU were *fused* into
one loop body instead of three passes over memory (Chapter 7). The `sum(1)` became an accumulator
`acc0` and an inner loop `ridx1` (Chapter 6). The index `ridx0*3 + ridx1` came from the 2×3 shape
being flattened onto a 1-D buffer, then simplified by algebra (Chapter 3). `relu` became `(0.0f<alu3)?alu3:0.0f`,
because the compiler has no ReLU, only a comparison and a select (Chapter 1). The loops were opened
and closed in the right places by the linearizer (Chapter 5). The C was compiled by `gcc`, cached on
disk by a hash of its source, loaded with `ctypes` and called on numpy arrays (Chapter 4). The 120 ms
of "schedule" is the Python compiler running for the first time; the second time this program shape
is seen, that cost disappears (Chapter 7).

Drop `OPT=0` and the default optimizer notices that the outer loop only has 2 iterations and
unrolls it into straight-line code with two accumulators, `acc0` and `acc1`. That is Chapter 10.

> **Key idea:** an ML compiler does not "translate" your program the way a C compiler translates a
> statement. It *rewrites a graph* over and over, each rewrite moving the description one step
> closer to loops over memory, until what is left can be printed as C.

## The map

@diagram 00-pipeline

Read the map in a snake: left to right across the top, right to left across the middle, left to
right across the bottom. Each box is one module of `uopc/`, and each is one or two chapters.

| Stage | What it does | Module(s) | Lines | Chapter |
|---|---|---|---|---|
| Tensor API | Lazy tensors: every op just builds graph nodes | `tensor.py`, `nn.py` | 326 + 100 | 9 |
| Shape graph | The IR: hash-consed DAG of UOps, type checking, text format | `uop.py`, `dtype.py`, `uir.py` | 319 + 72 + 177 | 1 |
| Rewriting | Pattern matching and the graph rewrite engine | `rewrite.py` | 115 | 2 |
| Symbolic | Constant folding, algebra, bounds, evaluation by rewriting | `symbolic.py` | 295 | 3 (and 2) |
| Autodiff | Build the gradient graph from the forward graph | `autodiff.py` | 82 | 8 |
| Kernelize | Cut the graph into kernels, schedule them, cache the plan | `kernelize.py` | 207 | 7 |
| Layout + opts | Upcast, unroll, thread, choose memory layouts | `opt.py` | 155 | 10, 11 |
| Rangeify | Shapes become loops: push indices through movement ops | `rangeify.py`, `composite.py` | 191 + 92 | 6 |
| Linearize | Order the graph into a list with properly nested loops | `linearize.py` | 127 | 5 |
| Render | Print C (or CUDA) | `render_c.py`, `gpu.py` | 143 + 170 | 4, 12, 13 |
| Compile + execute | gcc/NVRTC, load, call | `runtime.py`, `cuda.py` | 66 + 90 | 4, 12 |

The line counts are real (`wc -l uopc/*.py` gives 2,740 in total, including docstrings). Nothing in
the pipeline is large. What makes a compiler hard is not the size of any one pass; it is that every
pass trusts every earlier pass, so a small mistake in week 1 surfaces as a wrong number in week 9.

## Who this book is for

You are a strong programmer. You have written a few thousand lines of C and know what a pointer, a
cache line and a segfault are (CMU's 15-213, *Computer Systems: A Programmer's Perspective*, is the
reference level). You are comfortable in Python, including classes, closures and `dataclasses`. You
have trained a neural network with PyTorch or similar and know roughly what a matrix multiply, a
softmax and backpropagation are.

You do *not* need to have taken a compilers course, to know any GPU programming, or to know anything
about term rewriting, interval arithmetic, autodiff internals or flash attention. Each of those is
taught from first principles when it is needed.

## How the book maps onto the course

The book is the companion text for a 14-week, project-based course (`book/SYLLABUS.md`). The course
gives a spec each week and tests to pass; the book explains how to meet the spec and why the reference
solution looks the way it does.

| Week | Topic | Chapters | Milestone |
|---|---|---|---|
| 1 | UOps, rewriting, symbolic | [1](01-uops.md), [2](02-rewriting.md), [3](03-symbolic.md) | Run programs by rewriting |
| 2 | Memory, rendering, runtime | [4](04-memory-and-c.md) | Elementwise programs compiled to C |
| 3 | Loops | [5](05-loops.md) | GEMM and conv written by hand |
| 4 | Movement ops, rangeify | [6](06-rangeify.md) | Any model compiles to C (slowly) |
| 5 | Kernels, autodiff, Tensor | [7](07-kernelize.md), [8](08-autodiff.md), [9](09-tensor-mnist.md) | MNIST trains |
| 6 | Fast CPU kernels, fusion | [10](10-cpu-performance.md), [11](11-fusion-flash.md) | SOTA-competitive CPU GEMM |
| 7–8 | GPUs, shared memory, tensor cores | [12](12-gpus.md), [13](13-shared-memory-tensor-cores.md) | CUDA kernels |
| 9–10 | Real models | [14](14-gpt.md), [15](15-beyond.md) | A GPT trains |
| 11–14 | Projects | [15](15-beyond.md), appendices | Your own extension |

@diagram 00-weeks

The course is cumulative in a way most courses are not. Week 5 does not replace week 1's code; it
calls it, thousands of times per training step. The syllabus calls the course "an exercise in slop
management", and it means it: if your week-1 rewrite engine is quadratic, your week-9 GPT compiles
overnight; if your week-1 hash consing confuses `1` with `1.0`, your week-5 gradients are silently
wrong.

## Setting up

You need Python 3.11 or newer, a C compiler with OpenMP, and (optionally) an NVIDIA GPU with the CUDA
toolkit. The reference machine for every number in this book is an Intel i7-13620H laptop
(6 performance + 4 efficiency cores) with Python 3.12.13 and gcc 15.2.0 (`book/outputs/machine.txt`).

```sh
git clone <your fork of the course repo> ml-compilers && cd ml-compilers
python3 -m venv .venv                 # or: uv venv
.venv/bin/pip install -e '.[dev]'     # or: uv pip install -e '.[dev]'
```

The only runtime dependency is numpy; `[dev]` adds pytest (tests), and markdown and pygments (to build
this book). The `pyproject.toml` is short enough to read in full:

@include pyproject.toml

Check that your C compiler understands the flags the runtime uses. `uopc/runtime.py` compiles every
kernel with `gcc -O3 -march=native -fopenmp -ffp-contract=fast -fassociative-math -fno-signed-zeros
-fno-trapping-math` (Chapter 10 explains each one). On Linux, gcc has OpenMP built in. On macOS,
Apple's clang does not support `-fopenmp` out of the box; install gcc from Homebrew and set
`CC=gcc-14` (or whatever version you installed).

Then run the test suite. On the reference machine:

```
...................................................................s...  [100%]
70 passed, 1 skipped in 10.78s
```

The one skipped test is `tests/test_w07_gpu.py::test_cuda_smem_gemm_on_gpu`, which needs a working
NVIDIA driver. The reference machine did not have one, so all CUDA in this book is tested by rendering
it and by running the same index math through a CPU emulator (Chapter 12). If you have a GPU, that
test will run.

## A tour of the repository

```
ml-compilers/
├── uopc/            the compiler (one module per stage; see the map)
├── tests/           test_w01_uops.py ... test_w09_models.py: one file per week
├── examples/        mnist_mlp.py, gpt.py, flash_attention.py, bench_gemm.py, datasets.py
├── book/
│   ├── SYLLABUS.md  the course spec
│   ├── src/         this book, one markdown file per chapter, plus diagrams/
│   ├── outputs/     captured runs quoted in the text (benchmarks, training logs)
│   └── site/        the built HTML
└── tools/
    └── build_book.py   builds book/src -> book/site
```

The package's own docstring is the one-paragraph version of the course:

@include uopc/__init__.py

### `uopc/`: the compiler

Every module starts with a docstring saying which week it belongs to and what it does. Read those
docstrings first when you open a file; they are written to be the summary you would want.
Imports only point from later weeks to earlier ones: `uop.py` imports nothing but `dtype.py`;
`rewrite.py` imports `uop.py`; `symbolic.py` imports both; `rangeify.py` imports `symbolic.py`; and so
on up to `tensor.py`, which imports almost everything. There are no cycles, which means you can build
and test the modules in the order the chapters present them.

### `tests/`: the oracle

| File | Tests | What it checks |
|---|---|---|
| `test_w01_uops.py` | 12 | hash consing, type checking, rewriting, symbolic, evaluation, `.uir` |
| `test_w02_render.py` | 4 | loopless C programs over memory |
| `test_w03_loops.py` | 3 | hand-written GEMM and conv in the loop IR |
| `test_w04_rangeify.py` | 6 | movement ops and reductions against numpy |
| `test_w05_tensor.py` | 5 | kernels, gradients (finite differences), the Tensor API |
| `test_w06_opt.py` | 5 | optimized kernels still give the same answers |
| `test_w07_gpu.py` | 6 | CUDA rendering, the emulator, shared memory, tensor cores |
| `test_w09_models.py` | 3 | small end-to-end models |

Run one week, one test, or stop at the first failure:

```sh
.venv/bin/python -m pytest tests/test_w01_uops.py -v
.venv/bin/python -m pytest tests/test_w01_uops.py -k hash_consing
.venv/bin/python -m pytest -x
```

The tests are an *oracle* in the course's sense: they define correct behavior. When you write your own
compiler, the strongest thing you can do is make these tests pass against your code, then write more.
Appendix B is about how.

### `examples/`

`mnist_mlp.py` (week 5), `bench_gemm.py` and `flash_attention.py` (week 6), and `gpt.py` (weeks 9–10)
are complete programs that use only the `Tensor` API. `datasets.py` downloads MNIST and Tiny
Shakespeare into `~/.cache/uopc/data` the first time you need them.

### `book/`

The markdown sources are in `book/src/`. Code listings are not copied into the text; they are pulled
out of the repository when the book is built, so a listing can never drift from the code it shows.

## Conventions

### Listings

A listing with a label above it, like this one, is the *real code*, extracted from the file named in
the label at build time:

@include uopc/uop.py#prod

The label `uopc/uop.py · prod` means "the definition named `prod` in `uopc/uop.py`". A label like
`uopc/uir.py:37-45` means a line range. Listings without a label are illustrations: REPL sessions,
wrong versions to contrast with right ones, or generated C. Any output shown in this book was produced
by running the code on the reference machine, and pasted unedited (timings will differ on your
machine).

The code style is dense: two-space indentation, one-line `if`s, little vertical whitespace. That is a
deliberate choice for a book. A function that fits on one screen can be read as a whole, and every
chapter's listings would be twice as long in a more conventional style. Your own code does not need to
look like this; it needs to be as *simple* as this.

### Environment variables

| Variable | Values | Effect | Where |
|---|---|---|---|
| `DEBUG` | `1` | one line per `realize`: kernel count, schedule time, run time | `kernelize.py` |
| | `2` | also one line per kernel: time and GFLOP/s | |
| | `3` | also run each kernel a second time (warm caches) and report that time | |
| | `4` | also print the generated source of every newly compiled kernel | |
| `OPT` | `0` | no kernel optimizations: plain loops | `tensor.py` |
| | `1` (default) | hand-written heuristics (`HandOpt`) | `opt.py` |
| | `2` | search: time candidate optimizations and keep the fastest (`SearchOpt`) | `opt.py` |
| `CC` | `gcc` (default) | the C compiler | `runtime.py` |
| `CFLAGS` | see above | override all compiler flags (e.g. strict IEEE ordering) | `runtime.py` |
| `UOPC_CACHE` | `~/.cache/uopc` | where compiled `.so` files are cached | `runtime.py` |

Levels are cumulative: `DEBUG=4` prints everything `DEBUG=2` does, plus the source. Note that
`DEBUG=4` only prints a kernel the first time it is *compiled* in this process; a cached plan is not
printed again.

### Words

A *UOp* (micro-op) is one node of the graph. An *op* is its kind (`Ops.ADD`). A *kernel* is one
generated C function. *Realize* means "actually compute this tensor now". Appendix C has the full
glossary; each chapter defines its terms the first time it uses them.

## How each chapter works

Every chapter after this one has the same skeleton:

1. **A hook**: the problem the chapter solves, in one paragraph.
2. **What you'll build** and **Where this fits**: the deliverables, and links to the neighbors.
3. **Concept sections**, each of which explains *why* before *how*, with real code, real output and a
   diagram wherever structure is easier to see than to read.
4. **Pitfalls**: bugs that actually happened while building the reference compiler. Not hypothetical
   "be careful with X" warnings: real failures with real symptoms, and the fix. If you are stuck on a
   bug, read the Pitfalls of the chapter you are in first.
5. **Exercises**, graded ★ (an hour or less), ★★ (an afternoon) and ★★★ (a project). Some are
   "implement X", some are "predict the output", some are "find the bug". ★ exercises have solutions in
   collapsed blocks. Try before you peek.
6. **A quiz** in the style of the course's in-person quizzes: short questions that check you understood
   the idea, not that you memorized the code.
7. **Further reading**: papers, books and source code worth your time.

### Three ways to read

- **Taking the course.** Read the chapters for the week *before* the week starts, implement the spec
  in your own repository, and use the reference code only when you are stuck or when you are done and
  want to compare. The exercises marked "implement" are often exactly the assignment.
- **Self-study.** Read in order. For each chapter, re-implement the module from the prose and the
  tests, then diff your design against `uopc/`. Expect weeks 1, 4 and 5 to take the longest.
- **Reading to understand ML compilers** (you have used XLA, TVM, Triton or `torch.compile` and want to
  know what is inside). Read Chapters 1, 2, 6, 7 and 10 in order, skim the rest. Chapter 15 compares
  `uopc` to the production compilers.

## Slop management

The syllabus grades style with three questions: *Did you write slop? Can I read your code? Can it be
simpler?* Those questions are the most important technical advice in the course, so here is what they
mean in practice.

**Small.** The whole compiler is 2,740 lines. When a module grows past a few hundred lines, something is
usually doing two jobs. The UOp, for example, is one class with three fields; there is no `AddUOp`,
no `ConstUOp`, no visitor hierarchy. Chapter 1 explains why that uniformity makes everything after it
shorter.

**Tested.** Every week adds a test file, and every test is an end-to-end check against an independent
oracle: numpy, finite differences, Python's own arithmetic, or the C program's actual output. A pass
that has not been run against an oracle is a pass that is wrong in a way you have not found yet.

**Readable.** Names say what things are (`call_params`, `inline_call`, `graph_rewrite`). Docstrings
say why, not what. Errors are raised where the mistake is made, with a message that names the problem
(`Ops.ADD: dtype mismatch dtypes.i32 vs dtypes.f32 (use CAST)`), never three passes later as a
`KeyError`.

**Honest about failure.** The file `book/NOTES.md` is a lab notebook of real bugs hit while building
`uopc`: hash-cache collisions between `1` and `1.0`, a linearizer that took four designs, an
unsimplified index that made attention O(N³), a fusion decision that recomputed a LayerNorm for every
output column. Each one is told in full in the Pitfalls section of the chapter it belongs to, with
the numbers. Learning to recognize these failure shapes is most of what experience in compilers is.

**Simple beats clever until measured.** Every performance trick in Part IV is introduced with a
benchmark that shows the problem first. Before that, the rule is: the simplest correct thing.

## Acknowledgments and lineage

The design of `uopc` is heavily inspired by [tinygrad](https://github.com/tinygrad/tinygrad), whose
UOp IR, pattern matcher and "rangeify" lowering are the direct ancestors of the ones in this book.
`uopc` is much smaller and makes different trade-offs (it is a teaching compiler; tinygrad runs real
workloads on many backends), but if you finish this book, reading tinygrad's source is the natural
next step, and Chapter 15 gives a guided route through it. The course syllabus (`book/SYLLABUS.md`)
defines the week-by-week spec that the chapters follow.

## Warm-up exercises

1. ★ Set up the environment and run the whole test suite. Then run
   `.venv/bin/python -m pytest tests/test_w01_uops.py -v` and read the names of the 12 tests. Which
   chapters do you expect each to belong to?
2. ★ Run the "first taste" program from this preface with `DEBUG=4` and then `OPT=0 DEBUG=4`. Find
   the two differences in the generated C and explain them in one sentence each.
3. ★ Change the program to `(x * 2 + 1).relu().sum(0)` (sum over the other axis). Before running it,
   predict the loop bounds and the index expression into `data0`. Then check.
4. ★ Run `wc -l uopc/*.py` and draw the import graph of `uopc/` by reading the `import` lines at the
   top of each file. Confirm that it has no cycles.

<details><summary>Hints and solutions</summary>

1. Hash consing, type inference and `.uir` belong to Chapter 1; `test_custom_pattern_matcher`,
   `test_run_stack_program` and `test_call_inlining_and_scoping` to Chapter 2; `test_const_fold_semantics`
   and the two `test_symbolic_*` tests to Chapter 3; the deep-graph test to Chapters 1 and 2 (both
   `toposort` and `graph_rewrite` must be iterative).
2. With the default `OPT=1`, the two-iteration outer loop is upcast: it disappears, and its two
   iterations become two accumulators (`acc0`, `acc1`) updated in one inner loop. The index into
   `data0` becomes `ridx0` and `ridx0+3` instead of `ridx0*3+ridx1`.
3. The output has 3 elements and the reduction runs over 2 rows, so (with `OPT=0`) the outer loop
   runs to 3, the inner loop to 2, and the index is `ridx1*3 + ridx0`: the reduction now strides
   through memory.
4. The spine is `dtype.py` ← `uop.py` ← `rewrite.py` ← `symbolic.py` ← `rangeify.py` ← `kernelize.py`
   ← `tensor.py` ← `nn.py`, with side branches (`render_c.py` ← `gpu.py`, `composite.py` ← `autodiff.py`,
   `runtime.py` ← `kernelize.py`). Every import points from a later week to an earlier one (or within
   a week), so there is no cycle. The one surprise is at the bottom of `tensor.py`: it imports `opt.py`
   *after* the `Tensor` class is defined, to pick the default optimizer.

</details>

## Further reading

- Randal E. Bryant and David R. O'Hallaron, *Computer Systems: A Programmer's Perspective* (3rd ed.,
  2015). The prerequisite level for this book, and the best reference for the memory hierarchy chapters.
- Keith D. Cooper and Linda Torczon, *Engineering a Compiler* (2nd ed., 2011). A conventional compilers
  textbook, useful as a contrast: most of it is about parsing and register allocation, which ML
  compilers barely need.
- The tinygrad repository, <https://github.com/tinygrad/tinygrad>. Start with `tinygrad/uop/ops.py`
  after Chapter 2.
- Tianqi Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018.
  The paper that made "ML compiler" a field; compare its design to `uopc` after Chapter 10.
