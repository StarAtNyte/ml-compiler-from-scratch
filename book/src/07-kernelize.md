# 7. Kernel fusion and fission

[Chapter 6](06-rangeify.md) ended with a compiler that can lower any shape graph into one loop
nest. "Any" is the problem. Ask it for `(A @ B) @ C` with three 128x128 matrices and it produces a
correct C function that takes **367 ms**. The same computation split into two loop nests, with the
intermediate `A @ B` written to a temporary buffer in between, takes **1.1 ms**. Nothing about the
arithmetic changed; only *where the loop boundaries are*. This chapter is about choosing those
boundaries: which parts of the graph become one kernel (**fusion**), where the graph is cut and an
intermediate is written to memory (**fission**), and how the resulting list of kernels is compiled
once, cached, and executed over and over by a training loop.

**What you'll build**

- The rule that decides where a graph is cut: **materialization points** (roots, `STAGE`,
  `REDUCE`).
- `make_plan`: cut the graph, replace each cut by a placeholder buffer, renumber buffers locally
  so identical kernels render identical C, and lower each piece with `rangeify`.
- The `Plan`: a list of compiled kernels plus a description of which buffers each one reads and
  writes, holding no UOps at all.
- Buffers owned by UOps, freed by the garbage collector.
- `realize()`, with a **structural JIT cache** keyed on the shape of the graph, not its data.
- `stage_broadcast_compute`, a one-rule cost model for the fusion/recompute trade-off, and
  `fuse_reduce`, the switch that lets reductions nest.

**Where this fits.** The kernelizer sits between the shape graph (built by the Tensor library of
[Chapter 9](09-tensor-mnist.md) and extended by the autodiff of [Chapter 8](08-autodiff.md)) and
rangeify. It calls `lower()` from Chapter 6 once per kernel and the renderer and runtime from
[Chapter 4](04-memory-and-c.md) once per *distinct* kernel. The optimizer of
[Chapter 10](10-cpu-performance.md) plugs into it through the `opt` argument.

Unless stated otherwise, the listings in this chapter were produced with `OPT=0` (no week-6
optimizer, so the C stays readable) and with the renderer's four-line prelude removed.

## 7.1 Why one giant loop nest is wrong

Here is `(A @ B) @ C` built with the movement ops of Chapter 6 and lowered as a single kernel
(`N = 4` to keep the listing short):

```python
def mm(a, b):                                    # (n, k) @ (k, m), movement ops only
  n, k = a.shape; _, m = b.shape
  ae = a.reshape((n, 1, k)).expand((n, m, k))
  be = b.permute((1, 0)).reshape((1, m, k)).expand((n, m, k))
  return (ae * be).reduce(Ops.ADD, 1)
A, B, C = (buf(s, 16).reshape((4, 4)) for s in range(3))
root = mm(mm(A, B), C)
```

```c
void abc(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*4);
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float acc1;
      acc1 = 0.0f;
      for (int ridx2 = 0; ridx2 < 4; ridx2++) {
        float val2 = acc1;
        float acc0;
        acc0 = 0.0f;
        float val3 = data2[((ridx2*4)+ridx1)];
        for (int ridx3 = 0; ridx3 < 4; ridx3++) {
          float val4 = acc0;
          float val5 = data0[(alu1+ridx3)];
          float val6 = data1[((ridx3*4)+ridx2)];
          acc0 = (val4+(val5*val6));
        }
        float val7 = acc0;
        acc1 = (val2+(val7*val3));
      }
      float val8 = acc1;
      data3[(alu1+ridx1)] = val8;
    }
  }
}
```

Read it with Chapter 6's rule in mind: an accumulator lives in the loops its index mentions. The
inner product `(A @ B)[i, k]` is needed by the outer reduction at index `(i, k)`, where `k`
(`ridx2`) is the *outer reduce's* loop counter. So the inner accumulator `acc0` lives inside the
`ridx2` loop, which lives inside the `j` loop (`ridx1`). Every element `(A @ B)[i, k]` is recomputed
once for every `j`: four nested loops, O(N^4) work for an O(N^3) problem. At `N = 128`:

```text
one loop nest :    367.4 ms
two kernels   :      1.1 ms
max |diff| vs numpy: 0.00018310547 0.00018310547
```

Both answers are right. One is over 300x slower.

The general principle: **a value that is consumed through an `EXPAND` is recomputed once per copy,
unless something stores it.** `EXPAND` is free as an index map precisely because it does not
remember anything; every broadcast copy re-derives its value from scratch. For a load from a
buffer, re-deriving is a single load and costs nothing extra. For a reduction (an entire loop) or a
long elementwise chain, it costs a multiple of the work.

Not every reduction under an `EXPAND` is a problem. In `x - max(x)` (Chapter 6, `sub_max`), the
max's index mentions only the row loop, so the linearizer computes it once per row in a sibling
loop. Trouble starts when the consumer of a reduction is itself inside a loop the producer does not
need, and the most common such consumer is *another reduction*: matmul chains, attention
(`softmax(QK^T) V`), a LayerNorm feeding a linear layer.

> **Key idea:** fusion saves memory traffic, fission saves recomputation. A kernel boundary costs
> one write and one read of the intermediate; not having one costs (copies) x (work per element).
> A kernelizer is a policy for choosing between them.

### What fusion buys: counting bytes

The other side of the trade is memory traffic, and it is worth putting numbers on it. Take softmax
of a `512 x 512` float32 matrix: 1 MiB of input. The Tensor library builds it from seven primitive
operations: a max-reduce, a subtract (an add of a negated, broadcast max), a multiply by `log2(e)`,
an `EXP2`, a sum-reduce, a reciprocal, and a multiply. Three ways to run it:

| Strategy | Kernels | Main-memory traffic (approx.) |
|---|---|---|
| one kernel per primitive op (an eager, unfused library) | 7 | six full-size ops read 1 MiB each and the four full-size elementwise ones write 1 MiB each: ~10 MiB |
| cut at every `REDUCE` (the `uopc` default) | 3 | max reads 1 MiB, sum reads 1 MiB, output reads 1 MiB and writes 1 MiB: ~4 MiB |
| everything fused (`fuse_reduce=True`) | 1 | one 2 KiB row at a time stays in L1 across the three passes: ~1 MiB read, 1 MiB written |

The row-wise reductions write only 2 KiB each, which is why cutting at them is nearly free. A
memory-bound kernel's time is proportional to its traffic, so the difference between the first and
last row is roughly 5x in run time, before any other optimization. This is the arithmetic behind
every "fused softmax" and "fused LayerNorm" kernel in production libraries.

### A vocabulary of fusions

Compiler papers name fusions by the kinds of ops on either side of the edge. In `uopc` terms:

| Producer -> consumer | Example | What `uopc` does |
|---|---|---|
| elementwise -> elementwise | `relu(x * 2 + 1)` | always fused (loop fusion) |
| elementwise -> reduce | `sum(exp(x - m))` | fused into the reduction's inner loop ("input fusion") |
| reduce -> elementwise | `x / sum(x)` | cut; the consumer reads the reduction's temp ("epilogue" fusion only with `fuse_reduce`) |
| reduce -> reduce through `EXPAND` | `(A @ B) @ C`, `softmax(QK^T) V` | cut (fusing them is the recompute disaster above) |
| elementwise -> `EXPAND` -> reduce | LayerNorm feeding a matmul | fused unless the recompute is expensive (`stage_broadcast_compute`) |
| movement ops anywhere | `x.T`, `x[1:3]`, `pad` | always fused; they are index maps, never kernels |

The last row deserves emphasis. A movement op never becomes a kernel by itself unless it is a root
(the user asked for a transposed copy) or the week-6 layout pass decides a copy pays for itself
([Chapter 10](10-cpu-performance.md)). Views are free; that was the point of Chapter 6.

## 7.2 Materialization points

The kernelizer's policy is deliberately simple. A node is **materialized** (computed by its own
kernel and written to a buffer) if it is:

- a **root**: something the user asked for, unless it already *is* a contiguous buffer;
- a **`STAGE`**: an explicit request for a temporary buffer (from the user via
  `Tensor.contiguous()`, or inserted by a rule, see below);
- a **`REDUCE`**, unless `fuse_reduce` is on.

Everything between materialization points is fused into the kernel of the nearest materialized
consumer.

@include uopc/kernelize.py#_materialize_points

Cutting at *every* `REDUCE` is a blunt rule, and it is exactly the right default. A reduction's
output is smaller than its input by the size of the reduced axes, so writing it costs little, and
it guarantees that no reduction is ever nested inside another reduction's loop. Elementwise chains,
on the other hand, are always fused: `((a * 2 + 1).relu() * 3)` becomes one kernel with one load and
one store per element, instead of four kernels with four round trips through memory. With
`DEBUG=4`:

```c
void e_2_4(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu1 = ((ridx0*4)+ridx1);
      float val2 = data0[alu1];
      float alu3 = ((val2*2.0f)+1.0f);
      data1[alu1] = (((0.0f<alu3)?alu3:0.0f)*3.0f);
    }
  }
}
```

Softmax is the standard example of a cut graph. `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
has two reductions, so it becomes three kernels: the max, the sum, and the elementwise output.

@diagram 07-fusion-cut

Note what the cut does *not* prevent: `exp(x - max)` is computed twice, once inside the sum kernel
and once inside the output kernel. That is a recomputation too, but a constant factor of 2 on a
cheap elementwise op, which is far cheaper than writing and re-reading an `(N, N)` temporary.
Fusion is a policy, not a law.

## 7.3 make_plan: from a graph to kernels

Here is the kernelizer in full. It takes the roots, returns a `Plan` and the list of input
`BUFFER`s the plan refers to:

@include uopc/kernelize.py#make_plan

Walk through it with softmax on a `(4, 8)` input.

**1. Inputs.** Every `BUFFER` in the graph, in topological order, gets an input number:
`("in", 0)`, `("in", 1)`, ... The plan never refers to a specific buffer object, only to "input
number `i`". This is what lets one plan run on different data.

**2. Rewrites.** First `strip_detach` removes `STAGE(arg="detach")` markers, which
`Tensor.detach()` inserts only to stop gradients ([Chapter 8](08-autodiff.md)); they must not become
kernels. Then `stage_broadcast_compute` (below) may wrap some nodes in `STAGE` (unless
`fuse_reduce` is on), and the optional `opt.layout` from Chapter 10 may add more. Each returns new
roots.

@include uopc/kernelize.py#strip_detach

**3. Points.** `_materialize_points` lists the cut nodes in topological order, so producers come
before consumers. That order *is* the schedule.

**4. References.** Every point that is not a root gets a temporary, `("tmp", j)`, with its dtype
and element count. Every root gets an output, `("out", r)`, unless it is already a contiguous
buffer, in which case its "output" is an alias of the input (`("in", i)`).

**5. Placeholders.** Each point `u` gets a placeholder `BUFFER` with a *negative* slot (real buffers
have positive slots, so they cannot collide), reshaped to `u`'s shape. A consumer that used `u`
will now read the placeholder instead.

**6. Kernel bodies.** For each point, take its expression (the `STAGE`'s source, or the node itself)
and substitute every *other* point by its placeholder. The body now reads only buffers. It is a
complete, self-contained shape graph: exactly what `lower()` accepts.

**7. Local renumbering.** The body's buffers are renumbered `0, 1, ..., n-1` in topological order,
and the output gets slot `n`. `refs` remembers, for each local slot, what it really is: an input, a
temp, or an output.

**8. Lowering.** `lower()` (or `opt.lower()`) rangeifies the body into a `SINK`, `render_c` turns it
into C, and the kernel is recorded with its name, source, refs and a FLOP estimate. One last
detail: the refs are filtered to the buffer slots that still appear in the lowered program
(`alive`). Simplification can delete a read entirely, and the rendered C function then has fewer
parameters than the body had buffers. Pitfall 7 is the bug this line fixes.

For softmax, the plan is:

```text
r_4 [('in', 0), ('tmp', 0)] 32
r_4 [('in', 0), ('tmp', 0), ('tmp', 1)] 132
e_4_8 [('in', 0), ('tmp', 0), ('tmp', 1), ('out', 0)] 136
temps: [(dtypes.f32, 4), (dtypes.f32, 4)]
outs: [(dtypes.f32, 32)]
inputs: [UOp(Ops.BUFFER, arg=ParamArg(slot=1, dtype=dtypes.f32, shape=(32,), addrspace=None))]
```

Read the refs as function signatures. Kernel 0 reads input 0 and writes temp 0 (the row max).
Kernel 1 reads input 0 and temp 0 and writes temp 1 (the row sum of `exp(x - max)`). Kernel 2 reads
all three and writes output 0. The generated code:

```c
void r_4(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float acc0;
    acc0 = -INFINITY;
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      float val1 = acc0;
      float val2 = data0[((ridx0*8)+ridx1)];
      acc0 = ((val1>val2)?val1:val2);
    }
    float val3 = acc0;
    data1[ridx0] = val3;
  }
}
```

```c
void r_4(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    float val1 = data1[ridx0];
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      float val2 = acc0;
      float val3 = data0[((ridx0*8)+ridx1)];
      acc0 = (val2+exp2f(((val3+(val1*-1.0f))*1.4426950408889634f)));
    }
    float val4 = acc0;
    data2[ridx0] = val4;
  }
}
```

```c
void e_4_8(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data1[ridx0];
    float val2 = data2[ridx0];
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      int alu3 = ((ridx0*8)+ridx1);
      float val4 = data0[alu3];
      data3[alu3] = (exp2f(((val4+(val1*-1.0f))*1.4426950408889634f))*(1.0f/val2));
    }
  }
}
```

A few things to notice. `exp` is `exp2(x * log2(e))`, because the IR's only exponential is `EXP2`
([Chapter 9](09-tensor-mnist.md)). The placeholder reads (`data1[ridx0]` of the max, a `(4,)` buffer
reshaped to `(4, 1)` and expanded to `(4, 8)`) became loads hoisted out of the column loop: to
rangeify, a placeholder is just another buffer behind a chain of movement ops. And each kernel's
parameters are numbered from `data0`, whatever the global buffer numbers were.

### Kernel names and FLOPs

Kernels are named in tinygrad's style: `r_` if the body contains a reduction, `e_` if it is purely
elementwise, followed by the output shape. Names are for humans reading `DEBUG` output and
profiles; nothing depends on them.

@include uopc/kernelize.py#_kernel_name

@include uopc/kernelize.py#_flops

The FLOP count is a rough "one per ALU element plus one per reduced element", good enough for the
GFLOP/s column of `DEBUG=2` and for comparing kernels with each other.
[Chapter 10](10-cpu-performance.md) counts FLOPs more carefully.

## 7.4 Identical kernels, identical C

Step 7, the local renumbering, looks like bookkeeping. It is an optimization with a large payoff.
Two kernels that do the same thing to different buffers, say the row sums of two different `(16, 16)`
matrices, produce the same body up to buffer identity. After renumbering, the bodies are *the same
UOp* (hash consing again), lower to the same `SINK`, and render to byte-identical C:

```text
a: 1  b: 2
r_16 [('in', 0), ('out', 0)]
r_16 [('in', 1), ('out', 1)]
same C source: True
void r_16(float* restrict data0, float* restrict data1) {
```

The runtime of [Chapter 4](04-memory-and-c.md) caches compiled shared objects by a hash of the
source and flags, so the second kernel costs no compilation. In a transformer, every layer has the
same shapes: twelve layers produce twelve copies of each kernel and one compilation. The kernel
name intentionally carries no buffer numbers, for the same reason.

## 7.5 The Plan

The output of kernelization is a `Plan`:

@include uopc/kernelize.py#Kernel

@include uopc/kernelize.py#Plan

Two design decisions are visible here.

First, **a plan contains no UOps**: kernel sources are strings, buffers are referred to by
`("in" | "tmp" | "out", number)`. A plan can therefore be cached forever without keeping any tensor
alive, and it can run on any set of inputs with the right shapes.

Second, **temporaries belong to the cached plan**: `realize` allocates `temp_arrays` the first time
a plan runs and reuses them on every later run. Temps never escape (only outputs are handed back,
in fresh `new_buffer`s), so reuse is safe as long as runs of one plan do not overlap, which they
cannot in a single-threaded Python loop. An earlier version allocated them with `np.empty` on every
run; that is simpler, but every step then paid page faults on freshly mapped memory, which the
`DEBUG=3` output below makes visible.

@diagram 07-plan

## 7.6 Buffers belong to UOps

Where does the data for "input 0" live? In the `BUFFER` UOp itself:

@include uopc/kernelize.py#new_buffer

@include uopc/kernelize.py#buffer_data

@include uopc/kernelize.py#base_buffer

`new_buffer` hands out a fresh, positive slot number per buffer (so two buffers with the same shape
are different UOps, and hash consing never merges them) and stores the NumPy array in the UOp's
`__dict__`. Recall from Chapter 1 that `UOp._cache` is a `WeakValueDictionary`: it does not keep
nodes alive. When the last `Tensor` that refers (directly or through a graph) to a `BUFFER` UOp goes
away, Python frees the UOp, its `__dict__`, and the array. Memory management for the whole system
is the Python garbage collector. There is no `free` anywhere in `uopc`.

The consequence for a training loop is pleasant: after a step, `Tensor.uop` of each parameter is
rebound to its new output buffer, the old graph becomes unreachable, and every intermediate buffer
of the previous step is freed.

## 7.7 realize(): running a plan

`realize` is the function the Tensor library calls when you ask for data (`.numpy()`, `.item()`,
`realize_all`):

@include uopc/kernelize.py#realize

It deduplicates the roots, looks the graph's *structure* up in `_plans`, builds and compiles a plan
on a miss, allocates temporaries and outputs, and calls every kernel with the right arrays. The
`DEBUG` levels print progressively more:

| `DEBUG` | Prints |
|---|---|
| 1 | one line per `realize`: number of kernels, schedule time (Python side, including compiles), run time |
| 2 | plus one line per kernel: name, time, GFLOP/s |
| 3 | plus a second, warm run of each kernel (no first-touch page faults, warm caches) |
| 4 | plus the C source of each kernel *when it is compiled* (cache misses only) |

Softmax of a `512 x 512` matrix, computed twice, with `DEBUG=2`:

```text
  r_512                             0.680 ms     0.39 GFLOP/s
  r_512                             1.498 ms     0.70 GFLOP/s
  e_512_512                         2.116 ms     0.50 GFLOP/s
realize: 3 kernels, schedule 238.1 ms, run 4.4 ms
[1.0000002 1.        0.9999997]
  r_512                             0.634 ms     0.41 GFLOP/s
  r_512                             1.427 ms     0.74 GFLOP/s
  e_512_512                         1.511 ms     0.69 GFLOP/s
realize: 3 kernels, schedule 0.1 ms, run 3.6 ms
[1.0000002 1.        0.9999997]
```

The first call spends 238 ms building the plan (rangeify, simplify, render, three `gcc`
invocations; these runs used an empty `UOPC_CACHE` directory, so nothing was precompiled). The
second spends 0.1 ms: the plan was found in the cache, and its temporaries were already allocated.
The kernels themselves are slow (well under 1 GFLOP/s, against a machine peak in the hundreds; that is
[Chapter 10](10-cpu-performance.md)'s job).

And with `DEBUG=3`, on a graph with an explicit `STAGE`:

```text
  e_256_256                         0.523 ms     0.38 GFLOP/s  (again: 0.353 ms)
  r_256                             0.034 ms     1.93 GFLOP/s  (again: 0.023 ms)
realize: 2 kernels, schedule 153.1 ms, run 1.1 ms
```

The first run of `e_256_256` is about 50% slower than the second because it writes to a freshly
allocated temporary: the first touch of every 4 KB page is a page fault. Benchmark with warm runs.

## 7.8 The structural JIT cache

A training loop builds a new graph every step: new parameter buffers, new data, same model. Running
rangeify, simplify, and the renderer for every step would cost seconds per step for a GPT (the
first compile of the GPT graph takes ~9 s even after optimization; NOTES 22). The cache key must
therefore identify graphs that are *the same program on different data*:

@include uopc/kernelize.py#_structure_key

It is the graph in topological order, with each node replaced by `(op, arg, source positions)`, so
the key captures the exact DAG shape including sharing. The one exception is `BUFFER`: its arg is
replaced by `(dtype, size)`, dropping the slot number. Two graphs that differ only in *which*
buffers they read have the same key, and since the plan refers to inputs by topological position, the
cached plan is correct for both. That is the whole JIT: a dictionary.

Everything else in the arg stays in the key, including the value of every `CONST`. That is
correct (a different constant is a different program) and it is a trap.

### Python numbers are structure

AdamW's bias corrections are `1 / (1 - b1**t)`, a Python float that changes every step. Multiply
a tensor by it and it becomes a `CONST` in the graph, hence part of the structure key, hence a
cache miss every step (NOTES 17). Here is the effect, at `DEBUG=1`:

```python
for t in range(1, 4):
  corr = 1 / (1 - 0.9 ** t)
  (w * corr).realize()
  print(f"step {t}: corr={corr:.4f}  cached plans={len(_plans)}")
for t in range(1, 4):
  corr = Tensor(np.array([1 / (1 - 0.9 ** t)], np.float32))
  (w * corr).realize()
  print(f"step {t}: cached plans={len(_plans)}")
```

```text
-- bias correction as a Python float (baked into the graph as a CONST)
realize: 1 kernels, schedule 85.1 ms, run 0.1 ms
step 1: corr=10.0000  cached plans=1
realize: 1 kernels, schedule 78.1 ms, run 0.1 ms
step 2: corr=5.2632  cached plans=2
realize: 1 kernels, schedule 76.9 ms, run 0.1 ms
step 3: corr=3.6900  cached plans=3
-- bias correction as a 1-element tensor (a BUFFER: data, not structure)
realize: 1 kernels, schedule 74.2 ms, run 0.1 ms
step 1: cached plans=4
realize: 1 kernels, schedule 0.1 ms, run 0.1 ms
step 2: cached plans=4
realize: 1 kernels, schedule 0.1 ms, run 0.0 ms
step 3: cached plans=4
```

Each Python-float step pays ~80 ms of compilation (with an empty on-disk `.so` cache) for a 0.1 ms
kernel. In a real model with
hundreds of kernels that is the difference between a fast training loop and a compiler benchmark.
The fix in `uopc/nn.py` is to pass the changing scalars as one-element tensors:

@include uopc/nn.py#AdamW.step

The same reasoning explains why the GPT sampler pads its context to a fixed length (NOTES 18):

```python
ctx = ctx + [0] * (a.ctx - len(ctx))                           # fixed shape -> one compiled graph
```

A shape is structure too. Feeding contexts of length 1, 2, 3, ... would compile a new plan for every
generated token. Padding to the maximum length and reading out the logits at the right position keeps
one plan. [Chapter 15](15-beyond.md) shows the real fix: symbolic shapes.

> **Key idea:** the JIT cache splits every value into *structure* (ops, shapes, constants) and
> *data* (buffer contents). Anything that changes between steps must be data.

## 7.9 STAGE: asking for a buffer

`STAGE` is the IR's fission knob. `UOp(Ops.STAGE, (x,))` means "materialize `x` into a temporary
and let consumers index into it". It has no semantics beyond identity: rangeify without a
kernelizer simply computes through it (Chapter 6), and its gradient is the identity
([Chapter 8](08-autodiff.md)). The Tensor library exposes it as `contiguous()`:

@include uopc/tensor.py#Tensor.contiguous

`(a.exp() * 2).contiguous().sum(1)` is two kernels: an `e_256_256` that writes `exp(a) * 2` to a
temp, and an `r_256` that sums it (the `DEBUG=3` output above). Without `.contiguous()` it would be
one fused `r_256`. Which is faster depends on the consumer, and the next section is about the one
case where the compiler should decide on its own.

## 7.10 The recompute trap: stage_broadcast_compute

Cutting at every `REDUCE` prevents reductions from being recomputed. It does nothing about
*elementwise* chains that are broadcast into a reduction. The canonical case is a LayerNorm feeding
a linear layer:

```python
x = Tensor.randn(128, 256); W = Tensor.randn(256, 256)
m = x.mean(-1, keepdim=True)
y = ((x - m) * ((x - m).square().mean(-1, keepdim=True) + 1e-5).rsqrt()) @ W
```

The normalized activation `xn = (x - m) * rsqrt(var + eps)` has shape `(128, 256)`. The matmul
expands it to `(128, 256, 256)` (one copy per output column) and reduces. Since `xn` is elementwise,
it is fused into the matmul kernel and recomputed for every output column:

```c
void r_128_256(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3, float* restrict data4) {
  for (int ridx0 = 0; ridx0 < 128; ridx0++) {
    int alu1 = (ridx0*256);
    float val2 = data1[ridx0];
    float val3 = data2[ridx0];
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 256; ridx2++) {
        float val4 = acc0;
        float val5 = data0[(alu1+ridx2)];
        float val6 = data3[((ridx2*256)+ridx1)];
        acc0 = (val4+(((val5+((val2*0.00390625f)*-1.0f))*(1.0f/sqrtf(((val3*0.00390625f)+1e-05f))))*val6));
      }
      float val7 = acc0;
      data4[(alu1+ridx1)] = val7;
    }
  }
}
```

The inner loop does the LayerNorm arithmetic (and a `sqrtf` and a division, which `gcc` may or may
not hoist) for every multiply-add of the matmul. In a GPT forward/backward step, this pattern cost
**374 ms of kernel time; staging the broadcast computations brought it to 125 ms** (NOTES 15).

The fix is a rewrite that runs before cutting:

@include uopc/kernelize.py#stage_broadcast_compute

@include uopc/kernelize.py#_fused_work

The rule, in words: find every `EXPAND` that multiplies the element count by at least `factor`
(4). Walk down through movement ops to the computed value being broadcast. If it is a real
computation (an ALU op other than a bare `CAST`, not a scalar) and recomputing it is expensive
(`_fused_work`, the number of ALU ops in the fused expression times its element count, is at least
4096), wrap it in a `STAGE`. On the LayerNorm example:

```text
STAGEs inserted: 1  shape: (128, 256)  fused work: 262144
kernels with staging:    ['r_128', 'r_128', 'e_128_256', 'r_128_256']
kernels without staging: ['r_128', 'r_128', 'r_128_256']
```

With staging, the LayerNorm output is written once by its own `e_128_256` kernel and the matmul
kernel becomes a plain GEMM:

```c
void e_128_256(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  for (int ridx0 = 0; ridx0 < 128; ridx0++) {
    float val1 = data1[ridx0];
    float val2 = data2[ridx0];
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      int alu3 = ((ridx0*256)+ridx1);
      float val4 = data0[alu3];
      data3[alu3] = ((val4+((val1*0.00390625f)*-1.0f))*(1.0f/sqrtf(((val2*0.00390625f)+1e-05f))));
    }
  }
}
```

```c
void r_128_256(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 128; ridx0++) {
    int alu1 = (ridx0*256);
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 256; ridx2++) {
        float val2 = acc0;
        float val3 = data0[(alu1+ridx2)];
        float val4 = data1[((ridx2*256)+ridx1)];
        acc0 = (val2+(val3*val4));
      }
      float val5 = acc0;
      data2[(alu1+ridx1)] = val5;
    }
  }
}
```

How much does it buy? It depends on what else limits the kernel, which is a lesson about
cost models. I timed the whole plan (best of 5 runs) both ways:

```text
OPT=0:  with staging: 1.99 ms   without: 1.91 ms
OPT=1:  with staging: 0.58 ms   without: 1.35 ms
```

At `OPT=0` the naive GEMM is bound by its stride-256 walk down `W`'s columns: the CPU waits on memory
either way, the recompute hides in the shadow of the cache misses, and the extra kernel makes the
staged version slightly slower. With the week-6 optimizer (layout pass, register tiling, threads),
the GEMM becomes compute-bound and the recompute is pure overhead: staging is 2.3x faster. The
week-6 kernel list gains a transposing copy of `W` (`e_1_256_256`) from the layout pass, explained in
[Chapter 10](10-cpu-performance.md). Optimizations interact; measure the configuration you ship.

### Why the thresholds

Both thresholds exist because staging has a cost: an extra kernel launch, a write and a read of the
intermediate, and a temporary allocation.

- `factor = 4`: an `EXPAND` that merely doubles a tensor recomputes each element twice. Writing and
  re-reading it is about as expensive. Fourfold reuse or more makes staging a clear win.
- `_fused_work >= 4096`: softmax's `1 / sum` is a tiny `(rows, 1)` computation broadcast over the
  row. The first version of the rule had no work threshold, staged it, and added **two useless
  kernels** to every softmax (NOTES 15). Recomputing a reciprocal per element is cheaper than a
  kernel launch.
- `CAST` is excluded because a bare type conversion is as cheap as the load it wraps.

This is a cost model in miniature: an estimate of recompute work, compared to a fixed threshold. It
is crude and it is right often enough. Production compilers (XLA, Inductor, tinygrad) have more
elaborate versions of exactly this trade-off.

### Wrapping bottom-up

`stage_broadcast_compute` collects a set of nodes to wrap, then rebuilds the graph with
`UOp.wrap`, not with the more obvious `substitute`. The difference matters when one target contains
another (NOTES 16):

@include uopc/uop.py#UOp.wrap

`root.substitute({inner: STAGE(inner), outer: STAGE(outer)})` replaces `outer` by a `STAGE` of the
*original* `outer`, which still contains the original, unstaged `inner`; the staging of `inner` is
lost. `wrap` rebuilds bottom-up and wraps the *rebuilt* node, so nested targets compose:

```python
inner = (x * 2.0 + 1.0)            # wants staging
outer = UOp(Ops.EXP2, (inner,)) * inner       # also wants staging, and contains inner
root = outer.reduce(Ops.ADD, 1)
```

```text
substitute: STAGE nodes = 1
wrap:       STAGE nodes = 2
```

## 7.11 fuse_reduce: letting reductions nest

The default cuts at every `REDUCE`. With `fuse_reduce=True`, reductions are not materialization
points; they fuse into their consumers, and rangeify's accumulator placement decides the loop
structure. `fuse_reduce` also turns off `stage_broadcast_compute`: the switch means "never write
the big intermediate", and staging a broadcast computation is exactly such a write. For softmax,
fusing is a strict improvement: one kernel instead of three, no temporaries,
and the row's max and sum computed in sibling loops:

```c
void r_4_8(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*8);
    float acc0;
    acc0 = -INFINITY;
    float acc1;
    acc1 = 0.0f;
    for (int ridx2 = 0; ridx2 < 8; ridx2++) {
      float val2 = acc0;
      float val3 = data0[(alu1+ridx2)];
      acc0 = ((val2>val3)?val2:val3);
    }
    float val4 = acc0;
    float alu5 = (val4*-1.0f);
    for (int ridx3 = 0; ridx3 < 8; ridx3++) {
      float val6 = acc1;
      float val7 = data0[(alu1+ridx3)];
      acc1 = (val6+exp2f(((val7+alu5)*1.4426950408889634f)));
    }
    float val8 = acc1;
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      int alu9 = (alu1+ridx1);
      float val10 = data0[alu9];
      data1[alu9] = (exp2f(((val10+alu5)*1.4426950408889634f))*(1.0f/val8));
    }
  }
}
```

For `(A @ B) @ C` it is the 367 ms disaster from the start of the chapter. So `fuse_reduce` is a
global switch (`Tensor.fuse_reduce`), off by default, and it pays off for one important case:
attention, where fusing the softmax reductions into the attention kernel and keeping the score row
in registers is the first step toward flash attention ([Chapter 11](11-fusion-flash.md)). A smarter
kernelizer would decide per reduction; Exercise 8 asks you to write that rule.

## 7.12 The whole step, end to end

Putting Chapters 6, 7 and 8 together on a tiny model (a linear layer, ReLU, mean of squares) and
kernelizing the loss *and* the gradient of the weight in one plan:

```python
X, W = Tensor.randn(32, 64), Tensor.randn(64, 10, requires_grad=True)
loss = (X @ W).relu().square().mean()
loss.backward([W])
plan, _ = make_plan([loss.uop, W.grad.uop])
```

```text
['r_32_10', 'r_1', 'e_1', 'r_10_64', 'e_64_10']
```

Five kernels: the forward matmul `X @ W`; the sum of squares; the scale by `1/320` for the mean; the
gradient matmul `X^T @ dL/dh` (with the ReLU and square derivatives fused into its inner loop); and
a transposing copy of that result into `W`'s `(64, 10)` layout. The forward matmul is computed
**once** and read by both the loss and the gradient kernels. That happened for free: the gradient
graph refers to the very same `REDUCE` node as the forward graph (hash consing), so it is one
materialization point with one temp. [Chapter 8](08-autodiff.md) explains how that gradient graph is
built.

## 7.13 Building it yourself: an order of work

If you are implementing week 5 from this chapter, this order keeps every step testable:

1. **One kernel per root, no cuts.** `make_plan` with `_materialize_points` returning only the roots
   is just `lower()` per root plus bookkeeping. Test: the week-4 tests, through the new path.
2. **Cut at `REDUCE`.** Add placeholders and substitution. Test: softmax has 3 kernels
   (`test_kernelize_fuses_elementwise_and_splits_reductions`) and matches NumPy; `(A @ B) @ C` at
   N=128 runs in milliseconds, not hundreds of them.
3. **Local renumbering.** Test: two same-shaped reductions of different buffers render identical C.
4. **The Plan, without UOps.** Make `realize` run a plan against a list of input arrays. Test: build
   the plan once, run it on two different inputs, compare both against NumPy.
5. **The structural cache.** Test: the second `realize` of a same-shaped graph reports a
   sub-millisecond schedule time at `DEBUG=1`.
6. **`STAGE` and `contiguous()`**, then `stage_broadcast_compute`. Test: the LayerNorm-into-matmul
   graph gains one kernel, and the softmax graph does not.
7. **The training loop** of [Chapter 9](09-tensor-mnist.md): `test_training_reduces_loss`.

At each step, keep `DEBUG=2` output from a fixed small model in a text file and diff it as you go:
kernel counts and names are the cheapest regression test a kernelizer has.

## Pitfalls

**1. The fusion/recompute trap (NOTES 15).** Fusing everything between reductions sounds free, but
an elementwise result that is `EXPAND`ed into a reduce is recomputed for every output column. A GPT
step spent 374 ms in kernels; `stage_broadcast_compute` brought it to 125 ms. Staging is not free
either: the first version staged softmax's tiny `1/sum` and added two useless kernels. Only stage
when the recompute is expensive (`_fused_work >= 4096`).

**2. `substitute` vs. bottom-up wrap (NOTES 16).** `graph.substitute({x: STAGE(x)})` wraps the
*original* `x`, so anything inside `x` that should also be staged is not. When rewrite targets can
nest, rebuild bottom-up and transform the rebuilt node (`UOp.wrap`).

**3. JIT cache misses from Python floats (NOTES 17).** A changing Python scalar becomes a `CONST`, a
`CONST` is structure, and every step recompiles. Symptom: `DEBUG=1` shows `schedule` times of
hundreds of milliseconds on every step instead of fractions of a millisecond. Fix: pass changing
scalars as one-element tensors.

**4. Changing shapes recompile too (NOTES 18).** Sampling from a GPT with a growing context compiles
a new plan per token. Pad to a fixed shape (or implement symbolic shapes, Chapter 15).

**5. `DEBUG=4` prints nothing on a cache hit.** Sources are printed when a plan is compiled. If you
run the same graph twice to "see the C", the second run is silent. This is a feature, but it has
confused everyone at least once. Clear `kernelize._plans` or change a shape.

**6. Measuring a fusion decision in the wrong configuration.** The LayerNorm staging looked like
a slight *loss* at `OPT=0` (1.99 ms vs 1.91 ms) and is a 2.3x win at `OPT=1`. A memory-bound kernel
hides recomputation behind its cache misses; once the optimizer makes it compute-bound, the
recompute is the bottleneck. Tune fusion heuristics with the optimizer on, on the shapes you care
about.

**7. A deleted read shifted every kernel argument (found and fixed).** `(t * 0)` for an integer
tensor `t` simplifies to the constant 0, so the lowered kernel no longer reads `t`, and `render_c`
emits a function with *one* parameter, the output. `refs`, however, was still computed from the
body's buffers before lowering: `[("in", 0), ("out", 0)]`. The runtime passed `t`'s array as the
first argument, which the kernel took to be its output, and `(Tensor([1, 2, 3]) * 0)` wrote its
zeros *into the input*. The fix binds exactly the slots that survive lowering. Now:

```text
e_3 [('out', 0)]
void e_3(int* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    data1[ridx0] = 0;
  }
}

[0, 0, 0] [1, 2, 3]
```

Note the parameter is still called `data1`: the renderer names parameters by slot, so the gap in the
numbering is the visible trace of the deleted read. The regression test is
`test_deleted_reads_do_not_shift_kernel_arguments`:

@include tests/test_w05_tensor.py#test_deleted_reads_do_not_shift_kernel_arguments

The general lesson: any two components that must agree on a calling convention (here, the list of
buffers) should derive it from the same object, the final program, not from an earlier stage that
a later optimization can invalidate.

## Exercises

1. ★ **Count kernels.** Without running anything, predict the number of kernels (default settings)
   for: `a + b`, `(a + b).sum()`, `a.softmax(-1)`, `a.var(-1)`, `(a @ b).relu() @ c`. Then check
   with `make_plan` and `len(plan.kernels)`.

2. ★ **Read a plan.** For `x.var(-1)` on a `(4, 8)` input, print the plan's refs and temps and
   describe in one sentence what each kernel computes.

3. ★★ **Measure the recompute.** Build `(A @ B) @ C` with `fuse_reduce=True` via
   `make_plan(..., fuse_reduce=True)` and time it against the default at N = 32, 64, 128. Fit the
   ratio to a power of N.

4. ★★ **A cache-miss detector.** Add a `DEBUG` mode (in your own copy) that, on a cache miss, finds
   the most similar cached key and prints the first position where they differ. Use it to find the
   `CONST` that changes in a training step that keeps recompiling.

5. ★★ **Find the bug.** A student renumbers kernel buffers by *slot* order instead of topological
   order: `bufs = sorted({...}, key=lambda b: b.arg.slot)`. Do identical kernels still render
   identical C? Construct a counterexample. *Hint:* `new_buffer` hands out slots in creation order.

6. ★★ **Placeholder slots.** Why are placeholder slots negative? What would break if they reused
   `0, 1, 2, ...`?

7. ★★ **Buffer reuse.** Each temporary of a plan gets its own array. Write a pass over the plan that
   computes each temp's live range (first and last kernel that touches it) and assigns temps with
   disjoint live ranges to the same storage. How much memory does it save on the softmax plan? On
   the five-kernel training step above?

8. ★★★ **Per-reduction fusion.** Replace the global `fuse_reduce` switch with a rule: fuse a `REDUCE`
   into its consumer unless the consumer would evaluate it inside a reduce loop it does not depend
   on. *Hint:* you can detect that from shapes alone: is there an `EXPAND` between the `REDUCE` and
   another `REDUCE` that consumes it? Check that softmax fuses to one kernel and `(A @ B) @ C` stays
   at two.

9. ★★★ **A better cost model.** `stage_broadcast_compute` ignores the cost of the extra write and
   read. Model each choice in bytes moved plus ALU ops, with a machine balance parameter (FLOPs per
   byte, from [Chapter 10](10-cpu-performance.md)'s roofline), and stage only when the model says
   so. Compare with the fixed thresholds on the LayerNorm example at `OPT=0` and `OPT=1`.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** `a + b`: 1 (elementwise, fused). `(a + b).sum()`: 1 (the add fuses into the reduction, and the
reduction is the root). `a.softmax(-1)`: 3 (max, sum, output). `a.var(-1)`: the mean is a `REDUCE`,
the squared deviations are summed by a second `REDUCE`, and the root multiplies that sum by `1/n`, so
3 (`r`, `r`, `e`). `(a @ b).relu() @ c`: 3, which surprises most people. The first matmul is a
`REDUCE` (one kernel) and the second is the root (one kernel), but the ReLU in between is an
elementwise computation broadcast 64-fold into the second matmul, and its fused work (2 ALU ops x
4096 elements = 8192) passes the 4096 threshold, so `stage_broadcast_compute` gives it its own
`e_64_64` kernel. The real output for 64x64 inputs at `OPT=0`:
`['r_64_64', 'e_64_64', 'r_64_64']`.
Check each with `len(make_plan([t.uop])[0].kernels)`; at `OPT=1`, the layout pass may add transposing
copies (Chapter 10).

**2.** Kernel 0 (`r_4`) reads the input, writes the row sums (temp 0). Kernel 1 (`r_4`) reads the
input and temp 0, writes the row sums of squared deviations from the mean (temp 1). Kernel 2 (`e_4`)
reads temp 1 and scales it by `1/8` into the output.

</details>

## Quiz

1. Name the three kinds of materialization points.
2. Why is a reduction's output cheap to materialize, compared to its input?
3. What does a kernel's `refs` list contain, and why does the `Plan` contain no UOps?
4. Two kernels compute row sums of two different matrices of the same shape. Why do they produce
   byte-identical C, and why does that matter?
5. What is in `_structure_key` for a `BUFFER`? For a `CONST`?
6. Why did AdamW recompile every step, and what was the fix?
7. When does `stage_broadcast_compute` insert a `STAGE`?
8. Who frees the memory of an intermediate buffer, and when?

<details>
<summary>Answers</summary>

1. Roots (unless already contiguous buffers), `STAGE` nodes, and `REDUCE` nodes (unless
   `fuse_reduce`).
2. It is smaller by the product of the reduced axes: a row sum of an `(N, N)` matrix writes `N`
   values after reading `N^2`.
3. One entry per local buffer slot: `("in", i)`, `("tmp", j)`, or `("out", r)`. Without UOps, a
   cached plan keeps no tensors alive and applies to any inputs with the same structure.
4. Local renumbering in topological order makes their bodies the same hash-consed UOp, so they lower
   and render identically; the runtime's content-addressed cache then compiles the source once.
5. `("buf", dtype name, shape)`: the slot (identity) is dropped. A `CONST` keeps its value (via
   `_argkey`), so a different constant is a different key.
6. Its bias corrections were Python floats, so they became `CONST`s that changed every step, changing
   the structure key. Passing them as one-element tensors makes them data.
7. When an `EXPAND` grows a tensor by at least 4x and the computed (non-`CAST`, non-scalar ALU)
   value under it has fused work (ALU ops x elements) of at least 4096.
8. The Python garbage collector, when the last reference to the `BUFFER` UOp disappears: the array
   lives in the UOp's `__dict__`, and `UOp._cache` holds only weak references.

</details>

## Further reading

- T. Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018,
  section 3 on operator fusion (injective, reduction, and "complex-out-fusable" operators).
- The XLA documentation's description of operator fusion (`openxla.org/xla`), and the "Operation
  Semantics" page for the HLO ops it fuses.
- J. Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode
  Transformation and Graph Compilation", ASPLOS 2024. TorchInductor's scheduler makes the same
  fuse-or-materialize decisions, with a much larger cost model.
- J. Ragan-Kelley et al., "Halide", PLDI 2013, on the three-way trade-off between parallelism,
  locality, and redundant recomputation. `compute_root` vs. `compute_inline` is exactly
  `STAGE` vs. fusion.
- tinygrad's scheduler (`tinygrad/engine/schedule.py` in older releases, the rangeify-based
  scheduler in newer ones): kernel naming, `DEBUG` levels, and the structural cache in this chapter
  all follow tinygrad's conventions.
