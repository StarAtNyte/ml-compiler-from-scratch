# 15. Symbolic Shapes, Inference, and Where to Go Next

You have a compiler that trains a GPT. Now ask it to *talk*. Generating text one token at a time means
running the model on a prompt of length 5, then 6, then 7, and so on, and `uopc` treats every one of
those lengths as a brand-new program: a new graph, a new plan, a new trip through gcc. A production
system would also keep the keys and values of past tokens in a cache instead of recomputing them, run
the matrix multiplies in half precision, and spread the work across several devices. None of that is in
the reference compiler. This chapter is about the distance between what you built and what real systems
do, measured honestly, with each missing piece turned into a concrete design you can implement as a final
project. It ends with a map of the real compilers you are now equipped to read, and with more than twenty
project ideas.

**What you'll build** (all of it as design exercises; nothing in this chapter changes `uopc/`):

- A loop with a run-time trip count, in the week-3 loop IR, and a measurement of what it costs the
  symbolic engine.
- A bounded `Variable` prototype (a few lines of monkeypatching) that shows exactly which simplifications
  symbolic shapes need.
- A fixed-capacity KV cache built from today's Tensor API, verified against full causal attention, that
  compiles once and decodes every later token from cache.
- A mixed-precision GEMM (f16 storage, f32 accumulation) and a simulated data-parallel all-reduce, both
  checked against numpy.
- A comparison of tinygrad, TVM, Triton, XLA, MLIR/IREE, torch.compile and Halide, and a reading guide to
  tinygrad's source.

**Where this fits:** [Chapter 14](14-gpt.md) trained a GPT and padded its sampling loop to a fixed length
to dodge recompilation (NOTES 18). This chapter explains why that trick was necessary and what replaces it.
After this come the appendices: the op reference ([Appendix A](16-appendix-reference.md)), testing and
debugging ([Appendix B](17-appendix-testing.md)), and the glossary and quiz bank
([Appendix C](18-appendix-glossary.md)).

> **Labeling convention for this chapter.** Code shown with an `@include` label is in the repository and
> tested. Code in a plain fenced block marked *scratch* is a small script written for this chapter: it
> imports `uopc` but is not part of it. Every output shown was produced by running that code on the
> reference laptop (i7-13620H, gcc 15, Python 3.12). Code marked *sketch* is a design proposal and has not
> been run.

## 15.1 An honest inventory

Before designing anything, list what is missing and where in the code the gap shows. Each row is a
section of this chapter or a project at the end.

| Missing feature | Where you see the gap in `uopc` | Section |
|---|---|---|
| Shapes known only at run time | `UOp.shape` is `tuple[int, ...]`; `_structure_key` hashes shapes | 15.2 |
| A KV cache for decoding | `examples/gpt.py` re-runs the whole context for every sampled token | 15.3 |
| In-place updates | `Tensor.assign` rebinds to a new buffer; there is no "store into an existing buffer" op at the Tensor level | 15.3 |
| bf16, loss scaling | `dtypes` has `f16` but no `bf16`; optimizers are f32 only | 15.4 |
| More than one device | `BUFFER` has no device; `Program` is one process, one address space | 15.5 |
| Real GPU execution | `cuda.py` is untested on the reference machine (NOTES 25) | Projects |
| Tensor cores from the graph | `Ops.WMMA` is reserved; `wmma_gemm` is a string template | Projects |

None of these are embarrassing. They are where the course stops and the field begins.

## 15.2 Symbolic shapes

### Why static shapes cost compiles

The JIT cache in `kernelize.py` keys every plan on the *structure* of the graph. Buffers are anonymized,
but their sizes are not, and neither are the shape arguments of movement ops:

@include uopc/kernelize.py#_structure_key

That is the right key for training, where every step has the same shapes. It is the wrong key for
generation. Here is a one-layer GPT run on three sequence lengths and then the first length again
(scratch script, cold kernel cache):

```python
# scratch: how many plans does a forward pass compile when the sequence length varies?
import sys, time
sys.path.insert(0, "examples")
import numpy as np
from gpt import GPT
from uopc.tensor import Tensor
from uopc import kernelize

Tensor.manual_seed(0)
model = GPT(vocab=65, ctx=16, dim=32, heads=2, layers=1)
for T in (5, 6, 7, 5):
  t0 = time.perf_counter()
  model(Tensor(np.zeros((1, T), np.int32))).numpy()
  print(f"T={T:2d}: {(time.perf_counter() - t0) * 1e3:6.0f} ms, plans so far: {len(kernelize._plans)}")
```

```text
T= 5:   2787 ms, plans so far: 8
T= 6:   2432 ms, plans so far: 9
T= 7:   2289 ms, plans so far: 10
T= 5:      7 ms, plans so far: 10
```

(The first eight plans are the model's parameter initializers. Timings in this chapter are single runs on
the reference laptop and vary by tens of percent between runs; plan counts and errors do not.) Each new length costs about two and a half
seconds of rangeify, rendering and gcc; a repeated length costs 7 ms. A chatbot that answers with 300 tokens
would compile 300 times. `examples/gpt.py` avoids this by padding:

@include examples/gpt.py:95-102

Padding works, but it has a price: every token pays for the full context length, even the first one. Fixed
shapes are the right tool when lengths cluster (pad to a few buckets: 64, 128, 256); they are the wrong tool
when lengths vary one token at a time.

@diagram 15-symbolic-shapes

### The loop IR is already symbolic

Look at what `RANGE` takes: its end is `src[0]`, an arbitrary `i32` UOp, not a Python int. So the week-3
loop language can already express loops whose trip count is only known at run time. This scratch program
reads `n` from a one-element buffer and sums the first `n` columns of each row:

```python
# scratch: a loop whose trip count is read from memory at run time: out[i] = sum(x[i, :n])
import numpy as np
from uopc.uop import UOp, Ops, ParamArg, sink, range_
from uopc.dtype import dtypes
from uopc.render_c import render_c
from uopc.runtime import run_c
from uopc.symbolic import vmin, vmax

ROWS, MAXN = 4, 8
X   = UOp(Ops.BUFFER, (), ParamArg(0, dtypes.f32, (ROWS * MAXN,)))
LEN = UOp(Ops.BUFFER, (), ParamArg(1, dtypes.i32, (1,)))
OUT = UOp(Ops.BUFFER, (), ParamArg(2, dtypes.f32, (ROWS,)))
n = LEN.index(0).load()                       # the symbolic size: a value, not a python int
i, j = range_(ROWS, 0), range_(n, 1)          # RANGE's end is any i32 UOp
acc = UOp(Ops.ALLOC, (), ParamArg(0, dtypes.f32, (1,))).after(i)
init = acc.index(0).store(UOp.const(0.0))
accj = acc.after(init, j)
upd = accj.index(0).store(accj.index(0).load() + X.index(i * MAXN + j).load())
prog = sink(OUT.index(i).store(acc.after(upd.end(j)).index(0).load()).end(i))

src = render_c(prog)
print(src[src.index("void"):])
x = np.arange(ROWS * MAXN, dtype=np.float32)
for length in (3, 8):
  out = np.zeros(ROWS, np.float32)
  run_c(prog, {0: x, 1: np.array([length], np.int32), 2: out})
  print(f"n={length}: {out}  numpy: {x.reshape(ROWS, MAXN)[:, :length].sum(1)}")
print("bounds of j:", (vmin(j), vmax(j)), " bounds of i*8+j:", (vmin(i * MAXN + j), vmax(i * MAXN + j)))
```

```text
void kernel(float* restrict data0, int* restrict data1, float* restrict data2) {
  int val1 = data1[0];
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < val1; ridx1++) {
      float val2 = acc0;
      float val3 = data0[((ridx0*8)+ridx1)];
      acc0 = (val2+val3);
    }
    float val4 = acc0;
    data2[ridx0] = val4;
  }
}

n=3: [ 3. 27. 51. 75.]  numpy: [ 3. 27. 51. 75.]
n=8: [ 28.  92. 156. 220.]  numpy: [ 28.  92. 156. 220.]
```

One compiled function, two lengths, correct answers. The linearizer even hoisted the load of `n` above
both loops, because the scope of `RANGE` is the scope of its end plus itself (Chapter 5). So the back end
needs nothing new. The trouble is upstream, in the algebra:

```text
bounds of j: (0, 2147483646)  bounds of i*8+j: (-2147483648, 2147483647)
```

`vmax(j)` is `vmax(n) - 1`, and a loaded `i32` can be anything, so `j` might be two billion. Then
`i*8 + j` might overflow `i32`, and `bounds()` does the only sound thing: it gives up and returns the full
range. That cascades. Here is the index `(i*8 + j) // 8` simplified with a static `j` (end 8) and with the
dynamic one (the same scratch script, continued):

```python
from uopc.symbolic import simplify
from uopc.uir import to_uir
j8 = range_(MAXN, 1)
print("static  j:", to_uir(simplify((i * MAXN + j8) // MAXN)).strip().splitlines()[-1])
print("dynamic j:"); print(to_uir(simplify((i * MAXN + j) // MAXN)), end="")
```

```text
static  j: %0 = range 4 : idx=0 kind=loop
dynamic j:
%0 = range 4 : idx=0 kind=loop
%1 = buffer 1 : dtype=i32 slot=1
%2 = index %1, 0
%3 = load %2
%4 = range %3 : idx=1 kind=loop
%5 = floordiv %4, 8
%6 = add %0, %5
```

With `j < 8` known, `j // 8` folds to 0 and the expression is just `i`. With `j` unbounded, `_div_sum`
still splits off `i` but `j // 8` survives. This is NOTES 12 again: an unsimplified index mentions a loop
variable it does not really depend on, and the reduce lowering puts accumulators inside loops they do not
need. Every `RESHAPE` of a symbolic axis produces exactly these `//` and `%` chains.

> **Key idea:** symbolic shapes are not a back-end problem. The loop IR and the C renderer already handle a
> run-time trip count. What breaks is *bounds analysis*, and everything that depends on it: div/mod
> simplification, the `/` versus `floordiv_i32` choice in the renderer, and dependence analysis in rangeify.

### A bounded Variable

The fix every production system converges on is to give a symbolic size *declared bounds*: "the sequence
length is between 1 and 2048". tinygrad calls this a `Variable` with a min and a max; torch.compile attaches
value ranges to its symbolic integers. In `uopc` the smallest version is a scalar `i32` leaf whose bounds
come from a declaration instead of from `_bounds`. Here is that idea as a short monkeypatch, using a
`PARAM` as the leaf (scratch):

```python
# scratch: a symbolic-shape variable is a PARAM with declared bounds. Not in uopc --
# we monkeypatch bounds analysis to show what the extra knowledge buys.
import uopc.symbolic as S
from uopc.uop import UOp, Ops, ParamArg, range_
from uopc.dtype import dtypes
from uopc.uir import to_uir

DECLARED: dict[UOp, tuple[int, int]] = {}
def Variable(name_slot: int, lo: int, hi: int) -> UOp:
  v = UOp(Ops.PARAM, (), ParamArg(name_slot, dtypes.i32))
  DECLARED[v] = (lo, hi)
  return v

_old = S._bounds
S._bounds = lambda u: DECLARED.get(u) or _old(u)

n = Variable(0, 1, 8)                    # sequence length: 1 <= n <= 8
i, j = range_(4, 0), range_(n, 1)
print("bounds of j:", (S.vmin(j), S.vmax(j)))
print(to_uir(S.simplify((i * 8 + j) // 8)), end="")
print(to_uir(S.simplify((i * 8 + j) % 8)), end="")
print(to_uir(S.simplify(j < n)), end="")
```

```text
bounds of j: (0, 7)
%0 = range 4 : idx=0 kind=loop
%0 = param : slot=0 dtype=i32
%1 = range %0 : idx=1 kind=loop
%0 = param : slot=0 dtype=i32
%1 = range %0 : idx=1 kind=loop
%2 = cmplt %1, %0
```

The existing rule `RANGE: (0, vmax(end) - 1)` now yields `j ∈ [0, 7]`, and both div/mod identities fire:
`(i*8 + j) // 8` is `i` and `(i*8 + j) % 8` is `j`. Nothing else in `symbolic.py` changed.

The last line is instructive. `j < n` is always true inside the loop, but interval arithmetic cannot see it:
`j ∈ [0, 7]` and `n ∈ [1, 8]` overlap. That fact is *relational* (it relates two variables), and intervals
only know about one variable at a time. You need a rule that knows what a `RANGE` means:

```python
# sketch: a relational rule for symbolic.py -- not in uopc, not run
(UPat(Ops.CMPLT, src=(UPat(Ops.RANGE, name="r"), UPat.var("e"))),
 lambda r, e: UOp.const(True) if r.src[0] is e else None),
```

Masks of the form `j < n` are everywhere once shapes are symbolic (causal attention, padding), so this
rule matters.

### The full design

A real implementation touches most modules. Here is the work list, roughly in the order you would do it.

1. **`uop.py`: an op and a shape type.** Add `Ops.VAR` with `arg = VarArg(name: str, lo: int, hi: int)`,
   dtype `i32`, shape `()`. Let shapes contain UOps: `tuple[int | UOp, ...]`. `_infer` then needs
   symbolic equality for its checks: `RESHAPE` must verify `prod(arg) == prod(src.shape)` where both sides
   may be expressions. Use `simplify(a - b)` and accept only if it folds to `CONST 0`; reject otherwise
   (a sound but incomplete check, like every real system's).
2. **`symbolic.py`: bounds and rules.** `_bounds(VAR) = (lo, hi)`; the relational `RANGE < end` rule above;
   and a rule that `x // n` with `0 <= x < n` is 0 when `n` is symbolic (the existing rule needs a `CONST`
   divisor).
3. **`rangeify.py`: almost free.** `ravel` and `unravel` (listed below) multiply and divide by shape
   entries using `UOp` arithmetic, so they already accept UOp sizes. `Ctx.rng(size)` must accept a UOp
   end, and `_axis_parts` must refuse to upcast a symbolic axis (an unroll factor must divide the size,
   and you cannot prove that for `n`). Reductions over a symbolic axis are just loops with a symbolic end.
4. **`kernelize.py`: plans with holes.** `_structure_key` must key a `VAR` by `(name, lo, hi)`, never by its
   current value; temp buffers are allocated at the upper bound `hi`. `realize` takes the bound values,
   checks `lo <= v <= hi` (the *guard*), and passes them to every kernel.
5. **`render_c.py` and `runtime.py`: one more parameter.** A `VAR` renders as an `int` parameter of the
   kernel (`void k(float* restrict data0, ..., int n)`), and `Program.__call__` passes it with
   `ctypes.c_int`. The CUDA renderer does the same; launch dimensions become expressions evaluated at launch.
6. **`tensor.py`: binding.** A user-facing `Tensor` axis can be a `VAR`. tinygrad's API binds a value at the
   call site (`Variable("start_pos", 1, MAX).bind(t)`); you can copy that shape of API.

Item 3 is the pleasant surprise. Here is `ravel`; nothing in it requires `s` to be a Python int, because
`flat * s` is `UOp.__mul__`, which accepts a UOp operand:

@include uopc/rangeify.py#ravel

What does this buy for GPT? One compiled plan for every prompt length up to the context size, and, with a
KV cache, decoding work proportional to the current position rather than to the maximum. Which brings us to
inference.

## 15.3 Inference and the KV cache

### What decoding recomputes

Causal attention has a useful property: the output at position `t` depends only on tokens `0..t`. So when
you append token `t+1`, nothing about the keys and values of tokens `0..t` changes. A sampler that re-runs
the model on the whole context recomputes all of them, at every layer, for every new token. For a context of
length `T` that is `O(T)` redundant work per token and `O(T²)` per generated sequence, before even counting
attention itself.

A **KV cache** stores, for each layer and head, the key and value vectors of every past token. Each decode
step then computes `q, k, v` for one new token, appends `k, v` to the cache, and attends with one query
against the cached keys. The memory cost is easy to count: two tensors (K and V) per layer, each
`heads × T × head_dim` elements:

```text
bytes = 2 · layers · T · dim · bytes_per_element
```

For the default model in `examples/gpt.py` (2 layers, `dim` 64, context 64, f32) that is 2·2·64·64·4 =
64 KiB. For a model with 32 layers, `dim` 4096 and a 4096-token context in f16 it is 2·32·4096·4096·2 bytes =
2 GiB *per sequence*, which is why serving systems treat cache memory as their scarcest resource (see
PagedAttention in Further reading).

### A fixed-capacity cache with today's API

You can build a correct KV cache right now, without symbolic shapes, by making the *position* data instead
of shape: a one-element `i32` tensor. Every decode step then has the same graph, so it reuses one plan.
AdamW uses the same trick for its bias corrections (NOTES 17). The cache has a fixed capacity `TMAX`; the
write is a functional scatter through a one-hot row; unused slots are masked to `-1e9` before the softmax.
One attention head, scratch:

```python
# scratch: a fixed-capacity KV cache built from today's Tensor API: no symbolic shapes needed,
# the position is *data* (a 1-element i32 tensor), so every decode step has the same graph.
import math, time
import numpy as np
from uopc.tensor import Tensor
from uopc import kernelize

TMAX, C, D = 64, 32, 16
rng = np.random.default_rng(0)
Wq, Wk, Wv = (Tensor(rng.standard_normal((C, D)).astype(np.float32) / math.sqrt(C)) for _ in range(3))
xs = rng.standard_normal((TMAX, C)).astype(np.float32)

def reference(T):                                   # causal attention over the first T tokens, in numpy
  q, k, v = (xs[:T] @ W.numpy() for W in (Wq, Wk, Wv))
  s = q @ k.T / math.sqrt(D) + np.triu(np.full((T, T), -np.inf), 1)
  p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
  return p @ v

K, V = Tensor.zeros(TMAX, D).realize(), Tensor.zeros(TMAX, D).realize()
slots = Tensor.arange(TMAX)
def decode_step(x: Tensor, pos: Tensor) -> Tensor:
  global K, V
  q, k, v = x @ Wq, x @ Wk, x @ Wv                  # (1, D) each
  here = slots.eq(pos).float().reshape(TMAX, 1)     # one-hot row `pos`: a functional scatter
  K = (K * (1 - here) + here * k).realize()
  V = (V * (1 - here) + here * v).realize()
  seen = slots.reshape(1, TMAX) < pos.reshape(1, 1) + 1   # slots 0..pos are valid
  s = (q @ K.T) * (1 / math.sqrt(D))
  p = seen.where(s, -1e9).softmax(-1)
  return p @ V                                      # (1, D)

outs, plans, times = [], [], []
for t in range(TMAX):
  t0 = time.perf_counter()
  outs.append(decode_step(Tensor(xs[t:t + 1]), Tensor(np.array([t], np.int32))).numpy())
  times.append(time.perf_counter() - t0)
  plans.append(len(kernelize._plans))
print("max |err| vs full causal attention:", float(np.abs(np.concatenate(outs) - reference(TMAX)).max()))
print("compiled plans after step 1, 2, 64:", plans[0], plans[1], plans[-1])
print(f"step 1: {times[0]*1e3:.0f} ms (compiles)   steps 2-{TMAX}: {np.mean(times[1:])*1e3:.2f} ms each")
```

```text
max |err| vs full causal attention: 8.156349239896166e-07
compiled plans after step 1, 2, 64: 3 3 3
step 1: 114 ms (compiles)   steps 2-64: 4.00 ms each
```

Three plans (the K update, the V update, and the attention output) are built on the first step and reused
for all 63 others. The 64 incremental outputs match full causal attention computed by numpy to within
`1e-6`. (Step 1 was cheap here because the kernels were already in the on-disk `.so` cache from an earlier
run; the plan still had to be built in Python.)

@diagram 15-kv-cache

The diagram shows what this version still wastes. Every step rewrites the whole `TMAX × D` cache (the
one-hot scatter is a full elementwise pass), and attention scores all `TMAX` slots, then throws away the ones
past `pos`. Both costs are `Θ(TMAX · D)` per token where `Θ(pos · D)` would do. Two features remove them:

- **An in-place store.** Writing one row should be one `STORE` at index `pos * D + d`, into the *existing*
  buffer. `uopc` can already express that in the loop IR (it is just `BUFFER.index(...).store(...)`); what is
  missing is a Tensor-level op that means "mutate this buffer", and a kernelizer that respects it: every
  kernel that reads the cache must be ordered after the write (an `AFTER` edge between kernels), and the
  structural cache must treat the cache buffer as an input *and* an output of the same plan.
- **A symbolic loop end.** With `pos` as a bounded `VAR`, the attention reduce runs `RANGE(end = pos + 1)`
  and the mask disappears.

> **Key idea:** the fixed-capacity cache is the "pad to a bucket" strategy applied to the time axis. It is
> correct and compiles once, but its cost is set by the capacity, not by the work. Symbolic shapes and
> in-place updates are what make the cost follow the work.

### Batching and the rest of an inference stack

Real serving adds two more ideas worth knowing by name. **Continuous batching** groups requests of different
lengths into one batch and lets them join and leave between decode steps; it needs either symbolic batch
sizes or padding to buckets. **Paged KV caches** (vLLM's PagedAttention) allocate the cache in fixed-size
blocks, like virtual memory pages, and keep a per-sequence block table; attention then gathers keys through
an indirection. In `uopc` terms, that indirection is an `INDEX` whose index is itself a `LOAD`: a
data-dependent address, which the loop IR already supports and the Tensor API does not (there is no gather;
`autodiff` refuses a data-dependent `INDEX`).

## 15.4 Mixed precision

Half-precision floats (f16) halve memory traffic and, on GPUs with tensor cores, multiply peak FLOP/s
several times over. The catch is range and precision: f16 has a 10-bit mantissa and a maximum of 65504. The
standard recipe is *store in low precision, accumulate in high precision*. `uopc` has an `f16` dtype
(rendered as `_Float16`, which gcc supports on x86-64, and as `half` by the CUDA renderer), so you can measure the recipe directly. Scratch:

```python
# scratch: mixed precision on a CPU: f16 storage, f32 arithmetic.
import numpy as np
from uopc.tensor import Tensor
from uopc.dtype import dtypes
from uopc.kernelize import make_plan
rng = np.random.default_rng(0)
a, b = rng.standard_normal((64, 64)).astype(np.float32), rng.standard_normal((64, 64)).astype(np.float32)
A16, B16 = Tensor(a, dtype=dtypes.f16), Tensor(b, dtype=dtypes.f16)
c = A16.float() @ B16.float()                     # load halves, multiply-accumulate in f32
ref = a.astype(np.float16).astype(np.float64) @ b.astype(np.float16).astype(np.float64)
print("max |err| vs f64 on the same f16 inputs:", float(np.abs(c.numpy() - ref).max()))
print("max |err| of f16 rounding of the inputs:", float(np.abs(a @ b - ref).max()))
cc = (A16 @ B16)                                  # everything in f16, including the accumulator
print("all-f16 GEMM max |err|:", float(np.abs(cc.numpy().astype(np.float64) - ref).max()))
plan, _ = make_plan([(A16.float() @ B16.float()).uop])   # unoptimized, to read it
src = plan.kernels[-1].src
print(src[src.index("void"):])
```

```text
max |err| vs f64 on the same f16 inputs: 8.231960237026215e-06
max |err| of f16 rounding of the inputs: 0.009531687945127487
all-f16 GEMM max |err|: 0.06902523338794708
void r_64_64(_Float16* restrict data0, _Float16* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    int alu1 = (ridx0*64);
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 64; ridx2++) {
        float val2 = acc0;
        _Float16 val3 = data0[(alu1+ridx2)];
        _Float16 val4 = data1[((ridx2*64)+ridx1)];
        acc0 = (val2+(((float)(val3))*((float)(val4))));
      }
      float val5 = acc0;
      data2[(alu1+ridx1)] = val5;
    }
  }
}
```

Read the three errors in order. Rounding the *inputs* to f16 changes the product by about `1e-2`; that is the
price of half-precision storage. Accumulating those f16 inputs in f32 adds only `8e-6` on top. Accumulating
in f16 adds `7e-2`, almost ten thousand times more, from just 64 additions per output. The kernel shows the
recipe at the source level: the loads are `_Float16`, the `CAST` nodes became `(float)` conversions fused
into the multiply, and the accumulator is a `float`. Tensor cores do exactly this in hardware: f16 or bf16
inputs, f32 accumulators ([Chapter 13](13-shared-memory-tensor-cores.md)).

**Training** adds a second problem: gradients are often tiny, and f16's smallest normal number is about
`6e-5`. Mixed-precision training (Micikevicius et al.) keeps an f32 *master copy* of the weights, computes
the forward and backward passes in f16, and multiplies the loss by a *loss scale* (say 1024) before
`backward` so that gradients land in f16's representable range; the optimizer divides it back out, in f32.
**bfloat16** sidesteps most of this: it keeps f32's 8-bit exponent and cuts the mantissa to 7 bits, so it
has f32's range and needs no loss scaling. It is not in `uopc`. Adding it is a good small project: C has no
portable `bf16` type, so store it as a 16-bit integer, convert to f32 by shifting left 16 bits, and convert
back with round-to-nearest-even, all of which you can express with `CAST`, `MUL` and bit tricks in the
renderer.

## 15.5 Multiple devices

Every `uopc` buffer lives in one address space and every kernel runs in one process. Scaling past one
device means choosing what to split.

- **Data parallelism** splits the batch. Each device holds a full copy of the weights, computes gradients on
  its shard, and the devices *all-reduce* (sum, then share) the gradients before the optimizer step.
- **Tensor parallelism** splits individual matrices. A linear layer `Y = X W` with `W` split by columns gives
  each device a slice of `Y` (concatenate to finish); split by rows, each device produces a partial sum of
  `Y` (all-reduce to finish). Megatron-LM pairs a column split with a row split so that an MLP needs only one
  all-reduce.
- **Pipeline parallelism** splits the layers: device 0 runs layers 1–8, device 1 runs 9–16, and micro-batches
  flow through like an assembly line.

Data parallelism is pure math, so you can check it today. The gradient of a mean loss over a batch is the
mean of the gradients over two equal halves (scratch):

```python
# scratch: data parallelism, simulated: two "devices" each see half the batch; averaging their
# gradients (an all-reduce) gives exactly the full-batch gradient.
import numpy as np
from uopc.tensor import Tensor
from uopc.nn import Linear

Tensor.manual_seed(0)
layer = Linear(8, 4)
rng = np.random.default_rng(0)
X, Y = rng.standard_normal((32, 8)).astype(np.float32), rng.integers(0, 4, 32).astype(np.int32)
params = [layer.weight, layer.bias]

def grads(x, y):
  for p in params: p.grad = None
  layer(Tensor(x)).cross_entropy(Tensor(y)).backward(params)
  return [p.grad.numpy() for p in params]

full = grads(X, Y)
shards = [grads(X[:16], Y[:16]), grads(X[16:], Y[16:])]        # device 0, device 1
allreduced = [(g0 + g1) / 2 for g0, g1 in zip(*shards)]         # the all-reduce
print("max |full - allreduced|:", max(float(np.abs(a - b).max()) for a, b in zip(full, allreduced)))
```

```text
max |full - allreduced|: 1.4901161193847656e-08
```

Equal up to f32 rounding of the summation order. The compiler work for a real implementation:

1. **A device on every buffer.** Add `device: str` to `ParamArg` for `BUFFER`. `_infer` rejects an ALU op
   whose operands live on different devices, exactly like it rejects mismatched dtypes today.
2. **A `COPY` op** (`src = (buffer,)`, `arg = target device`), the only op allowed to cross devices.
3. **Kernelize cuts at copies.** A `COPY` is a materialization point, like `STAGE`, and is executed by the
   runtime (a `memcpy`, `cuMemcpyPeer`, or a socket send), not by a generated kernel.
4. **All-reduce as a composite** of copies and adds. The bandwidth-optimal *ring* all-reduce sends each
   gradient around a ring of `p` devices in `2(p − 1)` steps of `N/p` elements, so each device transfers
   about `2N` elements regardless of `p`.

## 15.6 Real compilers compared

You now know enough to read the design documents of production ML compilers and see your own decisions in
them. They all do the same five jobs: **capture** a program, represent it in a **graph IR**, decide
**fusion and scheduling**, **generate kernels**, and **run** them. They differ in where they draw the
boundaries, and in which jobs they leave to you.

@diagram 15-compiler-landscape

The table summarizes the public designs as of this writing. These projects move quickly; treat specifics as
pointers to check in their documentation, not as gospel.

| System | You write | Main IR(s) | Who fuses | Schedule and tuning | Dynamic shapes | Targets |
|---|---|---|---|---|---|---|
| **uopc** | lazy Python Tensor code | one UOp graph, rewritten in stages | `kernelize` rules + a recompute cost rule | `hand_opt` heuristics; `search_opt` timing (`OPT=2`) | none (static) | C + OpenMP; CUDA (render-tested) |
| **tinygrad** | lazy Python Tensor code | one UOp graph, rewritten in stages | automatic, rule-based | heuristics; BEAM search over optimizations | bounded symbolic `Variable`s | many: CPU, CUDA, Metal, AMD, OpenCL, WebGPU, … |
| **TVM** | imported models (ONNX, PyTorch, …) or tensor expressions | Relay / Relax (graph), TensorIR (loops) | graph-level fusion passes | schedule primitives; AutoTVM, Ansor, MetaSchedule | symbolic shapes in Relax | LLVM CPUs, CUDA, ROCm, Vulkan, Metal, … |
| **Triton** | one kernel per function, over *blocks* | Triton's MLIR dialects → LLVM | you do (a kernel is a fusion) | `@triton.autotune` over configs you list | sizes are runtime arguments | NVIDIA and AMD GPUs |
| **XLA** | JAX, TensorFlow, PyTorch/XLA programs | HLO (StableHLO for exchange) | automatic fusion passes | heuristics and cost models | mostly static; recompiles per shape | CPU, GPU, TPU |
| **MLIR / IREE** | models via front ends (StableHLO, TOSA, torch-mlir) | MLIR dialects (linalg, vector, gpu, …) | dialect-level fusion in IREE | pass pipelines | dynamic dimensions in tensor types | CPU, CUDA, ROCm, Vulkan/SPIR-V, Metal |
| **torch.compile** | ordinary PyTorch, captured from Python bytecode | FX graphs → Inductor's loop IR | Inductor (pointwise and reduction fusion) | heuristics; `max-autotune` tries kernel templates | symbolic ints with guards; recompile on guard failure | Triton (GPU), C++/OpenMP (CPU) |
| **Halide** | an algorithm plus a separate schedule (C++ DSL) | Halide IR | the schedule (`compute_at`, `compute_root`, inline) | hand schedules; autoschedulers | extents are run-time; bounds inference | CPUs, GPUs, DSPs |

### What to steal from each

**tinygrad** is `uopc`'s direct ancestor: the hash-consed `UOp`, `UPat`/`PatternMatcher`, `graph_rewrite`,
rangeify, a linearizer and C-style renderers all have tinygrad counterparts, and the course syllabus is
built around its design. Read it for symbolic `Variable`s with bounds (its LLaMA example uses one for the
KV-cache position), for its many renderers sharing one C-style base class, and for BEAM search.

**TVM** made the *compute/schedule split* (from Halide) the center of an ML compiler: you describe *what* to
compute, then apply schedule primitives (`split`, `reorder`, `tile`, `vectorize`, `bind` to GPU threads) that
change *how* without changing the result. `uopc`'s `Opts` is a tiny schedule language with four primitives.
TVM's Ansor and MetaSchedule search a much larger space with a learned cost model. It is a much larger version of
`search_opt`.

**Triton** moves the boundary the other way. You write one kernel at the level of *blocks* (`tl.load` of a
whole tile, `tl.dot` of two tiles), and the compiler handles what Chapters 12 and 13 did by hand: coalescing,
shared-memory staging, and tensor-core instructions. Fusion is your job, because a Triton kernel is itself a fusion.
torch.compile's Inductor generates Triton code, which is a nice division of labor: Inductor decides what to
fuse, Triton decides how to run a block.

**XLA** pioneered aggressive automatic fusion of whole-program graphs (HLO) for TPUs and GPUs. It prefers
static shapes, as `uopc` does, and pays with recompilation when shapes change. You measured the same
trade-off in 15.2.

**MLIR** is not a compiler but a framework for building them: many *dialects* (IRs) at different levels,
with conversions between them. `uopc` uses one IR at every level, which is simpler but means a single op set
must serve the Tensor API and the C renderer (see the op-lifetime diagram in
[Appendix A](16-appendix-reference.md)). IREE is an end-to-end compiler and runtime built on MLIR.

**torch.compile** solves the capture problem that every lazy-tensor design (including `uopc`) dodges:
real PyTorch code is eager, full of Python control flow. TorchDynamo captures graphs from Python bytecode and
inserts *guards* (checks on shapes, types and values) that trigger recompilation when they fail;
AOTAutograd traces the backward pass ahead of time. Its symbolic shapes are the closest production analogue
of the design in 15.2.

**Halide** is the origin of the algorithm/schedule split, from image processing. Its *bounds inference*,
which computes how much of each input a tile of output needs, is the same reasoning as `uopc`'s `vmin`/`vmax`
over index expressions, applied to whole regions.

## 15.7 Reading tinygrad after this book

tinygrad is the best next codebase: it is Python, it is small for what it does, and you already know its
vocabulary. Its file layout changes between releases, so navigate by names, not paths:

| In `uopc` | Look for in tinygrad | Notes |
|---|---|---|
| `uop.py`: `UOp`, `Ops`, `_infer` | `class UOp`, `class Ops` | far more ops (and more dtypes); shape tracking has changed over versions |
| `rewrite.py`: `UPat`, `PatternMatcher`, `graph_rewrite` | the same names | adds compiled matchers and rewrite contexts |
| `symbolic.py`: `simplify`, `vmin`/`vmax` | symbolic rule sets, `vmin`/`vmax` | plus `Variable` with bounds |
| `rangeify.py`: `rangeify`, `lower` | rangeify | the same idea, with many more cases |
| `kernelize.py`: materialization, `realize` | the scheduler | kernel boundaries, buffer assignment |
| `opt.py`: `Opts`, `search_opt` | kernel optimization actions, BEAM | `BEAM=<n>` turns search on |
| `linearize.py` | the linearizer | |
| `render_c.py`, `gpu.py` | renderers (a C-style base class with per-backend subclasses) | compare with `CRenderer`/`CUDARenderer` |
| `DEBUG=1..4` | `DEBUG=` levels | the same convention: higher prints more |

Three habits make reading it productive. First, run a three-line program with high `DEBUG` and read the
kernels before reading any source. Second, read the *tests* for a component before the component: they tell
you what it is for. Third, follow one tensor from `Tensor.__add__` down to the generated code with a
debugger; you will recognize every stage.

## 15.8 Final projects

Weeks 11 onward are yours. The syllabus says "implement a paper, port to strange hardware, mostly
anything." Good projects have a **measurable** outcome (a speedup, a model that runs, a test suite that
passes) and touch **one or two** modules deeply rather than everything shallowly. Difficulty: ★ about one
week, ★★ two to three weeks, ★★★ the whole project period.

| # | Project | ★ | What you change | How you know it works |
|---|---|---|---|---|
| 1 | Fixed-capacity KV-cache sampler for `examples/gpt.py` (15.3, generalized to all layers and heads) | ★ | `examples/` only | samples match the padded sampler token for token; per-token time vs padding |
| 2 | Graph visualizer: UOp graph → Graphviz `dot`, with kernel boundaries colored | ★ | new tool | renders the GPT step; used in your report |
| 3 | Cross-compile to a Raspberry Pi or RISC-V board (`CC`, `CFLAGS`) and benchmark GEMM | ★ | `runtime.py` flags | test suite passes on the board; roofline for it |
| 4 | Relational symbolic rules (`RANGE < end`, `x // n` for symbolic `n`) with a property-test suite | ★ | `symbolic.py` | property tests (Appendix B) with 1,000+ seeds |
| 5 | Bounded symbolic `Variable` end to end (15.2) | ★★★ | `uop`, `symbolic`, `rangeify`, `kernelize`, `render_c` | GPT forward on lengths 1..64 compiles one plan |
| 6 | In-place `ASSIGN` op and a true KV cache | ★★★ | `tensor`, `kernelize` | per-token cost grows with `pos`, not `TMAX` |
| 7 | bf16 dtype with correct round-to-nearest-even | ★★ | `dtype`, `render_c`, `symbolic` | bit-exact against a numpy reference conversion |
| 8 | Mixed-precision training with dynamic loss scaling | ★★ | `nn.py` | GPT loss curve within noise of f32 |
| 9 | int8 weight quantization with per-channel scales | ★★ | new dtype, `nn.py` | perplexity delta vs f32; GEMM speed |
| 10 | Run and debug the CUDA path on a real GPU; compare with cuBLAS/PyTorch | ★★ | `cuda.py`, `gpu.py` | `test_cuda_smem_gemm_on_gpu` passes; GFLOP/s table |
| 11 | A HIP (AMD) or OpenCL renderer | ★★ | new renderer subclass | the week-7 tests pass through it |
| 12 | A Metal renderer (Apple GPUs) | ★★ | new renderer subclass | same |
| 13 | WebGPU/WGSL renderer running in a browser | ★★★ | new renderer + JS runtime | MNIST inference in a web page |
| 14 | Lower `WMMA` from the graph (pattern-match GEMM tiles, emit fragments) | ★★★ | `gpu.py`, `render_c.py` | tensor-core GEMM matches f32 within f16 error |
| 15 | Double-buffered shared-memory GEMM with asynchronous copies | ★★★ | `gpu.py` | GFLOP/s vs `smem_gemm_ir` |
| 16 | Explicit SIMD: render vector types or intrinsics for UPCAST groups | ★★ | `render_c.py` | GEMM GFLOP/s beyond the 114 of Chapter 10 |
| 17 | Persistent tuning cache: store `search_opt` winners on disk keyed by kernel source | ★ | `opt.py` | second run of `OPT=2` compiles no candidates |
| 18 | Learned cost model for `search_opt` (features of the AST → predicted time) | ★★★ | `opt.py` | ranks candidates better than `hand_opt` on held-out kernels |
| 19 | Memory planner: reuse temp buffers by liveness | ★★ | `kernelize.py` | peak temp memory of a GPT step drops; same outputs |
| 20 | Gradient checkpointing (recompute activations in backward) | ★★ | `autodiff.py`, `tensor.py` | peak memory vs extra FLOPs, measured |
| 21 | Gather/scatter ops: real embedding lookup and its gradient | ★★ | `uop`, `rangeify`, `autodiff` | `Embedding` without the one-hot matmul; speedup |
| 22 | Direct convolution kernels (no im2col) or Winograd F(2,3) | ★★★ | `tensor.py`, `opt.py` | conv2d GFLOP/s vs im2col |
| 23 | Data-parallel training across processes with shared-memory all-reduce | ★★ | new runtime layer | 2-process loss curve equals 1-process curve |
| 24 | Tensor-parallel linear layers across two "devices" (`COPY` op) | ★★★ | `uop`, `kernelize`, `runtime` | outputs equal single-device; copy volume measured |
| 25 | Implement FlashAttention-2's backward pass in the loop IR | ★★★ | `examples/` | gradient check against unfused attention |
| 26 | Whole-compiler fuzzer: random Tensor programs vs numpy | ★★ | new test file | finds (and you fix or report) at least one bug |
| 27 | ONNX importer: load a small model into uopc Tensors | ★★ | new front end | ResNet-18 or similar matches onnxruntime outputs |
| 28 | Cut compile time: memoize rangeify per kernel structure, profile `make_plan` | ★★ | `kernelize`, `rangeify` | GPT first-step compile time (Chapter 14's ~9 s) halved |

Whatever you choose, write the tests first, keep a lab notebook of measurements the way `book/NOTES.md`
does, and keep the code as small as you can. The course rule still applies: if you let slop in now, it
compounds.

## Pitfalls

- **Python values in the graph are shapes in disguise.** Anything that varies per call and is baked into
  the graph as a `CONST` or a shape makes every call structurally different and recompiles everything. It
  happened with AdamW's bias corrections (NOTES 17) and with sampling lengths (NOTES 18); the fixes were the
  same: make the varying thing *data* (a one-element tensor) or make the shape fixed (padding).
- **Unbounded symbolic values poison bounds analysis.** A trip count loaded from memory gives `j` an upper
  bound of `2^31 - 2`, `i*8 + j` might overflow, and `bounds()` correctly returns "anything" (15.2). Every
  rule gated on bounds then silently stops firing; the code stays correct but gets slower. Always attach
  declared bounds to symbolic sizes.
- **Interval arithmetic cannot prove relational facts.** `j < n` inside `for (j = 0; j < n; j++)` is
  obviously true and unprovable with intervals. You need a rule that knows what `RANGE` means.
- **Accumulating in low precision.** An all-f16 64-term dot product had an error of `7e-2` versus `8e-6` with
  an f32 accumulator, on the same f16 inputs (15.4). Cast up before the reduce, not after it.
- **Masking is not skipping.** A masked fixed-capacity cache computes scores for empty slots and throws them
  away; correct output can hide `Θ(TMAX)` waste. Measure per-token time as `pos` grows to see it.

## Exercises

1. ★ *(predict the output)* In the scratch program of 15.2, what does `vmin(i * MAXN + j)` become if `LEN`
   is replaced by a `CONST 8`? Why?
2. ★ Generalize the fixed-capacity KV cache of 15.3 to `H` heads by giving `K` and `V` shape
   `(H, TMAX, D)`. Check against the numpy reference per head.
3. ★ Measure the masking waste: time `decode_step` for `TMAX = 64, 256, 1024` at `pos = 3`. Plot time vs
   `TMAX`.
4. ★★ Implement the relational rule `RANGE r < r.src[0] → True` in a scratch copy of `symbolic.py`, and
   write a test that `simplify(j < n)` folds for both a `CONST` and a `PARAM` end.
5. ★★ *(find the bug)* A student adds `Ops.VAR` and makes `_structure_key` include `u.arg` for it, where
   `arg = VarArg(name, lo, hi, value)`. Every decode step recompiles. Why, and what is the fix?
6. ★★ Implement bf16 conversion functions in C (`float bf16_to_f32(unsigned short)` and the reverse with
   round-to-nearest-even) and test them exhaustively against numpy on all 65,536 bit patterns.
7. ★★ Add dynamic loss scaling to a scratch copy of `AdamW`: halve the scale on any non-finite gradient and
   skip the step; double it after 1,000 clean steps.
8. ★★ Write tensor parallelism for one linear layer by hand: split `W` by columns into two `Linear`s, run
   both, concatenate with `Tensor.cat`, and compare against the unsplit layer. Then split by rows and use a
   sum instead.
9. ★★★ Design the `VAR` op on paper: its `_infer` rule, its `_bounds`, its rendering, and the changes to
   `_structure_key` and `realize`. Identify every place in `uopc/` that assumes a shape entry is an `int`
   (hint: grep for `prod(`).

<details markdown="1"><summary>Solutions to the ★ exercises</summary>

**1.** With `end = CONST 8`, `_bounds(RANGE) = (0, 7)`, so `i*8 + j` has bounds `(0*8 + 0, 3*8 + 7) = (0, 31)`;
`vmin` is `0`. Nothing can overflow, so bounds are exact. (This is the static case tested in
`test_symbolic_ranges`.)

**2.** Make `here` shape `(1, TMAX, 1)`, `k` shape `(H, 1, D)` (reshape the `(1, H·D)` projection), and the
update `K * (1 - here) + here * k`. Scores are `q (H, 1, D) @ K.transpose() (H, D, TMAX)`, masked with
`seen` reshaped to `(1, 1, TMAX)`. The numpy reference is the single-head one applied to each head's
`D`-wide slice of the projections.

**3.** Wrap the loop in a function of `TMAX`; skip the first (compiling) call; time 20 calls at a fixed
`pos`. Expect time to grow roughly linearly in `TMAX` once the Python overhead (about 4 ms per step at
`TMAX = 64`, from 15.3) is no longer dominant: that slope is the masking waste.

</details>

## Quiz

1. Why does `uopc` compile a new plan when only the sequence length changes?
2. The week-3 loop IR already supports a loop whose end is loaded from memory. What part of the compiler
   stops working well in that case, and why?
3. What two quantities does a KV cache store, per layer, and why is it correct to reuse them?
4. Compute the KV-cache size for 24 layers, `dim` 2048, 2048 tokens, f16.
5. Why accumulate in f32 when inputs are f16? Give the order of magnitude from 15.4.
6. What is the difference between bf16 and f16, and which one needs loss scaling?
7. In data parallelism, what is communicated each step, and with which collective?
8. Name one thing Triton leaves to the programmer that `uopc`'s kernelizer decides automatically.

<details markdown="1"><summary>Answers</summary>

1. `_structure_key` includes every shape and movement-op argument, and a different length changes them, so
   the graph is structurally new and misses the plan cache.
2. Bounds analysis: a loaded `i32` has full-range bounds, so index expressions might overflow, `bounds()`
   gives up, and div/mod simplification, the `/` vs `floordiv_i32` choice, and rangeify's dependence analysis
   all degrade.
3. The key and value vectors of every past token. Causal attention at earlier positions never sees later
   tokens, so past keys and values never change when a token is appended.
4. 2 · 24 · 2048 · 2048 · 2 bytes = 402,653,184 bytes = 384 MiB per sequence.
5. f16 has a 10-bit mantissa; summing many products loses precision quickly. In 15.4, a 64-term dot product
   had error ~`7e-2` with an f16 accumulator vs ~`8e-6` with f32 on the same inputs.
6. bf16 has an 8-bit exponent (f32's range) and a 7-bit mantissa; f16 has a 5-bit exponent and a 10-bit
   mantissa. f16 needs loss scaling because small gradients underflow its range; bf16 usually does not.
7. Gradients, combined with an all-reduce (sum or mean), so every replica applies the same update.
8. Fusion: in Triton, the programmer decides what goes in one kernel; `uopc` decides kernel boundaries in
   `kernelize` (materialization points and `stage_broadcast_compute`).

</details>

## Further reading

- tinygrad, the source code and its examples (github.com/tinygrad/tinygrad). The LLaMA example shows a KV
  cache with a bounded symbolic position.
- J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand, S. Amarasinghe. *Halide: A Language and Compiler
  for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines.* PLDI 2013.
- T. Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI 2018.
- L. Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020.
- P. Tillet, H. T. Kung, D. Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network
  Computations.* MAPL 2019.
- C. Lattner et al. *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation.* CGO 2021.
- J. Ansel et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph
  Compilation.* ASPLOS 2024.
- The XLA and StableHLO documentation at openxla.org.
- P. Micikevicius et al. *Mixed Precision Training.* ICLR 2018.
- M. Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.*
  arXiv 2019.
- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- R. Pope et al. *Efficiently Scaling Transformer Inference.* MLSys 2023.
