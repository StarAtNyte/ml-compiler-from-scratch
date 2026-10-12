# 6. Movement ops and rangeify

By the end of [Chapter 5](05-loops.md) you can write a matrix multiply in the loop IR: three
`RANGE`s, an `ALLOC` for the accumulator, a handful of `AFTER`s to order the stores, and a lot of
index arithmetic like `ridx0*256 + ridx2`. It works. It is also miserable. Nobody wants to write
`for` loops for every `softmax` in a transformer, and nobody can write them correctly by hand for
long. Tensor programmers think in *shapes*: "reshape this to `(N, 1, K)`, broadcast it against
that, multiply, and sum the last axis." This chapter adds that vocabulary to the IR (four movement
ops, a `REDUCE`, a general `STACK`) and then writes the compiler pass that removes it again:
**rangeify** turns a graph of shaped tensors into a graph of scalars inside loops. After this
chapter, "compile any model to C (slow, but correct)" (the syllabus milestone for weeks 3 and 4) is
done.

**What you'll build**

- The movement ops `RESHAPE`, `EXPAND`, `PERMUTE`, `SHRINK`, their type rules, and the *index map*
  each one stands for.
- `REDUCE` (pop the last *n* axes) and `STACK` with arbitrary inputs.
- `ravel`/`unravel`, the two functions that convert between multi-dimensional and flat indices.
- `rangeify(u, idx)`: "what is the value of `u` at index `idx`?", answered by pushing the index
  down the graph, one rule per op.
- The `REDUCE` lowering: a fresh loop, an accumulator, its initialization and read-back.
- `lower(root, out)`: a whole shaped graph becomes a `SINK` of stores you can render and run.
- GEMM and convolution written with nothing but movement ops, and the composite ops `PAD` and
  `FLIP` built from primitives in `uopc/composite.py`.

**Where this fits.** Rangeify sits between the shape world of the frontend (the Tensor library of
[Chapter 9](09-tensor-mnist.md), which is a thin layer over the ops in this chapter) and the loop
world of [Chapter 5](05-loops.md). Its output goes through the symbolic simplifier of
[Chapter 3](03-symbolic.md), the linearizer, and the C renderer. In
[Chapter 7](07-kernelize.md) we stop lowering the whole graph as one loop nest and cut it into
kernels; in [Chapter 10](10-cpu-performance.md) the `Opts` that `lower()` accepts start to matter.

> **A note on listings.** Every generated C function in this chapter was produced by `render_c`
> on the output of `lower()`. The renderer always emits the same four-line prelude (two includes
> and the `floordiv_i32`/`floormod_i32` helpers from [Chapter 4](04-memory-and-c.md)); it is
> omitted from the listings to save space. Everything else is verbatim.

## 6.1 Shapes are functions, not memory

Start with the question every array library has to answer: *what is a tensor?* The obvious answer
is "a block of memory plus a shape". A `float[12]` buffer with shape `(3, 4)` stores element
`(i, j)` at offset `4*i + j`. This is **row-major** order: the last axis is contiguous, and the
**stride** of axis `k` is the product of the sizes after it, here `(4, 1)`.

A more useful answer for a compiler is: *a tensor of shape `s` is a function from index tuples in
`s` to values.* The buffer above is the function

```text
x(i, j) = data[4*i + j]        for 0 <= i < 3, 0 <= j < 4
```

Under this view, a **movement op** does not move any data. It builds a new function by composing
the old one with an **index map**. Transposing `x` gives `t(i, j) = x(j, i)`. Taking rows 1..2 gives
`s(i, j) = x(i + 1, j)`. Broadcasting a row vector `b` of shape `(1, 4)` to `(3, 4)` gives
`e(i, j) = b(0, j)`. None of these needs a single byte of new memory; each is a cheap rewrite of
the index before it reaches the buffer.

NumPy implements the same idea with *strides*: a view is `(base pointer, shape, strides, offset)`
and a transpose just swaps two strides. Strides are fast (one multiply-add per axis) but not closed
under composition: `x.T.reshape(12)` cannot be expressed as any single stride vector, so NumPy
silently copies. Index maps are closed under composition, because a composition of functions is a
function. The price is that some compositions contain `//` and `%`, which the symbolic simplifier
then has to clean up.

| Representation | Transpose | Broadcast | Slice | Reshape of a transposed view |
|---|---|---|---|---|
| Strides (NumPy, early tinygrad `View`) | swap strides | stride 0 | offset + shape | **not expressible**, copy |
| Index maps (this book) | permute indices | index -> 0 | index + begin | `unravel(ravel(idx))`, always works |

> **Key idea:** a shaped graph describes *what* every element is, as a composition of index maps
> ending at buffers and constants. Lowering it to loops is the act of evaluating that composition
> symbolically, for a symbolic index made of loop counters.

## 6.2 The four movement ops

`uopc` has exactly four primitive movement ops. Each has an argument (`arg`), a type rule that
computes its output shape from its input shape, and an index map. Here are the type rules, straight
from the IR's type checker in `uopc/uop.py` (Chapter 1 built `_infer`; this is the week-4 part):

@include uopc/uop.py:286-303

Every malformed movement op dies at construction time with a message saying what was wrong. That
is the same "the type checker is the first test" philosophy as in Chapter 1, and it catches most
shape bugs in the frontend long before anything is lowered.

### RESHAPE

`x.reshape(s)` reinterprets the elements of `x`, in row-major order, as a tensor of shape `s`. The
only requirement is that the element count is the same. To find the source element of output index
`idx`, compute its position in row-major order (**ravel**) and convert that position back into an
index of the source shape (**unravel**):

```text
reshape(x, s)[idx] = x[unravel(ravel(idx, s), x.shape)]
```

Example: `x` has shape `(3, 4)`; `y = x.reshape((2, 6))`. Then `y[1, 2]` is flat position
`1*6 + 2 = 8`, which in shape `(3, 4)` is `(8 // 4, 8 % 4) = (2, 0)`. So `y[1, 2] = x[2, 0]`.

### EXPAND

`x.expand(s)` repeats size-1 axes. It requires the same rank, and each axis either keeps its size or
grows from 1. The index map sends the index on an expanded axis to 0:

```text
expand(x, s)[i0, ..., in] = x[j0, ..., jn]    where jk = 0 if x.shape[k] == 1 else ik
```

`EXPAND` does not add axes. To broadcast a vector of shape `(4,)` to `(3, 4)` you first `RESHAPE`
it to `(1, 4)` and then `EXPAND`. Keeping the two jobs separate keeps each op's rule trivial.

### PERMUTE

`x.permute(p)` reorders axes: output axis `k` is input axis `p[k]`, so the output shape is
`(x.shape[p[0]], x.shape[p[1]], ...)`. To evaluate it we need the *inverse* direction: given an
output index, which input index? Output coordinate `k` is input coordinate `p[k]`, so

```text
permute(x, p)[idx] = x[inv]    where inv[p[k]] = idx[k] for every k
```

For a 2D transpose `p = (1, 0)` the permutation is its own inverse, which is exactly why bugs in
this rule survive 2D tests (see Pitfalls).

### SHRINK

`x.shrink(((b0, e0), (b1, e1), ...))` keeps the half-open window `[bk, ek)` on every axis. The
output shape is `(e0 - b0, e1 - b1, ...)` and the index map is an offset:

```text
shrink(x, arg)[idx] = x[i0 + b0, i1 + b1, ...]
```

`SHRINK` is slicing with step 1. There is no strided slice primitive; `Tensor._stride2d` in
[Chapter 9](09-tensor-mnist.md) builds `x[::2]` from `PAD`, `RESHAPE` and `SHRINK`.

## 6.3 Broadcasting is EXPAND plus RESHAPE

NumPy's broadcasting rule: align shapes on the right, prepend 1s to the shorter one, and on every
axis the sizes must be equal or one of them must be 1; the result takes the larger size. In
`uopc`, broadcasting is not an op at all. It is sugar that inserts `RESHAPE` (to prepend 1s) and
`EXPAND` (to grow 1s). The `.uir` parser does it for binary ops:

@include uopc/uir.py#_broadcast

The Tensor frontend has the same logic in `_broadcast_shape` and `Tensor.broadcast_to`. Because
the ALU type rule demands identical shapes (`ADD: shape mismatch (3, 4) vs () (broadcast with
EXPAND)`), every broadcast is explicit in the graph. That is a feature: rangeify never has to think
about broadcasting, only about `EXPAND`.

## 6.4 REDUCE: pop the last n axes

`x.reduce(op, n)` combines the last `n` axes with `op` (`ADD` or `MAX`) and removes them. Shape
`(2, 3, 4)` reduced with `n = 2` becomes `(2,)`; with `n = 0` it is the identity; with
`n = len(shape)` it becomes a scalar of shape `()`.

Why only the *last* axes? Because it is a canonical form. Any reduction over any set of axes is a
`PERMUTE` that moves those axes to the end, followed by a `REDUCE`. The Tensor library does exactly
that:

@include uopc/tensor.py#Tensor._reduce

One op with one argument is easier to lower, to differentiate ([Chapter 8](08-autodiff.md)) and to
pattern-match than a family of reductions over arbitrary axis sets. The permute costs nothing: it is
an index map, and it disappears into the index arithmetic of the load.

A reduction needs an **identity element** to initialize its accumulator: 0 for `ADD`, and the
smallest representable value for `MAX` (`-inf` for floats, `INT_MIN` for `i32`).

@include uopc/rangeify.py#identity

## 6.5 STACK, generalized

In week 1, `STACK` could only stack constants (so the const folder could run programs). Now it can
stack any same-shaped, same-dtype values: `STACK(a, b, c)` has shape `(3, *a.shape)`, and
element `[k, *rest]` is element `rest` of the `k`-th source. `INDEX(x, i)` is its inverse: it pops
the leading axis by selecting entry `i`. Together they make the IR able to build tensors out of
pieces, which is how `PAD` will be built below.

## 6.6 Rangeify: computing one element

Now the main event. We have a shaped graph and want a loop program. The idea is to stop thinking
about whole tensors and ask a much smaller question:

> **Key idea:** *what is the value of node `u` at index `idx`?* If `idx` is made of loop counters,
> the answer is a scalar expression that, placed inside those loops, computes every element.

For the root of shape `(M, N)`, create two loops `i in [0, M)` and `j in [0, N)` and ask for the
root's value at `(i, j)`. Every op answers by asking its sources the same question at some
*transformed* index. The index flows down the graph, from the root toward the leaves, transformed by
each movement op's index map, until it hits a `BUFFER` (which answers with a `LOAD`) or a `CONST`
(which answers with itself). Here is the whole pass:

@include uopc/rangeify.py#rangeify

Go through it branch by branch:

- **CONST**: a constant is the same at every index.
- **BUFFER**: buffers are flat (`BUFFER is flat` in the type rule), so `idx` has exactly one
  component. The value is `LOAD(INDEX(buf, i))`, the same address-then-load pair you used in
  [Chapter 4](04-memory-and-c.md).
- **ALU**: elementwise ops have no index map. `f(a, b)` at `idx` is `f(a[idx], b[idx])`. Note
  `u.replace(src=...)`: the op and its arg are reused, only the sources change, and the result has
  shape `()` because its sources do.
- **INDEX**: `x.index(i0, ...)` pops leading axes, so `x.index(i)[idx] = x[i, *idx]`. The index
  values themselves are scalars, rangeified at the empty index `()`.
- **STACK**: `STACK(a, b, c)[k, *rest]` is "`a[rest]` if `k < 1`, else `b[rest]` if `k < 2`, else
  `c[rest]`", built as a chain of `WHERE`s from the last source backward.
- **RESHAPE**, **EXPAND**, **PERMUTE**, **SHRINK**: exactly the four index maps from the previous
  section. The `if b else i` in `SHRINK` avoids building `i + 0` nodes for untouched axes.
- **STAGE** is the kernelizer's materialization marker ([Chapter 7](07-kernelize.md)). Without a
  kernelizer there is nothing to materialize into, so it is transparent.
- **REDUCE** is the only op that creates new loops. It gets its own section.

### Watching an index flow

Here is a small chain: a `(3, 4)` matrix, transposed, rows 1 and 2 of the transpose kept, times a
broadcast vector of length 3.

```python
x = buf(0, 12).reshape((3, 4))                       # a 3x4 matrix
y = x.permute((1, 0)).shrink(((1, 3), (0, 3)))       # rows 1..2 of x^T: shape (2, 3)
b = buf(1, 3).reshape((1, 3)).expand((2, 3))
root = y * b
```

I wrapped `rangeify` in a tracer that prints each call before it recurses (the tracer is a
scratch script, not repo code; `i` and `j` are `RANGE 0` and `RANGE 1`):

```text
MUL      shape=(2, 3)   idx=(i, j)
  SHRINK   shape=(2, 3)   idx=(i, j)
    PERMUTE  shape=(4, 3)   idx=((i+1), j)
      RESHAPE  shape=(3, 4)   idx=(j, (i+1))
        BUFFER   shape=(12,)    idx=(((i+(j*4))+1))
  EXPAND   shape=(2, 3)   idx=(i, j)
    RESHAPE  shape=(1, 3)   idx=(0, j)
      BUFFER   shape=(3,)     idx=(j)
```

Read it top to bottom: `SHRINK` adds the begin offset 1 to axis 0; `PERMUTE` swaps the two
coordinates; `RESHAPE` ravels `(j, i+1)` in shape `(3, 4)` into `j*4 + i + 1` (already simplified, see
below). On the other branch, `EXPAND` zeroes the broadcast axis and the `RESHAPE` from `(3,)` to
`(1, 3)` drops it. The generated C is the whole story in two loads:

```c
void view_chain(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val1 = data0[((ridx0+(ridx1*4))+1)];
      float val2 = data1[ridx1];
      data2[((ridx0*3)+ridx1)] = (val1*val2);
    }
  }
}
```

@diagram 06-index-pushing

Five movement ops, zero copies, and the C compiler sees ordinary strided loads. This is the payoff
of the function view: movement ops cost nothing at run time *if* the index arithmetic simplifies.

### ravel and unravel

`RESHAPE` is the only rule that needs real arithmetic. Two helpers do the work:

@include uopc/rangeify.py#ravel

@include uopc/rangeify.py#unravel

`ravel` is Horner's rule: for shape `(s0, s1, s2)`, `((0*s0 + i0)*s1 + i1)*s2 + i2`. `unravel`
goes the other way, peeling off the last axis first: coordinate `k` is
`(flat // stride_k) % s_k`, where `stride_k` is the product of the sizes after axis `k`. A size-1
axis always has coordinate 0, so it gets a `CONST 0` instead of a pointless `% 1`.

Both build *UOp expressions*, not numbers: the flat index is a graph of `ADD`, `MUL`, `FLOORDIV`,
`FLOORMOD` over `RANGE`s. Here is what they produce for a `(4, 6)` index, printed with a small
formatting helper (`i` is `RANGE 0`, `j` is `RANGE 1`):

```text
ravel((i, j), (4, 6))      = ((((0*4)+i)*6)+j)
simplified                 = ((i*6)+j)
unravel(.., (4, 6))        = ['((((((0*4)+i)*6)+j)//6)%4)', '((((((0*4)+i)*6)+j)//1)%6)']
simplified                 = ['i', 'j']
unravel(.., (2, 12)) simpl = ['(((i*6)+j)//12)', '(((i*6)+j)%12)']
unravel(.., (24, 1)) simpl = ['((i*6)+j)', '0']
```

Raveling and unraveling in the *same* shape is the identity, and the symbolic rules of
[Chapter 3](03-symbolic.md) prove it: `((i*6 + j) // 6) % 4` becomes `(i + j//6) % 4` by the
floor-division rule `(a*c + b)//c = a + b//c`, then `j//6 = 0` because bounds analysis knows
`0 <= j < 6`, then `i % 4 = i` because `0 <= i < 4`. Reshaping into a genuinely different shape,
like `(2, 12)`, leaves real `//` and `%` in the index. Those are the cost of the function view, and
they are cheap.

### Memoization: rangeify is a DAG walk

The graph is a DAG, not a tree: `x * x` uses `x` twice, and `softmax` uses its input three times.
Without memoization, rangeify would walk a shared subgraph once per path to it, and a chain like
`a = x + x; b = a + a; c = b + b; ...` would cost 2^n calls. `ctx.memo` caches the answer per
`(node, index)` pair. Because UOps are hash consed ([Chapter 1](01-uops.md)), two index
expressions that are structurally equal are the *same Python object*, so the tuple `idx` is a cheap,
exact dictionary key. Hash consing also gives free common-subexpression elimination of the
*output*: two paths that compute `LOAD(INDEX(buf0, i*4 + j))` produce one node, which the
renderer emits once.

The memo is keyed on the index as well as the node. The same node evaluated at two different
indices (say `y[i]` and `y[i+1]`) is two different scalar computations.

## 6.7 Lowering REDUCE: accumulators and loops

Every other op maps an index to an index. `REDUCE` is different: element `idx` of
`x.reduce(ADD, n)` is a *sum over* all values of the popped axes. In loop language that is a new
loop, an accumulator, an initialization before the loop, an update inside it, and a read after it.
Chapter 5 showed how to write that by hand with `ALLOC`, `AFTER`, `STORE` and `END`. `_reduce`
writes it automatically:

@include uopc/rangeify.py#_reduce

Ignore `_axis_parts`, `_combos` and the balanced tree for now. With default options (no unrolling,
which is a week-6 feature, see [Chapter 10](10-cpu-performance.md)) they create exactly one
`REDUCE`-kind `RANGE` per popped axis (or `CONST 0` for a size-1 axis), and `vals` has one element:
the source's value at `idx + (r0, r1, ...)`. Then the accumulator is built in five steps:

1. `acc = ctx.alloc(dtype)`: a fresh one-element `ALLOC`, rendered as a local `float acc0;`.
2. `acc0 = acc.after(*ranges_in(*idx))`: *this* accumulator lives inside every loop the output
   index mentions. An `AFTER` whose dependencies are `RANGE`s is how the IR says "a fresh instance
   per iteration of these loops". The linearizer reads it as the scope of the `ALLOC`.
3. `init = acc0.index(0).store(identity)`: write 0 (or `-inf`) into it.
4. `accl = acc0.after(init, *real)`: the accumulator *after* the init and *inside* the reduce
   loops. Load it, combine with the new value, store it back: `upd`.
5. `acc0.after(upd.end(*real)).index(0).load()`: after the reduce loops have ended, load the result.
   This is the scalar value that `REDUCE` returns to its consumer.

Why all the `AFTER`s? The IR is a pure graph with no statement order. Two `LOAD`s of `acc0[0]` with
no ordering edge are the same node (hash consing!), so "the accumulator before the loop" and "the
accumulator after the loop" *must* be different nodes. `AFTER` makes them different and tells the
linearizer which effects must happen first. This is the memory-ordering discipline of
[Chapter 4](04-memory-and-c.md), applied mechanically.

Here is a row sum, `x.reduce(ADD, 1)` on a `(3, 4)` matrix, as `.uir` after `lower()`:

```text
%0 = buffer 3 : dtype=f32 slot=1
%1 = range 3 : idx=0 kind=loop
%2 = index %0, %1
%3 = alloc 1 : dtype=f32 slot=0
%4 = after %3, %1
%5 = index %4, 0
%6 = store %5, 0.0
%7 = range 4 : idx=1 kind=reduce
%8 = after %4, %6, %7
%9 = index %8, 0
%10 = load %9
%11 = buffer 12 : dtype=f32 slot=0
%12 = mul %1, 4
%13 = add %12, %7
%14 = index %11, %13
%15 = load %14
%16 = add %10, %15
%17 = store %9, %16
%18 = end %17, %7
%19 = after %4, %18
%20 = index %19, 0
%21 = load %20
%22 = store %2, %21
%23 = end %22, %1
%24 = sink %23
```

and as C:

```c
void rowsum(float* restrict data0, float* restrict data99) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float val1 = acc0;
      float val2 = data0[((ridx0*4)+ridx1)];
      acc0 = (val1+val2);
    }
    float val3 = acc0;
    data99[ridx0] = val3;
  }
}
```

Match them up: `%4` (`after %3, %1`) is the `float acc0;` inside the `ridx0` loop; `%6` is
`acc0 = 0.0f`; `%8`..`%17` is the loop body; `%18` closes the reduce loop; `%19`..`%21` is the read
after it.

@diagram 06-reduce-lowering

### Where the accumulator lives decides the loop structure

Step 2 is the most consequential line in the file. The accumulator is placed in the loops that
`idx` *mentions*, found syntactically by `ranges_in`:

@include uopc/rangeify.py#ranges_in

Consider `x - max(x, axis=1)` broadcast back over the row, the first half of a softmax:

```python
x = buf(0, 12).reshape((3, 4))
m = x.reduce(Ops.MAX, 1).reshape((3, 1)).expand((3, 4))
root = x - m
```

The output loops are `i` (rows) and `j` (columns). The index reaching the `REDUCE` went through
`EXPAND` (column index replaced by 0) and `RESHAPE` (`(i, 0)` in `(3, 1)` to `(i,)` in `(3,)`), so it
is just `(i,)`. The accumulator lives in loop `i` only, and the linearizer makes the max loop a
*sibling* of the `j` loop instead of nesting it inside:

```c
void sub_max(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    int alu1 = (ridx0*4);
    float acc0;
    acc0 = -INFINITY;
    for (int ridx2 = 0; ridx2 < 4; ridx2++) {
      float val2 = acc0;
      float val3 = data0[(alu1+ridx2)];
      acc0 = ((val2>val3)?val2:val3);
    }
    float val4 = acc0;
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu5 = (alu1+ridx1);
      float val6 = data0[alu5];
      data1[alu5] = (val6+(val4*-1.0f));
    }
  }
}
```

The max is computed once per row, not once per element. Note the range numbering: the output's
`j` is `RANGE 1` (`ridx1`), the reduce's is `RANGE 2`, but the reduce loop comes first because the
`j` loop reads its result through an `END`. That ordering decision is the linearizer's
(Chapter 5, the "sibling loops" story in NOTES 8c); rangeify only has to produce the right graph.

The flip side: `ranges_in` is syntactic. If the index *mentions* `j` without depending on it, the
accumulator lands inside the `j` loop and the reduce is recomputed for every `j`. That is exactly
what happened in a real bug, and it is why `RESHAPE` simplifies its indices immediately (see
"Index simplification at RESHAPE" below).

### Unrolled copies share one loop

`ctx.reduce_ranges` caches the reduce loops per `(REDUCE node, set of ranges in idx)`. Its purpose
is week 6's register tiling: when an output axis is unrolled by 4, rangeify asks for the same
`REDUCE` at four indices that differ only by constants, and those four accumulators must share
*one* reduce loop (that is the whole point of a register tile). It has a pleasant side effect even
without unrolling. Here two shifted views of the same row sum share a loop:

```python
x = buf(0, 20).reshape((4, 5))
y = x.reduce(Ops.ADD, 1)
z = y.shrink(((0, 3),)) + y.shrink(((1, 4),))        # s[i] + s[i+1]
```

```c
void neighbors(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    float acc1;
    acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      float val1 = acc0;
      int alu2 = ((ridx0*5)+ridx1);
      float val3 = data0[alu2];
      acc0 = (val1+val3);
      float val4 = acc1;
      float val5 = data0[(alu2+5)];
      acc1 = (val4+val5);
    }
    float val6 = acc0;
    float val7 = acc1;
    data1[ridx0] = (val6+val7);
  }
}
```

Two accumulators, one loop. Both copies `END` the same `RANGE`; `merge_ends` (below) merges those
two `END`s into one.

## 6.8 lower(): from a shaped root to a SINK

`rangeify` answers "the value at `idx`". `lower` wraps it into a complete program: create the
output loops, ask for the root's value, store it into the output buffer at the raveled index, close
the loops, simplify.

@include uopc/rangeify.py#lower

On the CPU path with default `Opts`, `_axis_parts` returns one `LOOP` range per output axis of size
greater than 1 and `CONST 0` for size-1 axes (a loop of one iteration is not a loop). `_combos` then
yields a single index tuple. The rest is short:

- the store address is `ravel(idx, root.shape)`: outputs are always contiguous row-major;
- `ranges_in(...)` collects the output loops, and `body.end(*loops)` closes them;
- `merge_ends` merges duplicate `END`s produced by unrolled copies (one `END` per loop, over a
  `SINK` of their bodies);
- `simplify` runs the symbolic rules over the whole program once more.

@include uopc/rangeify.py#merge_ends

The `gpu` branch and the `THREAD`/`UPCAST` handling are for [Chapter 10](10-cpu-performance.md) and
[Chapter 12](12-gpus.md). Notice that they only change *which* ranges exist and how an output index
is assembled from them (`outer * factor + c`). The rangeify rules themselves never change. That
separation (what to compute versus how to map it onto the machine) is the central design decision
of the compiler, borrowed from Halide.

The test helper that every week-4 test uses is `lower` plus the runtime:

@include tests/helpers.py#run_shaped

## 6.9 Index simplification at RESHAPE

Look again at the `RESHAPE` branch of `rangeify`:

```python
r = rangeify(u.src[0], tuple(simplify(i) for i in unravel(ravel(idx, u.shape), u.src[0].shape)), ctx)
```

The `simplify` call is not cosmetic. `lower` simplifies everything at the end anyway, so the C
would have the same arithmetic either way. What changes is the *loop structure*, because
`_reduce` places accumulators according to the ranges an index mentions. To see it, take the
`sub_max` example and add a flatten/unflatten pair at the top, the kind of reshape round trip
Tensor code produces all the time (`x.reshape(-1)` then back):

```python
x = buf(0, 16).reshape((4, 4))
rowmax = x.reduce(Ops.MAX, 1).reshape((4, 1)).expand((4, 4))
y = (x - rowmax).reshape((16,)).reshape((4, 4))
```

The index reaching the `REDUCE` is, unsimplified, `((i*4 + j) // 4) % 4`: mathematically `i`,
syntactically mentioning `j`. I monkeypatched `simplify` to the identity during rangeify (and ran
the real one afterwards, so the arithmetic is cleaned up but the structure is frozen):

```c
void no_simplify(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*4);
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu2 = (alu1+ridx1);
      float val3 = data0[alu2];
      float acc0;
      acc0 = -INFINITY;
      for (int ridx2 = 0; ridx2 < 4; ridx2++) {
        float val4 = acc0;
        float val5 = data0[(alu1+ridx2)];
        acc0 = ((val4>val5)?val4:val5);
      }
      float val6 = acc0;
      data1[alu2] = (val3+(val6*-1.0f));
    }
  }
}
```

The row max is now recomputed for every column: O(N^3) work for an O(N^2) problem. With the real
code:

```c
void with_simplify(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*4);
    float acc0;
    acc0 = -INFINITY;
    for (int ridx2 = 0; ridx2 < 4; ridx2++) {
      float val2 = acc0;
      float val3 = data0[(alu1+ridx2)];
      acc0 = ((val2>val3)?val2:val3);
    }
    float val4 = acc0;
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu5 = (alu1+ridx1);
      float val6 = data0[alu5];
      data1[alu5] = (val6+(val4*-1.0f));
    }
  }
}
```

The same mistake in the real compiler made fused attention at N=512 take 1787 ms instead of 7 ms
(NOTES 12; the story is told in full in [Chapter 11](11-fusion-flash.md)). The memo benefits too:
`(u, (i,))` and `(u, (((i*4+j)//4)%4,))` are different keys for the same value.

> **Key idea:** in a compiler, *syntactic* facts (which variables an expression mentions) drive
> structural decisions (which loop a computation lives in). Keep expressions in normal form at the
> moment you inspect them, not only at the end.

## 6.10 GEMM from movement ops

Now the payoff: matrix multiplication, written without a single loop. Start from the math,

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

and think about the three-dimensional **iteration space** `(i, j, k)`, of shape `(N, N, K)`. The
summand `A[i, k] * B[k, j]` is a function on that cube. Build each factor as a tensor of shape
`(N, N, K)`:

- `A` has shape `(N, K)`. Reshape to `(N, 1, K)` (insert a `j` axis of size 1) and expand to
  `(N, N, K)`: the value at `(i, j, k)` is `A[i, k]` for every `j`.
- `B` has shape `(K, N)`. Permute to `(N, K)` (now indexed `[j, k]`), reshape to `(1, N, K)`, expand:
  the value at `(i, j, k)` is `B[k, j]` for every `i`.
- Multiply elementwise and `REDUCE(ADD, 1)` pops `k`, leaving `(N, N)`.

@diagram 06-gemm-views

The week-4 test does exactly this in `.uir` (the parser's broadcasting sugar inserts the `EXPAND`s):

@include tests/test_w04_rangeify.py#test_gemm_from_movement_ops

With `N = 4`, the generated C is the textbook triple loop, with the accumulator in the `j` loop
and `A`'s row offset hoisted:

```c
void gemm(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*4);
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 4; ridx2++) {
        float val2 = acc0;
        float val3 = data0[(alu1+ridx2)];
        float val4 = data1[((ridx2*4)+ridx1)];
        acc0 = (val2+(val3*val4));
      }
      float val5 = acc0;
      data2[(alu1+ridx1)] = val5;
    }
  }
}
```

Two loads per multiply-add, and `B` is read down a column (stride 4). That stride is a performance
disaster at N=1024 (about 2–5 GFLOP/s depending on the run, NOTES 19), and fixing it is [Chapter 10](10-cpu-performance.md)'s
job. Correctness first.

The Tensor library's `matmul` is the same construction with the reduction axis last
(`(..., M, 1, K) * (..., 1, N, K)`), which also handles batches for free, because extra leading axes
just ride along. See [Chapter 9](09-tensor-mnist.md).

### The syllabus GEMM puzzle

The course syllabus opens with this program, called `gemm_256x256.uir`:

```text
%0 = buffer 65536 : dtype=f32 slot=0
%1 = buffer 65536 : dtype=f32 slot=1
%2 = reshape %0, (256, 256, 1)
%3 = reshape %1, (256, 1, 256)
%4 = mul %2, %3
%5 = reduce %4 : op=add pop=1
```

Before reading on, work out what it computes with index maps. The broadcast shape is
`(256, 256, 256)`; call its axes `(i, j, k)`.

- `%2 = A.reshape(256, 256, 1)` expanded: the value at `(i, j, k)` is `A[i, j]` (the size-1 axis is
  `k`, so `k` is dropped).
- `%3 = B.reshape(256, 1, 256)` expanded: the value at `(i, j, k)` is `B[i, k]` (the size-1 axis is
  `j`).
- Reducing `k`: `C[i, j] = sum_k A[i, j] * B[i, k] = A[i, j] * (sum_k B[i, k])`.

That is `A * rowsum(B)`, an elementwise product with a broadcast row sum. It is not a GEMM
(NOTES 11). The generated C (N=4) agrees: `A`'s load `val3` sits *outside* the `k` loop because its
index does not mention `k`, and the loop only sums a row of `B`:

```c
void syllabus(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*4);
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu2 = (alu1+ridx1);
      float acc0;
      acc0 = 0.0f;
      float val3 = data0[alu2];
      for (int ridx2 = 0; ridx2 < 4; ridx2++) {
        float val4 = acc0;
        float val5 = data1[(alu1+ridx2)];
        acc0 = (val4+(val3*val5));
      }
      float val6 = acc0;
      data2[alu2] = val6;
    }
  }
}
```

The repository keeps a test that pins this down, so nobody "fixes" the oracle to match a wrong
belief:

@include tests/test_w04_rangeify.py#test_syllabus_gemm_is_not_a_gemm

The correct version reshapes `A` to `(N, 1, K)` and `B` *transposed* to `(1, N, K)`. (The docstring
of `uopc/uir.py` uses `(256, 1, 256)` and `(1, 256, 256)` without the transpose, which computes
`A @ B^T`, a perfectly good GEMM if `B` is stored transposed.) The lesson generalizes: write the
index map down. "Reshape and multiply" looks right whatever the axes are.

## 6.11 Convolution from movement ops

A 1D valid convolution (strictly, a cross-correlation, which is what ML frameworks call
convolution) with a length-`K` kernel:

```text
out[i] = sum_k x[i + k] * w[k]        for 0 <= i < L - K + 1
```

The factor `x[i + k]` is a "sliding window". Each fixed `k` is a `SHRINK` of `x` (the window
`[k, k + Lo)`), so a `STACK` of `K` shrinks gives a `(K, Lo)` tensor `win[k, i] = x[i + k]`:

```python
L, K = 8, 3; Lo = L - K + 1
x, w = buf(0, L), buf(1, K)
win = UOp(Ops.STACK, tuple(x.shrink(((k, k + Lo),)) for k in range(K)))   # (K, Lo): win[k, i] = x[i+k]
cols = win.permute((1, 0))                                                  # (Lo, K)
out = (cols * w.reshape((1, K)).expand((Lo, K))).reduce(Ops.ADD, 1)          # (Lo,)
```

This is **im2col** in miniature: build the matrix of windows, then it is a matrix-vector product.

```c
void conv1d(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 6; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    float val1 = data0[ridx0];
    float val2 = data0[(ridx0+1)];
    float val3 = data0[(ridx0+2)];
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val4 = acc0;
      float val5 = data1[ridx1];
      acc0 = (val4+(((ridx1<1)?val1:((ridx1<2)?val2:val3))*val5));
    }
    float val6 = acc0;
    data2[ridx0] = val6;
  }
}
```

The `STACK` became a `WHERE` chain selecting among three loads, all hoisted out of the `k` loop. On
`x = 0..7` and `w = [1, 10, 100]` it prints `[210. 321. 432. 543. 654. 765.]`, the same as
`np.convolve(x, w[::-1], "valid")`. `Tensor.conv2d` ([Chapter 9](09-tensor-mnist.md)) is the 2D
version: `KH*KW` shrinks, a `STACK`, a permute, and a matmul.

## 6.12 Composite ops: PAD and FLIP

The syllabus says PAD and FLIP are *composite*: not primitives, but built from the primitives.
That is a design choice worth arguing about. tinygrad has both as primitive movement ops, because a
primitive `PAD` lowers to a single bounds check. Making them composite keeps rangeify and autodiff
smaller (four movement rules instead of six) and is a superb exercise in index maps. The cost is
uglier generated code, as you will see.

### Concatenation

Everything starts with concatenating two same-shaped tensors along an axis: `STACK` them (new
leading axis of size 2), move that axis next to `axis`, and merge the two with a reshape.

@include uopc/composite.py#concat2

Check the index map for 1D: `s = STACK(a, b)` has shape `(2, n)`, `s[t, i]` is `a[i]` if `t = 0`
else `b[i]`. Reshape to `(2n,)`: element `p` is `s[p // n, p % n]`, which is `a[p]` for `p < n` and
`b[p - n]` for `p >= n`. Concatenation.

### PAD

Padding with zeros after the data is "concatenate with a zero tensor of the same shape, then shrink
away the excess". Padding before is the mirror image. If the padding is larger than the axis, do it
in rounds, doubling each time:

@include uopc/composite.py#zeros_like

@include uopc/composite.py#pad_axis

@include uopc/composite.py#pad

`pad_axis(v, 0, 1, 2)` on a length-3 vector builds this graph:

```text
%0 = reshape 0.0, (1,)
%1 = expand %0, (5,)
%2 = buffer 3 : dtype=f32 slot=0
%3 = expand %0, (3,)
%4 = stack %2, %3
%5 = reshape %4, (6,)
%6 = shrink %5, ((0, 5),)
%7 = stack %1, %6
%8 = reshape %7, (10,)
%9 = shrink %8, ((4, 10),)
```

and rangeify plus the simplifier turn it into a bounds-checked select:

```c
void pad1d(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 6; ridx0++) {
    int alu1 = (ridx0+4);
    int alu2 = (alu1%5);
    float val3 = data0[(alu2%3)];
    data1[ridx0] = (((alu1/5)<1)?0.0f:(((alu2/3)<1)?val3:0.0f));
  }
}
```

Two observations. First, the `/` and `%` here are C's truncating operators, which the renderer
emits only because bounds analysis proved the operands non-negative (NOTES 4). Second, the load
`data0[(alu2%3)]` happens *unconditionally*, even for padding positions. It is safe because the
index map of a `STACK` branch is always in bounds for that branch's shape (here `% 3`), so a
speculative load never reads outside the buffer. A primitive `PAD` would emit
`(0 <= i-1 && i-1 < 3) ? x[i-1] : 0`, which is shorter. Exercise 9 asks you to build it.

### FLIP, and the diagonal trick

Reversing an axis cannot be written with `SHRINK` and `RESHAPE` alone, since none of them maps
`i` to `n - 1 - i`. The composite builds it as a matrix-vector product with the **anti-identity**
`M[i, j] = [i + j == n - 1]`:

```text
y[i] = sum_j x[j] * M[i, j] = x[n - 1 - i]
```

How do you build `M` from movement ops? With a classic trick. Let `u` be the one-hot vector of
length `2n - 1` with a 1 at position `n - 1` (a 1 padded with `n - 1` zeros on each side). Expand it
to `n + 1` rows, flatten, keep the first `n * 2n` elements, and view them as `(n, 2n)`. Element
`(i, j)` of that view is at flat position `2n*i + j`, which comes from row-length-`(2n-1)` data, so it
reads

```text
u[(2n*i + j) mod (2n - 1)] = u[(i + j) mod (2n - 1)]       because 2n = 1 (mod 2n - 1)
                           = u[i + j]                       because i + j <= 2n - 2 for j < n
```

and `u[i + j] = 1` exactly when `i + j = n - 1`. Shrinking to the first `n` columns gives `M`.
Taking a row-stride that is one longer than the data shears the data by one position per row; that
is the "diagonal trick".

@include uopc/composite.py#anti_eye

@include uopc/composite.py#flip_axis

`anti_eye(4)` evaluated by the week-1 constant folder (`run` + `to_python`) is

```text
[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
```

and flipping a length-3 vector generates:

```c
void flip3(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val1 = acc0;
      float val2 = data0[ridx1];
      int alu3 = ((((ridx0*6)+ridx1)%5)+1);
      int alu4 = (alu3%3);
      acc0 = (val1+(val2*(((alu3/3)<1)?0.0f:(((alu4/2)<1)?(((alu4%2)<1)?1.0f:0.0f):0.0f))));
    }
    float val5 = acc0;
    data1[ridx0] = val5;
  }
}
```

Correct, and O(n^2): the anti-identity is computed with index arithmetic for every `(i, j)` and
multiplied into a sum. For a course whose goal is "correct first, fast later", that is acceptable,
and it keeps the primitive set small. Flipping is rare in models (it appears in the gradient of a
convolution with respect to its input, which the autodiff of [Chapter 8](08-autodiff.md) never
needs because it differentiates the im2col construction instead). Exercise 5 asks for an O(n)
composite.

The week-4 tests check `PAD` and `FLIP` against NumPy, including padding larger than the axis:

@include tests/test_w04_rangeify.py#test_pad_flip_composites

## 6.13 Testing rangeify: random movement chains

Every rule of rangeify is a few characters long, which means a typo in any of them produces
plausible but wrong indices. Hand-written tests catch the cases you thought of. The best test in
the week-4 file instead generates random chains of movement ops and compares against NumPy, which
implements the same semantics with strides:

@include tests/test_w04_rangeify.py#test_random_movement_chains

Six random ops on a 24-element tensor, five seeds, about thirty lines. It exercises permutations of
rank 3 and 4 (which catch the inverse-permutation bug that 2D transposes hide), shrinks of
expanded tensors, and reshapes of non-contiguous views, which is precisely where strides would need
a copy and index maps produce `//` and `%`. Appendix B ([Chapter 17](17-appendix-testing.md))
discusses this kind of differential fuzzing in general.

## Pitfalls

**1. The syllabus GEMM is not a GEMM (NOTES 11).** `reshape(A, (N, N, 1)) * reshape(B, (N, 1, N))`
reduced over the last axis computes `A * rowsum(B)`. It type-checks, it runs, and it produces numbers
of a plausible magnitude. Only an oracle catches it. Derive the index map of every test program by
hand once, and test against NumPy, never against your own expectations.

**2. Unsimplified indices change the loop structure (NOTES 12).** Rangeify of a `RESHAPE` produces
`((i*N + j)//N) % N`, which mentions `j` but is `i`. Because the reduce lowering places accumulators
in the loops an index mentions, row reductions got nested inside the column loop: O(N^3). Fused
attention at N=512 took 1787 ms; simplifying at each `RESHAPE` made it 7 ms. The C above shows the
before and after. The symptom to look for in generated code: an `acc = -INFINITY` *inside* a loop
whose counter its reduction never uses.

**3. `const_like` must broadcast (NOTES 13).** `-x` is `x * -1`. If the constant is created as a
scalar `CONST`, the type checker rejects the multiply:

```text
UOpError: Ops.MUL: shape mismatch (3, 4) vs () (broadcast with EXPAND)
```

`UOp.const_like` therefore returns `CONST.reshape((1, 1)).expand((3, 4))` for a `(3, 4)` node.
Rangeify turns that back into the scalar constant for free (`EXPAND` maps the index to 0,
`RESHAPE` of `(1, 1)` drops it, `CONST` ignores the index), so the broadcast costs nothing.

**4. Distribution is needed before div/mod can simplify (NOTES 14).** A merge-then-split chain
produces `(((i*4 + j)*4 + k) // 16) % 4`. The div rule `(a*c + b)//c = a + b//c` needs to see the
`i*16` term, which only exists after distributing `(i*4 + j)*4` to `i*16 + j*4`. With the full rule
set the expression simplifies to `i`. With the distribution rule filtered out of `symbolic`:

```text
full rules:           i
without distribution: (((((i*4)+j)*4)+k)//16)
```

The leftover mentions `j` and `k`, which triggers Pitfall 2 all over again. Without this rule, GPT
attention kernels had `((x/4096)%4)` chains in their inner loops.

**5. The inverse permutation.** The rule is `inv[p[k]] = idx[k]`. The tempting
`inv[k] = idx[p[k]]` applies the permutation instead of its inverse. For every permutation of
two axes, and for any permutation that is its own inverse, the two agree, so a test suite of
transposes passes. A rank-3 rotation like `(1, 2, 0)` exposes it. This is why the random-chain test
uses rank-3 and rank-4 shapes.

**6. Speculative loads must stay in bounds.** Composite `PAD` evaluates both branches of its
`WHERE` and loads unconditionally. That is only safe because every `STACK` branch's index stays in
that branch's shape. If you add a primitive `PAD` whose out-of-range index is computed as `i - 1`,
the load must be guarded, or you read `data0[-1]`. C will not tell you.

## Exercises

1. ★ **Predict the index.** `x` is a `(3, 4)` buffer. Write down the load index (in terms of the
   loop counter `r`) for `x.permute((1, 0)).reshape((12,))`, then check with `lower` + `render_c`.

2. ★ **Unravel by hand.** Compute `unravel(17, (2, 3, 4))` using the formula
   `(flat // stride) % size`. Then compute `ravel` of your answer.

3. ★★ **Predict the loop structure.** What C does
   `x.reshape((1, 3, 4)).expand((2, 3, 4)).reduce(Ops.ADD, 2)` generate (shape `(2,)`)? How many
   times is the sum computed, and why? *Hint:* which ranges does the index reaching the `REDUCE`
   mention?

4. ★★ **2D convolution.** Write a `(H, W)` by `(KH, KW)` valid convolution as a `.uir` program or
   with UOp builders, using only `SHRINK`, `STACK`, `PERMUTE`, `RESHAPE`, `EXPAND`, `MUL` and
   `REDUCE`. Test it against `scipy.signal.correlate2d(x, w, "valid")` or a NumPy loop.

5. ★★ **An O(n) FLIP.** `flip_axis` is O(n^2). Write a composite that reverses a length-`n` vector
   with a `STACK` of `n` one-element `SHRINK`s, and compare the generated C with `flip3` above.
   *Hint:* `x.shrink(((n-1-k, n-k),))` has shape `(1,)`.

6. ★★ **Find the bug.** A student's `PERMUTE` rule is
   `r = rangeify(u.src[0], tuple(idx[p] for p in u.arg), ctx)`. Which of these does it get wrong:
   `x.permute((1, 0))`, `x.permute((0, 2, 1))`, `x.permute((2, 0, 1))`? Prove it with a
   NumPy example.

7. ★★ **Max-pooling with movement ops.** Write 2x2 max pooling of an `(H, W)` image
   (`H`, `W` even) as one `RESHAPE`, one `PERMUTE` and one `REDUCE(MAX, 2)`. Compare with
   `Tensor.max_pool2d` in `uopc/tensor.py`, which uses `max((3, 5))`.

8. ★★ **The syllabus GEMM, fixed.** Rewrite `gemm_256x256.uir` so that it computes `A @ B` with
   `A` and `B` both stored row-major. Run it through `run_shaped` at `N = 256` against NumPy.

9. ★★★ **Primitive PAD.** Add `Ops.PAD` (arg: `((before, after), ...)`) with a type rule and a
   rangeify rule that produces `valid.where(src[idx - before], 0)` with a guarded index. Compare the
   C for `pad(x, ((1, 1), (0, 2)))` with the composite's. Then decide, with numbers, whether the
   course was right to make it composite.

10. ★★★ **Views as strides.** Write `as_strided(chain) -> (shape, strides, offset) | None` that
    decides whether a chain of movement ops over a buffer can be expressed as a single strided view
    (NumPy-style), returning `None` when a `RESHAPE` makes that impossible. Use
    `_last_axis_stride` in `uopc/opt.py` for inspiration: it evaluates the index at 0 and 1.
    Fuzz it against `test_random_movement_chains`.

11. ★★★ **Bounds on unrolled reduce sharing.** `_reduce` keys its loop cache on
    `(node, frozenset(ranges_in(*idx)))`. Construct a graph where two *different* index tuples
    mentioning the same ranges ask for the same `REDUCE` (as in `neighbors`) and argue why sharing
    the loop is still correct. Then find a case where sharing would be wrong if the cache were
    keyed on the node alone.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** `reshape` ravels `r` in shape `(12,)` (just `r`) and unravels it in the transposed shape
`(4, 3)`: `(r // 3, r % 3)`. `PERMUTE (1, 0)` swaps them: `(r % 3, r // 3)` in `x`'s shape `(3, 4)`.
The buffer index is `(r % 3) * 4 + r // 3`. The generated C:

```c
void flat_t(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 12; ridx0++) {
    float val1 = data0[(((ridx0%3)*4)+(ridx0/3))];
    data1[ridx0] = val1;
  }
}
```

**2.** Strides of `(2, 3, 4)` are `(12, 4, 1)`. `17 // 12 % 2 = 1`, `17 // 4 % 3 = 4 % 3 = 1`,
`17 // 1 % 4 = 1`. So `(1, 1, 1)`, and `ravel` gives `((0*2 + 1)*3 + 1)*4 + 1 = 17`.

</details>

## Quiz

1. Why can't `EXPAND` add a new axis? What do you use instead?
2. `x` has shape `(2, 3, 4)`. What are the output shape and the index map of
   `x.permute((2, 0, 1))`?
3. Why does `rangeify` memoize on `(node, idx)` and not on `node` alone?
4. What decides which loops a `REDUCE`'s accumulator lives in?
5. Why does the reduce lowering need three different `AFTER` nodes on the same `ALLOC`?
6. Give the index map of `REDUCE`, and say why only the *last* axes are reduced.
7. What does the syllabus's `gemm_256x256.uir` compute?
8. Why is the unconditional load in the composite `PAD` safe?

<details>
<summary>Answers</summary>

1. Its type rule requires equal rank, and each axis keeps its size or grows from 1. Insert the axis
   with a `RESHAPE` to size 1 first, then `EXPAND` it. One job per op keeps each index map trivial.
2. Shape `(4, 2, 3)`. Output axis `k` is input axis `p[k]`, so `out[a, b, c] = x[b, c, a]`
   (`inv[2] = a`, `inv[0] = b`, `inv[1] = c`).
3. The same node evaluated at different indices is a different scalar computation (`y[i]` vs
   `y[i+1]`). Keying on the index too makes sharing exact; hash consing makes the key cheap.
4. The ranges that the index reaching the `REDUCE` *mentions* (`ranges_in(*idx)`), which is why
   indices must be simplified before that point.
5. The accumulator before the loop (with the init store), inside the loop (after the init, per
   reduce iteration), and after the loop (after the `END`) are three different memory states. In a
   pure hash-consed graph, loads of the same address with the same dependencies are one node, so the
   states must be distinct nodes, and the dependencies tell the linearizer the order.
6. It does not have one: `REDUCE` creates new loops over the popped axes and accumulates. Last axes
   only is a canonical form; any axis set is a `PERMUTE` away, and one simple op is easier to lower,
   differentiate and pattern-match.
7. `C[i, j] = A[i, j] * sum_k B[i, k]`, an elementwise product with a broadcast row sum.
8. The select picks the branch by position, but the load's index is computed through the `STACK`
   branch's own index map, which always lands inside that branch's shape (`% 3` in the example),
   so it never leaves the buffer even when its value is discarded.

</details>

## Further reading

- J. Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism, Locality,
  and Recomputation in Image Processing Pipelines", PLDI 2013. The algorithm/schedule split that
  `lower(root, out, Ctx(opts=...))` imitates.
- C. R. Harris et al., "Array programming with NumPy", *Nature* 585, 2020. Strides, views, and
  broadcasting, explained by the people who designed them.
- The NumPy documentation on broadcasting (`numpy.org/doc/stable/user/basics.broadcasting.html`).
- tinygrad (`github.com/tinygrad/tinygrad`): `tinygrad/shape/` (ShapeTracker and View, the
  strides-plus-masks representation) and, in recent versions, the rangeify-based scheduler. Reading
  it after this chapter is the best way to see what a production version adds.
- T. Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018,
  for tensor expressions as index-space lambdas (`te.compute`).
