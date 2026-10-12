# 9. A Tensor library and your first MNIST

You can now compile any shape graph into C, cut it into kernels, and differentiate it. But nobody
wants to *write* shape graphs. Typing `A.reshape((M, 1, K)).expand((M, N, K))` every time you mean
`A @ B` is how compiler people punish themselves, not how machine-learning people work. This chapter
builds the thin layer that turns the compiler into something that feels like PyTorch: a `Tensor`
class with broadcasting, reductions over any axes, matrix multiplication, convolution, softmax and
cross-entropy; a handful of layers; two optimizers. Then we use it to train a classifier on MNIST,
the course's week-5 milestone, and look closely at the 32 C kernels that make up one training step.
None of this needs a new compiler feature. Every operation in this chapter is *a few lines of graph
building*, and that is the point: if your IR is right, the frontend is small.

**What you'll build**

- `uopc/tensor.py`: a lazy `Tensor` that wraps a shaped UOp graph, with NumPy-style broadcasting,
  movement ops, reductions over arbitrary axes, `matmul`, `conv2d`, `softmax`, `log_softmax`,
  `argmax`, `one_hot` and `cross_entropy`.
- `uopc/nn.py`: `Linear`, `Conv2d`, `LayerNorm`, `Embedding`, a `parameters()` walker, and the `SGD`
  and `AdamW` optimizers.
- `examples/mnist_mlp.py`: a 784-128-10 MLP that trains to 94% test accuracy in about nine seconds.

**Where this fits.** [Chapter 7](07-kernelize.md) taught the compiler to split a graph into kernels and
[Chapter 8](08-autodiff.md) taught it to differentiate one. This chapter is the user-facing API on top of
both. [Chapter 10](10-cpu-performance.md) then makes the kernels it produces fast.

## 9.1 Why a lazy Tensor

A NumPy array is *eager*: `a + b` computes and stores a new array immediately. An eager design is
simple, but it hides the program from the compiler. If `y = (x * 2 + 1).relu().sum(1)` ran eagerly,
it would be four separate loops with three temporary arrays, and no later pass could fuse them,
because by the time `sum` runs the temporaries already exist.

Our `Tensor` is *lazy*. It is a thin wrapper around a UOp, and each operator builds a bigger UOp graph.
Nothing runs until somebody asks for numbers (`.numpy()`, `.item()`, or `.realize()`). At that moment
the whole pending graph goes through the pipeline from [Chapter 7](07-kernelize.md): kernelize,
rangeify, render, compile (or fetch from cache), run.

Laziness buys three things, each of which this chapter relies on:

1. **Fusion.** The compiler sees the whole expression and decides where to cut it into kernels.
2. **Autodiff on graphs.** `gradient()` from [Chapter 8](08-autodiff.md) needs the forward graph as data.
   An eager library would have to record a tape; a lazy one already *is* the tape.
3. **Whole-step JIT caching.** A training step builds the same graph shape every iteration, so its
   compiled plan can be reused (`_structure_key` in `kernelize.py`).

The price is that errors and costs show up late. A shape mismatch is caught when you build the graph
(the UOp type checker runs at construction), but a slow kernel only shows up at `realize()`.

Here is a session, run with `DEBUG=1` so that every `realize` prints one line:

```text
>>> x = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
>>> y = (x * 2 + 1).relu().sum(1)
>>> y
<Tensor (2,) f32>
>>> y.uop.op, y.uop.src[0].op, y.uop.src[0].arg
Ops.REDUCE Ops.WHERE None
>>> y.numpy()
realize: 1 kernels, schedule 6.1 ms, run 0.1 ms
[ 9. 27.]
>>> y.uop.op, y.uop.arg
Ops.BUFFER ParamArg(slot=2, dtype=dtypes.f32, shape=(2,), addrspace=None)
```

Before `.numpy()`, `y` is a REDUCE over a WHERE (that is `relu`) over arithmetic over a BUFFER. Building
it ran no C at all. After `.numpy()` the tensor has been *rebound*: `y.uop` is now a BUFFER that holds
the result, so asking again costs nothing, and any graph built on `y` later starts from data rather than
recomputing the whole chain.

## 9.2 The Tensor class

The class starts with a constructor that accepts either a UOp (the internal path, used by every
operator) or anything NumPy can turn into an array (the user path):

@include uopc/tensor.py#Tensor.__init__

User data is copied into a fresh BUFFER UOp (`new_buffer` in `kernelize.py` attaches the NumPy array to
the UOp itself, so the memory lives exactly as long as the node does; see [Chapter 7](07-kernelize.md)).
BUFFERs are flat, so a 2-D array becomes `BUFFER(6).reshape((2, 3))`. dtypes are inferred the obvious
way: floats become `f32`, bools `bool`, everything else `i32`.

Realization is where the frontend hands over to the compiler:

@include uopc/tensor.py#Tensor.realize_all

Three details matter. First, tensors that are *already* data (a BUFFER, or a reshape of a whole BUFFER)
are skipped; realizing them would launch a useless copy kernel. Second, `realize_all` realizes many
tensors in *one* call. That is how an optimizer step updates every parameter, every moment estimate and
the loss in a single schedule: one structural cache lookup, one batch of kernels. Realizing them one by
one would work, but it would compute shared subexpressions (the gradients) once per tensor. Third, before
rebinding a tensor to its result BUFFER, `realize_all` keeps the graph it came from in `t._graph`, so that
`loss.item()` followed by `loss.backward(...)` still has a graph to differentiate (the Pitfalls tell how
the first version got this wrong).

`assign` is the only "mutation" in the library, and it is not really one:

@include uopc/tensor.py#Tensor.assign

It rebinds the Python object to a new graph. Buffers are never written in place by user code; a
parameter update builds `p - lr * g` and then makes `p` *point at* that expression. When the expression
is realized, `p.uop` becomes the new BUFFER and the old one is garbage-collected along with its UOp. This
functional style keeps autodiff honest (no node ever changes its value) and lets the kernelizer
reason purely about graphs.

## 9.3 Broadcasting

Elementwise ops in the IR are strict: `ADD` demands that both operands have the same shape. The type
checker from [Chapter 1](01-uops.md) says so, loudly. Users expect NumPy's *broadcasting* instead:
`(2, 3) + (3,)` adds the vector to every row. The rules are:

1. Line up the shapes from the right; pad the shorter one with leading 1s.
2. In each position the sizes must be equal, or one of them must be 1.
3. The result takes the larger size in each position.

@include uopc/tensor.py#_broadcast_shape

Turning a shape into a broadcast shape takes exactly two primitive movement ops: a RESHAPE to add the
leading 1s, and an EXPAND to stretch the 1s:

@include uopc/tensor.py#Tensor.broadcast_to

and every binary operator funnels through one helper:

@include uopc/tensor.py#Tensor._binop

So `a + b` with `a` of shape `(2, 3)` and `b` of shape `(3,)` becomes
`ADD(a, EXPAND(RESHAPE(b, (1, 3)), (2, 3)))`. In the REPL:

```text
>>> c = a + b
>>> c.uop.op, [s.op for s in c.uop.src]
Ops.ADD [Ops.RESHAPE, Ops.EXPAND]
>>> c.uop.src[1].op, c.uop.src[1].shape, c.uop.src[1].src[0].op, c.uop.src[1].src[0].shape
Ops.EXPAND (2, 3) Ops.RESHAPE (1, 3)
```

(`a` itself shows up as a RESHAPE because it is a flat BUFFER viewed as `(2, 3)`.) Does the EXPAND cost
anything? Not a byte. Rangeify ([Chapter 6](06-rangeify.md)) turns `EXPAND` into "use index 0 on the
stretched axis", so the generated C simply never looks at `ridx0` when it reads `b`:

```c
void e_2_3(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu1 = ((ridx0*3)+ridx1);
      float val2 = data0[alu1];
      float val3 = data1[ridx1];
      data2[alu1] = (val2+val3);
    }
  }
}
```

> **Key idea:** broadcasting is free because it is *indexing*, not copying. A broadcast operand is read
> through an index expression that ignores the broadcast loop. The same fact is what makes matmul,
> one-hot and argmax cheap to express below.

### The rest of the elementwise vocabulary

With `_binop` in hand, the arithmetic operators are one-liners. Several are *derived* rather than
primitive, because the IR deliberately has a small ALU (ADD, MUL, MAX, CMPLT, CMPNE, RECIP, EXP2, LOG2,
SQRT, SIN, WHERE, CAST):

@include uopc/tensor.py:129-153

The first line of `_binop` (shown above) is NumPy-style type promotion: an integer tensor combined with
a Python `float` is cast to `f32` first, so `Tensor([1, 2, 3]) * 0.5` is `[0.5, 1.0, 1.5]` (see the
Pitfalls for what happened without it). Subtraction is addition of a negation, division is multiplication by a reciprocal, `>` is `<` with the
operands swapped, `==` is "not `!=`". Why not add SUB and DIV to the IR? Because every op you add must be
handled by the renderer, the symbolic simplifier, the bounds analysis *and* autodiff. Deriving `a - b`
as `a + b * -1` costs nothing at runtime (gcc sees `x + y*-1.0f` and emits a subtraction) and saves four
rules in four files.

Transcendentals follow the same philosophy:

@include uopc/tensor.py:155-170

The only exponential in the IR is `EXP2`, because that is what hardware has (`exp2f` on CPUs,
`ex2.approx` on NVIDIA GPUs). `exp(x)` is `exp2(x * log2(e))` and `log(x)` is `log2(x) * ln 2`. You will
see the constant `1.4426950408889634f` (that is log2 e) all over the generated C in this chapter. `relu`
is a WHERE, `sigmoid` is built from `exp` and a reciprocal, and `tanh` from `sigmoid`. Every one of these
gets a gradient for free, because autodiff only needs rules for the primitives.

## 9.4 Movement ops

The movement methods are thin conveniences over the four primitive movement ops from
[Chapter 6](06-rangeify.md). `reshape` accepts a `-1` and works out the missing size, `permute` accepts
negative axes, `transpose` swaps two axes, `unsqueeze` inserts a 1, `flatten` merges trailing axes.
Indexing with ints and step-1 slices is a SHRINK followed by a RESHAPE that drops the integer-indexed axes:

@include uopc/tensor.py#Tensor.__getitem__

`pad` and `flip` call the composite constructions from `composite.py` (built from STACK, RESHAPE and
SHRINK; see [Chapter 6](06-rangeify.md)). Concatenation reuses them: each piece is zero-padded into its
place along the axis, and the padded pieces are added.

@include uopc/tensor.py#Tensor.cat

Adding zeros sounds wasteful, but after rangeify each padded piece becomes a bounds-checked WHERE, so
`cat` costs a few comparisons per element, not extra memory traffic.

## 9.5 Reductions over arbitrary axes

The IR's REDUCE has a deliberately narrow contract: it reduces the *last n* axes. Users want
`x.sum((0, 2))`. The bridge is a permutation: move the kept axes to the front, the reduced axes to the
back, then pop:

@include uopc/tensor.py#Tensor._reduce

For `x` of shape `(2, 3, 4)` and `axis=(0, 2)`, `keep = [1]`, so we permute by `(1, 0, 2)` to shape
`(3, 2, 4)` and REDUCE the last 2 axes, giving shape `(3,)`. With `keepdim=True` a reshape puts the 1s
back where the reduced axes were, which is what makes `x - x.max(-1, keepdim=True)` broadcast correctly.

```text
>>> x = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
>>> x.sum((0, 2)).numpy(), x.sum((0, 2)).uop.src[0].arg
[ 60.  92. 124.] (1, 0, 2)
```

The permutation, like the expand, is free. It only changes which loop variable multiplies which stride:

```c
void r_3(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      for (int ridx2 = 0; ridx2 < 4; ridx2++) {
        float val1 = acc0;
        float val2 = data0[(((ridx1*12)+(ridx0*4))+ridx2)];
        acc0 = (val1+val2);
      }
    }
    float val3 = acc0;
    data1[ridx0] = val3;
  }
}
```

The outer loop is the kept axis (stride 4 in memory), the two inner loops are the reduced axes (strides
12 and 1). The accumulator `acc0` lives inside the output loop and outside the reduce loops, exactly the
placement rule from [Chapter 6](06-rangeify.md).

The derived reductions are short:

@include uopc/tensor.py#Tensor.mean

@include uopc/tensor.py#Tensor.var

`mean` sums in `f32` (casting integer tensors first) and multiplies by `numel(result) / numel(input)`,
a Python float that becomes a CONST. `var` is the
mean squared deviation. `min` is a negated max of the negation (the IR has only ADD and MAX reductions).
`softmax`, `log_softmax` and `logsumexp` get their own section below.

## 9.6 Matrix multiplication is broadcast, multiply, reduce

Here is the most important three-line function in the library:

@include uopc/tensor.py#Tensor.matmul

Read it as geometry. A matrix product `C[i, j] = sum_k A[i, k] * B[k, j]` has three indices, so lay
the computation out on an `(M, N, K)` box. `A` has no `j` index, so it is reshaped to `(M, 1, K)` and
broadcast along N. `B` has no `i` index; transposed to `(N, K)` and reshaped to `(1, N, K)` it broadcasts
along M. Multiplying gives the full box of products `A[i, k] * B[k, j]`, and summing the last axis (K)
collapses the box onto the `(M, N)` face, which is `C`.

@diagram 09-matmul-broadcast

The box is never stored. It is a *description* of the iteration space, and rangeify turns it into a loop
nest in which each multiply happens once, right before it is accumulated. For a `(4, 8) @ (8, 3)` product
with the optimizer switched off (`OPT=0`), the graph and the code are:

```text
[('REDUCE', (4, 3), (Ops.ADD, 1)), ('MUL', (4, 3, 8), None), ('EXPAND', (4, 3, 8), (4, 3, 8)),
 ('RESHAPE', (4, 1, 8), (4, 1, 8)), ('RESHAPE', (4, 8), (4, 8))]
```

```c
void r_4_3(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 8; ridx2++) {
        float val1 = acc0;
        float val2 = data0[((ridx0*8)+ridx2)];
        float val3 = data1[((ridx2*3)+ridx1)];
        acc0 = (val1+(val2*val3));
      }
      float val4 = acc0;
      data2[((ridx0*3)+ridx1)] = val4;
    }
  }
}
```

That is the textbook triple loop, produced by a library function that never mentions a loop. (The first
line above follows `src[0]` down from the result; the max error against NumPy was `2.4e-07`.) Look at the
two loads in the inner loop. `A` is read at `ridx0*8 + ridx2`: consecutive `k` are adjacent in memory.
`B` is read at `ridx2*3 + ridx1`: consecutive `k` are a whole row apart. For a 3-wide matrix that does
not matter; for a 1024-wide one it is the difference between 1.9 and 8 GFLOP/s, and it is the first
thing [Chapter 10](10-cpu-performance.md) fixes.

Batched matmul falls out of the same code. All leading axes (`...`) ride along untouched, so
`(B, M, K) @ (B, K, N)` broadcasts to `(B, M, N, K)` and reduces K. The test
`test_ops[matmul_batched]` checks `a @ b.transpose()` on `(2, 3, 4)` tensors against NumPy.
`linear` is `x @ w + b`, with the bias broadcast across the batch.

## 9.7 Convolution as im2col of shrinks

A 2-D convolution with a `KH x KW` kernel computes, for every output pixel, a dot product between the
kernel and the `KH x KW` window of input under it, summed over input channels. The classic way to make
convolution a matrix multiply is **im2col**: gather every window into a row of a big "patches" matrix,
then multiply by the flattened weights. Libraries usually implement im2col as a copy. We get it from
movement ops alone:

@include uopc/tensor.py#Tensor.conv2d

For each kernel offset `(ky, kx)` there is a SHRINK of the (padded) input that selects the pixels that
offset touches for every output position; each has shape `(N, C, OH, OW)`. STACKing the `KH*KW` shrinks
gives `(KH*KW, N, C, OH, OW)`. Permuting to `(N, OH, OW, C, KH*KW)` and merging the last two axes gives
the patches matrix `(N, OH, OW, C*KH*KW)`, whose column order (channel-major, then `ky`, then `kx`)
matches `w.reshape(O, C*KH*KW)`. One matmul later, a permute puts channels back in NCHW order.

Strided convolution needs every `s`-th window. With no strided SHRINK in the IR, `_stride2d` pads the
window up to a multiple of `s`, reshapes `(OH*s)` into `(OH, s)`, and takes index 0 of the `s` axis:

@include uopc/tensor.py#Tensor._stride2d

Max pooling uses the same reshape trick without the pad: `(H, W)` becomes `(H/k, k, W/k, k)` and the two
`k` axes are max-reduced.

Is it right? A `(1, 2, 5, 5)` input, `(3, 2, 3, 3)` weights, `padding=1`, checked against a NumPy loop:

```text
(1, 3, 5, 5) max err 1.4305115e-06
```

Is it fast? Here is the reduce kernel, with `OPT=0`. It is worth staring at, because it is the honest
output of a compiler that has not yet learned anything about convolutions:

```c
void r_1_5_5_3(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    int alu1 = (ridx0*5);
    int alu2 = ((ridx0+5)%6);
    int alu3 = (alu2*5);
    int alu4 = ((ridx0+7)%6);
    int alu5 = (alu4*5);
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      int alu6 = (ridx1+5);
      int alu7 = (alu6%6);
      int alu8 = (alu7%5);
      int alu9 = ((ridx1+7)%6);
      int alu10 = (alu9%5);
      int alu11 = (ridx1+17);
      int alu12 = (alu11%6);
      int alu13 = ((ridx1+19)%6);
      int alu14 = (ridx1+29);
      int alu15 = (alu14%6);
      int alu16 = (alu15%5);
      int alu17 = ((ridx1+31)%6);
      int alu18 = (alu17%5);
      for (int ridx2 = 0; ridx2 < 3; ridx2++) {
        float acc0;
        acc0 = 0.0f;
        for (int ridx3 = 0; ridx3 < 18; ridx3++) {
          float val19 = acc0;
          int alu20 = (ridx3%9);
          int alu21 = (ridx3/9);
          int alu22 = (alu21*25);
          int alu23 = (alu22+((alu2%5)*5));
          float val24 = data0[(alu23+alu8)];
          float val25 = data0[(alu23+ridx1)];
          float val26 = data0[(alu23+alu10)];
          int alu27 = (alu22+alu1);
          float val28 = data0[(alu27+(alu12%5))];
          float val29 = data0[(alu27+ridx1)];
          float val30 = data0[(alu27+(alu13%5))];
          int alu31 = (alu22+((alu4%5)*5));
          float val32 = data0[(alu31+alu16)];
          float val33 = data0[(alu31+ridx1)];
          float val34 = data0[(alu31+alu18)];
          float val35 = data1[((((ridx2*18)+(alu21*9))+(((ridx3/3)%3)*3))+(ridx3%3))];
          acc0 = (val19+(((alu20<1)?(((alu6/6)<1)?0.0f:(((alu7/5)<1)?(((((alu1+alu8)+25)/30)<1)?0.0f:((((alu3+alu8)/25)<1)?val24:0.0f)):0.0f)):((alu20<2)?(((((alu1+ridx1)+25)/30)<1)?0.0f:((((alu3+ridx1)/25)<1)?val25:0.0f)):((alu20<3)?(((alu9/5)<1)?(((((alu1+alu10)+25)/30)<1)?0.0f:((((alu3+alu10)/25)<1)?val26:0.0f)):0.0f):((alu20<4)?((((alu11/6)%2)<1)?0.0f:(((alu12/5)<1)?val28:0.0f)):((alu20<5)?val29:((alu20<6)?(((alu13/5)<1)?val30:0.0f):((alu20<7)?((((alu14/6)%2)<1)?0.0f:(((alu15/5)<1)?((((alu5+alu16)/25)<1)?val32:0.0f):0.0f)):((alu20<8)?((((alu5+ridx1)/25)<1)?val33:0.0f):(((alu17/5)<1)?((((alu5+alu18)/25)<1)?val34:0.0f):0.0f)))))))))*val35));
        }
        float val36 = acc0;
        data2[(((ridx0*15)+(ridx1*3))+ridx2)] = val36;
      }
    }
  }
}
```

Three things are visible. The 18-long reduce loop is `C * KH * KW` flattened, and `ridx3 % 9` picks which
of the nine kernel taps this iteration belongs to: that is the STACK, lowered to a chain of WHERE
(`alu20<1 ? ... : alu20<2 ? ...`). The `%6`, `/30` and `/25` terms are the composite PAD, which
[Chapter 6](06-rangeify.md) builds from concatenation with zeros; each tap carries its own bounds check.
And all nine candidate loads are hoisted *above* the select, so every iteration performs nine loads to
use one. The kernel is correct and general, and roughly nine times more memory-hungry than it should be.
A convolution-aware compiler would make the tap index a real loop (or unroll it) instead of selecting
from a STACK; that is exercise 9.10.

## 9.8 Softmax, log-softmax and cross-entropy

Softmax turns a vector of scores into a probability distribution:
`softmax(x)_i = exp(x_i) / sum_j exp(x_j)`. Computed literally in `f32`, it overflows as soon as a score
exceeds about 88 (`exp(89)` is larger than the largest float). The cure is an identity: for any constant
`m`, `exp(x_i - m) / sum_j exp(x_j - m)` equals `exp(x_i) / sum_j exp(x_j)`, because the `exp(-m)` factors
cancel. Choosing `m = max_j x_j` makes every exponent at most 0, so nothing overflows, and the largest
term is exactly 1, so the sum is at least 1 and nothing underflows to a 0/0.

Classification losses want the *log* of the probability of the right class. Taking the log of a
softmax is numerically poor (a tiny probability rounds to 0 and its log to `-inf`), so we compute it
directly:

`log_softmax(x)_i = (x_i - m) - log(sum_j exp(x_j - m))`.

@include uopc/tensor.py#Tensor.softmax

@include uopc/tensor.py#Tensor.log_softmax

@include uopc/tensor.py#Tensor.logsumexp

Cross-entropy for a batch of `B` examples with integer labels `y_b` is
`L = -(1/B) * sum_b log_softmax(z_b)[y_b]`. Picking "the entry at index `y_b`" is a gather, and gathers
with data-dependent indices are exactly what the IR cannot differentiate (`_rule` for INDEX refuses).
So we turn the label into a **one-hot** row and use a multiply-and-sum, which is an ordinary reduce:

@include uopc/tensor.py#Tensor.one_hot

@include uopc/tensor.py#Tensor.cross_entropy

`one_hot` is itself a broadcast: labels of shape `(B,)` become `(B, 1)`, `arange(n)` has shape `(n,)`, and
comparing them broadcasts to `(B, n)` with a 1 exactly where the column equals the label:

```text
>>> Tensor(np.array([2, 0, 3], np.int32)).one_hot(4).numpy()
[[0. 0. 1. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]]
>>> logits.cross_entropy(labels).item(), numpy_reference
0.31853973865509033 0.3185397696491856
```

How many kernels does `softmax` become? REDUCE is a materialization point ([Chapter 7](07-kernelize.md)),
so: one kernel for the row max, one for the row sum of exponentials (which reads the max), and one
elementwise kernel for the output. The test suite pins this down:

@include tests/test_w05_tensor.py#test_kernelize_fuses_elementwise_and_splits_reductions

Here are the three kernels for a `(4, 8)` softmax, `OPT=0`:

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

Notice that `exp2f(...)` appears in both the second and the third kernel. The exponentials are not
materialized; they are *recomputed* in the consumer, because the kernelizer only cuts at reductions and
roots. Recomputing one `exp2f` per element is cheaper than writing and re-reading a temporary, so here
fusion wins. When the recomputation is a whole matmul instead of one `exp2f`, the answer flips, and that
trade-off is the subject of [Chapter 11](11-fusion-flash.md).

## 9.9 argmax without an argmax op

To measure accuracy we need the index of the largest logit. The IR has MAX-reduce but no ARGMAX, and
adding one would mean a reduction that carries two values (the best value and its index), touching
rangeify and the renderer. Instead, a trick with ops we already have:

@include uopc/tensor.py#Tensor.argmax

Step by step, for a row of length `n`:

1. `is_max` is a 0/1 mask: where does the row equal its maximum?
2. `is_max.where(n - idx, 0)` puts `n - idx` at every maximal position and 0 elsewhere. Since
   `idx < n`, the maximal positions all get positive values, and the *smallest* index gets the
   *largest* value.
3. MAX-reducing picks `n - first_idx`, and `* -1 + n` turns it back into `first_idx`.

So ties go to the first occurrence, matching NumPy:

```text
>>> x = Tensor(np.array([[0.1, 3.0, 3.0, -1.0], [2.0, 0.0, 5.0, 5.5]], np.float32))
>>> x.argmax(-1).numpy(), x.argmax(-1).dtype
[1 3] dtypes.i32
```

and it compiles to two reductions and a tiny epilogue (`OPT=0`):

```c
void r_2(float* restrict data0, float* restrict data1, int* restrict data2, int* restrict data3) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    int acc0;
    acc0 = (-2147483647-1);
    float val1 = data1[ridx0];
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int val2 = acc0;
      float val3 = data0[((ridx0*4)+ridx1)];
      int val4 = data2[ridx1];
      acc0 = ((val2>(((val3!=val1)!=1)?((val4*-1)+4):0))?val2:(((val3!=val1)!=1)?((val4*-1)+4):0));
    }
    int val5 = acc0;
    data3[ridx0] = val5;
  }
}
```

The `int` accumulator starting at `INT_MIN` is the `identity()` of an integer MAX-reduce from rangeify,
and `(val3!=val1)!=1` is how `eq` renders: "not not-equal". `arange(n)` is a real 4-element BUFFER
(`data2`) here; a slightly smarter frontend would make it a RANGE-derived value and save the load.

## 9.10 Gradients

The Tensor-level autodiff entry point is short because [Chapter 8](08-autodiff.md) did the work:

@include uopc/tensor.py#Tensor.backward

`backward` takes the parameter list explicitly, calls `gradient(loss, [p.uop ...])`, and stores each
result as a *lazy* Tensor in `p.grad`. Nothing is computed. Gradients are accumulated (`p.grad + g`) if
`backward` is called twice, as in PyTorch. Continuing the session from 9.1:

```text
>>> l = Linear(3, 4)
realize: 1 kernels, schedule 1.0 ms, run 0.1 ms
>>> [p.shape for p in parameters(l)]
[(3, 4), (4,)]
>>> loss = l(x).square().mean(); loss.backward(parameters(l))
>>> l.weight.grad, l.bias.grad
<Tensor (3, 4) f32> <Tensor (4,) f32>
>>> l.bias.grad.numpy()
realize: 4 kernels, schedule 24.1 ms, run 0.2 ms
[ 0.65893304  1.2755529   0.39464036 -1.0922788 ]
```

(The one kernel during `Linear(3, 4)` is the zero-initialized bias being realized into a buffer.) The
gradient tests compare every interesting op against central finite differences in float64, with the same
`numgrad` helper you met in [Chapter 8](08-autodiff.md):

@include tests/test_w05_tensor.py#test_gradients

## 9.11 Layers

`nn.py` is written against the Tensor API only; it never touches a UOp. A layer is a Python object that
holds parameter Tensors (created with `requires_grad=True`) and has a `__call__`:

@include uopc/nn.py#Linear

The weight is stored as `(n_in, n_out)` so that `x @ w` needs no transpose in the user's head (the
compiler will decide where a transpose is worth making; see [Chapter 10](10-cpu-performance.md)). The
initialization is PyTorch's default for linear layers: uniform in `+-1/sqrt(n_in)`, which keeps the
variance of activations roughly constant from layer to layer. The bias starts at zero and is realized
immediately, so it is a BUFFER rather than an EXPAND-of-a-CONST; otherwise "the bias" would be a constant
baked into every kernel and its gradient would have nowhere to go.

`Conv2d` is the same pattern around `conv2d`, `LayerNorm` normalizes the last axis
(`(x - mean) / sqrt(var + eps) * weight + bias`), and `Embedding` is a one-hot matmul:

@include uopc/nn.py#Embedding

A real embedding layer is a gather (`weight[idx]`), and its gradient is a scatter-add. Our IR has neither,
and autodiff refuses data-dependent INDEX. The one-hot product computes the same thing with ops autodiff
understands: `one_hot(idx) @ W` selects rows, and its gradient `one_hot^T @ dY` scatters into them. It
costs `vocab` times more FLOPs than a gather, which is fine for a character-level GPT
([Chapter 14](14-gpt.md)) and terrible for a 50,000-token vocabulary.

Finally, the optimizer needs a list of parameters. Rather than make every layer register its
parameters, `parameters()` walks Python attributes, lists and dicts and collects every Tensor with
`requires_grad`:

@include uopc/nn.py#parameters

The `seen` set handles shared sub-objects (and cycles). The order is the attribute order of each object,
which is deterministic, so the optimizer's state lines up with the same parameters every run.

## 9.12 Optimizers

An optimizer turns gradients into parameter updates. Every update in this library is built as a graph
and realized in one batch, together with any extra tensors the caller wants (the loss).

### SGD with momentum

Plain stochastic gradient descent is `p <- p - lr * g`. Momentum keeps a velocity `v <- mu * v + g` and
steps along it, `p <- p - lr * v`, which smooths noisy gradients and speeds progress along consistent
directions:

@include uopc/nn.py#SGD

The test checks the arithmetic by hand for the loss `sum(w^2)` (gradient `2w`):

@include tests/test_w05_tensor.py#test_sgd_momentum_step

### AdamW

Adam keeps two running averages per parameter: the first moment `m` (mean of the gradient) and the
second moment `v` (mean of its square). With `t` counting steps from 1:

- `m <- b1 * m + (1 - b1) * g`
- `v <- b2 * v + (1 - b2) * g^2`
- `m_hat = m / (1 - b1^t)`, `v_hat = v / (1 - b2^t)` (bias correction: both start at 0, so early averages
  are too small; dividing by `1 - b^t` undoes that exactly for a constant gradient)
- `p <- p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)`

Dividing by `sqrt(v_hat)` gives every parameter its own step size: parameters with large, noisy gradients
take smaller steps. The `(1 - lr * wd)` factor is *decoupled* weight decay, the "W" in AdamW: it shrinks
weights directly instead of adding `wd * p` to the gradient, which in Adam would get divided by
`sqrt(v_hat)` and lose its meaning.

@include uopc/nn.py#AdamW

The only surprising lines are `c1` and `c2`. The bias corrections depend on `t`, which changes every step.
Written as Python floats they would become CONST nodes in the graph, the graph would be *structurally
different* every step, and the structural JIT cache would miss every time. Passed as one-element tensors
they are BUFFERs, whose contents do not participate in the cache key. Here is the difference, measured
by a scratch subclass `AdamWFloats` whose `step` is identical except that `c1, c2` are floats (time per
training step of the MNIST MLP, in ms, first eight steps):

```text
AdamW        plans cached  5   step ms: 409 14 16 16 12 11 11 13
AdamWFloats  plans cached 12   step ms: 228 169 187 190 212 179 164 188
```

Same math, same results, more than ten times slower: every step misses the plan cache, so it is
re-kernelized, re-rangeified and re-rendered in Python. (In this run gcc did not have to recompile: the
`.so` files for these exact constants were already on disk from an earlier run of the same script. On a
truly cold cache each step also pays gcc for the update kernels, and an earlier run measured 600-800 ms
per step.) This is pitfall 17 in
`book/NOTES.md`, and the general lesson is: **anything that changes between iterations must be data, not
structure.**

## 9.13 Training MNIST

MNIST is 70,000 grayscale 28x28 images of handwritten digits (60,000 train, 10,000 test). The loader
downloads the four gzip files once into `~/.cache/uopc/data` and returns flat `float32` images scaled to
`[0, 1]` and `int32` labels:

@include examples/datasets.py#mnist

The model is a two-layer perceptron, 784 inputs, 128 hidden units with ReLU, 10 outputs, trained with
AdamW on random minibatches of 128:

@include examples/mnist_mlp.py

A training step is five lines: build the forward graph (`model(x).cross_entropy(y)`), clear the old
gradients, build the backward graph, and let the optimizer realize the new parameters, moments and loss
in one batch. Evaluation runs the model on the test set in chunks of 1000 and compares `argmax` with the
labels. The real output, from a run with the current code (`book/outputs/mnist.txt` holds an earlier
capture with identical losses and slightly slower steps: 460.7 ms for step 0 and 14.4 ms/step at the end):

```text
step    0  loss 2.3134  284.8 ms/step
step  100  loss 0.3881  14.2 ms/step
step  200  loss 0.3272  13.1 ms/step
step  300  loss 0.2496  12.4 ms/step
step  400  loss 0.2643  11.9 ms/step
step  500  loss 0.2517  11.6 ms/step
step  599  loss 0.2162  11.6 ms/step
test accuracy 94.06%
```

(The ms/step column is a running average since step 0, so the expensive first step inflates the early
rows.) The initial loss 2.31 is about `ln 10 = 2.30`, the loss of a uniform guess over ten classes, which
is a good sanity check that initialization is sane. After 600 steps (about 1.3 epochs) the test accuracy
is 94.06%. A longer run, a learning-rate schedule or a CNN would do better; the milestone is that a model
written in *your* compiler learns.

**Step 0 versus the rest.** The first step takes about 300 ms because it plans and renders 32 kernels (and,
on a cold cache, runs gcc 32 times: with an empty `UOPC_CACHE` the same step took 3.7 s). Every later step
has the same graph *structure*: same ops, same shapes, different buffers. `_structure_key`
([Chapter 7](07-kernelize.md)) maps it to the cached plan, so the steady-state step only builds graphs,
hashes them, and calls 32 already-loaded C functions.

## 9.14 Anatomy of one training step

`DEBUG=2` prints every kernel with its wall time and a rough GFLOP/s. Here is the second training step
(the first warm one) of the MLP above, from a run made while this chapter was being written (a rerun
with the current code produces the same 32 kernels in the same order; individual times move by tens of
percent from run to run):

```text
  e_1_128_784                       0.279 ms     0.00 GFLOP/s
  r_128_128                         0.431 ms    59.65 GFLOP/s
  r_128                             0.047 ms     0.19 GFLOP/s
  e_128_128                         0.027 ms     1.84 GFLOP/s
  e_1_10_128                        0.013 ms     0.00 GFLOP/s
  r_128_10                          0.180 ms     1.82 GFLOP/s
  r_128                             0.051 ms     0.05 GFLOP/s
  r_128                             0.030 ms     0.26 GFLOP/s
  r_128                             0.041 ms     0.65 GFLOP/s
  r_128                             0.015 ms     0.43 GFLOP/s
  e_128_10                          0.048 ms     0.75 GFLOP/s
  r_128_128                         0.132 ms     2.48 GFLOP/s
  e_128_128                         0.052 ms     0.95 GFLOP/s
  r_128_784                         0.626 ms    41.01 GFLOP/s
  e_784_128                         0.375 ms     0.80 GFLOP/s
  e_784_128                         0.484 ms     0.83 GFLOP/s
  e_784_128                         0.597 ms     1.51 GFLOP/s
  r_128                             0.052 ms     0.32 GFLOP/s
  e_128                             0.035 ms     0.01 GFLOP/s
  e_128                             0.015 ms     0.03 GFLOP/s
  e_128                             0.025 ms     0.05 GFLOP/s
  r_10_128                          0.187 ms     1.75 GFLOP/s
  e_128_10                          0.038 ms     0.10 GFLOP/s
  e_128_10                          0.017 ms     0.30 GFLOP/s
  e_128_10                          0.029 ms     0.40 GFLOP/s
  r_10                              0.010 ms     0.13 GFLOP/s
  e_10                              0.014 ms     0.00 GFLOP/s
  e_10                              0.013 ms     0.00 GFLOP/s
  e_10                              0.023 ms     0.00 GFLOP/s
  r_128                             0.038 ms     0.41 GFLOP/s
  r_1                               0.010 ms     0.01 GFLOP/s
  e_1                               0.015 ms     0.00 GFLOP/s
realize: 32 kernels, schedule 1.6 ms, run 4.4 ms
```

Kernel names encode the output shape: `r_` kernels contain a reduction, `e_` kernels are purely
elementwise. By reading the C (`DEBUG=4`) you can assign every kernel to a role:

@diagram 09-training-step

Summed, the kernels take 3.95 ms; the `run 4.4 ms` in the `realize` line also includes the Python and
ctypes cost of 32 calls. The six kernels that touch `W1`, the 784x128 matrix that holds 99% of the
parameters, account for 2.79 ms of it, and the twelve optimizer kernels together cost more than the
whole forward pass.

**Forward (7 kernels).** `e_1_128_784` copies `W1` into a transposed temporary and `r_128_128` is `x @ W1`;
the copy is the layout pass from [Chapter 10](10-cpu-performance.md), which makes the reduce axis
contiguous for both operands. `e_128_128` is `relu(h + b1)`, materialized by `stage_broadcast_compute`
([Chapter 7](07-kernelize.md)) because it is broadcast into the next matmul and would otherwise be
recomputed for every output column. `e_1_10_128` and `r_128_10` are the copy-and-matmul pair for `W2`.
Two `r_128` kernels compute the row max and the sum of exponentials of `log_softmax`.

**Backward (10 kernels).** Four small kernels for the gradient of cross-entropy with respect to the
logits (the last of them, `e_128_10`, writes `dL/dlogits`), `r_128_128` and `e_128_128` for `dL/dh`
through `W2` and the ReLU, `r_128_784` for `dW1`, `r_10_128` for `dW2`, and two small reductions over the
batch for the bias gradients. The scheduler is free to interleave phases: the third kernel of the step
(`r_128` reading the labels) is already a backward kernel, because it depends only on the labels. The ReLU backward kernel is a one-liner:

```c
void e_128_128(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  for (int ridx0 = 0; ridx0 < 128; ridx0++) {
    for (int ridx1 = 0; ridx1 < 128; ridx1++) {
      int alu1 = ((ridx0*128)+ridx1);
      float val2 = data0[alu1];
      float val3 = data1[ridx1];
      float val4 = data2[alu1];
      data3[alu1] = ((0.0f<(val2+val3))?val4:0.0f);
    }
  }
}
```

It recomputes `h + b1 > 0` from the pre-activation instead of storing a mask: that is the WHERE rule from
`autodiff.py` (`s[0].where(g, 0)`) fused with the forward expression that produced the condition.

**Update (12 kernels).** Three per parameter: new `m`, new `v`, new `p`. They are separate kernels because
`m`, `v` and `p` are all roots of the `realize_all` call, and roots are materialization points. The `W1`
update reads the freshly written `m` and `v`:

```c
void e_784_128(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3, float* restrict data4, float* restrict data5) {
  float val1 = data2[0];
  float val2 = data4[0];
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 784; ridx0++) {
    for (int ridx1 = 0; ridx1 < 128; ridx1++) {
      int alu3 = ((ridx0*128)+ridx1);
      float val4 = data0[alu3];
      float val5 = data1[alu3];
      float val6 = data3[alu3];
      data5[alu3] = (val4+(((val5*val1)*(1.0f/(sqrtf((val6*val2))+1e-08f)))*-1.0f));
    }
  }
}
```

`data2[0]` and `data4[0]` are `c1` and `c2`: one-element BUFFERs, loaded once, outside the loop. With
`wd = 0` the decay factor `(1 - lr * wd)` folded to 1 and vanished.

**Loss (3 kernels).** Per-example cross-entropy, the sum over the batch, and the final scale by `-1/128`.

Two observations are worth making before moving on. First, two of the backward kernels (a sum over
classes and a tie count) and part of a third compute the gradient that flows through the `max` in
`log_softmax`. Mathematically that
gradient is exactly zero: shifting all logits by a constant does not change `log_softmax`, so the
derivative with respect to the shift vanishes. Autodiff does not know that, and faithfully computes a
zero the expensive way. Second, the `m` and `v` updates for `W1` read the gradient at
`data1[((ridx1*784)+ridx0)]`: the gradient is produced as `(128, 784)` (that is `dW1` transposed) and
read with a stride of 784. Both are opportunities, not bugs.

## 9.15 What is slow, and why

The steady-state step costs about 12 ms (the last line of the run in 9.13), and the 32 kernels
account for roughly 4 to 6 ms of it. Where does the rest go? A scratch script that times the three phases
of 50 warm steps (graph building for the forward pass, `backward`, and `opt.step`) printed:

```text
forward graph                  1.87 ms/step
backward graph                 1.71 ms/step
step (structure key+run)       9.81 ms/step
```

and `cProfile` over 50 warm steps (the profiler inflates everything, but the ranking is what matters):

```text
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1600    0.168    0.000    0.198    0.000 uopc/runtime.py:53(__call__)
      150    0.080    0.001    0.157    0.001 uopc/uop.py:131(toposort)
62950/32100    0.076    0.000    0.126    0.000 uopc/uop.py:100(_argkey)
    11750    0.075    0.000    0.125    0.000 uopc/uop.py:230(_infer)
    18650    0.050    0.000    0.326    0.000 uopc/uop.py:111(__new__)
       50    0.037    0.001    0.190    0.004 uopc/kernelize.py:180(_structure_key)
```

`Program.__call__` is the 32 C calls per step (the kernels themselves plus ctypes argument marshalling).
Everything else is Python: about 370 UOp constructions per step (18,650 over 50 steps) (each one hash-consed and type-checked by
`_infer`), topologically sorting the step's graph, and hashing its structure for the plan cache. This is
the tax of a Python compiler that rebuilds the graph every step. NOTES 22 tells how it was once much worse:
`UOp.__eq__`/`__hash__` written in Python, and `Enum.__hash__` being a Python function, made the GPT
compile take 163 s for 12 steps until identity hashing (in C) cut it to 47 s. Production systems avoid the
rebuild entirely by capturing the step once and replaying it (tinygrad's `TinyJit`, CUDA graphs).

What about the kernels themselves? The two big matmuls run at 41-60 GFLOP/s, the elementwise kernels at
around 1 GFLOP/s or less. For elementwise work that is expected: they are bound by memory and by fixed
per-call overhead, as [Chapter 10](10-cpu-performance.md) will quantify with the roofline model. Running
the whole training with the optimizer switched off (`OPT=0`, `book/outputs/mnist_opt0.txt`) gives the same
losses digit for digit and 17.2 ms/step instead of 14.4 (both from the earlier captures): on a model this small the Python overhead hides
most of the kernel speedup. On the 1024x1024 GEMM of the next chapter it is the other way around, and the
optimizer is worth a factor of more than 50.

## 9.16 Building it yourself: the week-5 milestone

If you are implementing this for the course, here is an order that keeps every step testable:

1. `Tensor.__init__`, `numpy()`, `realize()` on top of your `realize` from [Chapter 7](07-kernelize.md).
   Test: `Tensor(a).numpy() == a`.
2. `_broadcast_shape`, `broadcast_to`, `_binop`, and the arithmetic dunders. Test against NumPy on
   random shapes, including `(2, 3, 4) + (4,)`, `(2, 1, 4) * (3, 1)` and scalar operands on both sides.
3. Movement methods and `__getitem__`. Test chains of them against NumPy (`test_ops[getitem_cat]`).
4. `_reduce` with arbitrary axes and `keepdim`; `mean`, `var`, `max`, `min`. Test `sum((0, 2))`.
5. `matmul`, then `softmax`/`log_softmax` (check `softmax(x + 1000)` does not produce NaN), `one_hot`,
   `cross_entropy`, `argmax` (check ties).
6. `backward`, then gradient-check every op you wrote with finite differences.
7. `Linear`, `parameters`, `SGD`; overfit 64 random points (`test_training_reduces_loss`). Only then
   `AdamW` and the real dataset.
8. MNIST. Watch the first loss (should be near 2.30) and the ms/step after step 0 (should drop by more
   than an order of magnitude; if it does not, your JIT cache is missing).

## Pitfalls

**A changing Python constant defeats the JIT cache (NOTES 17).** Shown in 9.12: AdamW's bias corrections
as floats made every step more than ten times slower. The symptom is "every step is as slow as the first". Check with
`DEBUG=1`: if the `schedule` time does not drop to about a millisecond after step 0, something structural
changes between steps. Learning-rate schedules, dropout masks and step counters must all be tensors.

**Forgetting to realize optimizer state.** `realize_all(*news, *self.m, *self.v, *extra)` is not
decoration. A scratch `LeakyAdamW` that realizes only the parameters still trains correctly, because the
new `m` and `v` get computed inside the parameter kernels. But `m` and `v` themselves stay lazy, so each
step's graph contains the previous step's graph:

```text
step 0:     385 ms   graph behind m[0]:  134 nodes
step 1:     263 ms   graph behind m[0]:  235 nodes
step 2:     517 ms   graph behind m[0]:  336 nodes
step 3:     616 ms   graph behind m[0]:  437 nodes
step 4:     771 ms   graph behind m[0]:  538 nodes
step 5:    1102 ms   graph behind m[0]:  639 nodes
```

The graph grows by 101 nodes per step, the structure never repeats, and every step recompiles a bigger
program. Any state that outlives a step must be realized every step.

**Python scalars took the tensor's dtype.** The first version of `_binop` built every Python scalar with
`self.dtype`, so `Tensor([1, 2, 3]) * 0.5` multiplied by `int(0.5) == 0`, and `Tensor([1, 2, 3]).mean()`
was `0`: NumPy promotes to float, we silently truncated. The fix is the promotion line at the top of
`_binop` (an int tensor meeting a Python float is cast to `f32`) and a `mean` that always sums in `f32`.
Now:

```text
>>> t = Tensor([1, 2, 3])
>>> (t * 0.5).numpy(), t.mean().item()
[0.5 1.  1.5] 2.0
```

@include tests/test_w05_tensor.py#test_python_float_promotes_int_tensors

**A kernel whose input simplified away overwrote that input.** Found while writing this chapter. In
`make_plan`, a kernel's argument list (`refs`) was every BUFFER of its *shaped* body, collected before
lowering. The renderer, however, emits a C parameter only for buffers the *simplified* program still
reads. When simplification deletes a read entirely, the two lists disagree and every pointer shifts by
one. `t * 0` on an integer tensor folds to the constant 0, the kernel's signature becomes
`void e_3(int* restrict data1)`, and ctypes passed `t`'s own buffer as `data1`: the first version printed
`[0 0 0] [0 0 0]` for `(t * 0).numpy(), t.numpy()`, writing the "output" into the input. (Floats escape,
because `x * 0.0` is not folded: it is not `0.0` for NaN or infinity.) The fix keeps only the slots that
survive in the lowered program:

@include uopc/kernelize.py:159-163

The generated kernel still has one parameter, but it is now bound to the right buffer:

```c
void e_3(int* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    data1[ridx0] = 0;
  }
}
```

```text
>>> (t * 0).numpy(), t.numpy()
[0 0 0] [1 2 3]
```

@include tests/test_w05_tensor.py#test_deleted_reads_do_not_shift_kernel_arguments

The general lesson: whenever two passes each compute "the list of parameters" independently, they will
eventually disagree. Derive one from the other.

**Unstable softmax.** If you write `x.exp() / x.exp().sum(-1, keepdim=True)`, it works on small random
inputs and returns NaN on real logits above about 88. Always subtract the max; test with `x + 1000`.

**Fusion can recompute a lot (NOTES 15).** Fusion recomputes anything elementwise that is not a
materialization point. In softmax that is one `exp2f`, which is cheap. In a transformer, a LayerNorm output
feeding a matmul gets recomputed for every output column: a GPT step spent 374 ms in kernels until
`stage_broadcast_compute` ([Chapter 7](07-kernelize.md)) started staging such operands, bringing it to
125 ms. If one kernel in `DEBUG=2` output is mysteriously slow, read its C and look for an expensive
expression inside the reduce loop.

**Reading the loss before `backward` used to zero the gradients.** `.item()` realizes the loss, and
realizing *rebinds* `loss.uop` to the BUFFER holding the result (9.1). In the first version that was the
end of the story: the graph that connected the loss to the parameters was gone, `gradient()` found no
path and returned zeros, with no error and no warning:

```text
>>> w = Tensor(np.array([1.0, -2.0], np.float32), requires_grad=True)
>>> loss = w.square().sum()
>>> loss.item()
5.0
>>> loss.backward([w]); w.grad.numpy()
[0. 0.]
```

The fix is the `t._graph = t.uop` line in `realize_all` (9.2): a realized tensor remembers the graph it
came from, and `backward` differentiates that graph if it exists:

@include uopc/tensor.py#Tensor.backward

The same session now prints `5.0` and then `[ 2. -4.]`. It is still good practice to read the loss
after `opt.step(loss)`, as `mnist_mlp.py` does, so the loss is computed in the same batch of kernels.

**`detach()` used to be a no-op for gradients.** The first `detach` returned a new Tensor around the
*same* UOp, and `gradient()` follows UOps, not Tensor objects, so the gradient flowed straight through.
Now `detach` wraps the value in a marker node:

@include uopc/tensor.py#Tensor.detach

Autodiff's rule for STAGE passes the gradient through unchanged, except when the STAGE's `arg` is
`"detach"`, where it returns no gradient. The kernelizer removes the marker before anything else, so it
costs nothing at run time:

@include uopc/kernelize.py#strip_detach

Both fixes are pinned by one test: realizing before `backward` keeps the gradient `2w`, and
`(w.detach() * w).sum()` has gradient `w`, not `2w`:

@include tests/test_w05_tensor.py#test_item_before_backward_and_detach

## Exercises

**9.1** ★ Predict how many kernels `(x - x.mean(-1, keepdim=True)).square().sum()` produces for `x` of shape
`(64, 64)`, then check with `make_plan`. *Hint:* count REDUCEs.

**9.2** ★ What does `Tensor(np.array([[3., 1., 3.]], np.float32)).argmax(-1).numpy()` print, and why? What
would happen if the trick used `idx` instead of `n - idx`?

**9.3** ★ Implement `Tensor.std(axis, keepdim)` and `Tensor.logsumexp`-based `cross_entropy` (use
`logsumexp` instead of `log_softmax`). Gradient-check both.

**9.4** ★★ Implement `Tensor.gather(axis, idx)` for integer index tensors using `one_hot` and a reduce, so
that it is differentiable with respect to the data. What does it cost in FLOPs compared to a real gather?

**9.5** ★★ Add a learning-rate schedule (linear warmup, then cosine decay) to `AdamW` without causing JIT
cache misses. Verify with `DEBUG=1` that `schedule` stays around a millisecond after step 0.

**9.6** ★★ Revert the `alive` filter in `make_plan` (in a scratch copy), watch
`test_deleted_reads_do_not_shift_kernel_arguments` fail, then fix it the *other* way: render every buffer
of the body as a C parameter even when it is unused. Which fix is better, and why? *Hint:* think about
the structural cache and about identical kernels rendering identical C.

**9.7** ★★ Implement dropout: `x.dropout(p)` in training mode multiplies by a random 0/(1/(1-p)) mask.
Where must the randomness come from so that the graph structure is the same every step?

**9.8** ★★ Train a small CNN on MNIST (`Conv2d(1, 8, 3)`, ReLU, `max_pool2d`, `Linear`). Report accuracy
and ms/step, and identify the slowest kernel with `DEBUG=2`. Explain its cost from its C source.

**9.9** ★★ Use `detach()` to stop the gradient through the max in `log_softmax` (it is mathematically
zero, see 9.14). Count the kernels per training step before and after, and check that the losses are
unchanged to the printed precision.

**9.10** ★★★ Make `conv2d` fast: instead of a STACK of shrinks selected with a WHERE chain, express the
kernel taps as extra reduce axes (shape `(N, O, OH, OW, C, KH, KW)` reduced over the last three). Compare
the generated C and the speed against 9.7's kernel on a `(64, 16, 28, 28)` input.

<details>
<summary>Solutions to the ★ exercises</summary>

**9.1** Two: `make_plan` reports `['r_64', 'r_1']`. The `mean` REDUCE is one materialization point
(`r_64`; its `* (1/64)` is elementwise and fuses into its consumer). The subtraction and the square are
elementwise and fuse into the final `sum()`, which is both a REDUCE and the root, so it is a single
point (`r_1`). Count materialization points, and remember that a root which is itself a REDUCE counts
once.

**9.2** `[0]`. The row max is 3, `is_max = [1, 0, 1]`, `where(is_max, 3 - idx, 0) = [3, 0, 1]`, max is 3,
and `3 * -1 + 3 = 0`. With `idx` instead of `n - idx` the max would pick the *last* maximal index (2), and
the zero for non-maximal positions would collide with index 0.

**9.3** `std = var.sqrt()`. For cross-entropy:
`(self.logsumexp(-1).reshape(-1) - (self * labels.one_hot(n)).sum(-1)).mean()`. Both match finite
differences to about `5e-8`. (With the first version of `realize_all`, reading the loss with `.item()`
before `backward` made this check fail with zero gradients; see the Pitfalls.)

</details>

## Quiz

1. Why is broadcasting free in this compiler, while in an eager library `np.broadcast_to(b, shape).copy()`
   costs memory traffic?
2. `REDUCE` only reduces the last `n` axes. How does `x.sum(0)` on a `(4, 5)` tensor become a valid graph?
3. Write `matmul` for `(M, K) @ (K, N)` as a sequence of movement ops, one MUL and one REDUCE, with the
   shape after each step.
4. Why does `log_softmax` subtract the row max, and why does doing so not change the result?
5. Why are AdamW's bias corrections passed as one-element tensors rather than Python floats?
6. How many kernels does a softmax over the last axis produce, and which values are recomputed rather
   than stored?
7. Why does `Embedding` use a one-hot matmul rather than indexing?
8. After `y.numpy()`, what is `y.uop`, and why does that matter for graphs built on `y` later?

<details>
<summary>Answers</summary>

1. EXPAND is lowered to an index expression that ignores the broadcast loop (index 0 on the stretched
   axis); no data is copied, the same element is simply re-read.
2. `_reduce` permutes the reduced axis to the end (`permute(1, 0)`, shape `(5, 4)`) and REDUCEs 1 axis.
3. `A.reshape(M, 1, K).expand(M, N, K)`, `B.permute(1, 0).reshape(1, N, K).expand(M, N, K)`, MUL gives
   `(M, N, K)`, `REDUCE(ADD, 1)` gives `(M, N)`.
4. To keep every exponent at most 0 (no overflow) with the largest term exactly 1 (no 0/0). The constant
   cancels between numerator and denominator (or, in log space, between the two terms).
5. They change every step; as CONSTs they would change the graph structure and miss the JIT cache, so
   every step would recompile.
6. Three: max, sum of exponentials, and the elementwise output. The exponentials are recomputed in the
   sum kernel and again in the output kernel.
7. Autodiff cannot differentiate a data-dependent INDEX (its gradient is a scatter). One-hot times
   weights is an ordinary broadcast-multiply-reduce whose gradient is another matmul.
8. A BUFFER (or a reshape of one) holding the result. Later graphs start from that data instead of
   re-including, and recomputing, the whole chain that produced it.

</details>

## Further reading

- NumPy documentation, "Broadcasting" (`numpy.org/doc/stable/user/basics.broadcasting.html`): the rules
  this chapter implements.
- D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015; I. Loshchilov and
  F. Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 (AdamW).
- K. Chellapilla, S. Puri, P. Simard, "High Performance Convolutional Neural Networks for Document
  Processing", 2006: the im2col formulation of convolution.
- Y. LeCun, C. Cortes, C. Burges, "The MNIST database of handwritten digits".
- tinygrad, `tinygrad/tensor.py`: a production-sized version of this file, with the same lazy design.
- A. Karpathy, *micrograd*: the smallest possible autodiff engine, a good contrast to graph-level autodiff.
