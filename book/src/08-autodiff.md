# 8. Automatic differentiation

Training a neural network means nudging every parameter in the direction that lowers the loss. For
that you need the gradient: one partial derivative of a scalar loss with respect to each of
(potentially) millions of parameters, every step. Deriving those by hand is out of the question,
and estimating them numerically costs one or two extra evaluations of the whole model *per
parameter*. **Reverse-mode automatic differentiation** computes all of them for a small constant
multiple of the cost of computing the loss once. In `uopc` it is a remarkably short graph
transformation: given the shape graph of the loss, build a second shape graph that computes the
gradient. That gradient graph is then kernelized, rangeified and compiled like any other graph, by
the machinery of the previous two chapters. This chapter derives the algorithm, then derives every
one of its rules.

**What you'll build**

- `gradient(root, targets)`: reverse-mode AD over a DAG of UOps, in 13 lines.
- `_rule(u, g)`: a local derivative rule for every op that can appear in a pure shape graph, each
  derived from first principles: ALU ops, the four movement ops, `REDUCE` (sum and max, with ties),
  `STACK` and `INDEX`.
- `Tensor.backward`, which hangs the gradients on the parameters as lazy tensors.
- Gradient checking with finite differences, and second derivatives for free.

**Where this fits.** Autodiff runs on the shape graph, *before* the kernelizer
([Chapter 7](07-kernelize.md)) and rangeify ([Chapter 6](06-rangeify.md)). It uses the composite
ops `PAD` and `sum_to` from `uopc/composite.py` (Chapter 6). [Chapter 9](09-tensor-mnist.md)
builds a training loop on top of `backward`, and [Chapter 14](14-gpt.md) trains a GPT with it.

## 8.1 Three ways to get a derivative

Take `f(x) = sum(x * y + x)` for vectors `x` and `y`. There are three ways to get `df/dx`.

**Symbolic differentiation** applies calculus rules to the *expression* and produces a new
expression: `df/dx_i = y_i + 1`. This is what a computer algebra system does. Applied naively to a
program, it suffers from expression swell: the derivative of a product of `n` factors has `n`
terms each with `n - 1` factors, and shared subexpressions get duplicated.

**Numerical differentiation** treats `f` as a black box:
`df/dx_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)`. It is easy and indispensable for *testing*
(see "Gradient checking" below), but it costs two evaluations of `f` per input element, and its
accuracy is limited by the tension between truncation error (eps too large) and floating-point
cancellation (eps too small).

**Automatic differentiation** works on the *program*, not the expression: it applies the chain rule
to the sequence (or graph) of primitive operations that computes `f`, reusing intermediate values.
Its result is exact up to floating-point rounding, and its cost is a constant multiple of the cost
of `f`. Because UOp graphs are hash consed, our "symbolic" output graph never duplicates a shared
subexpression either: AD on a DAG gets the benefits of both.

## 8.2 Forward mode and reverse mode

Think of the program as a composition of functions,
`f = f_3 . f_2 . f_1`, with `x -> h1 -> h2 -> L`. The chain rule says its Jacobian is a product of
the Jacobians of the pieces:

```text
J_f = J_3 J_2 J_1          (J_k is the Jacobian of f_k at its input)
```

You never want to build these matrices: a `(1000,)`-to-`(1000,)` elementwise op has a 10^6-entry
Jacobian that is diagonal. You want products of them with vectors, and the order in which you
multiply decides the cost.

- **Forward mode** multiplies right to left, starting from a direction `v` in input space:
  `J_3 (J_2 (J_1 v))`. Each pass gives one *column* of `J_f`, the derivative of every output with
  respect to one input direction. Getting the full gradient of a scalar loss with respect to `n`
  parameters takes `n` passes. (Dual numbers, `a + b*eps` with `eps^2 = 0`, are forward mode.)
- **Reverse mode** multiplies left to right, starting from a covector `w` in output space:
  `((w^T J_3) J_2) J_1`. Each pass gives one *row*: the derivative of one output with respect to
  every input. For a scalar loss, `w = 1` and one pass gives the entire gradient.

A loss is a scalar and a model has millions of parameters, so ML uses reverse mode. The price is
memory: the backward pass multiplies by `J_1` *last*, and `J_1` depends on `x` (for `MUL`, the
derivative with respect to one factor is the other factor), so every intermediate value needed by a
local derivative must be kept around until the backward pass reaches it. In our lazy graph world,
"kept around" just means "the gradient graph refers to the forward node", and the kernelizer
decides whether that node becomes a buffer (Chapter 7).

> **Key idea:** reverse mode propagates `dL/d(node)`, the sensitivity of the one scalar output to
> each intermediate value, from the output back to the inputs. Each op only needs to know how to turn
> the sensitivity of its output into sensitivities of its inputs: a vector-Jacobian product.

## 8.3 The chain rule on a DAG

Programs are not chains; they are DAGs. In `x * y + x`, `x` is used twice. The multivariate chain
rule handles fan-out: if `L` depends on `x` through several uses `u_1, ..., u_k`, then

```text
dL/dx = sum over uses k of  (dL/du_k) * (du_k/dx)
```

Contributions from different paths *add up*. Reverse-mode AD on a DAG is therefore:

1. Seed `dL/dL = 1`.
2. Visit nodes in reverse topological order, so that when a node is visited every one of its
   consumers has already been visited, and its gradient is complete.
3. For each node `u` with gradient `g`, use `u`'s local rule to compute a contribution for each
   source, and *add* it to that source's running gradient.

For `L = sum(x * y + x)`:

```text
forward:   m = x * y      a = m + x      L = sum(a)
backward:  dL/da = expand(1)                        REDUCE(ADD) -> EXPAND
           dL/dm += dL/da ;   dL/dx += dL/da        ADD passes g to both sources
           dL/dx += dL/dm * y ; dL/dy += dL/dm * x  MUL: each source gets g times the other
result:    dL/dx = 1 + y ,  dL/dy = x
```

@diagram 08-reverse-mode

## 8.4 Why only pure shape graphs

The syllabus says autodiff works "only on pure shape graphs (not STORE, CALL, AFTER, RANGE,
END)". The reasons are worth spelling out.

- `STORE` and `AFTER` are effects and orderings on memory. The derivative of "write `v` to address
  `p`" is not a value; differentiating through memory requires recording what was overwritten
  (a "tape"), which is what imperative AD systems like PyTorch's autograd do at run time.
- `RANGE` and `END` are loops with an iteration count. Differentiating a loop in reverse means
  running it backward, which needs every iteration's state (or recomputation, "checkpointing").
  Compilers like Enzyme do this on LLVM IR, at considerable complexity.
- `CALL` could be inlined first; the shape graph never contains one.

A shape graph has none of those problems. Every node is a pure function of its sources, applied to
whole tensors, so every local derivative is again a pure function on whole tensors, expressible in
the same ops. That is why autodiff runs *before* rangeify: after rangeify, the graph is full of
loops and stores. This placement is the same as in JAX (which differentiates `jaxpr`s, a pure
array IR, before XLA lowers them) and tinygrad.

## 8.5 gradient(): the algorithm

Here is the whole driver:

@include uopc/autodiff.py#gradient

Line by line:

- `topo = root.toposort()` lists every node the loss depends on, children before parents.
- `on_path` is the set of nodes that depend on *some target*. A node not on the path cannot carry
  gradient to a target, so its rule is never applied. In a training step this prunes the entire
  data-preparation part of the graph: there is no point computing a gradient for the input images.
  It is computed in one forward sweep: a node is on the path if any source is.
- `grads` starts with the seed: a tensor of ones shaped like the root (for a scalar loss, `CONST
  1.0`). `_full` builds a broadcast constant, like `const_like` (NOTES 13: the constant must have the
  node's shape, or the ALU type rule rejects it).
- The loop walks `reversed(topo)`, parents before children. It skips nodes that have no gradient
  (nothing on the path consumed them), nodes off the path, and leaf targets (a `BUFFER` has no
  sources to pass gradient to).
- For each source, `_rule` returns a gradient contribution or `None` ("this input is not
  differentiable", like the condition of a `WHERE`). Contributions *add*: `grads[x] + gx` is an
  `ADD` UOp, which is the fan-out sum of the chain rule.
- Finally, one gradient per target. A target the loss does not depend on gets zeros of its shape,
  not an error (`gradient(f, [z])` for an unrelated `z` returns a `(4,)`-shaped `EXPAND` of 0.0).

Note what this function does *not* do: it computes no numbers. Everything it builds is a UOp graph.
Here is the gradient of `sum(x * y + x)` with respect to `x`, printed as `.uir`:

```text
%0 = reshape 1.0, (1,)
%1 = expand %0, (3,)
%2 = buffer 3 : dtype=f32 slot=1
%3 = mul %1, %2
%4 = add %1, %3
```

`%1` is `dL/da`, the seed broadcast back over the reduced axis. `%3` is the `MUL` rule's
contribution `g * y`, and `%4` adds the `ADD` rule's contribution `g`. Lowered, it is one line of C:

```c
void grad_x(float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val1 = data1[ridx0];
    data2[ridx0] = (val1+1.0f);
  }
}
```

The simplifier removed the multiplication by one. Note the parameter list: the gradient with respect
to `x` does not read `x` at all, so the kernel takes only `y` (`data1`) and the output. On
`x = [1, 2, 3]`, `y = [10, 20, 30]`:

```text
dloss/dx = [11. 21. 31.]  expected y+1 = [11. 21. 31.]
dloss/dy = [1. 2. 3.]  expected x = [1. 2. 3.]
```

### Sharing between forward and backward

The gradient graph refers to forward nodes by identity: `dL/dy = g * x` contains the very `x` node
of the forward graph, and the gradient of a matmul contains the forward activations. When the loss
and the gradients are kernelized together, any forward node that both need is one node, materialized
at most once. [Chapter 7](07-kernelize.md) showed it on a training step: the forward `X @ W` is one
kernel whose output is read by both the loss kernel and the gradient kernel. No "saved tensors"
machinery is needed; hash consing is the tape.

## 8.6 The rules

Every rule maps `(u, g)`, a node and the gradient of the loss with respect to it (same shape as
`u`), to one gradient per source (same shape as that source), or `None`.

@include uopc/autodiff.py#_rule

The helper `_full(like, v)` is a constant `v` broadcast to `like`'s shape:

@include uopc/autodiff.py#_full

Below, each rule is derived. Notation: `u` is the node's output, `a`, `b` (or `x`) its sources, `g`
is `dL/du` and the job is `dL/da = g * du/da` for elementwise ops (a vector-Jacobian product with a
diagonal Jacobian), or the appropriate generalization for ops that move elements around.

### ALU ops

**ADD.** `u = a + b`, `du/da = du/db = 1`. Both sources get `g`. Addition is a gradient
*distributor*.

**MUL.** `u = a * b`, `du/da = b`, `du/db = a`. So `(g * b, g * a)`: each source gets the gradient
times the *other* factor. This is the rule that makes reverse mode keep forward values alive. For
`x * x`, both contributions go to `x` and add up to `2 g x`, which is the derivative of `x^2`: fan-out
summation at work.

**RECIP.** `u = 1/x`, `du/dx = -1/x^2 = -u^2`. The rule reuses the forward output `u` instead of
recomputing `1/x`: `-(g * u * u)`.

**EXP2.** `u = 2^x = e^(x ln 2)`, so `du/dx = ln 2 * 2^x = u * ln 2`. Again the output is reused.

**LOG2.** `u = ln(x) / ln 2`, so `du/dx = 1 / (x ln 2)`: `g * (1/x) * (1/ln 2)`.

**SQRT.** `u = x^(1/2)`, `du/dx = 1 / (2 sqrt(x)) = 0.5 / u`.

**SIN.** `du/dx = cos(x)`. There is no `COS` op, so it is written `sin(x + pi/2)`, the same trick
`Tensor.cos` uses.

The Tensor library builds everything else from these. `exp(x) = exp2(x * log2(e))` gets its
derivative from the chain rule through `MUL` and `EXP2`: `exp2(x log2 e) * ln 2 * log2(e) =
exp(x)`. `tanh`, `sigmoid`, `gelu`, `log_softmax`: none of them has a rule of its own. Their
gradients are assembled from the primitive rules by `gradient()`, and they are tested numerically
(below).

**CAST.** Converting between float types is the identity up to rounding, so the gradient is `g`
converted back: `g.cast(s[0].dtype)`. Casting *from* an integer or bool has no derivative (the
input is discrete), so the rule returns `None`.

**WHERE.** `u = c ? a : b`. For each element, `u` is either `a` or `b`, so the gradient goes to
whichever was selected: `dL/da = c ? g : 0`, `dL/db = c ? 0 : g`. The condition is a bool and gets
`None`.

**CMPLT, CMPNE, FLOORDIV, FLOORMOD.** Comparisons produce bools; integer division produces
integers. Piecewise constant functions have derivative zero wherever they have one, and the
derivative is undefined at the jumps. Returning `None` for every source says "no gradient flows
here", which is the useful convention.

### MAX, and the problem of ties

`u = max(a, b)` is differentiable wherever `a != b`: the gradient goes entirely to the larger input.
At `a == b` the function has a kink and no derivative; any convex combination of the two one-sided
derivatives is a valid *subgradient*. The rule picks the symmetric one, half each:

```text
dL/da = g     if a > b,   0 if a < b,   g/2 if a == b
```

That is what the nested `WHERE`s implement. The choice is not cosmetic: it means `abs(x) =
max(x, -x)` has gradient `g/2 - g/2 = 0` at `x = 0`, the conventional subgradient, while an
asymmetric rule would give `+g` or `-g`. Two definitions of the same mathematical function can
still differ at the kinks:

```text
d|x|/dx  at [-2, 0, 3]: [-1.  0.  1.]
relu'    at [-2, 0, 3]: [0. 0. 1.]
max(x,0)' at [-2, 0, 3]: [0.  0.5 1. ]
```

`Tensor.relu` is `(x > 0).where(x, 0)`, so its derivative at 0 comes from the `WHERE` rule (the
condition is false, gradient 0). `x.maximum(0)` goes through the `MAX` rule and splits the tie.
Both are "correct". With randomly initialized float weights, exact ties are rare in practice, but
they are not rare in tests that use integers, or after a ReLU has produced many exact zeros. Pick one
definition, use it everywhere, and know what it does at the kink.

### Movement ops: transposes of index maps

Every movement op is *linear*, and better: each output element is a copy of exactly one input
element (or several outputs are copies of the same input, for `EXPAND`). For such a map, the
vector-Jacobian product is "send each output's gradient back to the input element it was copied
from, and add up the gradients of all outputs that copied the same input". Each rule is the dual of
its forward op.

@diagram 08-movement-duals

**RESHAPE.** A reshape is a bijection between element positions (same row-major order). The
gradient is reshaped back: `g.reshape(s[0].shape)`.

**PERMUTE.** `u = x.permute(p)`; output axis `k` is input axis `p[k]`. The gradient has `u`'s axes
and must be brought back to `x`'s: input axis `m` is output axis `k` where `p[k] = m`. That is the
inverse permutation `inv[p[k]] = k`, so the rule is `g.permute(inv)`. (Same inverse as in
rangeify's `PERMUTE` rule, for the same reason: a permutation's Jacobian is a permutation matrix,
whose transpose is its inverse.)

**EXPAND.** `u[i, j] = x[i, 0]` for every `j`: one input element feeds every output along the
expanded axis. By the fan-out rule, its gradient is the *sum* of their gradients:
`dL/dx[i, 0] = sum_j g[i, j]`. So the gradient of a broadcast is a reduction, written as `sum_to`:

@include uopc/composite.py#sum_to

`sum_to` finds the axes where the target shape has 1 and `g` does not, permutes them to the end,
reduces them (recall `REDUCE` pops the last axes), and reshapes back to the size-1 form. This is the
rule that turns a bias added to every row into a column sum in the backward pass, and the one that
turns the broadcast inside a matmul into the *other* matmul of the backward pass.

**SHRINK.** `u = x[b:e]` (per axis). Elements inside the window are copied, elements outside are
dropped. The gradient of a dropped element is 0, so the gradient is `g` placed back at offset `b`
inside a zero tensor of `x`'s shape: `pad(g, (b, n - e))`. SHRINK and PAD are each other's duals.
Since `pad` is the composite from Chapter 6, the gradient graph of a slice contains `STACK`s,
reshapes and shrinks:

```text
%0 = reshape 0.0, (1,)
%1 = expand %0, (4,)
%2 = reshape 1.0, (1,)
%3 = expand %2, (2,)
%4 = expand %0, (2,)
%5 = stack %3, %4
%6 = reshape %5, (4,)
%7 = stack %1, %6
%8 = reshape %7, (8,)
%9 = shrink %8, ((3, 8),)
```

That is `d sum(v[1:3]) / dv` for a length-5 `v`, and it lowers to a single bounds-checked
select. Note that the gradient doesn't read `v` at all (the slice is linear), so the kernel has no
input:

```c
void grad_shrink(float* restrict data0) {
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    int alu1 = (ridx0+3);
    data0[ridx0] = (((alu1/4)<1)?0.0f:((((alu1/2)%2)<1)?1.0f:0.0f));
  }
}
```

```text
d sum(v[1:3])/dv = [0. 1. 1. 0. 0.]
```

This is also why autodiff never needs `FLIP`: the gradient of a convolution built from shrinks
(im2col, Chapter 6) is a sum of pads.

### REDUCE

**Sum.** `u[i] = sum_k x[i, k]`, so `du[i]/dx[i, k] = 1`: every element of the reduced group gets
the group's gradient. `REDUCE(ADD)` and `EXPAND` are duals, mirror images of the previous rule. In
code: reshape `g` from `u.shape` to `u.shape + (1,)*n` and expand to `x.shape`.

**Max.** `u[i] = max_k x[i, k]`. Away from ties, the gradient goes to the arg-max element only. With
ties, the rule splits it evenly among all maximal elements, the same convention as elementwise
`MAX`:

```text
mask[i, k] = (x[i, k] == u[i])          as a float
cnt[i]     = sum_k mask[i, k]
dL/dx      = g[i] * mask[i, k] / cnt[i]
```

The code builds exactly that: `mx` is `u` broadcast back, `mask` compares, `cnt` is a sum-reduce of
the mask broadcast back, and the result is `ge * mask * (1 / cnt)`. On `x = [1, 3, 3]`:

```text
d max(x)/dx at [1. 3. 3.] = [0.  0.5 0.5]
```

and for the elementwise version with ties at index 1:

```text
d sum(max(x,y)) / dx = [1.  0.5 0. ]  / dy = [0.  0.5 1. ]
```

(For comparison, PyTorch's `torch.amax` documents the same even split; `torch.max(dim=...)` sends the
gradient to the single returned index.)

Note the cost: the backward of a max-reduce contains *another* reduction (the count), so it becomes
an extra kernel. Softmax uses `max` only for numerical stability and its gradient through the max is
mathematically zero (softmax is invariant to shifting its input), but the graph does not know that
and computes it anyway.

### STACK and INDEX

**STACK.** `u = STACK(a_0, ..., a_{n-1})`, so `u[k] = a_k`. Source `k`'s gradient is slice `k` of
`g`: `g.index(k)`.

**INDEX with constant indices.** `u = x.index(c)` selects `x[c]`. The dual of selecting one slice is
placing the gradient into a zero tensor at that slice, which is a `STACK` with `g` at position `c`
and zeros elsewhere. For several indices the code does that one axis at a time, innermost first.
The indices themselves are integers and get `None`.

**INDEX with data-dependent indices** (`x.index(i)` where `i` is computed from data, like an
embedding lookup with token ids) is not supported:

```text
NotImplementedError: gradient of a data-dependent INDEX (use a one-hot matmul)
```

The dual of a gather is a *scatter-add* (`dL/dx[i] += g` for the selected `i`, accumulating when the
same index is selected twice), and scatter-add needs an op this IR does not have. The workaround in
the error message is how the GPT of [Chapter 14](14-gpt.md) does embeddings: multiply a one-hot
matrix by the embedding table. A one-hot matmul is a gather written with ops we can differentiate,
at the cost of `vocab` times more arithmetic. (Exercise 9 asks for a real scatter.)

@include uopc/nn.py#Embedding

The one-hot matrix is built by comparison (`idx.unsqueeze(-1).eq(arange(n))`), whose gradient is
`None`, and cast to float, so the only gradient path is through the matmul into `weight`: the `EXPAND`
rule's `sum_to` adds up the gradients of every row that selected the same token, which is exactly the
scatter-add, computed densely.

**STAGE.** Materialization is the identity function, so the gradient passes through: `(g,)`.
With one exception: a `STAGE` whose arg is `"detach"` returns `(None,)`. That is how
`Tensor.detach()` is implemented:

@include uopc/tensor.py#Tensor.detach

The marker means "same value, but treat it as a constant when differentiating". It costs nothing at
run time: the kernelizer's `strip_detach` removes it before planning ([Chapter 7](07-kernelize.md)),
so it never becomes a kernel. Detaching is how you write a stop-gradient, for example in a target
that should not be trained toward. With `w = [1, 2]`, `sum(w * w.detach())` treats the second factor
as a constant, so its gradient is `w` rather than `2w`:

```text
grad with detach:    [1. 2.]
```

**CONST, BUFFER, PARAM** are leaves: no sources, nothing to return.

Anything else (`STORE`, `RANGE`, ...) raises
`NotImplementedError("no gradient rule for ... (autodiff only works on pure shape graphs)")`.

## 8.7 Worked examples: the gradients you will actually train with

The rules are local. It is worth checking, twice, that composing them gives the textbook formulas
for the two gradients that dominate every training run.

### Matrix multiplication

The Tensor library computes `Y = X @ W` for `X` of shape `(M, K)` and `W` of shape `(K, N)` as

```text
Xe = X.reshape(M, 1, K).expand(M, N, K)
We = W.T.reshape(1, N, K).expand(M, N, K)          (W.T is PERMUTE (1, 0))
Y  = (Xe * We).reduce(ADD, 1)                       shape (M, N)
```

Push a gradient `G = dL/dY` of shape `(M, N)` back through it with the rules:

1. `REDUCE(ADD)`: `G` is reshaped to `(M, N, 1)` and expanded to `(M, N, K)`. Call it `Ge`; its
   value at `(i, j, k)` is `G[i, j]`.
2. `MUL`: the `Xe` side gets `Ge * We`, the `We` side gets `Ge * Xe`.
3. `EXPAND` on the `X` side: `sum_to` sums `Ge * We` over axis 1 (the `N` axis that `X` was
   broadcast along): `dL/dX[i, k] = sum_j G[i, j] * W[k, j]`. That is `G @ W^T`.
4. `EXPAND` on the `W` side sums `Ge * Xe` over axis 0 (the `M` axis):
   `sum_i G[i, j] * X[i, k]`, of shape `(N, K)`. The `RESHAPE` and inverse `PERMUTE` rules bring it
   back to `(K, N)`: `dL/dW = X^T @ G`.

Two matmuls, each the same size as the forward one, both produced by the `EXPAND` rule: the
broadcast that turned a matrix product into an elementwise product turns, in the backward pass, into
the reduction of the *other* matrix product. Nobody wrote a "matmul backward". This is also the
`r_10_64` kernel of the training step in [Chapter 7](07-kernelize.md) (`X^T @ G`, of shape
`(10, 64)` before the final transposing copy).

### Softmax cross-entropy

For logits `z` (one row) and a one-hot label `y`, `Tensor.cross_entropy` computes
`L = -sum_c y_c log_softmax(z)_c` and averages over the batch, with
`log_softmax(z) = (z - m) - log(sum_c exp(z_c - m))` and `m = max(z)`. By hand: the derivative of
`log(sum exp)` with respect to `z_c` is `softmax(z)_c`, the `m` terms cancel (shifting all logits by a
constant changes nothing), so

```text
dL/dz_c = (softmax(z)_c - y_c) / B
```

for a batch of `B` rows. The autodiff graph knows none of this. It applies `MUL`, `ADD`, `LOG2`,
`EXP2`, `REDUCE(ADD)`, `REDUCE(MAX)` (with its mask and count) and `EXPAND` rules one by one, and the
result must agree:

```python
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], np.float32)
labels = np.array([0, 1], np.int32)
t = Tensor(logits, requires_grad=True)
t.cross_entropy(Tensor(labels)).backward([t])
```

```text
autodiff:
 [[-0.17049944  0.12121649  0.04928295]
 [ 0.05430186 -0.09876046  0.0444586 ]]
(softmax - onehot) / B:
 [[-0.17049944  0.12121648  0.04928295]
 [ 0.05430187 -0.09876049  0.04445861]]
```

Equal to the last float32 digit or so. The max's contribution is present in the graph (it is not
simplified away) and sums to zero numerically, which is the "mathematically zero" gradient through
the max mentioned above.

## 8.8 From gradient() to Tensor.backward

The Tensor library's `backward` is a loop over `gradient`'s results:

@include uopc/tensor.py#Tensor.backward

Three details matter to users. The loss must be a scalar (the seed `1.0` only makes sense for one
output). The gradients are lazy `Tensor`s: nothing is computed until the optimizer realizes them,
together with the parameter updates, in one `realize_all` (one plan, one cache entry). And gradients
*accumulate*: a second `backward` adds to `p.grad`, which is why optimizers have `zero_grad`.

The `getattr(self, "_graph", self.uop)` line handles a subtle interaction with the kernelizer. When
a tensor is realized (by `.item()`, `.numpy()`, or an optimizer's `realize_all`), its `uop` is
rebound to the output `BUFFER`, and a buffer is a leaf with no gradient path to anything. Printing
the loss before calling `backward` would therefore silently produce zero gradients. `realize_all`
keeps the pre-realization graph in `t._graph`, and `backward` differentiates that:

```python
loss = (w * w).sum()
print("loss.item() =", loss.item())
loss.backward([w]); print("grad after .item():", w.grad.numpy())
```

```text
loss.item() = 5.0
grad after .item(): [2. 4.]
```

The price is that the gradient graph refers to the forward *graph*, not to the buffer `.item()`
produced, so the forward work the gradients need is computed again when they are realized (in the
same plan as the updates, so it is still one realization per step). Realizing loss and gradients
together, as the optimizers do, avoids the repeat.

## 8.9 Gradient checking

Every rule above is one line, which means every rule is one typo away from a plausible wrong
answer. The oracle is numerical differentiation with **central differences**:

```text
df/dx_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)
```

Its truncation error is O(eps^2) (the even terms of the Taylor expansion cancel), against O(eps) for
the one-sided difference. The week-5 tests compute it in float64 with NumPy, as an independent
implementation of the same function:

@include tests/test_w05_tensor.py#numgrad

@include tests/test_w05_tensor.py#test_gradients

Three points of craft. The reference function `npfn` is plain NumPy, not `uopc`, so a bug in a
shared component cannot cancel itself out. The differences are taken in float64, so `eps = 1e-3`
gives truncation errors around 1e-6 without cancellation trouble; `uopc` computes in float32, hence
the tolerances `rtol=2e-2, atol=2e-3`. And the test functions are chosen to cover rules: `tanh`
(EXP2, RECIP, MUL), `log_softmax` with a slice (MAX-reduce, SHRINK, LOG2, EXPAND), a max of sums
(both reductions), pad-then-shrink (SHRINK and the composite PAD), `sqrt` and `recip`, and `sin` of a
matmul (PERMUTE, EXPAND, REDUCE).

Here is the same check for a graph that exercises `STACK` and `INDEX` (via `t.stack_with(t * t)[1]`):

```python
x = np.random.default_rng(0).standard_normal((2, 3)).astype(np.float32)
t = Tensor(x, requires_grad=True)
loss = (t.stack_with(t * t)[1].sum(0) * Tensor(np.array([1., 2., 3.], np.float32))).sum()
loss.backward([t])
```

```text
autodiff:
 [[ 0.25146043 -0.52841944  3.842536  ]
 [ 0.20980023 -2.1426775   2.1695704 ]]
finite differences:
 [[ 0.25146043 -0.52841944  3.842536  ]
 [ 0.20980023 -2.1426775   2.1695704 ]]
```

Gradient checks should avoid kinks: a finite difference across `max(a, b)` at `a == b` averages the
two one-sided slopes and agrees with the tie rule only by coincidence. Random float inputs avoid
exact ties with probability 1.

## 8.10 Higher-order derivatives

`gradient` takes a graph and returns a graph. Nothing stops you from differentiating its output:

```python
x = buf(0, 4)
f = (x * x * x).reduce(Ops.ADD, 1)                    # sum(x^3)
(g,) = gradient(f, [x])                               # 3x^2, a shaped graph like any other
(h,) = gradient(g.reduce(Ops.ADD, 1), [x])            # d/dx sum(3x^2) = 6x
```

```text
g = [ 3. 12. 27. 48.]
h = [ 6. 12. 18. 24.]
```

```c
void second_derivative(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    float alu2 = (val1*2.0f);
    data1[ridx0] = (((alu2+alu2)+val1)+val1);
  }
}
```

`(2x + 2x) + x + x = 6x`: the second derivative arrives as a sum of the contributions of every path,
just as the chain rule on a DAG says, and float reassociation (which the simplifier refuses to do;
NOTES 7) leaves it in that form. Because the gradient graph uses only ops that have rules, every
gradient graph is differentiable again. Systems with a hand-written backward kernel per op (as in
many early frameworks) have to provide a second set of rules for double backward; graph-to-graph AD
gets it for free.

## 8.11 What gradients cost

A rule of thumb: the backward pass costs about twice the forward. Each `MUL` in the forward becomes
two in the backward (one per source). Each matmul (an `EXPAND`-`MUL`-`REDUCE`) becomes two matmuls in
the backward: the `EXPAND`s turn into reductions via `sum_to`, and the `REDUCE` turns into an
`EXPAND`. And every elementwise op in the forward pass leaves a trail of elementwise ops in the
backward pass, which the kernelizer fuses into the neighboring matmul kernels. In the training step
of Chapter 7, the ReLU's backward (`WHERE` of a comparison) and the square's backward (`g * x + g * x`)
ended up inside the inner loop of the weight-gradient matmul:

```c
        _Bool alu3 = (0.0f<val2);
        float alu4 = ((alu3?val2:0.0f)*0.0031250000465661287f);
        float val5 = data1[((ridx2*64)+ridx1)];
        acc0 = (val1+((alu3?(alu4+alu4):0.0f)*val5));
```

(`0.0031250000465661287` is `1/320` rounded to float32 by the constant folder, NOTES 5; the forward
kernel printed the same constant as `0.003125f`.) Making this fast is the subject of Part IV.

## 8.12 Building it yourself: an order of work

1. Write `gradient` with only `ADD`, `MUL` and the leaves, and test `sum(x * y + x)` against the
   hand-derived `1 + y` and `x`. Include a function that uses an input twice.
2. Write `numgrad` (in float64 NumPy) *before* adding more rules. From here on, every new rule gets
   a gradient-check test the moment it is written.
3. Add the unary ALU rules (`RECIP`, `EXP2`, `LOG2`, `SQRT`, `SIN`), `CAST` and `WHERE`; test
   `tanh`, `sigmoid`, `relu`.
4. Add `REDUCE(ADD)` and `EXPAND` together (they are duals; one without the other cannot be tested on
   anything interesting), then `RESHAPE` and `PERMUTE`. Test a matmul and a non-self-inverse
   permutation.
5. Add `SHRINK` (needs `pad`), `STACK`, constant `INDEX`, then `REDUCE(MAX)` and `MAX` with ties.
   Test `log_softmax` with slicing, and a dedicated tie test (`max` of `[1, 3, 3]`).
6. Hook it into `Tensor.backward`, write SGD, and watch a loss go down
   (`test_training_reduces_loss`).

## Pitfalls

**1. Fan-out must accumulate.** The single most common bug in a first AD implementation is
`grads[x] = gx` instead of `grads[x] = grads[x] + gx`. Everything that uses a value once still works;
`x * x` gets gradient `x` instead of `2x`. Test a function where some input is used twice.

**2. The visiting order.** Rules must be applied in reverse topological order. A naive recursive
"propagate to my sources as soon as I get a gradient" visits a shared node before all its
consumers have contributed, and propagates an incomplete gradient (or propagates twice). The
`reversed(topo)` loop makes the order a non-issue, and it is iterative, so deep graphs do not hit the
recursion limit (NOTES 3).

**3. Constants must have the node's shape (NOTES 13).** The seed and the constants inside rules
(`_full(u, LN2)`, the `0.5` in `MAX`) must be broadcast to the node's shape, or the ALU type rule
rejects the graph with `shape mismatch ... vs ()`. `_full` exists for this.

**4. Kinks are conventions.** `relu` and `maximum(x, 0)` differ at 0 (gradient 0 vs 0.5). Neither is
wrong. Tests with integer-valued inputs hit kinks constantly; use random floats for gradient checks,
and pin the convention in a dedicated test.

**5. Data-dependent indexing has no gradient here.** Embedding lookups written with `INDEX` of token
ids fail with `NotImplementedError`. Use a one-hot matmul, or add a scatter op (Exercise 9).

**6. Python floats in the gradient graph.** Learning rates, bias corrections and loss scales that
change per step become `CONST`s in the update graph and miss the JIT cache every step (NOTES 17,
[Chapter 7](07-kernelize.md)). The gradient graph is built fresh every step; keep its *structure*
identical.

**7. Gradients accumulate across `backward` calls.** `Tensor.backward` adds to an existing
`p.grad`. Forgetting `opt.zero_grad()` doubles the effective learning rate on the second step, and
the graph grows by one `ADD` per step, so the structural cache misses every step too.

**8. Realizing the loss destroyed its graph (found and fixed).** `Tensor.realize` rebinds `uop` to
the output buffer. `loss.item(); loss.backward(params)` used to differentiate a `BUFFER` and return
zeros for every parameter, without any error. Logging the loss before the backward pass is the most
natural thing in the world, so this would have bitten everyone. `realize_all` now keeps `t._graph`
and `backward` uses it; `test_item_before_backward_and_detach` pins the behavior.

## Exercises

1. ★ **Derive by hand.** Compute `dL/dx` for `L = sum(exp2(x) * x)` by hand, then build it with
   `gradient` and compare on `x = [0, 1, 2]`.

2. ★ **Predict the graph.** What does `gradient(x.reshape((2, 3)).permute((1, 0)).reduce(Ops.ADD, 2), [x])`
   return, for a `(6,)` buffer `x`? Predict the ops in the result, then print it with `to_uir`.

3. ★★ **Forward mode.** Implement `jvp(root, target, tangent)`: forward-mode AD that returns the
   directional derivative of `root` in direction `tangent`. Write the forward rule for `ADD`, `MUL`,
   `EXP2`, `RESHAPE`, `EXPAND` and `REDUCE`. Check `jvp` against `gradient` via
   `sum(gradient * tangent) == jvp`.

4. ★★ **Find the bug.** A student's `PERMUTE` rule is `return (g.permute(u.arg),)`. Which tests in
   `test_gradients` catch it? Write the smallest test that does.

5. ★★ **The softmax gradient.** Derive `d softmax(x)_i / dx_j = s_i (delta_ij - s_j)` by hand. Then
   compare the size (node count) of the gradient graph `gradient()` builds for
   `sum(softmax(x) * w)` with a hand-written rule `g_x = s * (g - sum(g * s))`. Is the max-reduce's
   gradient in there? Why is it mathematically zero?

6. ★★ **Ties in MIN.** `Tensor.min` is `-((-x).max())`. What gradient does `x.min()` give at
   `x = [2, 1, 1]`? Predict, then check.

7. ★★ **Gradient checking with float32 only.** Change `numgrad` to evaluate the function with `uopc`
   (float32) instead of NumPy (float64). Find the `eps` that minimizes the error for `tanh_sum`, and
   explain the U-shaped error curve.

8. ★★★ **Checkpointing.** The forward activations kept alive for the backward pass dominate training
   memory. Design a `checkpoint(f)` for the Tensor API that recomputes `f`'s forward during the
   backward pass instead of keeping its outputs. *Hint:* in this lazy, hash-consed world, what does
   "keeping" even mean? Look at what the kernelizer materializes.

9. ★★★ **Scatter-add.** Design an op `SCATTER_ADD(x_shape, idx, g)` (or a rangeify-level lowering)
   that makes the gradient of a data-dependent `INDEX` possible, give its type rule and its rangeify
   rule, and explain why it needs either atomics or a reduction over the index axis.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** `d/dx (2^x x) = 2^x ln2 x + 2^x = 2^x (x ln 2 + 1)`. At `x = [0, 1, 2]`:
`[1, 2(ln 2 + 1), 4(2 ln 2 + 1)] = [1, 3.386, 9.545]`. In the graph, `MUL` sends `g * x` to the `EXP2`
node and `g * exp2(x)` to `x`; the `EXP2` rule turns its gradient into `g * x * exp2(x) * ln 2`, and the
two contributions to `x` are added.

**2.** The `REDUCE(ADD, 2)` rule produces `RESHAPE` of the seed to `(1, 1)` and an `EXPAND` to
`(3, 2)`; the `PERMUTE` rule permutes that back with the inverse `(1, 0)` to `(2, 3)`; the `RESHAPE`
rule reshapes to `(6,)`. So: a constant 1.0, reshape, expand, permute, reshape. All ones, as it must
be, since every element of `x` contributes once to the total sum.

</details>

## Quiz

1. Why does ML use reverse mode rather than forward mode?
2. State the chain rule for a node with two consumers. Where in `gradient()` does it appear?
3. Why is `gradient()` applied to shape graphs and not to rangeified loop programs?
4. What is the gradient rule for `EXPAND`, and why?
5. What is the gradient rule for `SHRINK`?
6. How does the `REDUCE(MAX)` rule handle ties, and what does `max(x, -x)` give at `x = 0`?
7. Why can `gradient` be applied to its own output?
8. Why is the gradient of a data-dependent `INDEX` not supported, and what is the workaround?

<details>
<summary>Answers</summary>

1. The loss is one scalar and the parameters are many. Reverse mode gets the whole gradient (one row
   of the Jacobian) in one backward pass; forward mode needs one pass per parameter.
2. `dL/dx = dL/du_1 * du_1/dx + dL/du_2 * du_2/dx`: contributions add. In
   `grads[x] = grads[x] + gx if x in grads else gx`.
3. Shape graphs are pure functions of whole tensors, so every local derivative is again a pure tensor
   expression in the same ops. Loop programs contain stores, orderings and loops, whose derivatives
   need a tape or reversed loops.
4. `sum_to(g, x.shape)`: sum the gradient over the broadcast axes, because one input element feeds
   every copy and the fan-out contributions add up.
5. `pad(g, ((b, n - e), ...))`: put the gradient back at its window inside zeros, since dropped
   elements have zero gradient.
6. It splits the gradient evenly among all maximal elements (`mask / count`); the elementwise `MAX`
   rule gives half to each side of a tie. So `max(x, -x)` (that is, `abs`) has gradient `g/2 - g/2 = 0` at 0.
7. Its output is an ordinary shape graph built only from ops that have rules.
8. Its dual is a scatter-add, which the IR cannot express. Use a one-hot matrix multiply instead.

</details>

## Further reading

- A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind, "Automatic Differentiation in Machine
  Learning: a Survey", JMLR 18, 2018. The best single overview of forward/reverse mode.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic
  Differentiation*, 2nd ed., SIAM, 2008. The reference text, including checkpointing.
- The JAX documentation, "The Autodiff Cookbook" (JVPs, VJPs, and higher-order derivatives as
  function transformations on a pure IR).
- A. Paszke et al., "Automatic differentiation in PyTorch", NIPS 2017 Autodiff Workshop, for the
  tape-based, run-time alternative.
- W. Moses and V. Churavy, "Instead of Rewriting Foreign Code for Machine Learning, Automatically
  Synthesize Fast Gradients" (Enzyme), NeurIPS 2020: AD on LLVM IR, loops, stores and all.
- tinygrad's `gradient.py` (in recent versions): the same `(node, grad) -> source grads` pattern
  matcher design, including the movement-op duals.
