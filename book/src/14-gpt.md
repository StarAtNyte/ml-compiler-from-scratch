# 14. Training a GPT

This is the chapter the course has been building toward. Every piece so far (UOps, rewriting, rangeify,
kernelization, autodiff, the layout pass, register tiles) was tested on small, isolated programs. A
transformer language model is the first workload that uses *all* of them at once, at a size where a bad
decision anywhere costs real time. We will write a GPT from scratch in about a hundred lines of Tensor code
(`examples/gpt.py`), follow its shapes through every layer, train it on Tiny Shakespeare until it writes
something that looks like a play, and then dissect what the compiler did with it: 243 kernels per training
step, a structural cache that compiles them once, and a performance history in which a first step went from 64
seconds to about 9, and a steady-state step from 1.2 seconds to about 0.24.

**What you'll build**

- A character-level GPT: token embeddings as a one-hot matmul, learned positions, causal multi-head
  self-attention with a mask, pre-LayerNorm residual blocks, a GELU MLP, a linear head, cross-entropy loss.
- AdamW whose per-step constants don't break the compile cache.
- A training loop with validation and a sampler with a fixed shape.
- A profiling method (DEBUG levels, per-kernel timings, `cProfile`, ablations) and the knowledge of which
  compiler decisions mattered for this model, with numbers.

**Where this fits.** Weeks 9–10. The model uses the Tensor API from [Chapter 9](09-tensor-mnist.md), autodiff
from [Chapter 8](08-autodiff.md), kernelization from [Chapter 7](07-kernelize.md) and the CPU optimizations
from [Chapter 10](10-cpu-performance.md). [Chapter 15](15-beyond.md) takes it further: symbolic shapes, KV
caches and mixed precision. All numbers in this chapter come from the reference laptop (i7-13620H, see
`book/outputs/machine.txt`) running on the CPU. Chapter 12 explains why the GPU backend isn't used here, and
Exercise 14.10 is about closing that gap.

## 14.1 The task: next-character prediction

The data is Tiny Shakespeare, about 1.1 million characters of plays, the classic char-rnn dataset. It is
downloaded once and cached:

@include examples/datasets.py#shakespeare

A **character-level language model** reads a window of T characters and predicts, at *every* position, the
next character. Each character is an integer in `[0, V)` with V = 65 distinct characters. A training example
is a pair of windows offset by one:

```text
x = "First Citizen:\nBefore we proce"     (T characters)
y = "irst Citizen:\nBefore we procee"     (the same, shifted left by one)
```

so that position *t* of the output is trained to predict `y[t] = x[t+1]` from `x[0..t]`. One window yields T
training signals, and the model must never look at `x[t+1..]` when predicting position *t*. That is what the
causal mask enforces.

The loss is the mean **cross-entropy**, the negative log-probability that the model assigns to the right next
character. A model that knows nothing assigns 1/65 to each character, and −ln(1/65) = ln 65 ≈ **4.174**.
That's the number to compare the first logged loss against.

## 14.2 The model at a glance, with every shape

Hyper-parameters of the default run (`examples/gpt.py` with no flags):

| symbol | meaning | value |
|---|---|---|
| B | batch size (windows per step) | 16 |
| T | context length | 64 |
| C | model width (embedding dim) | 64 |
| H | attention heads | 4 |
| D | head dim, C / H | 16 |
| V | vocabulary (characters) | 65 |
| L | transformer blocks | 2 |

@diagram 14-gpt-architecture

The whole forward pass, with the shape after each step:

| step | operation | shape |
|---|---|---|
| tokens | integer ids | (B, T) = (16, 64) i32 |
| embed | `one_hot(idx) @ W_tok + pos[:T]` | (16, 64, 64) |
| block × 2 | `x = x + attn(ln1(x))`; `x = x + mlp(ln2(x))` | (16, 64, 64) |
| final norm | `ln(x)` | (16, 64, 64) |
| head | `x @ W_head + b` | (16, 64, 65) logits |
| loss | `cross_entropy(logits.reshape(1024, 65), y.reshape(1024))` | () |

The parameters, counted by hand, which is a good habit before trusting any model code:

| tensor | shape | count |
|---|---|---|
| token embedding | 65 × 64 | 4,160 |
| position table | 64 × 64 | 4,096 |
| per block: 2 LayerNorms | 2 × (64 + 64) | 256 |
| per block: QKV projection | 64 × 192 + 192 | 12,480 |
| per block: output projection | 64 × 64 + 64 | 4,160 |
| per block: MLP up | 64 × 256 + 256 | 16,640 |
| per block: MLP down | 256 × 64 + 64 | 16,448 |
| 2 blocks | 2 × 49,984 | 99,968 |
| final LayerNorm | 64 + 64 | 128 |
| head | 64 × 65 + 65 | 4,225 |
| **total** | | **112,577** |

The script prints `112.6k parameters, vocab 65`, which matches. It is a tiny model, one thousandth of GPT-2
small, chosen so that a full training run takes a few minutes on a laptop CPU. Nothing in the code depends on
the size. `--layers 4 --dim 128` gives a bigger model with the same code paths.

## 14.3 Embeddings as a one-hot matmul

An embedding is a table lookup: row `idx` of a `(V, C)` matrix. Our IR has no gather and no scatter. Its
memory ops are the `INDEX`/`LOAD` that rangeify generates, and autodiff only works on pure shape graphs
(Chapter 8). So we write the lookup as a matrix product with a one-hot matrix:

@include uopc/nn.py#Embedding

@include uopc/tensor.py#Tensor.one_hot

`idx.unsqueeze(-1)` has shape (B, T, 1), and `Tensor.arange(V)` has shape (V,). Comparing them broadcasts to
(B, T, V): entry `[b, t, v]` is 1.0 if `idx[b,t] == v`. Multiplying by `W` (V, C) then selects one row per
token. The gradient comes for free from the matmul rule: `dW = one_hotᵀ @ dY`, which adds each position's
gradient into the row of the token that was there. That is a scatter-add, derived by autodiff with no
special case.

It sounds wasteful, since it builds a (16, 64, 65) tensor of mostly zeros. It isn't, because nothing builds it.
Here is the embedding kernel the compiler generates (`r_16_64_64`, from the forward plan). It is excerpted to
the first of its sixteen 4×4 register-tile accumulators, and the `//` comments are mine:

```c
void r_16_64_64(int* restrict data0, int* restrict data1, float* restrict data2, float* restrict data3) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    ...
      int val6 = data0[(alu2+alu5)];                 // the token id idx[b, t]
      ...
        for (int ridx3 = 0; ridx3 < 65; ridx3++) {
          float val16 = acc0;
          int val17 = data1[ridx3];                  // arange(65)[v]
          float alu18 = ((float)(((val6!=val17)!=1)));   // one_hot: (idx == v) as 0.0/1.0
          int alu19 = ((ridx2*260)+ridx3);
          float val20 = data2[alu19];                // W_tok, transposed by the layout pass
          acc0 = (val16+(alu18*val20));
          ...
```

The one-hot is computed on the fly *inside the reduction*, as a comparison and a cast, and multiplied straight
into the accumulator. Rangeify pushed the index of the broadcast into the comparison, and kernelization fused
it into the reduce. The `(B, T, V)` tensor exists only as an expression. The cost is B·T·V·C = 4.3 M
multiply-adds per forward pass, against a lookup's 65 K copies, which is negligible at V = 65. For a 50,000-token
vocabulary it would not be, and a real compiler would need a gather op (Chapter 15 discusses this). The small
`e_1_1_64_65` kernel before it is the layout pass from Chapter 10 copying `W_tok` to a K-contiguous layout.

The positional embedding is a plain learned `(T, C)` tensor `pos`. `self.pos[:T].reshape(1, T, -1)` is a
`SHRINK` and a `RESHAPE`, and the add broadcasts it over the batch. As you will see in 14.5, that add never
gets its own kernel either.

## 14.4 Causal self-attention

Attention lets each position read from every earlier position, with weights it computes from the data. For a
single head with queries Q, keys K and values V, all (T, D):

```text
scores = Q Kᵀ / √D                 (T, T)   how much position i wants position j
scores[i, j] = -inf  for j > i              causality: no peeking at the future
P = softmax(scores, axis=-1)       (T, T)   each row sums to 1
out = P V                          (T, D)   a weighted average of values
```

Multi-head attention runs H such heads in parallel on D = C/H slices of the model width and concatenates the
results. Here it is in `uopc`:

@include examples/gpt.py#Attention

Shapes, line by line, with B=16, T=64, C=64, H=4, D=16:

| expression | shape | ops in the graph |
|---|---|---|
| `x` | (16, 64, 64) | |
| `self.qkv(x)` | (16, 64, 192) | matmul (REDUCE over C) + bias |
| `.reshape(B, T, 3, H, D)` | (16, 64, 3, 4, 16) | RESHAPE |
| `qkv[:, :, i]` | (16, 64, 4, 16) | SHRINK + RESHAPE |
| `.permute(0, 2, 1, 3)` → q, k, v | (16, 4, 64, 16) | PERMUTE |
| `k.transpose()` | (16, 4, 16, 64) | PERMUTE |
| `q @ k.transpose()` | (16, 4, 64, 64) | broadcast to (16, 4, 64, 64, 16), MUL, REDUCE |
| `* (1 / math.sqrt(D))` | (16, 4, 64, 64) | MUL by 0.25 |
| `+ self.mask[:T, :T].reshape(1, 1, T, T)` | (16, 4, 64, 64) | SHRINK, RESHAPE, broadcast ADD |
| `.softmax(-1)` | (16, 4, 64, 64) | MAX, ADD reductions, EXP2, RECIP |
| `@ v` | (16, 4, 64, 16) | broadcast to (16, 4, 64, 16, 64), MUL, REDUCE |
| `.permute(0, 2, 1, 3).reshape(B, T, C)` | (16, 64, 64) | PERMUTE + RESHAPE |
| `self.proj(...)` | (16, 64, 64) | matmul + bias |

Splitting into heads and merging them back are *views*: reshapes and permutes, which rangeify turns into index
arithmetic (Chapter 6). No data is moved for them. The two products are batched matmuls, which
`Tensor.matmul` builds by broadcasting to a 5-D `(…, M, N, K)` tensor and reducing the last axis:

@include uopc/tensor.py#Tensor.matmul

The broadcast product for `q @ kᵀ` would have 16·4·64·64·16 = 4.2 M elements, and like the one-hot, it is
never materialized. It lives inside the reduce loop of one kernel.

@diagram 14-attention-shapes

**The mask** is an ordinary constant tensor, built once in numpy: `-1e9` above the diagonal and 0 elsewhere.
Adding it before the softmax drives `exp` of the future positions to exactly 0. We use −10⁹ rather than −∞
because infinities invite NaNs: a row that is entirely masked (possible with padding masks, though not with
this causal one) would compute `-inf - (-inf)` in the max-subtraction, and `0 · inf` can appear in a backward
pass. With a large finite number the arithmetic stays finite, and `exp(-1e9)` is still exactly 0.

**The softmax** is the numerically stable version from Chapter 9, `exp(x - max(x)) / sum(...)`:

@include uopc/tensor.py#Tensor.softmax

Here are the kernels the scale, mask and softmax become, taken from the forward plan. First the row max
(`r_16_4_64`, excerpted to one of its 16 unrolled accumulators, with comments added):

```c
      for (int ridx2 = 0; ridx2 < 64; ridx2++) {
        float val7 = acc0;
        int alu8 = ((alu2+alu5)+ridx2);
        float val9 = data0[alu8];                    // raw score q·k
        int alu10 = (alu5+ridx2);
        float val11 = data1[alu10];                  // mask[i, j]
        acc0 = ((val7>((val9*0.25f)+val11))?val7:((val9*0.25f)+val11));
```

The `0.25f` is 1/√16, folded to a constant, and the mask add is fused into the max. Then a second reduction
`r_16_4_64` computes the row sums of `exp2((s*0.25 + mask - max) * log2(e))`, and one elementwise kernel
writes the probabilities:

```c
void e_16_4_64_64(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3, float* restrict data4) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      for (int ridx2 = 0; ridx2 < 64; ridx2++) {
        int alu1 = (ridx2*64);
        int alu2 = (((ridx0*256)+(ridx1*64))+ridx2);
        float val3 = data2[alu2];
        float val4 = data3[alu2];
        for (int ridx3 = 0; ridx3 < 64; ridx3++) {
          int alu5 = ((((ridx0*16384)+(ridx1*4096))+alu1)+ridx3);
          float val6 = data0[alu5];
          float val7 = data1[(alu1+ridx3)];
          data4[alu5] = (exp2f(((((val6*0.25f)+val7)+val3)*1.4426950408889634f))*val4);
        }
      }
    }
  }
}
```

`val3` is −max and `val4` is 1/sum, both computed once per row, outside the inner loop, by the linearizer's
scope rules. Two tiny kernels `e_16_4_64_1` compute them (`data1 = data0 * -1.0f` and `data1 = 1.0f/data0`),
for a reason that turns out to be an accident of the cost model (Pitfall 9). This is the *unfused* attention of
Chapter 11: it materializes the (B, H, T, T) probabilities, 1 MB here. At T = 64 that is fine. At T = 2048 it
is what flash attention exists to avoid.

The test that matters most for attention is not numerical. It checks that the future does not leak:

@include tests/test_w09_models.py#test_attention_is_causal

Changing positions 4 and 5 of the input must leave outputs 0–3 untouched. A missing mask or a transposed one
would fail this test even though every individual kernel is "correct".

## 14.5 LayerNorm

LayerNorm normalizes each position's C-vector to zero mean and unit variance, then applies a learned per-channel
scale and shift:

```text
mu  = mean(x, axis=-1)                       (B, T, 1)
var = mean((x - mu)^2, axis=-1)              (B, T, 1)
y   = (x - mu) / sqrt(var + eps) * w + b     (B, T, C)
```

@include uopc/nn.py#LayerNorm

It is checked against numpy:

@include tests/test_w09_models.py#test_layernorm_matches_numpy

The compiler turns the first LayerNorm of block 1 into four kernels: a sum (`r_16_64`), a sum of squared
deviations (`r_16_64`, which recomputes `x - mu` inside the reduce rather than storing it), the reciprocal
standard deviation (`e_16_64_1`), and the normalization:

```c
void e_16_64_1(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      int alu1 = ((ridx0*64)+ridx1);
      float val2 = data0[alu1];
      data1[alu1] = (1.0f/sqrtf(((val2*0.015625f)+1e-05f)));
    }
  }
}
void e_16_64_64(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3, float* restrict data4, float* restrict data5, float* restrict data6) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      int alu1 = (ridx1*64);
      int alu2 = ((ridx0*64)+ridx1);
      float val3 = data2[alu2];
      float val4 = data3[alu2];
      for (int ridx2 = 0; ridx2 < 64; ridx2++) {
        int alu5 = (((ridx0*4096)+alu1)+ridx2);
        float val6 = data0[alu5];
        float val7 = data1[(alu1+ridx2)];
        float val8 = data4[ridx2];
        float val9 = data5[ridx2];
        data6[alu5] = (((((val6+val7)+((val3*0.015625f)*-1.0f))*val4)*val8)+val9);
      }
    }
  }
}
```

Read the last line of `e_16_64_64`. `val6 + val7` is the token embedding plus the position embedding,
`val3*0.015625f` is sum/64 = mean, `val4` is 1/σ, and `val8`, `val9` are `w` and `b`. The embedding add, the
centering, the scaling and the affine transform are all one expression. The embedding sum `x` is never
written to memory on its own.

Why is this LayerNorm output written to memory at all, as `e_16_64_64` does, when it could be fused into the
QKV matmul that consumes it? Because that matmul broadcasts it 192 times (once per output column), and
recomputing a LayerNorm 192 times per element costs more than one round trip through memory. That decision
is `stage_broadcast_compute`, and it is the subject of Section 14.10.

## 14.6 GELU and the MLP

The MLP widens to 4C = 256, applies a nonlinearity, and projects back. GPT-2 uses GELU, `x · Φ(x)` with Φ the
standard normal CDF, in its tanh approximation:

@include uopc/tensor.py#Tensor.gelu

`tanh` itself is built from `sigmoid`, which is built from `exp`, which is `exp2` times a constant (Chapter 9).
The compiler flattens the whole tower into one expression, with the MLP-up bias add fused in:

```c
void e_16_64_256(float* restrict data0, float* restrict data1, float* restrict data2) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      for (int ridx2 = 0; ridx2 < 256; ridx2++) {
        int alu1 = (((ridx0*16384)+(ridx1*256))+ridx2);
        float val2 = data0[alu1];
        float val3 = data1[ridx2];
        float alu4 = (val2+val3);
        data2[alu1] = ((alu4*0.5f)*((((1.0f/(exp2f((((((alu4+(((alu4*0.044715f)*alu4)*alu4))*0.7978845608028654f)*2.0f)*-1.0f)*1.4426950408889634f))+1.0f))*2.0f)+-1.0f)+1.0f));
      }
    }
  }
}
```

`0.7978845608028654` is √(2/π). The derivative of all of this, which the backward pass needs, is derived by
`gradient()` from the same ops. Nobody wrote a GELU backward.

## 14.7 Residual blocks and the full model

@include examples/gpt.py#Block

@include examples/gpt.py#GPT

This is the **pre-LayerNorm** arrangement used by GPT-2 and later models: each branch normalizes its *input*,
and the residual stream `x` itself is never normalized until the final `self.ln`. Gradients flow back through
the `+` of each residual connection unchanged, which is what makes deep stacks trainable.

The `Linear` layer is `x @ W + b`, with `W` stored as (in, out):

@include uopc/nn.py#Linear

## 14.8 The loss

@include uopc/tensor.py#Tensor.log_softmax

@include uopc/tensor.py#Tensor.cross_entropy

`log_softmax` subtracts the max before the `exp`, so it stays finite even for large logits, and it never
computes `log(softmax(x))`, which would underflow to `log(0)` for unlikely classes. The labels are one-hot
encoded and the dot product picks out the log-probability of the right class. That is the same one-hot trick
as the embedding, and again the one-hot is never materialized.

## 14.9 AdamW, and constants that change every step

AdamW keeps two running averages per parameter, the mean of the gradients `m` and of their squares `v`, and
updates with their ratio:

```text
m ← β1·m + (1-β1)·g
v ← β2·v + (1-β2)·g²
p ← p·(1 - lr·wd) - lr · m̂ / (√v̂ + ε),     m̂ = m / (1-β1ᵗ),  v̂ = v / (1-β2ᵗ)
```

The `(1-β^t)` terms correct the bias of averages that start at zero. "W" means the weight decay is *decoupled*
(Loshchilov & Hutter): it shrinks `p` directly rather than being added to the gradient.

@include uopc/nn.py#AdamW

The comment in `step` is a bug story (NOTES 17). Written as Python floats, the bias corrections become
`CONST`s whose values change every step, so the structural cache misses every step and recompiles all 243
kernels. Passing them as 1-element tensors keeps the graph's structure fixed (see
[Chapter 7](07-kernelize.md), Section 7.8). Section 14.12 measures what the naive version costs.

The update is realized together with everything else. `Tensor.realize_all(*news, *self.m, *self.v, *extra)` is
called with the loss as `extra`, so the forward pass, the backward pass, the optimizer update and the loss
value are all *one* schedule: one call to `realize`, one cache lookup, 243 kernels.

## 14.10 Training

@include examples/gpt.py#main

`batch` samples B random windows. Every 50 steps the script also evaluates one batch from the held-out last 10%
of the text. Here is the full run, as captured in `book/outputs/gpt_train.txt` (`python examples/gpt.py`, 600
steps):

```text
112.6k parameters, vocab 65
step     0  train 4.266  val 3.827    8933 ms/step
step    50  train 2.841  val 2.855     368 ms/step
step   100  train 2.616  val 2.559     288 ms/step
step   150  train 2.503  val 2.577     263 ms/step
step   200  train 2.444  val 2.514     249 ms/step
step   250  train 2.357  val 2.491     240 ms/step
step   300  train 2.265  val 2.336     240 ms/step
step   350  train 2.325  val 2.271     241 ms/step
step   400  train 2.156  val 2.286     239 ms/step
step   450  train 2.274  val 2.276     240 ms/step
step   500  train 2.074  val 2.223     235 ms/step
step   550  train 2.065  val 2.163     235 ms/step
step   599  train 2.015  val 2.110     235 ms/step

ICGLOUTery man!

For apbeng mantound rear:

Whals pant lachtiow't,
Htaw be pect she sto now thy sost soo with ble oving dity:
Went song rief tin of taut thurfs me dat aournclint,
Goant?


FORE:
Se wat, more-ks in sterdide be tom:
Whinenten wer I thent nor ifor ond he;
Wegon:
If to hard, of his do to
```

How to read it:

- **Step 0's loss is 4.266**, close to the ln 65 = 4.174 of a uniform guess. Random small weights give nearly
  uniform predictions, which is a sanity check that the loss is wired correctly. A first loss of 40 or 0.1
  would mean a bug.
- **The validation loss tracks the training loss** (2.110 against 2.015 at the end), so this small model is
  not overfitting in 600 steps. Each logged loss is from a single batch, so the steps are noisy: train 2.325
  at step 350 after 2.265 at step 300.
- **For scale:** in Andrej Karpathy's "Let's build GPT" lecture a bigram model, which predicts from the
  previous character only, plateaus around 2.5 on this data, and nanoGPT's character-level Shakespeare
  config (a 10.7M-parameter model trained on a GPU) reaches a validation loss of about 1.47. Our 112.6k
  parameters at 2.11 sit in between, where you would expect.
- **The sample** has learned the *format*: speaker names in capitals followed by a colon, line breaks,
  capitalized verse lines, real short words ("man", "be", "she", "now", "thy", "with", "to", "of", "his"),
  and plausible pseudo-words. More steps, a wider model (`--dim 128 --layers 4`) or a longer context all
  improve it.
- **The ms/step column is a running average since step 0**, so it includes the first step's compile. It
  drops from 8933 ms to 235 ms because steps 1 onward reuse the cached plan. Section 14.11 explains both
  numbers.

Training itself is checked by a fast test, a tiny GPT on a trivially predictable sequence that must cut its loss
by 4× in 30 steps:

@include tests/test_w09_models.py#test_tiny_gpt_learns

### Sampling with a fixed shape

Generation feeds the last T characters and draws the next one from the softmax of the logits at the last real
position. The prompt grows by one character per step. Feeding the *actual* prompt would make every step's
graph a different shape, (1, 1), (1, 2), (1, 3), …, so the first 64 samples would each compile a new plan.
The sampler instead pads the context to T with zeros and reads the logits at position `len(idx) - 1` (NOTES
18). Causality makes the padding harmless: position *t*'s output never sees positions after *t*. With the
padding every forward pass has the same graph and compiles once. In the `DEBUG=2` log, the first sampling
forward pass (52 kernels at batch 1) shows `schedule 4423.1 ms`, and every later one shows about 5 ms (that
run was on a busy machine, so only the ratio matters). The proper fix is symbolic shapes plus a KV cache, the
subject of [Chapter 15](15-beyond.md).

## 14.11 What the compiler did: anatomy of a training step

### Counting kernels

The forward pass alone is 61 kernels. With autodiff and the optimizer, one training step is 243. A scratch
script that runs three steps and inspects the structural cache:

```python
# scratch/plans.py (after building the default model and optimizer)
n0 = len(K._plans); step(); step(); step()
print("plans cached after 3 steps:", len(K._plans), "(new:", len(K._plans) - n0, ")")
big = max(K._plans.values(), key=lambda p: len(p.kernels))
print("training-step plan:", len(big.kernels), "kernels,", len({k.src for k in big.kernels}), "distinct C sources,", len(big.temps), "temp buffers")
print("temp memory: %.1f MB" % (sum(n * 4 for _, n in big.temps) / 1e6))
```

```text
plans cached after 3 steps: 14 (new: 1 )
training-step plan: 243 kernels, 128 distinct C sources, 152 temp buffers
temp memory: 31.7 MB
```

(A re-run after the compiler's later fixes, such as reusing temp arrays across calls, gave the same counts,
61 forward kernels and identical losses, and `examples/gpt.py --steps 51` reproduced the logged losses
exactly. Timings from that re-run were taken on a busy machine and are not comparable, so the numbers below
are the quiet-machine ones.)

Three steps produced *one* new plan, so the cache works. Of the 243 kernels only 128 have distinct C source.
The rest are byte-identical, the payoff of renumbering buffers locally per kernel (Chapter 7): the two
blocks' identical layers render to identical C, which compiles once and is shared through the
content-addressed `.so` cache.

### DEBUG=2: per-kernel timings

`DEBUG=2` prints one line per kernel launch with its time and an estimated GFLOP/s, and `DEBUG=1` a summary
per `realize`. To get a clean listing of a *steady-state* step (not the first, which compiles), the scratch
script flips `uopc.kernelize.DEBUG` at run time after three warm-up steps:

```python
# scratch/debug_step.py (excerpt)
for _ in range(3): step()
K.DEBUG = 2
buf = io.StringIO()
with contextlib.redirect_stdout(buf): step()
```

```text
  r_1024                            0.336 ms     1.39 GFLOP/s
  e_1_1_64_65                       0.047 ms     0.00 GFLOP/s
  r_16_64_64                        0.290 ms    73.52 GFLOP/s
  r_16_64                           0.247 ms     0.53 GFLOP/s
  r_16_64                           0.272 ms     1.45 GFLOP/s
  e_16_64_1                         0.036 ms     0.11 GFLOP/s
  e_16_64_64                        0.141 ms     3.26 GFLOP/s
  e_1_1_192_64                      0.027 ms     0.00 GFLOP/s
  r_16_64_192                       0.254 ms    99.02 GFLOP/s
  e_16_64_192                       0.137 ms     1.44 GFLOP/s
  r_16_4_64_64                      0.276 ms    30.43 GFLOP/s
  r_16_4_64                         0.176 ms     4.46 GFLOP/s
  ...
  r_1024                            0.363 ms     2.20 GFLOP/s
  r_1                               0.049 ms     0.02 GFLOP/s
  e_1                               0.036 ms     0.00 GFLOP/s
realize: 243 kernels, schedule 8.1 ms, run 112.1 ms
243 kernel launches; sum of kernel times 108.4 ms
top kernel names by total time:
  r_16_64_64             x12    24.65 ms
  r_16_4_64              x10    14.84 ms
  e_16_4_64_64           x4     12.74 ms
  r_192_64               x2     12.51 ms
  r_16_64                x20     5.05 ms
  r_16_4_64_64           x4      4.66 ms
  r_64                   x14     4.11 ms
  r_16_64_256            x4      3.16 ms
  r_256_64               x2      2.69 ms
  e_16_64_64             x10     2.40 ms
```

Kernel names are `r_` (contains a reduce) or `e_` (elementwise) followed by the output shape (Chapter 7).
Matching them to the model is a useful exercise:

- `r_16_64_192` is the QKV projection, `(B, T, 3C)`. At 99 GFLOP/s it is near the CPU GEMM rate of Chapter 10.
  `e_1_1_192_64` before it is the layout pass transposing the weight.
- `r_16_4_64_64` is `q @ kᵀ`, and `r_16_4_64` (×2 per layer forward, plus backward) are the softmax reductions.
- `r_192_64`, `r_256_64`, `r_64`, … are **weight gradients**: `dW = xᵀ @ dy` reduces over all B·T = 1024
  positions, and a bias gradient sums over them. `r_192_64` takes 6 ms per call for 25 MFLOP, about
  4 GFLOP/s, among the slowest kernels per FLOP in the step. Its reduce axis (B·T) is the *outermost* axis of
  both activations, which is the worst case for the layout heuristics of Chapter 10 (Exercise 14.7).
- `r_1024`, `r_1`, `e_1` are the loss and its scalar plumbing.

The arithmetic: the standard estimate for one training step of a transformer is 6 × parameters × tokens
FLOPs, here 6 × 112,577 × 1,024 ≈ 0.69 GFLOP. At about 108 ms of kernel time that's 6.4 GFLOP/s, 5% of what the
same laptop reaches on a large GEMM. The model is simply too small to be compute-bound. Most kernels touch a
few KB and finish in tens of microseconds, so launch overhead, memory traffic and serial reductions dominate.
A bigger model raises the fraction spent in big GEMMs, and with it the efficiency.

### Where the rest of the time goes

The kernels take about 110 ms, but a steady-state step takes longer. Splitting the step (from a scratch
script, on a quiet machine):

```text
build forward+backward graph:   20.4 ms
optimizer graph + realize   :  143.0 ms   (243 kernels)
```

About 20 ms is Python building the forward graph and differentiating it, about 8 ms is computing the
structure key and looking up the plan (`schedule 8.1 ms` above), and the remainder is kernel execution
plus the Python loop that calls 243 `ctypes` functions. On a quiet machine the steady state measures
140–150 ms per step. The training log's 235 ms includes the running-average effect of step 0 and periodic
validation passes, and was measured while the machine was doing other work.

### The first step: profiling the compiler

The first step is a different story. It has to build the plan: kernelize, run the layout pass, rangeify,
simplify and render 243 kernels, then compile whatever is not already in the `.so` cache with gcc. Two
measurements of the same script:

```text
$ UOPC_CACHE=scratch/coldcache python ablate.py baseline   # empty .so cache
baseline      first step   25.6 s   steady     148 ms/step   loss 4.286 -> 3.335
$ python ablate.py baseline                                          # warm .so cache
baseline      first step    5.8 s   steady     139 ms/step   loss 4.286 -> 3.335
```

With an empty cache, gcc compiles about 140 shared objects (the 128 distinct kernels of the step plus the
initialization kernels) and the first step takes 26 s. With a warm cache it's 6 s, all of it Python. To find
out *which* Python, use `cProfile`:

```python
# scratch/profile_step.py (excerpt)
pr = cProfile.Profile()
t0 = time.perf_counter(); pr.enable(); step(); pr.disable(); t1 = time.perf_counter()
print(f"step 0 (builds + compiles): {(t1-t0):.1f} s")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14); print(s.getvalue()[:3500])
```

```text
step 0 (builds + compiles): 20.8 s
         25845652 function calls (23894366 primitive calls) in 20.502 seconds

   Ordered by: cumulative time
   List reduced from 374 to 14 due to restriction <14>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   20.813   20.813 scratch/profile_step.py:21(step)
        1    0.001    0.001   20.756   20.756 uopc/nn.py:87(step)
        1    0.001    0.001   20.726   20.726 uopc/tensor.py:63(realize_all)
        1    0.005    0.005   20.724   20.724 uopc/kernelize.py:178(realize)
        1    0.080    0.080   20.527   20.527 uopc/kernelize.py:111(make_plan)
     2168    1.271    0.001   11.119    0.005 uopc/rewrite.py:90(graph_rewrite)
      243    0.007    0.000    9.888    0.041 uopc/opt.py:109(hand_opt)
      243    0.012    0.000    9.845    0.041 uopc/rangeify.py:170(lower)
29919/1702    0.146    0.000    8.641    0.005 uopc/rangeify.py:74(rangeify)
    11144    0.033    0.000    8.511    0.001 uopc/symbolic.py:282(simplify)
   244507    0.685    0.000    7.839    0.000 uopc/rewrite.py:82(rewrite)
    16254    0.011    0.000    6.871    0.000 uopc/rangeify.py:95(<genexpr>)
     1214    0.030    0.000    6.654    0.005 uopc/rangeify.py:125(_reduce)
17080/3752    0.013    0.000    6.544    0.002 uopc/rangeify.py:83(<genexpr>)
```

`cProfile` makes the step about 3.5× slower (20.8 s against 5.8 s unprofiled, since it instruments 26 million
calls), but the proportions are what matter. Almost everything is under `make_plan`. More than half is
`graph_rewrite`, called 2,168 times, mostly from `simplify` (11,144 calls) running symbolic rules over index
expressions inside `rangeify`, in particular at every `RESHAPE` (line 95) and in `_reduce`. This is the cost
of NOTES 12's fix. Simplifying indices eagerly is what keeps reductions out of the wrong loops, and it is paid
for in compile time. `hand_opt` → `lower` is the CPU optimizer rangeifying each kernel with its upcasts. Four
rules for reading a profile: `tottime` is time in the function itself and `cumtime` includes callees. Sort by
`cumulative` to find the subsystem, then by `tottime` to find the hot function. The `ncalls` column tells you
whether a function is slow or just called too often.

## 14.12 The performance story

When the GPT first ran end to end, the first step took **64 seconds** and each further step about **1.2
seconds**. The shipped compiler takes about **9 s** and **0.24 s** under the conditions of
`book/outputs/gpt_train.txt`. The fixes, from the development log (`book/NOTES.md`), in the order of the
pipeline:

| # | symptom | diagnosis | fix | effect |
|---|---|---|---|---|
| NOTES 12 | reductions in attention far too slow | an unsimplified index `((i*N + j)//N) % N` *mentions* j, so the reduce lowering put the accumulator inside the j loop: O(N³) | `simplify` each index at `RESHAPE` in rangeify | fused attention at N=512: 1787 ms → 7 ms |
| NOTES 14 | `((x/4096)%4)` chains in GPT's inner loops | div/mod rules couldn't see individual terms of `(a+b)*c` | distribute integer `*` over `+` in the symbolic rules | div/mod folds, inner loops shrink |
| NOTES 15 | 374 ms of kernels per step | LayerNorm output EXPANDed into a matmul was *recomputed* per output column | `stage_broadcast_compute` with a cost model | 374 ms → 125 ms of kernel time |
| NOTES 16 | staging missed nested candidates | `substitute({x: STAGE(x)})` wraps the *original* x | `UOp.wrap` rebuilds bottom-up | staging becomes complete |
| NOTES 17 | every step as slow as the first | AdamW's bias corrections were CONSTs, so the structure key changed each step | pass them as tensors | one plan for all steps |
| NOTES 21 | a 21 ms kernel staging a (128,784,128) product | the layout pass staged a whole broadcast product | `push_views` + only stage operands smaller than the reduce input | the MNIST kernel goes away |
| NOTES 22 | 163 s of compilation for 12 steps | `UOp.__eq__`/`__hash__` and `Enum.__hash__` in Python | identity hashing (in C) and memoized `simplify` | 47 s; first GPT step ~9 s |

The intermediate step times between fixes were not all recorded, so instead of reconstructing them, here is an
**ablation** of the shipped compiler: remove one fix at a time with a monkeypatch in a scratch script (nothing
in `uopc/` changes), and measure steady-state ms/step on the default model:

```python
# scratch/ablate.py (the switches)
if mode == "no_stage": K.stage_broadcast_compute = lambda roots, factor=4: roots
if mode == "no_layout":
  class NoLayout(O.HandOpt): layout = staticmethod(lambda roots: roots)
  Tensor.opt = NoLayout()
# mode == "opt0" sets OPT=0 before importing uopc; mode == "float_consts" uses an AdamW subclass
# whose step() bakes c1, c2 into the graph as python floats, as the first version did.
```

Three rounds of the four cheap configurations, interleaved, plus the expensive one:

```text
baseline      first step    8.0 s   steady     215 ms/step   loss 4.286 -> 3.335
no_stage      first step   19.8 s   steady     682 ms/step   loss 4.286 -> 3.335
no_layout     first step    6.3 s   steady     186 ms/step   loss 4.286 -> 3.335
opt0          first step    2.9 s   steady     517 ms/step   loss 4.286 -> 3.335
baseline      first step    7.3 s   steady     165 ms/step   loss 4.286 -> 3.335
no_stage      first step   16.9 s   steady     412 ms/step   loss 4.286 -> 3.335
no_layout     first step    5.2 s   steady     182 ms/step   loss 4.286 -> 3.335
opt0          first step    2.8 s   steady     499 ms/step   loss 4.286 -> 3.335
baseline      first step   10.8 s   steady     229 ms/step   loss 4.286 -> 3.335
no_stage      first step   20.4 s   steady     423 ms/step   loss 4.286 -> 3.335
no_layout     first step    7.1 s   steady     183 ms/step   loss 4.286 -> 3.335
opt0          first step    3.0 s   steady     508 ms/step   loss 4.286 -> 3.335
float_consts  first step    6.1 s   steady    4699 ms/step   loss 4.286 -> 3.550
```

@diagram 14-perf-ablation

(The losses agree to three decimals across configurations, a free correctness check: every variant computes
the same numbers. `float_consts` ran 4 steps instead of 9, hence its different final loss.) These runs shared
the machine with other jobs (load average 5–10), so the absolute numbers are noisy: the baseline ranged from
165 to 229 ms. The ordering was stable across rounds, though, and it says:

1. **The compile cache is the biggest single factor.** Baking the bias corrections in as floats makes *every*
   step a first step: about 4.7 s/step (5.5 s in an earlier run), roughly 20–30× slower. One line of
   `nn.py`.
2. **Broadcast staging is worth about 2×.** Without it, LayerNorm and GELU outputs are recomputed inside every
   matmul that consumes them. Removing it also makes step 0 about 2.5× slower (20 s against 8 s), because the
   fused recomputing kernels are much larger graphs to rangeify and simplify.
3. **Register tiles and threads (OPT=0 vs. baseline) are worth about 2.4×.** OPT=0 compiles fastest (no
   upcasts to unroll) and runs slowest.
4. **The layout pass doesn't pay off on this model.** Removing it was, if anything, slightly *faster* (183
   against 215 ms in this set, and 161–218 ms against 140–258 ms in other rounds). At K = 64 a strided operand
   fits in L1 anyway, so the transposing copies cost more than they save. The same pass was essential for
   MNIST's 784-wide GEMMs (NOTES 21) and for the 1024³ GEMM of Chapter 10. An optimization is a bet about the
   workload. Measure it on each workload you care about.

## Pitfalls

**1. The O(N³) attention (NOTES 12).** An unsimplified `RESHAPE` index put the softmax reductions inside the
j loop, so fused attention did N times too much work (see [Chapter 11](11-fusion-flash.md), Section 11.5).

**2. Undistributed products hide div/mod opportunities (NOTES 14).** Without distributing `(a + b) * c`,
GPT's attention kernels carried `((x/4096)%4)` chains in their inner loops (see [Chapter 6](06-rangeify.md),
Pitfalls). Some residue remains. The fused `q @ kᵀ` kernel still computes `((alu14+256)%192)` in its inner
loop where `alu14 ∈ [0, 63]`, so the value is just `alu14 + 64`. A bounds-based rule (if `vmin(x)//c ==
vmax(x)//c` then `x % c = x - c*(vmin(x)//c)`) would remove it (Exercise 14.8).

**3. The fusion/recompute trap (NOTES 15).** A LayerNorm output broadcast into a matmul is recomputed for every
output column; staging such tensors took kernel time from 374 ms to 125 ms per step (see
[Chapter 7](07-kernelize.md), Pitfalls).

**4. `substitute` wraps the original (NOTES 16).** Staging with `substitute` leaves candidates nested inside a
target unstaged; `UOp.wrap` rebuilds bottom-up instead (see [Chapter 7](07-kernelize.md), Pitfalls).

**5. Python floats break the JIT cache (NOTES 17).** Anything that changes every step and enters the graph as a
`CONST` (bias corrections, a learning-rate schedule, a dropout seed, a temperature) makes every step a cache
miss, about 20–30× slower in the ablation. Pass such values as tensors (see [Chapter 7](07-kernelize.md),
Pitfalls).

**6. Variable shapes break the JIT cache (NOTES 18).** A sampler that feeds a growing prompt compiles a new plan
per length. Pad to a fixed shape, or use symbolic shapes (Chapter 15).

**7. Staging a whole broadcast product (NOTES 21).** An early layout pass staged the entire (128, 784, 128)
broadcast product, a 21 ms kernel to save a few (see [Chapter 10](10-cpu-performance.md), Section 10.6).

**8. Python overhead is compiler overhead (NOTES 22).** Python-level `__eq__`/`__hash__` made compiling 12 GPT
steps take 163 s; identity hashing and memoized `simplify` brought it to 47 s (see [Chapter 1](01-uops.md)).
Profile the compiler as seriously as the kernels.

**9. A threshold on the wrong quantity.** `stage_broadcast_compute` stages a broadcast tensor when
`_fused_work(base) >= 4096`, and `_fused_work` is (ALU ops) × (elements of the *small* tensor):

@include uopc/kernelize.py#_fused_work

The cost of *not* staging is recomputation, which is proportional to the elements of the *broadcast* result.
At B = 16 several tiny tensors hit the threshold exactly: softmax's `-max` and `1/sum` (1 op × 16·4·64 = 4096
elements) and LayerNorm's `1/σ` (4 ops × 1024 = 4096). They become nine extra kernels (`e_16_4_64_1`,
`e_16_64_1`), and at B = 8 they disappear: the forward pass has 52 kernels instead of 61. In this model the
effect on step time was within noise (halving `_fused_work` gave 142–146 ms against 144–149 ms), so it is
harmless here. It is still a cost model whose decisions flip with the batch size for reasons unrelated to cost,
which is a good thing to find before it matters. (This is NOTES 15's "useless kernels" coming back by another
door.)

## Exercises

1. ★ **Count parameters.** Derive the 112,577 of Section 14.2 for `--dim 128 --layers 4 --heads 4` (V = 65,
   T = 64) by hand, then check with `sum(p.numel() for p in parameters(model))`.
2. ★ **Shapes.** For B=2, T=5, C=8, H=2, write the shape after every line of `Attention.__call__`, including
   the 5-D broadcast inside each matmul.
3. ★ **Read a kernel.** In `e_16_64_64` (Section 14.5), which `dataN` is the position table, which is
   LayerNorm's weight, and how can you tell from the index expressions alone?
4. ★★ **The cache, broken on purpose.** Add a cosine learning-rate schedule to AdamW. First pass `lr` as a
   Python float and measure ms/step. Then make it a tensor so the plan is reused. Explain the difference using
   `_structure_key`.
5. ★★ **Weight tying.** GPT-2 ties the head to the token embedding (`logits = x @ W_tokᵀ`). Implement it.
   How many parameters does it save here? What happens to the embedding's gradient, and why does autodiff
   handle the sum of two contributions without any change?
6. ★★ **Find the bug.** A student's causal mask is `np.tril(np.full((ctx, ctx), -1e9), -1)`. Their loss falls
   fast, to well under 1.0 in a few hundred steps, and their samples are gibberish. Explain both observations,
   and which test in `tests/test_w09_models.py` catches it.
7. ★★ **Weight-gradient kernels.** `r_192_64` reduces over B·T = 1024 positions at about 4 GFLOP/s. Dump its C
   with `DEBUG=4`, identify the strided operand, and change the layout heuristic (or add a rule) so that
   reductions over a leading axis transpose first. Measure the step time before and after.
8. ★★ **The leftover modulo.** Add the bounds-based FLOORMOD/FLOORDIV rule of Pitfall 2 to `uopc/symbolic.py`.
   Verify that the attention kernel's `% 192` disappears, run the whole test suite, and measure the compile
   time (it runs on every index, so it must be cheap).
9. ★★★ **A better cost model.** Change `stage_broadcast_compute` to estimate the actual recomputation cost
   (ops × elements of the broadcast consumer) against the cost of staging (one write and the reads), so that
   the decision doesn't depend on crossing 4096. Show that the nine small kernels disappear at B=16 and that
   LayerNorm and GELU outputs are still staged.
10. ★★★ **GPT on the GPU.** Make `make_plan` able to target the GPU path of Chapter 12: lower each kernel with
    `gpu_lower` (folding axes for 4-D kernels such as `r_16_4_64_64`), render with `render_cuda`, and run each
    kernel through `emulate` on the CPU so that `test_tiny_gpt_learns` passes on the emulator. With a GPU, run
    it through `cuda.py` and compare ms/step with the CPU.
11. ★★★ **Gradient clipping.** Add global-norm gradient clipping to the optimizer *without* an extra realize:
    the norm, the scale factor and the update must all be in the same schedule. Why does it have to be a
    tensor expression and not `float(norm)`?

<details>
<summary>Solutions to ★ exercises</summary>

**1.** Token embedding 65·128 = 8,320. Positions 64·128 = 8,192. Per block: two LayerNorms 4·128 = 512, QKV
128·384 + 384 = 49,536, projection 128·128 + 128 = 16,512, MLP up 128·512 + 512 = 66,048, MLP down 512·128 +
128 = 65,664, which totals 198,272, and ×4 = 793,088. Final LayerNorm 256, head 128·65 + 65 = 8,385. Total:
8,320 + 8,192 + 793,088 + 256 + 8,385 = **818,241** (≈ 818.2k).

**2.** `x` (2,5,8). `qkv(x)` (2,5,24), broadcast to (2,5,24,8) inside the matmul. Reshape (2,5,3,2,4).
`qkv[:,:,i]` (2,5,2,4). Permute → q, k, v (2,2,5,4). `k.transpose()` (2,2,4,5). `q @ kᵀ` broadcasts to
(2,2,5,5,4) and reduces to (2,2,5,5). The scale and mask keep (2,2,5,5), with the mask reshaped (1,1,5,5).
Softmax (2,2,5,5). `@ v` broadcasts to (2,2,5,4,5) and reduces to (2,2,5,4). Permute (2,5,2,4), reshape
(2,5,8), proj (2,5,8).

**3.** `data1[(alu1+ridx2)]` is indexed by (t, c) only, with no batch term: the position table, (T, C).
`data4[ridx2]` and `data5[ridx2]` are indexed by c only: LayerNorm's weight and bias, in the order they
appear in the expression (`*val8` then `+val9`), so `data4` is the weight. `data0` has the batch term
`ridx0*4096`: the token embeddings (B, T, C). `data2` and `data3` are indexed by (b, t): the row sums and
1/σ.

</details>

## Quiz

1. What loss should an untrained character model with V = 65 report, and why?
2. Why does `Embedding` use a one-hot matmul, and why doesn't it materialize a (B, T, V) tensor?
3. Why is the mask −10⁹ and not −∞?
4. In the generated softmax kernels, where did the constant `0.25f` come from?
5. Why must AdamW's bias corrections be tensors rather than Python floats in `uopc`?
6. A training step is 243 kernels but only 128 distinct C sources. Why?
7. Why does the sampler pad its context to T?
8. In the ablation, removing broadcast staging made both the steady state and the *first step* slower. Why
   the first step?

<details>
<summary>Answers</summary>

1. About ln 65 ≈ 4.17. Near-uniform predictions give each character probability 1/65, and cross-entropy is
   −ln(1/65).
2. The IR has no gather or scatter, and autodiff works on pure shape graphs. A one-hot matmul expresses the
   lookup, and its gradient (a scatter-add) comes from the matmul rule. The one-hot is an expression
   (`idx == v`) pushed into the reduce loop by rangeify and fused by kernelization, so it is never stored.
3. A fully masked row would compute `-inf - (-inf)` = NaN in the max-subtraction. A large finite number gives
   `exp(-1e9)` = 0 without NaNs.
4. 1/√D with D = 16, `1 / math.sqrt(D)` in `Attention.__call__`, a Python float baked into the graph as a
   constant and folded into the multiply.
5. They change every step. As CONSTs they change the structure key, so every step misses the plan cache and
   recompiles (about 20–30× slower in the ablation). As 1-element buffers only their contents change.
6. Kernels are renumbered with local buffer slots, so identical layers (two blocks, repeated
   LayerNorms, forward and backward patterns) render to byte-identical C, and the content-addressed cache
   compiles each distinct source once.
7. So that every forward pass has the same graph shape and reuses one compiled plan. Causality guarantees the
   padding can't affect earlier positions.
8. Without staging, the broadcast computations (LayerNorm, GELU) are inlined into every consumer kernel,
   making much larger graphs for rangeify and `simplify` to process, which is where compile time goes (the
   cProfile in 14.11).

</details>

## Further reading

- A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- A. Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), OpenAI 2019: pre-LayerNorm,
  GELU, weight tying.
- A. Karpathy, *nanoGPT* and *minGPT* (GitHub), and the lecture "Let's build GPT: from scratch, in code,
  spelled out" (2023). Our `examples/gpt.py` follows the same architecture. Also *char-rnn*, the source of the
  Tiny Shakespeare dataset.
- J. Ba, J. Kiros and G. Hinton, "Layer Normalization", 2016.
- D. Hendrycks and K. Gimpel, "Gaussian Error Linear Units (GELUs)", 2016.
- D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015. I. Loshchilov and F. Hutter,
  "Decoupled Weight Decay Regularization", ICLR 2019.
- J. Kaplan et al., "Scaling Laws for Neural Language Models", 2020: the 6·N·D FLOP estimate.
- tinygrad: `examples/` (GPT-2 and LLaMA implementations) and its JIT (`TinyJit`), which caches whole
  schedules much like `_plans`.
