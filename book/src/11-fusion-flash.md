# 11. Fusion, recompute, and flash attention

Every transformer layer computes `softmax(Q K^T / sqrt(D)) V`. Written the obvious way, it builds an
`N x N` matrix of scores, where `N` is the sequence length: 4 MB at `N = 1024`, 64 MB at 4096, a gigabyte
at 16,384. That matrix is pure intermediate: nobody wants it, it is consumed as soon as it is produced,
and moving it to memory and back is most of the work. [Chapter 10](10-cpu-performance.md) asked how to
make one kernel fast. This chapter asks the question one level up: *where should the kernel boundaries
be?* Cutting less (fusing more) saves memory traffic, but a fused consumer may have to recompute what it
no longer reads. We will implement attention three ways in `uopc`, measure all three, take apart a bug
that made fused attention cubic, and derive the trick that makes attention fusable in a single pass:
the **online softmax** at the heart of FlashAttention.

**What you'll build**

- `examples/flash_attention.py`: unfused attention (the Tensor code from
  [Chapter 9](09-tensor-mnist.md)), fused-with-recompute attention (the same code with
  `fuse_reduce=True`), and a hand-written FlashAttention-style kernel in the week-3 loop IR.
- An understanding of `fuse_reduce`, `stage_broadcast_compute` and accumulator placement good enough to
  predict, from the shape graph, how many times a kernel will recompute `Q K^T`.

**Where this fits.** This chapter uses the kernelizer from [Chapter 7](07-kernelize.md), the
reduce lowering from [Chapter 6](06-rangeify.md) and the opts from
[Chapter 10](10-cpu-performance.md). [Chapter 13](13-shared-memory-tensor-cores.md) supplies the GPU
machinery (shared memory, tensor cores) that real FlashAttention kernels are built from, and
[Chapter 14](14-gpt.md) puts attention inside a GPT.

## 11.1 Fusion is a trade

[Chapter 7](07-kernelize.md) cut the graph into kernels at *materialization points*: the roots, every
STAGE, and every REDUCE. Everything between two points fuses into one loop nest:

@include uopc/kernelize.py#_materialize_points

Each cut has a price and a benefit. The price of a cut is memory traffic: the producer writes its result
to a buffer and the consumer reads it back, `2 * size` bytes plus a kernel launch. The benefit is that
the value is computed once. Fusing across the cut removes the traffic, but if the consumer reads each
producer value more than once (because of a broadcast), the fused consumer recomputes it every time.

So the decision for a candidate fusion is roughly

`fuse if  (bytes saved) / (memory bandwidth)  >  (FLOPs recomputed) / (compute rate)`

and both sides can be large. In [Chapter 9](09-tensor-mnist.md)'s softmax, fusing recomputes one `exp2f`
per element in two kernels instead of storing the exponentials, a clear win. NOTES 15 records the other
extreme: in a GPT, a LayerNorm output EXPANDed into a matmul was recomputed for every output column, and
one training step spent 374 ms in kernels until `stage_broadcast_compute` started staging such tensors
(125 ms; see [Chapter 7](07-kernelize.md), Section 7.10):

@include uopc/kernelize.py#stage_broadcast_compute

@include uopc/kernelize.py#_fused_work

The rule is a crude cost model: stage a computed tensor if it is broadcast by a factor of 4 or more *and*
recomputing it would cost at least 4096 ALU-op-elements. Remember this function: the first version of
fused attention ran straight into it (11.4).

REDUCE is the interesting boundary. A REDUCE consumed through an EXPAND (`x - x.max(-1, keepdim=True)`)
must be materialized, or the reduction would be recomputed for every element it is broadcast to. But
with `fuse_reduce=True`, `_materialize_points` no longer cuts at REDUCEs: rangeify then lowers a reduce
*inside* the consumer's loops, and each output position recomputes the reduction it needs. That is
exactly the "fused attention" we want, provided the recomputation happens once per *row*, not once per
*element*. Whether it does is decided by accumulator placement, as section 11.5 will show the hard way.

## 11.2 Attention, and why it is memory-bound

Single-head attention over a sequence of `N` positions with head dimension `D`:

- `S = Q K^T / sqrt(D)`: scores, `(N, N)`. `S[i, j]` is how much query `i` attends to key `j`.
- `P = softmax(S)` along each row: `(N, N)`, each row a probability distribution.
- `O = P V`: `(N, D)`, each output row a weighted average of the value rows.

The FLOPs are two matmuls, `2 N^2 D` each, `4 N^2 D` in total (268 million at `N = 1024, D = 64`), plus
`O(N^2)` for the softmax. The memory is dominated by `S` and `P`: `N^2` floats each, 4.2 MB at `N = 1024`
and 16.8 MB at 2048, against 256 KB for each of `Q`, `K`, `V`, `O`. On the roofline of
[Chapter 10](10-cpu-performance.md), the matmuls have intensity `O(D)` and the softmax has intensity
`O(1)`; the more `N^2` intermediates we write and re-read, the further left the whole computation slides.

The Tensor code is one line:

@include examples/flash_attention.py#attention

@diagram 11-attention-memory

## 11.3 Implementation 1: unfused

Kernelized normally, the one-liner becomes six kernels. Here are the first and the fifth `realize` of
the same attention in one process (`DEBUG=2`, `N = 1024`, `D = 64`):

```text
  r_1024_1024                       5.380 ms    24.95 GFLOP/s
  r_1024                            0.786 ms     2.67 GFLOP/s
  r_1024                            1.807 ms     3.48 GFLOP/s
  e_1024_1024                       3.174 ms     2.31 GFLOP/s
  e_1_64_1024                       0.370 ms     0.00 GFLOP/s
  r_1024_64                         1.694 ms    79.22 GFLOP/s
realize: 6 kernels, schedule 139.8 ms, run 13.6 ms
...
  r_1024_1024                       1.980 ms    67.77 GFLOP/s
  r_1024                            0.903 ms     2.32 GFLOP/s
  r_1024                            2.288 ms     2.75 GFLOP/s
  e_1024_1024                       1.869 ms     3.93 GFLOP/s
  e_1_64_1024                       0.283 ms     0.00 GFLOP/s
  r_1024_64                         1.503 ms    89.30 GFLOP/s
realize: 6 kernels, schedule 0.4 ms, run 9.1 ms
```

In order: `S = Q K^T` (a tiled GEMM, the same 4x4 tile as in [Chapter 10](10-cpu-performance.md); the
layout pass found `K` already contiguous along `D`, so no copy), the row max of `S` scaled by `0.125`, the
row sum of exponentials, `P` itself, a transposed copy of `V` (the layout pass at work, because `P V`
reduces over `j` and `V` is `(N, D)`), and `O = P V`. The max and sum kernels each re-read all of `S`:

```c
void r_1024(float* restrict data0, float* restrict data1) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 256; ridx0++) {
    int alu1 = (ridx0*4);
    float acc0;
    acc0 = -INFINITY;
    float acc1;
    acc1 = -INFINITY;
    float acc2;
    acc2 = -INFINITY;
    float acc3;
    acc3 = -INFINITY;
    for (int ridx1 = 0; ridx1 < 1024; ridx1++) {
      float val2 = acc0;
      int alu3 = ((ridx0*4096)+ridx1);
      float val4 = data0[alu3];
      acc0 = ((val2>(val4*0.125f))?val2:(val4*0.125f));
      float val5 = acc1;
      float val6 = data0[(alu3+1024)];
      acc1 = ((val5>(val6*0.125f))?val5:(val6*0.125f));
      float val7 = acc2;
      float val8 = data0[(alu3+2048)];
      acc2 = ((val7>(val8*0.125f))?val7:(val8*0.125f));
      float val9 = acc3;
      float val10 = data0[(alu3+3072)];
      acc3 = ((val9>(val10*0.125f))?val9:(val10*0.125f));
    }
    float val11 = acc0;
    data1[alu1] = val11;
    float val12 = acc1;
    data1[(alu1+1)] = val12;
    float val13 = acc2;
    data1[(alu1+2)] = val13;
    float val14 = acc3;
    data1[(alu1+3)] = val14;
  }
}
```

and `P` re-reads `S` once more, recomputing the scale and the exponential, and writes the second
`N x N` buffer:

```c
void e_1024_1024(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 1024; ridx0++) {
    float val1 = data1[ridx0];
    float val2 = data2[ridx0];
    for (int ridx1 = 0; ridx1 < 1024; ridx1++) {
      int alu3 = ((ridx0*1024)+ridx1);
      float val4 = data0[alu3];
      data3[alu3] = (exp2f((((val4*0.125f)+(val1*-1.0f))*1.4426950408889634f))*(1.0f/val2));
    }
  }
}
```

Count the passes over `N x N` data: `S` is written once and read three times, `P` is written once and
read once. Six passes, 25 MB of traffic at `N = 1024`, for a computation whose inputs and output total
1 MB. And compare the first lines of the two runs: the GEMM writing `S` took 5.4 ms the first time and
2.0 ms the fifth. The difference is not computation, it is the operating system: the first write to each
4 KB page of a fresh 4 MB buffer takes a page fault. Large intermediates cost twice: once in bandwidth,
once in page faults.

The first version of the runtime paid the page faults on *every* call, because `realize` allocated the
plan's temporaries with `np.empty` each time; a `DEBUG=3` run then showed the `S` kernel at 9.7 ms on its
first execution and 1.1 ms when re-run immediately on the same (now mapped) memory. The fix is one field
on the cached `Plan`: the temporary arrays are allocated on the first run and reused by every later one.

@include uopc/kernelize.py#Plan

and in `realize`:

@include uopc/kernelize.py:203-204

With that change the captured unfused time at `N = 1024` dropped from 28.4 ms to 9.8 ms (11.8).

## 11.4 Implementation 2: fused, with recompute

The second implementation changes no math, only the kernelizer's options: `fuse_reduce=True` stops
cutting at REDUCEs, and a custom optimizer upcasts the last output axis and threads the first:

@include examples/flash_attention.py#FlashOpt

With `fuse_reduce`, the reductions of softmax are lowered *inside* the loops of their consumer. The plan
shrinks from six kernels and two `N x N` temporaries to two kernels and none:

```text
fuse_reduce=False: kernels ['r_1024_1024', 'r_1024', 'r_1024', 'e_1024_1024', 'e_1_64_1024', 'r_1024_64'], temps [1048576, 1024, 1024, 1048576, 65536]
fuse_reduce=True: kernels ['e_1_64_1024', 'r_1024_64'], temps [65536]
```

The only temporary left is the transposed copy of `V` (65,536 floats). Everything else happens in one
kernel, `r_1024_64`, which produces `O` directly. Here is its first part:

```c
void r_1024_64(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 1024; ridx0++) {
    int alu1 = (ridx0*64);
    float acc2;
    acc2 = -INFINITY;
    float acc4;
    acc4 = 0.0f;
    for (int ridx4 = 0; ridx4 < 1024; ridx4++) {
      float val2 = acc2;
      float acc1;
      acc1 = 0.0f;
      for (int ridx5 = 0; ridx5 < 64; ridx5++) {
        float val3 = acc1;
        float val4 = data0[(alu1+ridx5)];
        float val5 = data1[((ridx4*64)+ridx5)];
        acc1 = (val3+(val4*val5));
      }
      float val6 = acc1;
      acc2 = ((val2>(val6*0.125f))?val2:(val6*0.125f));
    }
    float val7 = acc2;
    float alu8 = (val7*-1.0f);
    for (int ridx6 = 0; ridx6 < 1024; ridx6++) {
      float val9 = acc4;
      float acc3;
      acc3 = 0.0f;
      for (int ridx7 = 0; ridx7 < 64; ridx7++) {
        float val10 = acc3;
        float val11 = data0[(alu1+ridx7)];
        float val12 = data1[((ridx6*64)+ridx7)];
        acc3 = (val10+(val11*val12));
      }
      float val13 = acc3;
      acc4 = (val9+exp2f((((val13*0.125f)+alu8)*1.4426950408889634f)));
    }
    float val14 = acc4;
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu15 = (ridx1*16);
      float acc5;
      acc5 = 0.0f;
      ...
      float acc20;
      acc20 = 0.0f;
      int alu16 = (alu15+alu1);
      for (int ridx2 = 0; ridx2 < 1024; ridx2++) {
        float val17 = acc5;
        float acc0;
        acc0 = 0.0f;
        int alu18 = ((ridx1*16384)+ridx2);
        float val19 = data2[alu18];
        ...
        for (int ridx3 = 0; ridx3 < 64; ridx3++) {
          float val50 = acc0;
          float val51 = data0[(alu1+ridx3)];
          float val52 = data1[((ridx2*64)+ridx3)];
          acc0 = (val50+(val51*val52));
        }
        float val53 = acc0;
        float alu54 = (exp2f((((val53*0.125f)+alu8)*1.4426950408889634f))*(1.0f/val14));
        acc5 = (val17+(alu54*val19));
        acc6 = (val20+(alu54*val21));
        ...
```

(the `...` elide the other fourteen accumulators; the whole kernel is 172 lines). This is a *nested*
reduction: an inner reduce (the dot product over `e < 64`) inside an outer reduce (max, sum, or the
`P V` sum over keys), inside the row loop. Rangeify produced it with no new code: the reduce lowering
simply recursed. Per query row `ridx0` there is a max pass over keys (`ridx4`), a sum pass (`ridx6`),
and then the output loop `ridx1 < 4`, each iteration of which makes *another* full pass over the keys
(`ridx2`), recomputing every dot product and every exponential and feeding sixteen output accumulators.
`FlashOpt` upcasts the head dimension by `min(D, 16) = 16`, so the 64 output columns come out in four
groups of sixteen, and each group pays for its own pass. That is six full computations of `q_i . k_j` per
row instead of one.

The first version of this experiment did not look like this. `stage_broadcast_compute` (11.1) saw `P`
EXPANDed by a factor of 64 into the `P V` matmul, measured its recompute cost, and *staged* it, exactly
as designed for LayerNorm. The "fused" plan therefore still wrote one `N x N` buffer (`P`), while the
script printed "(no N*N buffer, QK^T recomputed)". Two cost-model rules disagreed: `fuse_reduce` meant
"never write the big intermediate", and the broadcast rule meant "never recompute an expensive
broadcast". The fix makes the user's explicit request win: with `fuse_reduce=True`, `make_plan` skips
`stage_broadcast_compute`:

@include uopc/kernelize.py:121-125

and the example's docstring now describes the recomputation accurately. The lesson is about debugging,
not attention: when a benchmark's description and its plan disagree, believe the plan
(`make_plan(...).temps`).

So does fusing win? The captured runs (11.8) say no, by a wide margin: 28.5 ms fused against 9.8 ms
unfused at `N = 1024`, and 103.8 against 35.3 ms at 2048. Fusion removed 25 MB of `N x N` traffic and
added five extra computations of `Q K^T`, each done without the register tiling that makes the unfused
`Q K^T` GEMM fast. On this machine the recomputation costs far more than the traffic did.

> **Key idea:** fusing a reduction into its consumer is only cheap if the reduction is computed once per
> *consumer row*, and if the consumer itself makes one pass. Every extra pass recomputes everything the
> fused reduction depends on. The fix is not a better cost model; it is a better algorithm (11.6).

## 11.5 The O(N^3) bug: when an index lies about its loops

Before the version above existed, fused attention had a much worse problem. At `N = 512` it took
1787 ms; after the fix described below, 7 ms (NOTES 12). This section tells the full story, because it is the
cleanest example in this book of a semantic property (*which loops does this value depend on?*) being
decided by a syntactic one (*which loop variables does this expression mention?*).

Recall how rangeify lowers a REDUCE ([Chapter 6](06-rangeify.md)):

@include uopc/rangeify.py#_reduce

The accumulator is placed with `acc.after(*ranges_in(*idx))`: it lives inside exactly the loops that
the output index `idx` *mentions*. For the softmax max inside attention, the output index is the row `i`,
so the accumulator sits inside the `i` loop and outside the key loops. One max per row: correct.

Now remember how rangeify handles a RESHAPE: the index is raveled into a flat offset and unraveled into
the source shape. When `x` of shape `(N,)` is viewed as `(N, 1)` and expanded to `(N, N)`, the reshape
asks for `x[unravel(ravel((i, j)))]`, and the first component is `((i*N + j) // N) % N`. Mathematically
that is `i`: `0 <= j < N`, so `j // N = 0`. Syntactically it mentions `j`. A scratch script that prints
rangeify's raw index expressions for an `(N, N)` index and what `simplify` makes of them:

```text
raw ((((((0*512)+i)*512)+j)//512)%512) mentions ['i', 'j']   simplified i
raw ((((((0*512)+i)*512)+j)//1)%512) mentions ['i', 'j']   simplified j
```

If the index reaching a REDUCE is the raw form, `ranges_in` reports both `i` and `j`, and the accumulator
for "max over the row" is placed inside the `j` loop of its consumer. The reduction is then recomputed
for every `j`. Each recomputation is itself a pass over `N` keys with a `D`-long dot product inside, so
the kernel does `O(N^2 * N * D)` work instead of `O(N * N * D)`: cubic in the sequence length.

The fix is in the RESHAPE rule of `rangeify`: simplify the new indices immediately, so that every later
decision sees the truth:

@include uopc/rangeify.py#rangeify

The comment on the RESHAPE branch says why: dependence analysis and the memo both need the simplified
form. (The memo matters too: `rangeify` caches on `(node, idx)`, and two spellings of the same index are
two different cache keys, so unsimplified indices also defeat memoization.)

We can watch the bug come back. A scratch script patches `uopc.rangeify.simplify` so that it does
nothing when called from the RESHAPE rule (and works normally everywhere else, including the final
cleanup in `lower`), runs the fused attention of 11.4, and prints
the loops and accumulator initializations of the attention kernel at `N = 512`:

```text
  for (int ridx0 = 0; ridx0 < 512; ridx0++) {
    acc2 = -INFINITY;
    acc4 = 0.0f;
    for (int ridx4 = 0; ridx4 < 512; ridx4++) {
      acc1 = 0.0f;
      for (int ridx5 = 0; ridx5 < 64; ridx5++) {
    for (int ridx6 = 0; ridx6 < 512; ridx6++) {
      acc3 = 0.0f;
      for (int ridx7 = 0; ridx7 < 64; ridx7++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      acc5 = 0.0f;
      ...
      acc20 = 0.0f;
      for (int ridx2 = 0; ridx2 < 512; ridx2++) {
        acc0 = 0.0f;
        for (int ridx3 = 0; ridx3 < 64; ridx3++) {
N=512 reshape-simplify on: 9.2 ms  err 3.4e-07
  for (int ridx0 = 0; ridx0 < 512; ridx0++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      acc7 = 0.0f;
      ...
      acc22 = 0.0f;
      for (int ridx2 = 0; ridx2 < 512; ridx2++) {
        acc0 = 0.0f;
        acc2 = -INFINITY;
        acc5 = -INFINITY;
        acc6 = 0.0f;
        for (int ridx3 = 0; ridx3 < 64; ridx3++) {
        for (int ridx4 = 0; ridx4 < 512; ridx4++) {
          acc1 = 0.0f;
          acc4 = 0.0f;
          for (int ridx5 = 0; ridx5 < 64; ridx5++) {
        for (int ridx6 = 0; ridx6 < 512; ridx6++) {
          acc3 = 0.0f;
          for (int ridx7 = 0; ridx7 < 64; ridx7++) {
N=512 reshape-simplify off: 2210.7 ms  err 3.4e-07
```

(lines listing `acc6` to `acc19` and `acc8` to `acc21` elided). With simplification, the row max
(`acc2 = -INFINITY`) and row sum (`acc4 = 0.0f`) are initialized once per row `ridx0`, before their passes
over keys. Without it, they are initialized inside `ridx2`, the key loop of the `P V` product, so both
passes over all keys run again for every key. Same graph, same answer to seven digits, 240 times slower (NOTES 12 recorded 250x):

@diagram 11-n3-loops

Two lessons. First, a compiler's analyses are only as good as the canonical form they see: every
"which variables does this depend on" question must be asked of a simplified expression, or it is
answered about the spelling instead of the value. Second, this bug produced *correct output*. No test
comparing values against NumPy could catch it. Only a performance test (or reading the generated C)
could. That is why the course grades performance, and why it pays to assert loop structure in tests, as
`test_upcast_shares_reduce_loop` does (see [Appendix B](17-appendix-testing.md) on testing).

## 11.6 Online softmax

The fused kernel needed at least three passes over the keys per row (six with `FlashOpt`) because softmax seems to need its row max
before it can compute any exponential, and its row sum before it can normalize. Online softmax
(Milakov and Gimelshein, 2018) removes both dependencies, and FlashAttention (Dao et al., 2022) applies
it to attention.

Fix one query row. Write `s_j` for its scores (`j = 1..N`), `v_j` for the value rows. The goal is

`o = sum_j softmax(s)_j v_j = (sum_j exp(s_j - m) v_j) / (sum_j exp(s_j - m))`, with `m = max_j s_j`.

(Subtracting `m` changes nothing mathematically and keeps every exponent at most 0, as in
[Chapter 9](09-tensor-mnist.md).) Now define the same quantities over only the first `j` keys:

- `m_j = max(s_1, ..., s_j)`
- `l_j = sum_{k <= j} exp(s_k - m_j)`
- `o_j = sum_{k <= j} exp(s_k - m_j) v_k`

so that `o_N / l_N` is the answer. Can we get from step `j - 1` to step `j` without revisiting old keys?
The new maximum is easy: `m_j = max(m_{j-1}, s_j)`. For the sum, every old term was computed relative to
the old maximum; relative to the new one it is smaller by the factor `exp(m_{j-1} - m_j)`:

`exp(s_k - m_j) = exp(s_k - m_{j-1}) * exp(m_{j-1} - m_j)`.

The factor does not depend on `k`, so it comes out of the sum:

`l_j = l_{j-1} * exp(m_{j-1} - m_j) + exp(s_j - m_j)`

and identically, since each `v_k` is just carried along,

`o_j = o_{j-1} * exp(m_{j-1} - m_j) + exp(s_j - m_j) v_j`.

Start from `m_0 = -inf`, `l_0 = 0`, `o_0 = 0` (the first correction factor is `exp(-inf) = 0`, which
multiplies zeros anyway). By induction, after `N` keys `l_N` and `o_N` are exactly the two sums above,
and one division finishes the job. Every key is visited once; the state is two scalars and one
`D`-vector per query row. When the maximum does not change, the correction factor is exactly 1; when it
grows, everything accumulated so far shrinks by the same factor, which is what makes the running sums
consistent.

A worked example with one query, three keys and `D = 1`, as a scratch NumPy script that prints the state
after each key and compares with the two-pass answer:

```python
# online softmax, one query, three keys, head dim 1: watch the running state
import numpy as np
s = np.array([1.0, 3.0, 2.0])          # scores q.k_j (already scaled)
v = np.array([10.0, 20.0, 30.0])       # values v_j
m, l, o = -np.inf, 0.0, 0.0
for j in range(3):
  m_new = max(m, s[j])
  corr = np.exp(m - m_new)             # rescale everything accumulated so far
  p = np.exp(s[j] - m_new)
  l = l * corr + p
  o = o * corr + p * v[j]
  m = m_new
  print(f"key {j}: s={s[j]:.0f}  m={m:.0f}  corr={corr:.4f}  p={p:.4f}  l={l:.4f}  o={o:.4f}")
w = np.exp(s - s.max()); w /= w.sum()
print(f"online o/l = {o / l:.6f}   two-pass softmax(s) @ v = {w @ v:.6f}")
```

```text
key 0: s=1  m=1  corr=0.0000  p=1.0000  l=1.0000  o=10.0000
key 1: s=3  m=3  corr=0.1353  p=1.0000  l=1.1353  o=21.3534
key 2: s=2  m=3  corr=1.0000  p=0.3679  l=1.5032  o=32.3897
online o/l = 21.546979   two-pass softmax(s) @ v = 21.546979
```

@diagram 11-online-softmax

The same algebra merges two *partial* states, as well as a state and one key. If `(m1, l1, o1)` summarizes
one set of keys and `(m2, l2, o2)` another, then with `m = max(m1, m2)` the union is
`(m, l1 e^{m1 - m} + l2 e^{m2 - m}, o1 e^{m1 - m} + o2 e^{m2 - m})`. That associativity is what lets GPU
implementations process keys in blocks, and split long sequences across thread blocks (Exercise 11.6).

## 11.7 Implementation 3: online softmax in the loop IR

The Tensor API cannot express a loop-carried running max (there is no "scan" op), so the third
implementation is written directly in the loop IR of [Chapter 5](05-loops.md), the way the week-3
assignment wrote GEMMs by hand:

@include examples/flash_attention.py#online_attention_ir

Walk through it with the online-softmax recurrence in mind.

- **State.** Four ALLOCs: `M` and `L` (one float each: the running max and sum), `O` (`D` floats: the
  running output row) and `S` (one float: the dot-product accumulator). They are call-scope storage,
  rendered as local variables.
- **The row loop.** `i` is a THREAD range: each OpenMP thread owns whole query rows. `Mi = M.after(i)`
  says "the M that lives inside loop `i`", which places the declaration inside the parallel loop: each
  thread gets private state (NOTES 9).
- **Initialization.** `init_m` stores `-inf`, `init_l` stores 0, `init_o` zeroes the `D` floats in a small
  loop `d0`.
- **The key loop `j`.** First the dot product `q_i . k_j`, a reduction over `e` written exactly like the
  week-3 GEMM: initialize `S`, accumulate in loop `e`, `END` it, read it after the loop. It is scaled by
  `scale = (1/sqrt(D)) * log2(e)`, because the IR's exponential is `EXP2`: `exp(x) = exp2(x * log2 e)`, so
  folding `log2 e` into the scale makes every later exponential a bare `exp2`.
- **The update.** `m_new = max(m_old, s)`, `corr = exp2(m_old - m_new)`, `p = exp2(s - m_new)`, then
  `l = l * corr + p` and, in loop `d`, `o[d] = o[d] * corr + p * v[j, d]`. Finally `m = m_new`.
  `Mj = Mi.after(init_m, j)` is the AFTER-edge idiom from [Chapter 4](04-memory-and-c.md): "read M inside
  loop `j`, after its initialization".
- **Closing the key loop.** `UOp(Ops.SINK, (st_l, st_o, st_m)).end(j)` closes `j` over all three stores
  at once; it is the same SINK-under-END shape that `merge_ends` builds for register tiles.
- **Normalization.** After the key loop, `out[i, d] = o[d] / l` in a last loop `d2`.

All the ordering subtleties ("store the new max only after the old one has been used") are carried by
data dependencies: `st_m` stores `m_new`, which is computed from `m_old`, which was loaded once; `corr`
and `p` are values, not memory reads. The linearizer ([Chapter 5](05-loops.md)) turns this into:

```c
void flash(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 1024; ridx0++) {
    int alu1 = (ridx0*64);
    float acc2[64];
    float acc1;
    acc1 = 0.0f;
    float acc0;
    acc0 = -INFINITY;
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      acc2[ridx1] = 0.0f;
    }
    for (int ridx2 = 0; ridx2 < 1024; ridx2++) {
      float val2 = acc1;
      float val3 = acc0;
      float acc3;
      acc3 = 0.0f;
      int alu4 = (ridx2*64);
      for (int ridx3 = 0; ridx3 < 64; ridx3++) {
        float val5 = acc3;
        float val6 = data0[(alu1+ridx3)];
        float val7 = data1[(alu4+ridx3)];
        acc3 = (val5+(val6*val7));
      }
      float val8 = acc3;
      float alu9 = (val8*0.18033688011112042f);
      float alu10 = ((val3>alu9)?val3:alu9);
      float alu11 = (alu10*-1.0f);
      float alu12 = exp2f((val3+alu11));
      float alu13 = exp2f((alu9+alu11));
      acc1 = ((val2*alu12)+alu13);
      acc0 = alu10;
      for (int ridx4 = 0; ridx4 < 64; ridx4++) {
        float val14 = acc2[ridx4];
        float val15 = data2[(alu4+ridx4)];
        acc2[ridx4] = ((val14*alu12)+(alu13*val15));
      }
    }
    float val16 = acc1;
    for (int ridx5 = 0; ridx5 < 64; ridx5++) {
      float val17 = acc2[ridx5];
      data3[(alu1+ridx5)] = (val17*(1.0f/val16));
    }
  }
}
```

`0.18033688011112042f` is `log2(e) / 8`. `acc2[64]` is the running output row, `acc1` the running sum,
`acc0` the running max, and `acc3` the dot product; every one of them is declared inside the parallel
loop. Each query row reads all of `K` and `V` (256 KB each), which stay in the 1.25 MB L2 cache of a core
across rows, and nothing `N x N` exists anywhere.

## 11.8 Results

The captured runs (`book/outputs/flash_1024.txt` and `flash_2048.txt`, `D = 64`; each time is the best of
three after a warm-up, and includes building the graph and copying `Q`, `K`, `V` into tensors):

```text
unfused      9.82 ms  err 3.3e-07   (materializes 4.2 MB of scores)
fused       28.48 ms  err 3.6e-07   (no N*N buffer, QK^T recomputed)
online      10.00 ms  err 4.0e-07   (one pass, running max/sum)
```

```text
unfused     35.28 ms  err 3.0e-07   (materializes 16.8 MB of scores)
fused      103.82 ms  err 3.9e-07   (no N*N buffer, QK^T recomputed)
online      33.46 ms  err 3.6e-07   (one pass, running max/sum)
```

Three stories are in these six numbers. The fully fused kernel is the slowest by a factor of about three
at both sizes: six unblocked computations of `Q K^T` per row cost more than 25 MB (or 100 MB at 2048) of
traffic. The unfused version is now competitive with the online kernel, because its `Q K^T` and `P V`
are register-tiled, threaded GEMMs and, since temporaries are cached, it no longer pays page faults on
every call. (With the first version of the runtime, the captured unfused times were 28.4 and 35.8 ms.)
And the online kernel, which does the least work and moves the least memory, only ties: at `N = 1024`
the two are within noise, at 2048 online is 5% ahead. Its `N x N`-free memory behaviour matters more as
`N` grows and as memory gets scarcer, which is why it wins decisively on GPUs, but on this CPU at these
sizes its arithmetic is simply less efficient.

Do not overlook how far the online kernel is from the roofline. It performs 268 MFLOP at `N = 1024` in
about 10 ms: 27 GFLOP/s, while the unfused version's register-tiled `Q K^T` GEMM ran at 68 GFLOP/s in the
noisy run of 11.3 and over 120 GFLOP/s in quieter ones. The
reason is everything [Chapter 10](10-cpu-performance.md) taught: each thread handles one query row, so
every `k_j` and `v_j` loaded from cache is used for exactly one query (no register tile across
queries), and the dot product is a single dependency chain. The natural next step is to process a *block*
of queries per thread, so that each loaded key and value row serves the whole block: the register tiling
of Chapter 10 applied to attention (Exercise 11.5). That step is precisely what FlashAttention does on a
GPU.

## 11.9 How FlashAttention tiles on a GPU

A GPU ([Chapter 12](12-gpus.md)) has far more arithmetic per byte of memory bandwidth than this CPU, so
an `N x N` intermediate in global memory (HBM) is even more ruinous there. It also has a small, fast,
programmer-managed memory per thread block, *shared memory* (SRAM), covered in
[Chapter 13](13-shared-memory-tensor-cores.md). FlashAttention (Dao et al., 2022) and FlashAttention-2
(Dao, 2023) organize the online softmax around it:

1. **Grid.** One thread block per (batch, head, block of `B_r` query rows). FlashAttention-2 made the
   query-block dimension the parallel one, so long sequences fill the GPU even at batch size 1.
2. **Load a query block.** The block's `B_r x D` slice of `Q` is loaded once and kept on chip.
3. **Loop over key blocks.** For each block of `B_c` keys: load `K_j` and `V_j` (`B_c x D` each) into
   shared memory; compute the score tile `S = Q_i K_j^T` (`B_r x B_c`) with tensor-core matrix
   multiplies; update each row's running max `m` and sum `l` from the tile; rescale the output accumulator
   by `exp(m_old - m_new)` and add `exp(S - m_new) V_j`.
4. **Finish.** Divide the accumulator by `l` and write the `B_r x D` output block. FlashAttention-2 keeps
   the accumulator unnormalized until this final step (as our kernel does), which saves a division per
   key block. It also stores each row's log-sum-exp `m + log(l)`.
5. **Backward.** The backward pass needs `P`, which was never stored. It *recomputes* `S` and `P` block by
   block from `Q`, `K` and the saved log-sum-exp. Recomputing is cheaper than reading an `N x N` matrix
   from HBM: the fusion trade of 11.1, decided in favour of FLOPs.

The structure is the online softmax of 11.6 with the "block of queries" fix of 11.8 and the shared-memory
tiling of a GEMM. Everything `N x N` lives only as a `B_r x B_c` tile in registers. Memory traffic is
`O(N D)` per head instead of `O(N^2)`, and the remaining cost is two matmuls per tile, which tensor cores
execute near peak. In `uopc` terms, the score tile is a WMMA-lowered GEMM
([Chapter 13](13-shared-memory-tensor-cores.md)) inside the key loop of `online_attention_ir`, with
`m`, `l` and the `O` tile held in LOCAL ALLOCs. Writing it is a fine final project.

## Pitfalls

**Unsimplified indices change the loop structure (NOTES 12).** An index that mentions a loop variable it
does not depend on (`((i*N + j)//N) % N`) makes the reduce lowering place an accumulator inside that loop.
Fused attention became cubic: 9.2 ms became 2211 ms at `N = 512` in the reproduction above, with identical
output. Simplify indices before any analysis that asks which loops an expression uses, and test
performance as well as values.

**A fused kernel's passes multiply its recompute.** With `D = 64` and `FlashOpt`'s upcast of 16, the
fully fused kernel makes four output passes plus two softmax passes, each recomputing `Q K^T` for the row,
and loses to the unfused version by a factor of three. Before fusing a reduction into a consumer, count how many times
the consumer's loops will run the fused reduction.

**The label said "no N*N buffer".** In the first version it was wrong: `stage_broadcast_compute` staged
`P` because it is broadcast into `P V`, so "fused" attention still wrote one `N x N` buffer. Cost-model
rules interact. The fix lets `fuse_reduce=True` skip the broadcast-staging rule (11.4); the general habit
is to believe the plan (`make_plan(...).temps`) when it disagrees with a benchmark's description.

**Fresh temporaries page-fault.** The first version allocated the plan's temporaries with `np.empty` on
every `realize`, so every call paid a page fault for each 4 KB page it wrote: the `S` GEMM took 9.7 ms on
its first execution and 1.1 ms when re-run on the same memory. Caching the arrays on the `Plan` (11.3)
cut the captured unfused time from 28.4 to 9.8 ms. When a kernel that writes a large buffer looks slow,
run `DEBUG=3` and compare the "again" column before optimizing its loops.

**Online softmax and fully masked rows.** With masking (causal attention sets future scores to `-inf`),
a row whose scores are all `-inf` gives `m_old = m_new = -inf` and `corr = exp2(-inf - (-inf)) = NaN`.
Real implementations special-case `m_new = -inf` (or use a large finite negative instead of `-inf`).
Causal attention never has an empty row, but padding masks can.

**Forgetting to rescale `o`.** Rescaling only `l` when the maximum grows produces outputs that are subtly
wrong (too large weights on early keys) and pass a sloppy test where scores are small and the maximum
rarely changes. Test with scores whose maximum appears late in the row, as in the worked example.

**Per-thread state must be declared inside the parallel loop (NOTES 9).** The running max, sum and output
row of `online_attention_ir` are ALLOCs placed by `.after(i)` inside the THREAD loop. Declared outside, all
threads would share one `acc2[64]` and the output would be a race.

## Exercises

1. ★ **Size the matrices.** How large are `S` and `P` for `N = 8192` in `f32`, per head? A GPT-style model with 12 heads,
   batch 8: how much memory would unfused attention need for one layer's `S`?
2. ★ **Count the passes.** Predict the number of full passes over the keys (each with a `D = 64` dot product) that the
   fully fused kernel of 11.4 makes per query row with `FlashOpt`'s upcast set to 8, 16, 32 and 64. Which one
   needs the most registers?
3. ★ **Merge two states.** Prove the merge rule for two partial states `(m1, l1, o1)` and `(m2, l2, o2)` from 11.6, and
   check it numerically by splitting the worked example's keys into `{0}` and `{1, 2}`.
4. ★★ **Causal mask.** Add a causal mask to `online_attention_ir`: query `i` may only attend to keys `j <= i`. Try
   two ways: mask the scores with a WHERE (what value must a masked score get, given the NaN pitfall?), and
   make the key loop's end depend on `i` (does the IR, the linearizer and the renderer accept a RANGE whose
   end is an expression of another RANGE?). Verify against NumPy and compare the speed.
5. ★★ **Query blocking.** Process a block of 4 query rows per iteration of the THREAD loop in `online_attention_ir`, so
   that each loaded `k_j` and `v_j` serves four queries. Measure at `N = 1024` and `2048`, and compute the new
   GFLOP/s.
6. ★★ **Flash-decoding.** For `N_q = 1` query and `N = 65536` keys, the row loop has one iteration and
   one thread does everything. Split the keys into `T` chunks processed by different threads, each producing a
   partial `(m, l, o)`, and merge them with Exercise 3's rule in a second kernel.
7. ★★ **Upcast the head.** Give `FlashOpt` an upcast of the full head dimension (`D = 64`), so the fused kernel makes
   three passes instead of six. Is it faster? Then try the other choice the first version made by accident:
   run with `fuse_reduce=True` but call `stage_broadcast_compute` anyway, so `P` is stored. Rank the three
   variants and explain using register pressure and recomputation.
8. ★★ **Find the bug.** A student's online kernel computes `l = l * corr + p` and `o[d] = o[d] + p * v[j, d]`.
   On which inputs does it pass a test against NumPy, and on which does it fail? Write the failing test.
9. ★★★ **Backward with recompute.** Implement the backward pass of attention with recomputation: store only `O` and the per-row
   log-sum-exp in the forward pass, and recompute `P` block by block in the backward pass to produce `dQ`,
   `dK`, `dV`. Check against `gradient()` applied to the unfused Tensor graph.

<details>
<summary>Solutions to ★ exercises</summary>

**1.** `8192^2 * 4 = 268 MB` per head for `S` (and as much again for `P`). For 12 heads and batch 8:
`96 * 268 MB = 25.8 GB` for one layer's `S`, which is why nobody materializes it at long context.

**2.** Two softmax passes plus `64 / upcast` output passes: 10, 6, 4 and 3 passes for upcasts of 8, 16,
32 and 64. The upcast of 64 needs the most registers: 64 output accumulators live across the whole key
loop (far more than AVX2's 16 vector registers, so they spill), which is the price of making only three
passes.

**3.** Each `l` is a sum of terms `exp(s_k - m_x)`; multiplying by `exp(m_x - m)` re-expresses every term
relative to `m`, and the union's sum is the sum of the two re-expressed sums (likewise for `o`).
Numerically: `{0}` gives `(1, 1, 10)`; `{1, 2}` gives `m = 3`, `l = 1 + e^{-1} = 1.3679`,
`o = 20 + 30 e^{-1} = 31.0364`. Merging with `m = 3`: `l = 1 * e^{-2} + 1.3679 = 1.5032`,
`o = 10 e^{-2} + 31.0364 = 32.3897`, matching the sequential result, and `o / l = 21.547`.

</details>

## Quiz

1. What are the two costs on either side of a fusion decision?
2. Why does unfused attention make six passes over `N x N` data? Name them.
3. With `fuse_reduce=True` and `FlashOpt` at `D = 64`, how many times is each `q_i . k_j` computed, and
   why?
4. Write the recurrences for `m_j`, `l_j` and `o_j` in online softmax.
5. Why is the correction factor `exp(m_{j-1} - m_j)` allowed to multiply the whole running sum?
6. In the O(N^3) bug, what did the unsimplified index mention, and why did that move the accumulator?
7. Why can a value-comparison test not catch the O(N^3) bug?
8. In FlashAttention's backward pass, why is recomputing `P` cheaper than storing it?

<details>
<summary>Answers</summary>

1. Memory traffic (writing and re-reading the intermediate, plus a launch) against recomputation (FLOPs
   the fused consumer repeats because it reads each producer value more than once).
2. `S` written by the `Q K^T` kernel and read by the max, sum and `P` kernels (four passes); `P` written
   by its kernel and read by `P V` (two passes).
3. Six: once in the max pass, once in the sum pass, and once in each of the four output passes, because
   the head dimension is upcast 16 at a time and each group of 16 output columns makes its own pass.
4. `m_j = max(m_{j-1}, s_j)`; `l_j = l_{j-1} e^{m_{j-1} - m_j} + e^{s_j - m_j}`;
   `o_j = o_{j-1} e^{m_{j-1} - m_j} + e^{s_j - m_j} v_j`; answer `o_N / l_N`.
5. Every term in the sum was computed relative to `m_{j-1}`; re-expressing each relative to `m_j` multiplies
   it by the same factor, independent of `k`, so the factor comes out of the sum.
6. `((i*N + j)//N) % N` mentions `j`; the reduce lowering places accumulators inside every loop the output
   index mentions, so the row max and sum were recomputed for every `j`.
7. The output is identical (the recomputed reductions give the same values); only the amount of work
   changes. Only timing or structural assertions on the generated code can detect it.
8. Storing `P` costs `O(N^2)` writes and reads of slow global memory; recomputing it from `Q`, `K` and the
   saved log-sum-exp costs `O(N^2 D)` FLOPs on fast tensor cores, block by block, entirely on chip.

</details>

## Further reading

- A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867, 2018.
- M. Rabe and C. Staats, "Self-attention Does Not Need O(n^2) Memory", arXiv:2112.05682, 2021.
- T. Dao, D. Fu, S. Ermon, A. Rudra, C. Re, "FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness", NeurIPS 2022.
- T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023.
- J. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024.
- The `flash-attention` repository (Dao-AILab on GitHub) for the reference CUDA implementation.
