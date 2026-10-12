# Appendix A. UOp and `.uir` Reference

This appendix is the reference card you keep open while writing compiler passes. It documents every op in
`uopc.uop.Ops` exactly as the reference implementation treats it: what its sources and `arg` are, how its
dtype, shape and address space are inferred, what it means, which week introduces it, and what C it turns
into. It also gives the full grammar of the `.uir` text format, the scheduling knobs (`AxisType`, `Opts`),
and the environment variables. When this appendix and the code disagree, the code wins: `_infer` in
`uopc/uop.py` is the source of truth for types and shapes, `uopc/uir.py` for the grammar, and
`uopc/render_c.py` for the generated C.

**Where this fits:** it collects in one place what Chapters 1–13 introduce one op at a time. For how ops are
*tested* see [Appendix B](17-appendix-testing.md); for definitions of the vocabulary see
[Appendix C](18-appendix-glossary.md).

## A.1 The op set at a glance

@include uopc/uop.py#Ops

The op groups that rewrite rules and passes test against:

@include uopc/uop.py:59-63

Ops do not all live equally long. The front end builds *shape graphs*; `kernelize` cuts them into kernels;
`rangeify` replaces shapes with loops and memory; the linearizer and renderer see only scalar code. Each op
family exists in a contiguous band of that pipeline:

@diagram 16-op-lifetimes

| Op | Week | Sources | `arg` | dtype | shape | addrspace | Rendered as |
|---|---|---|---|---|---|---|---|
| `CONST` | 1 | none | `bool`, `int` or `float` | from the Python type | `()` | ALU | literal (`2.0f`, `1`, `INFINITY`) |
| `PARAM` | 1 | none | `ParamArg` | `arg.dtype` | `arg.shape` | ALU | not rendered (inlined by `run`) |
| `CALL` | 1 | body, args… | `CallArg` | body's | body's | body's | not rendered (inlined by `run`) |
| `STACK` | 1, 4 | ≥ 1, same shape and dtype | none | src's | `(n,) + src.shape` | ALU | `WHERE` chain after rangeify |
| `INDEX` | 1, 2 | x, i32 scalars… | none | x's | x.shape minus leading axes | x's | `x[i]` (memory) or index math |
| `CAST` | 1 | 1 | target `DType` | `arg` | src's | ALU | `((T)(x))` |
| `RECIP` | 1 | 1 float | none | src's | src's | ALU | `(1.0f/x)` |
| `EXP2` `LOG2` `SQRT` `SIN` | 1 (as needed) | 1 float | none | src's | src's | ALU | `exp2f(x)` etc. |
| `ADD` `MUL` | 1 | 2, same shape and dtype | none | src's | src's | ALU | `(a+b)`, `(a*b)` |
| `MAX` | 1 (as needed) | 2, same shape and dtype | none | src's | src's | ALU | `((a>b)?a:b)` |
| `CMPLT` `CMPNE` | 1 | 2, same shape and dtype | none | `bool` | src's | ALU | `(a<b)`, `(a!=b)` |
| `FLOORDIV` `FLOORMOD` | 1 | 2 `i32` | none | `i32` | src's | ALU | `/`, `%` or `floordiv_i32`/`floormod_i32` |
| `WHERE` | 1 | bool cond, a, b | none | a's | common | ALU | `(c?a:b)` |
| `BUFFER` | 2 | none | `ParamArg` (1-D) | `arg.dtype` | `arg.shape` | GLOBAL | kernel parameter `dataN` |
| `ALLOC` | 2 | none | `ParamArg` (1-D) | `arg.dtype` | `arg.shape` | LOCAL or `arg.addrspace` | `float accN[n];` / `float accN;` / `__shared__` |
| `LOAD` | 2 | memory | none | src's | src's | ALU | `T valN = ptr;` |
| `STORE` | 2 | memory, value | none | `void` | `()` | ALU | `ptr = value;` |
| `AFTER` | 2 | memory, deps… | none | src[0]'s | src[0]'s | src[0]'s | nothing (an ordering edge) |
| `SINK` | 2 | effects… | none | `void` | `()` | ALU | nothing |
| `RANGE` | 3 | end (`i32`) | `RangeArg` | `i32` | `()` | ALU | `for (...) {`, `#pragma omp`, `blockIdx`/`threadIdx` |
| `END` | 3 | passthrough, ranges… | none | `void` | `()` | ALU | `}` |
| `RESHAPE` | 4 | 1 | new shape | src's | `arg` | src's | index arithmetic |
| `EXPAND` | 4 | 1 | new shape | src's | `arg` | src's | index arithmetic |
| `PERMUTE` | 4 | 1 | permutation | src's | permuted | src's | index arithmetic |
| `SHRINK` | 4 | 1 | `((begin, end), …)` | src's | `end - begin` per axis | src's | index arithmetic |
| `REDUCE` | 4 | 1 | `(Ops.ADD or Ops.MAX, n)` | src's | src.shape minus last `n` | ALU | accumulator + loop |
| `STAGE` | 5 | 1 | none, or `"detach"` | src's | src's | ALU | a kernel boundary (temp buffer); `detach` is stripped |
| `BARRIER` | 7–8 | deps… | none | `void` | `()` | ALU | `__syncthreads();` (CUDA only) |
| `WMMA` | 8 | a, b, c | (reserved) | src[2]'s | src[2]'s | ALU | not rendered |

The rest of this appendix expands every row.

## A.2 Types: `DType`, `AddrSpace`, and the args

@include uopc/dtype.py#dtypes

| dtype | itemsize | C type | numpy | Notes |
|---|---|---|---|---|
| `void` | 0 | `void` | none | type of `STORE`, `SINK`, `END`, `BARRIER` |
| `bool` | 1 | `_Bool` | `bool_` | result of `CMPLT`/`CMPNE`; `WHERE` condition |
| `i32` | 4 | `int` | `int32` | indices, `RANGE` values; folding wraps like two's complement |
| `f32` | 4 | `float` | `float32` | folding rounds through `struct.pack('f')` (NOTES 5) |
| `f16` | 2 | `_Float16` (`half` in CUDA) | `float16` | storage type for mixed precision and tensor cores |

`dtypes.min`/`dtypes.max` give `(-2**31, 2**31 - 1)` for `i32`, `(False, True)` for `bool`, and
`(-inf, inf)` for floats; bounds analysis uses them as "anything".

@include uopc/dtype.py#AddrSpace

Address space is the third inferred property. It is what makes `LOAD` and `STORE` type-checkable: you can
only load from, or store to, something that is not `ALU`. `INDEX`, `AFTER` and the movement ops *preserve*
their source's address space, so `BUFFER.index(i)` is a GLOBAL pointer and `BUFFER.reshape(s)` is a
GLOBAL view. `REDUCE`, `STAGE` and every ALU op produce plain values (ALU).

The structured args:

@include uopc/uop.py#ParamArg

@include uopc/uop.py#CallArg

@include uopc/uop.py#RangeArg

`ParamArg.addrspace` is consulted only by `ALLOC` (LOCAL by default, SHARED for GPU shared memory).
`RangeArg` is `order=True`: ranges are compared by `idx` first, and *loops nest in increasing `idx` order*, an
invariant the linearizer relies on ([Chapter 5](05-loops.md)).

## A.3 The type checker

Every UOp is type-checked once, at construction, by `_infer`. A malformed node never exists:

@include uopc/uop.py#_infer

`call_params` finds the PARAMs that belong to a CALL body without descending into a nested CALL's body
(NOTES 2):

@include uopc/uop.py#call_params

Here is every `UOpError` message, triggered on purpose (scratch script, real output):

```python
# scratch: one malformed construction per check in _infer
from uopc.uop import UOp, Ops, ParamArg, CallArg, UOpError, range_
from uopc.dtype import dtypes
B = lambda n, d=dtypes.f32: UOp(Ops.BUFFER, (), ParamArg(0, d, (n,)))
P = lambda s, d=dtypes.i32, sh=(): UOp(Ops.PARAM, (), ParamArg(s, d, sh))
cases = [
 ("add i32 + f32", lambda: UOp.const(1) + UOp.const(1.0)),
 ("shape mismatch", lambda: B(4).reshape((2,2)) + B(4)),
 ("floordiv float", lambda: UOp(Ops.FLOORDIV, (UOp.const(1.0), UOp.const(2.0)))),
 ("exp2 of int", lambda: UOp(Ops.EXP2, (UOp.const(3),))),
 ("where non-bool", lambda: UOp(Ops.WHERE, (UOp.const(1), UOp.const(1), UOp.const(2)))),
 ("stack mixed", lambda: UOp(Ops.STACK, (UOp.const(1), UOp.const(2.0)))),
 ("index float", lambda: B(4).index(UOp.const(1.0))),
 ("too many idx", lambda: B(4).index(0, 0)),
 ("load of value", lambda: UOp.const(1.0).load()),
 ("store dtype", lambda: B(4).index(0).store(UOp.const(1))),
 ("reshape count", lambda: B(6).reshape((4,))),
 ("expand non-1", lambda: B(6).reshape((2,3)).expand((4,3))),
 ("permute bad", lambda: B(6).reshape((2,3)).permute((0,0))),
 ("shrink oob", lambda: B(6).reshape((2,3)).shrink(((0,3),(0,3)))),
 ("reduce mul", lambda: B(6).reduce(Ops.MUL, 1)),
 ("buffer 2d", lambda: UOp(Ops.BUFFER, (), ParamArg(0, dtypes.f32, (2,3)))),
 ("range f32 end", lambda: UOp(Ops.RANGE, (UOp.const(4.0),), None)),
 ("call slots", lambda: UOp(Ops.CALL, (P(0) + P(1), UOp.const(1)))),
 ("call arg type", lambda: UOp(Ops.CALL, (P(0) * P(0), UOp.const(1.0)))),
]
for name, f in cases:
  try: f(); print(f"{name:16s} OK?!")
  except UOpError as e: print(f"{name:16s} UOpError: {e}")
```

```text
add i32 + f32    UOpError: Ops.ADD: dtype mismatch dtypes.i32 vs dtypes.f32 (use CAST)
shape mismatch   UOpError: Ops.ADD: shape mismatch (2, 2) vs (4,) (broadcast with EXPAND)
floordiv float   UOpError: Ops.FLOORDIV is integer only
exp2 of int      UOpError: Ops.EXP2 needs a float, got dtypes.i32
where non-bool   UOpError: WHERE condition must be bool
stack mixed      UOpError: STACK srcs must agree in shape and dtype
index float      UOpError: INDEX indices are i32 scalars
too many idx     UOpError: INDEX: 2 indices into shape (4,)
load of value    UOpError: LOAD needs memory
store dtype      UOpError: STORE of dtypes.i32 into dtypes.f32
reshape count    UOpError: RESHAPE (6,) -> (4,)
expand non-1     UOpError: EXPAND (2, 3) -> (4, 3)
permute bad      UOpError: PERMUTE (0, 0) of (2, 3)
shrink oob       UOpError: SHRINK ((0, 3), (0, 3)) of (2, 3)
reduce mul       UOpError: REDUCE (Ops.MUL, 1) of (6,)
buffer 2d        UOpError: BUFFER is flat
range f32 end    UOpError: RANGE end is i32
call slots       UOpError: CALL needs params 0..0, body has [0, 1]
call arg type    UOpError: CALL arg 0: dtypes.f32() != param dtypes.i32()
```

What `_infer` deliberately does **not** check, so that passes stay simple: the source *count* of `LOAD`,
`STORE`, `AFTER`, `END` and `BARRIER`; that a `STORE`'s pointer is scalar; that `BUFFER` never appears in a
CALL body (a course rule, not enforced); and anything about `WMMA`. Those properties are guaranteed by the
code that builds such nodes (`rangeify`, hand-written loop IR) and by tests.

## A.4 Every op

**How the examples were produced.** Each example below is a `.uir` program parsed with `uir.parse` and
printed back with `uir.to_uir` (so you see the canonical form, including desugaring), followed by its
inferred type, and then either its value (via `symbolic.run`) or its C. Shaped graphs are lowered with
`rangeify.lower` into a fresh output buffer; loop programs are rendered directly with `render_c`. The
harness (scratch):

```python
# scratch: the harness behind every example in A.4
from uopc.uop import UOp, Ops, ParamArg
from uopc.uir import parse, to_uir
from uopc.rangeify import lower
from uopc.render_c import render_c
from uopc.symbolic import run, to_python

def show(name, src, mode):
  g = parse(src)
  print(to_uir(g), end="")
  print(f"# -> dtype={g.dtype.name} shape={g.shape} addrspace={g.addrspace.name}")
  assert parse(to_uir(g)) is g, "roundtrip failed"
  if mode == "run": print("# run ->", to_python(run(g)))
  elif mode == "lower":
    n = 1
    for s in g.shape: n *= s
    nb = len({u for u in g.toposort() if u.op is Ops.BUFFER})
    out = UOp(Ops.BUFFER, (), ParamArg(nb, g.dtype, (max(n, 1),)))
    c = render_c(lower(g, out)); print(c[c.index("void kernel"):], end="")
  elif mode == "c":
    c = render_c(g); print(c[c.index("void kernel"):], end="")
```

The `BARRIER` example is rendered with `gpu.render_cuda` instead, since only CUDA has barriers.
Every example also passed the assertion `parse(to_uir(g)) is g`: printing and re-parsing returns the very
same hash-consed node. The C examples omit the shared prelude (`#include`s and the floor-division helpers,
shown in A.6).

### Values (week 1)

#### `CONST`

- **Sources:** none. **Arg:** a Python `bool`, `int` or `float`.
- **Type:** dtype from the Python type (`bool`, `i32`, `f32`), shape `()`, ALU. An `f16` constant is built as
  `CAST(CONST f32, f16)` because Python has no f16 values. Shaped constants are `CONST` → `RESHAPE` → `EXPAND`
  (`UOp.const_like`, NOTES 13).
- **Hash consing:** the cache key uses `_argkey`, which adds the type and uses `repr` for floats, so `1`,
  `1.0`, `True`, `0.0` and `-0.0` are five different nodes (NOTES 1).
- **C:** `cconst` (A.6): `1`/`0` for bools, decimal ints (with `(-2147483647-1)` for `INT_MIN`), `repr` plus
  `f` for f32, `NAN`, `INFINITY`.

```text
%0 = const 2.5
# -> dtype=f32 shape=() addrspace=ALU
# run -> 2.5
```

#### `PARAM` and `CALL`

- **`PARAM`**: sources: none; arg: `ParamArg(slot, dtype, shape)`; type straight from the arg; ALU. An input
  to the *enclosing* CALL body.
- **`CALL`**: sources: `src[0]` is the body, `src[1:]` the arguments; arg: `CallArg(name)`. The body's own
  PARAMs (found by `call_params`, which skips nested CALL bodies) must be exactly slots `0..len(args)-1`, and
  argument `i` must match `PARAM i` in shape and dtype. The CALL has the body's type.
- **Semantics:** beta reduction. `symbolic.inline_call` substitutes arguments for PARAMs, scoped so it never
  rewrites inside a nested CALL's body (NOTES 2). Rewriting with `symbolic + evaluate` inlines bottom-up.
- **C:** none. `rangeify` raises `NotImplementedError` on PARAM; CALL/PARAM exist for the week-1
  "run by rewriting" interpreter.

```text
%0 = param : slot=0 dtype=i32
%1 = mul %0, %0
%2 = call %1, 7 : name=sq
# -> dtype=i32 shape=() addrspace=ALU
# run -> 49
```

A shaped PARAM takes a shaped argument:

```text
%0 = param : slot=0 dtype=f32 shape=(3,)
%1 = add %0, %0
%2 = stack 1.0, 2.0, 3.0
%3 = call %1, %2 : name=f
# -> dtype=f32 shape=(3,) addrspace=ALU
# run -> [2.0, 4.0, 6.0]
```

#### `STACK`

- **Sources:** one or more, all the same shape and dtype. **Arg:** none.
- **Type:** shape `(len(src),) + src[0].shape`, dtype of the sources, ALU.
- **Semantics:** a new leading axis. In week 1 its sources are constants or other STACKs (a literal
  array); from week 4 on they may be any shaped value.
- **Lowering:** `rangeify` turns `stack(a, b, c)[i, rest]` into `where(i < 1, a[rest], where(i < 2, b[rest],
  c[rest]))`. `symbolic.evaluate` pushes ALU ops through STACKs and resolves `INDEX` of a STACK by a constant.

```text
%0 = stack 1, 2, 3
%1 = stack %0, %0
# -> dtype=i32 shape=(2, 3) addrspace=ALU
# run -> [[1, 2, 3], [1, 2, 3]]
```

Shaped, lowered to C (note the `WHERE` chain that STACK became):

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = mul %0, %0
%2 = stack %0, %1
# -> dtype=f32 shape=(2, 4) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float val1 = data0[ridx1];
      data1[((ridx0*4)+ridx1)] = ((ridx0<1)?val1:(val1*val1));
    }
  }
}
```

#### `INDEX`

- **Sources:** `src[0]` is the indexed value `x`; `src[1:]` are indices, each an `i32` scalar. At most
  `len(x.shape)` of them.
- **Type:** dtype of `x`; shape `x.shape[len(indices):]` (indices pop axes off the *left*); address space of
  `x`. So `BUFFER.index(i)` is a GLOBAL scalar: a pointer you can `LOAD` and `STORE`.
- **Two roles.** On a value (week 1), `INDEX` selects a sub-array; `rangeify` prepends the index values to the
  index tuple it pushes down. On memory (week 2 on), `INDEX(buffer, flat_index)` is an address; the renderer
  prints `name[index]` (a scalar `ALLOC` of size 1 renders as the bare variable name).

```text
%0 = stack 10, 20, 30
%1 = index %0, 2
# -> dtype=i32 shape=() addrspace=ALU
# run -> 30
```

On a shaped buffer view, lowered:

```text
%0 = buffer 6 : dtype=f32 slot=0
%1 = reshape %0, (2, 3)
%2 = index %1, 1
# -> dtype=f32 shape=(3,) addrspace=GLOBAL
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val1 = data0[(ridx0+3)];
    data1[ridx0] = val1;
  }
}
```

### ALU ops (week 1, extended as needed)

All ALU ops are elementwise on shaped values and scalar after rangeify. Their reference semantics, which the
renderer must match, are `exec_alu`:

@include uopc/symbolic.py#exec_alu

and their C forms are `_unop` and `_binop`:

@include uopc/render_c.py#_unop

@include uopc/render_c.py#_binop

The `/` vs `floordiv_i32` decision is made per node from bounds analysis: C's truncating division agrees with
floor division only when the numerator is provably non-negative and the divisor is a positive constant
(NOTES 4).

#### `CAST`

- **Sources:** 1, any dtype. **Arg:** the target `DType`. **Type:** `arg`, source shape, ALU.
- **Semantics:** C conversion. Float to `i32` truncates toward zero; casting `inf`/`nan` to `i32` is undefined
  behavior in C, so the folder refuses to fold it (NOTES 6). `UOp.cast` returns the node itself when the
  dtype already matches.

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = cast %0 : dtype=i32
# -> dtype=i32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, int* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = ((int)(val1));
  }
}
```

#### `RECIP`

- **Sources:** 1, float. **Type:** source's. **Semantics:** `1/x`; folds `1/0` to `±inf` by the sign of zero.
  Division `a / b` in the Tensor API is `a * recip(b)`.

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = recip %0
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = (1.0f/val1);
  }
}
```

#### `EXP2`, `LOG2`, `SQRT`, `SIN`

- **Sources:** 1, float. **Type:** source's. **Semantics:** `2^x`, `log2 x` (`-inf` at 0, `nan` below),
  `sqrt x` (`nan` below 0), `sin x`. The Tensor API builds `exp`, `log`, `cos`, `tanh`, `sigmoid` and `gelu`
  from these (`exp(x) = exp2(x · log2 e)`).
- **C:** `exp2f`, `log2f`, `sqrtf`, `sinf` for f32 (the un-suffixed double versions for other float types).

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = exp2 %0
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = exp2f(val1);
  }
}
```

The other three differ only in the function name:

```c
    data1[ridx0] = log2f(val1);
    data1[ridx0] = sqrtf(val1);
    data1[ridx0] = sinf(val1);
```

#### `ADD` and `MUL`

- **Sources:** 2, identical shape and dtype (no implicit broadcasting or promotion in the IR; the error
  message tells you to use `EXPAND` or `CAST`). Commutative. **Type:** the sources'.
- **Semantics:** `i32` wraps (two's complement), `f32` rounds to f32; `inf * 0` folds to `nan`. Float
  *inputs* are rounded to f32 before folding too, as C rounds a literal: `16777217.0 + 2.0` folds to
  `16777218.0`. Constants created by reassociation rules wrap like `i32` (`_i32`), so `(x * 65536) * 65536`
  simplifies to `0` (`test_folding_matches_c_semantics`).
- **Sugar:** in `.uir`, binary ops broadcast and bare literals take the other operand's dtype. Below,
  `mul %a, 2` became a reshape/expand of the constant `2.0`:

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = buffer 4 : dtype=f32 slot=1
%2 = add %0, %1
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    float val2 = data1[ridx0];
    data2[ridx0] = (val1+val2);
  }
}
```

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = reshape 2.0, (1,)
%2 = expand %1, (4,)
%3 = mul %0, %2
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = (val1*2.0f);
  }
}
```

#### `MAX`

- **Sources:** 2, same shape and dtype. Commutative. **Semantics:** the larger value. Used for `relu`-style
  ops, `abs`, `minimum` (as `-max(-a, -b)`) and `REDUCE(MAX)`.

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = reshape 0.0, (1,)
%2 = expand %1, (4,)
%3 = max %0, %2
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = ((val1>0.0f)?val1:0.0f);
  }
}
```

#### `CMPLT` and `CMPNE`

- **Sources:** 2, same shape and dtype. **Type:** `bool`. `CMPNE` is commutative. Equality is
  `x.ne(y).ne(True)`; `>` swaps operands of `CMPLT`; `>=` is `not <`.

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = buffer 4 : dtype=f32 slot=1
%2 = cmplt %0, %1
# -> dtype=bool shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1, _Bool* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    float val2 = data1[ridx0];
    data2[ridx0] = (val1<val2);
  }
}
```

```text
%0 = buffer 4 : dtype=i32 slot=0
%1 = reshape 0, (1,)
%2 = expand %1, (4,)
%3 = cmpne %0, %2
# -> dtype=bool shape=(4,) addrspace=ALU
```

```c
void kernel(int* restrict data0, _Bool* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val1 = data0[ridx0];
    data1[ridx0] = (val1!=0);
  }
}
```

#### `FLOORDIV` and `FLOORMOD`

- **Sources:** 2 `i32`. **Semantics:** *floor* division and modulo (Python's `//` and `%`): `-7 // 2 = -4`,
  `-7 % 2 = 1`. Division by zero is not folded, and neither is `INT_MIN // -1`, which traps in C. Floor semantics make `(a*c + b) // c = a + b // c`
  unconditional, which is what index simplification needs.
- **C:** `/` and `%` when `vmin(numerator) >= 0` and the divisor is a positive constant; otherwise the prelude
  helpers. Values loaded from a buffer can be negative, so here the helpers appear:

```text
%0 = buffer 4 : dtype=i32 slot=0
%1 = reshape 3, (1,)
%2 = expand %1, (4,)
%3 = floordiv %0, %2
# -> dtype=i32 shape=(4,) addrspace=ALU
```

```c
void kernel(int* restrict data0, int* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val1 = data0[ridx0];
    data1[ridx0] = floordiv_i32(val1,3);
  }
}
```

```text
%0 = buffer 4 : dtype=i32 slot=0
%1 = reshape 3, (1,)
%2 = expand %1, (4,)
%3 = floormod %0, %2
# -> dtype=i32 shape=(4,) addrspace=ALU
```

```c
void kernel(int* restrict data0, int* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val1 = data0[ridx0];
    data1[ridx0] = floormod_i32(val1,3);
  }
}
```

Index arithmetic over `RANGE`s, whose bounds are known, renders as plain `/` and `%` (see `PERMUTE` below).

#### `WHERE`

- **Sources:** `(cond, a, b)`: `cond` is `bool`, `a` and `b` share a dtype, all three share a shape.
- **Type:** `a`'s dtype and shape. **Semantics:** select. It is how STACK, PAD (bounds-checked loads) and
  masks are expressed after rangeify.
- **`.uir` sugar:** all three operands broadcast; a bare literal takes the dtype of the *branches*, not of the
  bool condition.

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = buffer 4 : dtype=f32 slot=1
%2 = cmplt %0, %1
%3 = where %2, %0, %1
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    float val2 = data1[ridx0];
    data2[ridx0] = ((val1<val2)?val1:val2);
  }
}
```

### Memory (week 2)

#### `BUFFER`

- **Sources:** none. **Arg:** `ParamArg(slot, dtype, (n,))`; the shape must be 1-D ("BUFFER is flat");
  shaped views are `RESHAPE`s of it. **Type:** GLOBAL.
- **Semantics:** persistent device memory, a function argument. In the Tensor runtime a BUFFER UOp owns its
  numpy array (`kernelize.new_buffer`); kernels see local slots renumbered `0..n-1`, with negative slots used
  as placeholders during kernelization.
- **C:** a parameter `T* restrict dataN`, parameters sorted by slot.

#### `ALLOC`

- **Sources:** none. **Arg:** `ParamArg(slot, dtype, (n,), addrspace)`, 1-D. **Type:** address space
  `arg.addrspace` or LOCAL.
- **Semantics:** call-scope scratch memory. `rangeify` creates one size-1 ALLOC per reduce accumulator. The
  linearizer declares it in the deepest scope shared by all its users, so each OpenMP thread gets a private
  copy (NOTES 9); SHARED allocs are pinned to the top of the kernel (NOTES 24).
- **C:** `float accN[n];`, or `float accN;` for a LOCAL `(1,)` alloc, or `__shared__ float smemN[n];` with
  the CUDA renderer.

#### `LOAD`, `STORE`, `AFTER`, `SINK`

- **`LOAD`**: sources: `(ptr,)`, where `ptr` is memory (not ALU). Type: the pointer's dtype and shape, ALU.
  Always rendered as a named variable `T valN = ptr;`, so a value is read exactly once where it is scheduled.
- **`STORE`**: sources: `(ptr, value)`, same dtype. Type: `void`. Renders `ptr = value;`.
- **`AFTER`**: sources: `(mem, *deps)`. Type: exactly `mem`'s (dtype, shape and address space). It means
  "`mem`, but only once `deps` have happened". SSA graphs have no implicit order between a write and a later
  read of the same memory; `AFTER` is that order, made explicit as an edge. It renders as nothing (the name of
  `mem`). `UOp.after()` with no deps returns `mem` itself.
- **`SINK`**: sources: any effects (`STORE`s, `END`s). Type: `void`. The root of a program; renders nothing.

All six in one loopless program (write `a[0] + a[1]` into a scratch slot, read it back after the write,
square it):

```text
%0 = buffer 1 : dtype=f32 slot=1
%1 = index %0, 0
%2 = alloc 1 : dtype=f32 slot=0
%3 = index %2, 0
%4 = buffer 2 : dtype=f32 slot=0
%5 = index %4, 0
%6 = load %5
%7 = index %4, 1
%8 = load %7
%9 = add %6, %8
%10 = store %3, %9
%11 = after %2, %10
%12 = index %11, 0
%13 = load %12
%14 = mul %13, %13
%15 = store %1, %14
%16 = sink %15
# -> dtype=void shape=() addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  float acc0;
  float val1 = data0[0];
  float val2 = data0[1];
  acc0 = (val1+val2);
  float val3 = acc0;
  data1[0] = (val3*val3);
}
```

### Loops (week 3)

#### `RANGE` and `END`

- **`RANGE`**: sources: `(end,)`, an `i32` UOp (usually a `CONST`, but any `i32` value works; see
  [Chapter 15](15-beyond.md)). Arg: `RangeArg(idx, kind)`. Type: `i32` scalar, ALU. Its *value* is the loop
  counter, so it can be used in arithmetic. Bounds: `[0, vmax(end) - 1]`.
- **`END`**: sources: `(passthrough, *ranges)`. Type: `void`. Closes the listed ranges; everything that
  depends on the END runs after those loops.
- **Scope rule** (the linearizer's contract): a node sits inside every RANGE it transitively depends on,
  except those closed by an END on the path.
- **C:** `for (int ridxN = 0; ridxN < end; ridxN++) {` ... `}`. With `kind=thread` a
  `#pragma omp parallel for` precedes the loop; GPU kinds are covered in A.7.

```text
%0 = buffer 8 : dtype=f32 slot=1
%1 = range 8 : idx=0 kind=loop
%2 = index %0, %1
%3 = buffer 8 : dtype=f32 slot=0
%4 = index %3, %1
%5 = load %4
%6 = mul %5, 2.0
%7 = store %2, %6
%8 = end %7, %1
%9 = sink %8
# -> dtype=void shape=() addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 8; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = (val1*2.0f);
  }
}
```

The accumulator pattern (`ALLOC` + `AFTER(range)` + `END`) that every reduction lowers to is shown under
`REDUCE` below; [Chapter 5](05-loops.md) builds GEMM and conv1d with it by hand.

### Movement and reduction (week 4)

Movement ops never compute; they change how indices map to data. `rangeify` pushes an index tuple through
them (the index maps are in the `rangeify` docstring, [Chapter 6](06-rangeify.md)). They preserve their
source's dtype and address space, so a view of a BUFFER is still GLOBAL.

#### `RESHAPE`

- **Arg:** the new shape. **Check:** same number of elements. **Index map:** ravel the output index in the new
  shape, unravel it in the old one, then simplify (NOTES 12).

```text
%0 = buffer 6 : dtype=f32 slot=0
%1 = reshape %0, (2, 3)
# -> dtype=f32 shape=(2, 3) addrspace=GLOBAL
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu1 = ((ridx0*3)+ridx1);
      float val2 = data0[alu1];
      data1[alu1] = val2;
    }
  }
}
```

#### `EXPAND`

- **Arg:** the new shape, same rank. **Check:** every axis is unchanged or grows from 1. **Index map:** an
  expanded axis reads index 0. This is broadcasting; its gradient is a sum ([Chapter 8](08-autodiff.md)).

```text
%0 = buffer 3 : dtype=f32 slot=0
%1 = reshape %0, (1, 3)
%2 = expand %1, (2, 3)
# -> dtype=f32 shape=(2, 3) addrspace=GLOBAL
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val1 = data0[ridx1];
      data1[((ridx0*3)+ridx1)] = val1;
    }
  }
}
```

#### `PERMUTE`

- **Arg:** a permutation of `range(ndim)`; output axis `k` is input axis `arg[k]`. **Index map:** scatter the
  index back through the inverse permutation.

```text
%0 = buffer 6 : dtype=f32 slot=0
%1 = reshape %0, (2, 3)
%2 = permute %1, (1, 0)
# -> dtype=f32 shape=(3, 2) addrspace=GLOBAL
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      float val1 = data0[((ridx1*3)+ridx0)];
      data1[((ridx0*2)+ridx1)] = val1;
    }
  }
}
```

#### `SHRINK`

- **Arg:** one `(begin, end)` pair per axis with `0 <= begin <= end <= n`. **Index map:** add `begin`. PAD is
  *not* a primitive: `composite.pad` builds it from STACK with zeros and SHRINK, and FLIP from an anti-diagonal
  matrix built with the expand/reshape/shrink trick ([Chapter 6](06-rangeify.md)).

```text
%0 = buffer 12 : dtype=f32 slot=0
%1 = reshape %0, (3, 4)
%2 = shrink %1, ((1, 3), (0, 2))
# -> dtype=f32 shape=(2, 2) addrspace=GLOBAL
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      float val1 = data0[(((ridx0*4)+ridx1)+4)];
      data1[((ridx0*2)+ridx1)] = val1;
    }
  }
}
```

#### `REDUCE`

- **Arg:** `(Ops.ADD or Ops.MAX, n)` with `0 <= n <= ndim`. **Type:** the source's dtype with the last `n`
  axes popped; ALU (a reduction produces a value, not a view). To reduce other axes, `PERMUTE` them to the end
  first (that is what `Tensor.sum(axis)` does).
- **Lowering:** an accumulator `ALLOC` placed inside the loops the output index uses, initialized to the
  identity (`0` or `-inf`), updated in new `REDUCE`-kind ranges, and read back after their `END`.

```text
%0 = buffer 12 : dtype=f32 slot=0
%1 = reshape %0, (3, 4)
%2 = reduce %1 : op=add pop=1
# -> dtype=f32 shape=(3,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float val1 = acc0;
      float val2 = data0[((ridx0*4)+ridx1)];
      acc0 = (val1+val2);
    }
    float val3 = acc0;
    data1[ridx0] = val3;
  }
}
```

Popping both axes with `MAX`:

```text
%0 = buffer 12 : dtype=f32 slot=0
%1 = reshape %0, (3, 4)
%2 = reduce %1 : op=max pop=2
# -> dtype=f32 shape=() addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  float acc0;
  acc0 = -INFINITY;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float val1 = acc0;
      float val2 = data0[((ridx0*4)+ridx1)];
      acc0 = ((val1>val2)?val1:val2);
    }
  }
  float val3 = acc0;
  data1[0] = val3;
}
```

### Kernels (week 5)

#### `STAGE`

- **Sources:** 1. **Arg:** `None`, or the string `"detach"`. **Type:** the source's dtype and shape, ALU.
- **Semantics:** "materialize this into a temporary buffer you can index into". It is a request to the
  kernelizer: every STAGE is a materialization point, so it ends one kernel and the consumers read a temp
  buffer. `Tensor.contiguous()` creates one; `stage_broadcast_compute` and the layout pass insert them
  automatically. Without a kernelizer, `rangeify` simply computes the source inline.
- **`arg="detach"`:** `Tensor.detach()` wraps its value in `STAGE(x, "detach")`. Autodiff's rule for it
  returns no gradient, so it is a stop-gradient; `kernelize` removes it before scheduling with the one-rule
  matcher `strip_detach`, so it never costs a kernel. Checked in a scratch run: for `x = [1, 2, 3]`,
  `(x * x.detach()).sum()` has gradient `[1. 2. 3.]` (only the undetached factor contributes), and
  `x.detach().uop` prints as `Ops.STAGE 'detach'`.

Lowered on its own, a STAGE is transparent:

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = exp2 %0
%2 = stage %1
# -> dtype=f32 shape=(4,) addrspace=ALU
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = exp2f(val1);
  }
}
```

Through `kernelize.make_plan`, it is a cut: two kernels and one temp buffer (scratch):

```python
# scratch: STAGE as a kernel boundary
from uopc.uir import parse
from uopc.kernelize import make_plan
g = parse("""%a = buffer 4 : dtype=f32 slot=0
%e = exp2 %a
%s = stage %e
%r = add %s, 1.0""")
plan, inputs = make_plan([g])
for k in plan.kernels:
  print(k.name, k.refs)
  print(k.src[k.src.index("void"):], end="")
print("temps:", plan.temps, " outs:", plan.outs)
```

```text
e_4 [('in', 0), ('tmp', 0)]
void e_4(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = exp2f(val1);
  }
}
e_4 [('tmp', 0), ('out', 0)]
void e_4(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    data1[ridx0] = (val1+1.0f);
  }
}
temps: [(dtypes.f32, 4)]  outs: [(dtypes.f32, 4)]
```

### GPUs (weeks 7–8)

#### `BARRIER`

- **Sources:** the effects that must complete first (usually `STORE`s into shared memory). **Type:** `void`.
- **Semantics:** a workgroup-wide barrier. Consumers depend on it through `AFTER(smem, barrier)`.
- **C:** `__syncthreads();` with the CUDA renderer. The plain C renderer has no barrier (`CRenderer.barrier` is
  `None`), because running GPU threads one after another on a CPU cannot emulate one (NOTES 25); `gpu.emulate`
  raises `NotImplementedError` for any kernel containing a `BARRIER` rather than produce a wrong program.

A rotate-by-one through shared memory, written by hand (it only renders as CUDA):

```text
%0 = buffer 16 : dtype=f32 slot=1
%1 = range 16 : idx=0 kind=local
%2 = index %0, %1
%3 = alloc 16 : dtype=f32 slot=0 space=shared
%4 = index %3, %1
%5 = buffer 16 : dtype=f32 slot=0
%6 = index %5, %1
%7 = load %6
%8 = store %4, %7
%9 = barrier %8
%10 = after %3, %9
%11 = add %1, 1
%12 = floormod %11, 16
%13 = index %10, %12
%14 = load %13
%15 = store %2, %14
%16 = end %15, %1
%17 = sink %16
# -> dtype=void shape=() addrspace=ALU
```

```c
extern "C" __global__ void __launch_bounds__(16) kernel(float* __restrict__ data0, float* __restrict__ data1) {
  __shared__ float smem0[16];
  int lidx0 = threadIdx.x; /* 16 */
  float val1 = data0[lidx0];
  smem0[lidx0] = val1;
  __syncthreads();
  float val2 = smem0[((lidx0+1)%16)];
  data1[lidx0] = val2;
}
```

Note the `%` (not `floormod_i32`): `lidx0 + 1` has bounds `[1, 16]`, provably non-negative.
`gpu.smem_gemm_ir` is the full-size example, with two barriers per tile step
([Chapter 13](13-shared-memory-tensor-cores.md)).

#### `WMMA`

- **Sources:** `(a, b, c)`. **Type:** `c`'s dtype and shape; nothing is checked.
- **Status:** reserved. No pass produces it and neither renderer implements it (`render_wmma` raises
  `NotImplementedError`). Week 8 reaches tensor cores by *pattern*: `gpu.match_gemm` recognizes a GEMM-shaped
  `REDUCE` and `gpu.wmma_gemm` emits a complete `nvcuda::wmma` kernel as text. Lowering the graph to WMMA
  nodes is a final project ([Chapter 15](15-beyond.md)).

```text
%0 = buffer 256 : dtype=f16 slot=0
%1 = buffer 256 : dtype=f16 slot=1
%2 = buffer 256 : dtype=f32 slot=2
%3 = wmma %0, %1, %2
# -> dtype=f32 shape=(256,) addrspace=ALU
```

## A.5 The `.uir` text format

`.uir` is the course's interchange format: one SSA statement per line. The parser is `uir.parse`, the printer
`uir.to_uir`, and `parse(to_uir(g)) is g` for every graph (the printer is canonical, and hash consing makes
equal graphs identical).

### Grammar

In EBNF, transcribed from `_parse_line`, `_literal` and `_attr_value`:

```text
program   = { line } ;                          (* the value of the LAST statement is the program *)
line      = [ stmt ] [ "#" { any } ] NEWLINE ;  (* "#" starts a comment anywhere on a line *)
stmt      = "%" name "=" opname [ operands ] [ ":" { attr } ] ;
name      = { any char except "=" } ;           (* "%0", "%acc", "%x1" ... *)
opname    = an Ops member name, any case ;      (* "add", "ADD", "reshape", ... *)
operands  = operand { "," operand } ;           (* commas inside parentheses do not split *)
operand   = "%" name | literal ;
literal   = int | float | "true" | "false" | "inf" | "-inf" | "nan" | tuple ;
                                                (* ints/floats/tuples use Python literal syntax *)
tuple     = "(" [ literal { "," literal } [ "," ] ] ")" ;
attr      = key "=" value ;                     (* separated by whitespace *)
key       = "dtype" | "slot" | "shape" | "op" | "pop" | "idx" | "kind" | "space" | "name" ;
value     = dtype-name | op-name | axis-kind | addrspace | identifier | literal ;
```

**Positional args.** For `buffer`, `alloc` and `const` the *first* operand is the arg (the size, or the
constant). For `reshape`, `expand`, `permute` and `shrink` the *last* operand is the arg tuple. Every other
operand is a source.

### Attributes

| Key | Used by | Value | Default |
|---|---|---|---|
| `dtype` | `buffer`, `alloc`, `param`, `const`, `cast` | `void`, `bool`, `i32`, `f32`, `f16` | `f32` (`cast` requires it) |
| `slot` | `buffer`, `alloc`, `param` | int | `0` (`param` requires it) |
| `shape` | `param` | tuple | `()` |
| `space` | `alloc` | `alu`, `global`, `local`, `shared` | none (LOCAL) |
| `op` | `reduce` | `add`, `max` | `add` |
| `pop` | `reduce` | int | `1` |
| `idx` | `range` | int | `0` |
| `kind` | `range` | `loop`, `reduce`, `upcast`, `thread`, `global`, `local` | `loop` |
| `name` | `call` | identifier | `f` |

### Sugar

1. **Literals become constants.** A bare literal in a source position becomes a `CONST`. Its dtype is that of
   the first `%name` operand (for `where`: the first `%name` *branch*), so `mul %f32val, 2` means `2.0`.
   Exceptions: operands of `index` and `range`, bools, and statements with no `%name` operand use the
   literal's own Python type.
2. **Binary ops and `where` broadcast.** Missing leading axes are added with `reshape`, size-1 axes are
   `expand`ed. The printer writes the desugared form, which is why `mul %0, 2` prints as a `reshape` of `2.0`,
   an `expand`, and the `mul`:

```text
%0 = buffer 4 : dtype=f32 slot=0
%1 = reshape 0.0, (1,)
%2 = expand %1, (4,)
%3 = cmplt %0, %2
%4 = reshape 1.0, (1,)
%5 = expand %4, (4,)
%6 = where %3, %0, %5
```

(that is the printout of `cmplt %0, 0.0` followed by `where %1, %0, 1.0`).

### Printer rules

`to_uir` walks the graph in topological order and numbers statements `%0, %1, ...`. Constants are printed
inline at their uses, never as statements, unless the whole program is a constant. `buffer`/`alloc` print
size, `dtype`, `slot` and (for allocs with one) `space`; `range` always prints `idx` and `kind`; `reduce`
prints `op` and `pop`. Sources appear in their order in `src`, which is why buffers may print in an order
that differs from their slots.

### Parse errors

Every error is reported with its line number and text, including type errors raised by `_infer`:

```text
ParseError: line 2: '%1 = add %0, %2': undefined %2
ParseError: line 2: '%1 = frob %0': 'FROB'
ParseError: line 3: '%2 = add %0, %1': can't broadcast (2,) with (3,)
ParseError: line 1: '%0 = buffer 4 : dtype=f64 slot=0': unknown dtype 'f64'
ParseError: line 2: '%1 = exp2 %0': Ops.EXP2 needs a float, got dtypes.i32
```

### The syllabus example

The syllabus's `gemm_256x256.uir` parses and has shape `(256, 256)`, but it computes `A * rowsum(B)`, not a
GEMM (NOTES 11, `test_syllabus_gemm_is_not_a_gemm`). The correct program reshapes `A` to `(N, 1, K)` and the
*transpose* of `B` to `(1, N, K)`:

@include tests/test_w04_rangeify.py#test_gemm_from_movement_ops

## A.6 The C prelude and constants

Every C kernel starts with this prelude; the CUDA renderer uses a `__device__` version of the same helpers:

@include uopc/render_c.py#PRELUDE

@include uopc/render_c.py#cconst

Naming in generated code: `dataN` for BUFFER slot `N`, `accN`/`smemN` for ALLOC slot `N`, `ridxN` for
RANGE `idx` `N` (`gidxK`/`lidxK` on GPUs), `valN` for loads, `aluN` for ALU results used more than once.
An ALU result used exactly once is inlined into its user; that is the renderer's only "optimization", and it
is enough for gcc to see clean expressions.

## A.7 Scheduling: `AxisType` and `Opts`

@include uopc/uop.py#AxisType

How each kind renders:

| `AxisType` | Made by | C renderer | CUDA renderer |
|---|---|---|---|
| `LOOP` | `lower` for output axes | `for` loop | `for` loop |
| `REDUCE` | `_reduce` for reduced axes | `for` loop | `for` loop |
| `UPCAST` | nothing (see below) | `for` loop | `for` loop |
| `THREAD` | `lower` for `Opts.thread` | `#pragma omp parallel for` + `for` | `for` loop |
| `GLOBAL` | `lower` with `Opts.gpu` | `for` loop (the emulator) | `int gidxK = blockIdx.x/y/z;` |
| `LOCAL` | `lower` with `Opts.local` | `for` loop (the emulator) | `int lidxK = threadIdx.x/y/z;` |

`UPCAST` exists as a kind but no pass creates UPCAST ranges: upcasting and unrolling are done *during*
rangeify by instantiating the body once per constant offset (`_axis_parts` and `_combos`), so the unrolled
copies never pass through a range at all. The `.x` dimension goes to the *last* (fastest-varying) GPU axis,
for coalescing (NOTES 23).

The knobs that choose those kinds are a small dataclass, filled in by `opt.hand_opt_choice`,
`opt.search_opt`, or `gpu.gpu_opt_choice`, and applied by `rangeify.lower`:

@include uopc/rangeify.py#Opts

| Field | Meaning | Effect on the code |
|---|---|---|
| `upcast` | output axis → factor | a register tile: `factor` accumulators sharing one reduce loop (`merge_ends`) |
| `thread` | one output axis | that axis becomes the outermost loop, kind THREAD (OpenMP) |
| `unroll` | factor for the innermost reduce axis | the reduce body is replicated and combined as a balanced tree |
| `gpu` | bool | output axes become GLOBAL ranges |
| `local` | output axis → threads per block | splits that axis into GLOBAL × LOCAL ranges |

A row sum over an `(8, 8)` matrix with `Opts(thread=0, upcast={0: 2}, unroll=4)`, and the same reduction for
a GPU with `Opts(gpu=True, local={0: 4})` (scratch):

```python
# scratch: the same REDUCE with CPU opts and with GPU opts
from uopc.uir import parse
from uopc.uop import UOp, Ops, ParamArg
from uopc.dtype import dtypes
from uopc.rangeify import lower, Ctx, Opts
from uopc.render_c import render_c
from uopc.gpu import render_cuda
g = parse("""%a = buffer 64 : dtype=f32 slot=0
%m = reshape %a, (8, 8)
%s = reduce %m : op=add pop=1""")
out = UOp(Ops.BUFFER, (), ParamArg(1, dtypes.f32, (8,)))
for o in [Opts(thread=0, upcast={0: 2}, unroll=4), Opts(gpu=True, local={0: 4})]:
  print("#", o)
  ast = lower(g, out, Ctx(opts=o))
  src = render_cuda(ast) if o.gpu else render_c(ast)
  print(src[src.index("void") if not o.gpu else src.index("extern"):])
```

```text
# Opts(upcast={0: 2}, thread=0, unroll=4, gpu=False, local={})
void kernel(float* restrict data0, float* restrict data1) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu1 = (ridx0*2);
    float acc0;
    acc0 = 0.0f;
    int alu2 = (ridx0*16);
    float acc1;
    acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      float val3 = acc0;
      int alu4 = (ridx1*4);
      int alu5 = (alu2+alu4);
      float val6 = data0[alu5];
      int alu7 = (alu4+alu2);
      float val8 = data0[(alu7+1)];
      float val9 = data0[(alu7+2)];
      float val10 = data0[(alu7+3)];
      acc0 = (val3+((val6+val8)+(val9+val10)));
      float val11 = acc1;
      float val12 = data0[(alu5+8)];
      float val13 = data0[(alu7+9)];
      float val14 = data0[(alu7+10)];
      float val15 = data0[(alu7+11)];
      acc1 = (val11+((val12+val13)+(val14+val15)));
    }
    float val16 = acc0;
    data1[alu1] = val16;
    float val17 = acc1;
    data1[(alu1+1)] = val17;
  }
}

# Opts(upcast={}, thread=None, unroll=1, gpu=True, local={0: 4})
extern "C" __global__ void __launch_bounds__(4) kernel(float* __restrict__ data0, float* __restrict__ data1) {
  int gidx0 = blockIdx.x; /* 2 */
  int lidx0 = threadIdx.x; /* 4 */
  float acc0;
  acc0 = 0.0f;
  for (int ridx2 = 0; ridx2 < 8; ridx2++) {
    float val1 = acc0;
    float val2 = data0[(((gidx0*32)+(lidx0*8))+ridx2)];
    acc0 = (val1+val2);
  }
  float val3 = acc0;
  data1[((gidx0*4)+lidx0)] = val3;
}
```

Read the CPU kernel against the table: the THREAD axis has 4 iterations (8 rows ÷ upcast 2), two scalar
accumulators (`acc0`, `acc1`) share one reduce loop of 2 iterations (8 columns ÷ unroll 4), and each
iteration adds a balanced tree of four loads. (`alu5` and `alu7` are the same sum in two operand orders; the
symbolic canonicalizer only orders constants, so this common subexpression is left for gcc.)

## A.8 Environment variables

| Variable | Read in | Default | Effect |
|---|---|---|---|
| `DEBUG` | `kernelize.py` | `0` | `1`: one line per `realize` (kernel count, schedule and run time). `2`: plus per-kernel time and GFLOP/s. `3`: plus a second, warm run of each kernel. `4`: plus the C source of each kernel when its plan is first built. |
| `OPT` | `tensor.py` | `1` | `0`: plain `lower`, no layout pass. `1`: `HandOpt` (layout pass + `hand_opt` heuristics). `2`: `SearchOpt` (times candidate `Opts` per kernel; slow to compile, fast to run). |
| `CFLAGS` | `runtime.py` | `-O3 -march=native -fopenmp -ffp-contract=fast -fassociative-math -fno-signed-zeros -fno-trapping-math` | whitespace-separated flags for the C compiler (NOTES 20 explains `-fassociative-math`). |
| `CC` | `runtime.py` | `gcc` | the C compiler. |
| `UOPC_CACHE` | `runtime.py` | `~/.cache/uopc` | directory of compiled `.so` files, named by a hash of source, compiler and flags. |

All of them are read once, at import time. Changing `CFLAGS` or `CC` changes the cache key, so the two
configurations never share binaries.

The `DEBUG` levels on a 3×4 softmax (scratch: `x = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4));
print(x.softmax(-1).numpy().round(3))`):

```text
--- DEBUG=1
realize: 3 kernels, schedule 140.1 ms, run 0.3 ms
[[0.032 0.087 0.237 0.644]
 [0.032 0.087 0.237 0.644]
 [0.032 0.087 0.237 0.644]]
--- DEBUG=2
  r_3                               0.076 ms     0.00 GFLOP/s
  r_3                               0.029 ms     0.00 GFLOP/s
  e_3_4                             0.033 ms     0.00 GFLOP/s
realize: 3 kernels, schedule 14.2 ms, run 0.3 ms
[[0.032 0.087 0.237 0.644]
 [0.032 0.087 0.237 0.644]
 [0.032 0.087 0.237 0.644]]
--- DEBUG=3
  r_3                               0.078 ms     0.00 GFLOP/s  (again: 0.022 ms)
  r_3                               0.034 ms     0.00 GFLOP/s  (again: 0.017 ms)
  e_3_4                             0.034 ms     0.00 GFLOP/s  (again: 0.018 ms)
realize: 3 kernels, schedule 17.0 ms, run 0.3 ms
[[0.032 0.087 0.237 0.644]
 [0.032 0.087 0.237 0.644]
 [0.032 0.087 0.237 0.644]]
```

Kernel names are `r_` (contains a reduction) or `e_` (elementwise) followed by the output shape. The schedule
time varies between runs because the first run also builds and caches the `.so` files. `DEBUG=4` with `OPT=0`
prints the three kernels (the max, the sum of exponentials, and the normalized output):

```text
--- DEBUG=4 OPT=0
...
void r_3(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = -INFINITY;
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float val1 = acc0;
      float val2 = data0[((ridx0*4)+ridx1)];
      acc0 = ((val1>val2)?val1:val2);
    }
    float val3 = acc0;
    data1[ridx0] = val3;
  }
}
...
void r_3(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    float val1 = data1[ridx0];
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      float val2 = acc0;
      float val3 = data0[((ridx0*4)+ridx1)];
      acc0 = (val2+exp2f(((val3+(val1*-1.0f))*1.4426950408889634f)));
    }
    float val4 = acc0;
    data2[ridx0] = val4;
  }
}
...
void e_3_4(float* restrict data0, float* restrict data1, float* restrict data2, float* restrict data3) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val1 = data1[ridx0];
    float val2 = data2[ridx0];
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu3 = ((ridx0*4)+ridx1);
      float val4 = data0[alu3];
      data3[alu3] = (exp2f(((val4+(val1*-1.0f))*1.4426950408889634f))*(1.0f/val2));
    }
  }
}
...
```

(Each `...` elides the four-line prelude, or the timing lines, which are as for `DEBUG=3`.)

## A.9 Module map

| Week | Module | Key entry points |
|---|---|---|
| 1 | `uop.py` | `UOp`, `Ops`, `_infer`, `UOp.toposort`, `UOp.substitute`, `UOp.wrap`, `sink`, `range_` |
| 1 | `rewrite.py` | `UPat`, `PatternMatcher`, `graph_rewrite` |
| 1 | `symbolic.py` | `exec_alu`, `vmin`/`vmax`, `simplify`, `run`, `to_python`, `inline_call` |
| 1 | `uir.py` | `parse`, `to_uir` |
| 2 | `render_c.py`, `runtime.py` | `render_c`, `CRenderer`, `compile_c`, `Program`, `run_c` |
| 2–3 | `linearize.py` | `linearize`, `scopes`, `loop_targets` |
| 4 | `rangeify.py`, `composite.py` | `rangeify`, `lower`, `Opts`, `Ctx`; `pad`, `flip_axis` |
| 5 | `kernelize.py`, `autodiff.py`, `tensor.py`, `nn.py` | `make_plan`, `realize`; `gradient`; `Tensor`; `Linear`, `AdamW` |
| 6 | `opt.py` | `layout_pass`, `hand_opt_choice`, `search_opt`, `HandOpt`, `SearchOpt` |
| 7–8 | `gpu.py`, `cuda.py` | `gpu_lower`, `render_cuda`, `emulate`, `launch_dims`, `smem_gemm_ir`, `wmma_gemm`; `CUDA`, `run_cuda` |
