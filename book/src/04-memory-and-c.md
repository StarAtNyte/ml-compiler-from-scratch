# 4. Memory, rendering C, and the runtime

Everything so far has been a *value*: constants, stacks of constants, arithmetic on them. Values are easy to
reason about because they never change. A real kernel is not like that. It receives pointers to arrays that
someone else allocated, reads from them, writes results into other arrays, and maybe keeps a scratch
variable that it overwrites as it goes. This chapter adds memory to the IR, which means facing the one
question a graph IR cannot dodge: *in what order do things happen?* Then we cross the boundary to the real
machine. We print the graph as a C function, compile it with gcc into a shared library, load it into the
Python process, and call it on numpy arrays. At the end of the chapter you have a working compiler for
loopless programs. The next chapter adds loops.

**What you'll build**

- Six memory ops: `BUFFER`, `ALLOC`, `INDEX` as an address, `LOAD`, `STORE`, `AFTER`, plus `SINK` as the
  root of a program with side effects.
- The `addrspace` recursive property, which turns "you can't store to a value" into a type error.
- `CRenderer` (`uopc/render_c.py`): one UOp becomes at most one line of C, with single-use pure
  expressions inlined, constants printed exactly, and floor division rendered correctly.
- The runtime (`uopc/runtime.py`): gcc with the right flags, a content-addressed cache of `.so` files, and
  calls through `ctypes` with numpy buffers.

**Where this fits.** [Chapter 3](03-symbolic.md) gave us `vmin`, which the renderer uses to decide how to
print division. [Chapter 5](05-loops.md) adds `RANGE`/`END` and the full linearizer. This chapter only needs
the loopless special case of linearization, which is a topological sort.

## 4.1 Values versus memory

Recall the two promises a UOp makes ([Chapter 1](01-uops.md)). It is *immutable*: its op, sources and
arg never change. And it is *hash consed*: building the same structure twice gives the same object. Those
promises are what make `simplify` memoizable and common-subexpression elimination free. They describe
*values*, things like `3`, `x+1` and `a[i]*2`. In compiler terms the graph is in *SSA form* (static single
assignment): every value is defined exactly once, by the node that computes it, and every use refers to
that definition directly.

Memory breaks both promises. An array has *identity*: two arrays with the same contents are still
different arrays. It also has *time*: `tmp[0]` before a store and `tmp[0]` after it are different values,
even though the expression that reads them is spelled the same. A pure DAG has neither notion. It has no
"before", only "uses".

You cannot avoid memory, since kernels exist to read and write arrays. So the IR keeps two worlds and makes
the crossing between them explicit:

- **Values** live in registers (conceptually). `addrspace` is `ALU`.
- **Memory** is a place that values can be loaded from and stored into. `addrspace` is `GLOBAL` (arrays
  passed in by the caller), `LOCAL` (scratch storage that lives for one call), or later `SHARED` (GPU
  shared memory, [Chapter 13](13-shared-memory-tensor-cores.md)).

@include uopc/dtype.py#AddrSpace

`addrspace` is a recursive property like `dtype` and `shape`. It is computed by `_infer` when the node is
built, and the type checker uses it to reject nonsense such as loading from a number.

## 4.2 The memory ops

Six new ops, each small:

| Op | Sources | Arg | dtype, shape, addrspace | Meaning | C |
|---|---|---|---|---|---|
| `BUFFER` | none | `ParamArg(slot, dtype, (n,))` | `dtype`, `(n,)`, `GLOBAL` | Caller's array, argument number `slot` | `float* data0` |
| `ALLOC` | none | `ParamArg(slot, dtype, (n,), space)` | `dtype`, `(n,)`, `LOCAL` | Call-scope scratch storage | `float acc0[4];` |
| `INDEX` | `(mem, i)` | none | `dtype`, `()`, mem's | The *address* of element `i` | `data0[i]` |
| `LOAD` | `(addr,)` | none | `dtype`, `()`, `ALU` | Read the value at an address | `float val1 = data0[i];` |
| `STORE` | `(addr, value)` | none | `void`, `()`, `ALU` | Write a value to an address | `data1[i] = v;` |
| `AFTER` | `(mem, *deps)` | none | same as `mem` | `mem`, but only once `deps` have happened | nothing |
| `SINK` | `(*effects)` | none | `void` | The root: these effects are the program | nothing |

Memory is always *flat*. A 2-D matrix is a 1-D buffer plus an index expression `i*N + j` that you write
yourself (and that rangeify writes for you in [Chapter 6](06-rangeify.md)). That keeps the memory model
trivial and puts all layout logic in index arithmetic, where the simplifier of
[Chapter 3](03-symbolic.md) can see it.

`BUFFER` and `ALLOC` take a `ParamArg`, the same record used for `PARAM`:

@include uopc/uop.py#ParamArg

The `slot` of a `BUFFER` is its position in the C function's argument list. The `slot` of an `ALLOC` is
just a number to make its name unique (`acc0`, `acc1`). The type rules for the memory ops are a few lines
of `_infer`:

@include uopc/uop.py:235-240

@include uopc/uop.py:252-256

@include uopc/uop.py:274-282

Read them as a type system. `INDEX` *preserves* the address space of what it indexes: indexing a `GLOBAL`
buffer gives a `GLOBAL` address, not a value. `LOAD` requires its source to be memory and produces an `ALU`
value. `STORE` requires memory, checks that the stored value has the buffer's dtype, and produces `void`,
which is nothing you can compute with. `AFTER` is transparent: it has the type of its first source.

The builder methods on `UOp` are one-liners:

@include uopc/uop.py:201-206

A short session shows the properties flowing through:

```text
>>> from uopc.uop import UOp, Ops, ParamArg, sink
>>> from uopc.dtype import dtypes
>>> from uopc.render_c import render_c
>>> def buf(slot, n, dtype=dtypes.f32): return UOp(Ops.BUFFER, (), ParamArg(slot, dtype, (n,)))
>>> a, b, c = buf(0, 4), buf(1, 4), buf(2, 4)
>>> a, a.dtype, a.shape, a.addrspace
(UOp(Ops.BUFFER, arg=ParamArg(slot=0, dtype=dtypes.f32, shape=(4,), addrspace=None)), dtypes.f32, (4,), AddrSpace.GLOBAL)
>>> p = a.index(2)
>>> p.dtype, p.shape, p.addrspace
(dtypes.f32, (), AddrSpace.GLOBAL)
>>> v = p.load()
>>> v.dtype, v.shape, v.addrspace
(dtypes.f32, (), AddrSpace.ALU)
>>> st = c.index(2).store(v * 2.0)
>>> st.dtype, st.shape, st.addrspace
(dtypes.void, (), AddrSpace.ALU)
>>> a.index(2) is p, a.index(2).load() is v
(True, True)
```

The last line is the crux of this chapter. Hash consing still applies to memory ops, so "load `a[2]`"
written twice is *one* node. For a buffer that nobody writes, that is exactly right: reading an input twice
gives the same value, and computing it once is free CSE. For a buffer that the program *does* write, it is
exactly wrong, and we need a way to say "the load after the store".

`buf` is the same helper as `tests/helpers.py` provides; we use it throughout:

@include tests/helpers.py#buf

## 4.3 AFTER: order in a graph that has none

Here is the smallest program where order matters, from the week-2 tests. It writes `a[0] + a[1]` into a
one-element scratch buffer, then reads it back and squares it:

@include tests/test_w02_render.py#test_alloc_and_after

The key line is `v = tmp.after(w).index(0).load()`. `tmp.after(w)` is a new node, `AFTER(tmp, w)`. It *is*
`tmp`: same dtype, shape and address space, and the renderer prints it as `tmp`. But it has an extra source,
the store `w`, and in a DAG an extra source is a dependency. Anything built on `tmp.after(w)` can only
run after `w`. And because `AFTER(tmp, w)` is a different node from `tmp`, the load built on it is a
different node from a load built on plain `tmp`. Hash consing can no longer merge "before" and "after".

@diagram 04-memory-model

Here is the program after linearization, one line per node, with sources as `%n` references. The
printer is a scratch helper, not part of `uopc`, and worth having in your own compiler from day one:

```python
from uopc.uop import ParamArg, RangeArg
from uopc.linearize import linearize
def listing(lin):
  n = {u: i for i, u in enumerate(lin)}
  for i, u in enumerate(lin):
    a = u.arg
    if isinstance(a, ParamArg): a = f"slot={a.slot} {a.dtype.name}{list(a.shape)}"
    elif isinstance(a, RangeArg): a = f"idx={a.idx}"
    elif a is None: a = ""
    print(f"%{i:<3d}{u.op.name:7s}{' '.join(f'%{n[s]}' for s in u.src):16s}{a}")
listing(linearize(sink(out.index(0).store(v * v))))
```

```text
%0  BUFFER                 slot=1 f32[1]
%1  CONST                  0
%2  INDEX  %0 %1           
%3  ALLOC                  slot=0 f32[1]
%4  INDEX  %3 %1           
%5  BUFFER                 slot=0 f32[2]
%6  INDEX  %5 %1           
%7  LOAD   %6              
%8  CONST                  1
%9  INDEX  %5 %8           
%10 LOAD   %9              
%11 ADD    %7 %10          
%12 STORE  %4 %11          
%13 AFTER  %3 %12          
%14 INDEX  %13 %1          
%15 LOAD   %14             
%16 MUL    %15 %15         
%17 STORE  %2 %16          
%18 SINK   %17             
```

and as C:

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

`AFTER` (`%13`) produced no C at all. Its whole job was to put `%12` before `%15` in the linear order.

### What happens without it

Delete `.after(w)` and load from `tmp` directly:

```python
v_bad = tmp.index(0).load()
print(render_c(sink(out.index(0).store(v_bad * v_bad))))
```

```c
void kernel(float* restrict data1) {
  float acc0;
  float val1 = acc0;
  data1[0] = (val1*val1);
}
```

The store `w` vanished, and with it every read of `data0`. Nothing reachable from the `SINK` depends on
`w`, so `w` is not part of the program. What remains reads an uninitialized variable, which is undefined
behavior in C. It "works" or fails depending on what happened to be on the stack. The function signature
even lost its first parameter, since `data0` is no longer used. (`run_c` computes the argument list from the
same graph, so it still calls the function correctly.)

This is the general rule of the memory model, and it is worth stating as an invariant:

> **Key idea:** every read of memory that the program itself writes must go through an `AFTER` that names
> the write. Reads of memory that the program never writes (inputs) need no `AFTER`, and get CSE for free.

### Why a dependency edge, and not something heavier

Compilers have several ways to put side effects into a graph. LLVM keeps instructions in ordered basic
blocks, so order is simply list position, and builds *MemorySSA* on top to recover which store a load can
see. Sea-of-nodes compilers (HotSpot C2, V8's TurboFan) thread an explicit *memory state* value through
every load and store, so a store consumes the old memory state and produces a new one. JAX threads *effect
tokens* through ordered side effects. All of these turn "happens before" into data flow, so that the usual
graph algorithms respect it.

`AFTER` is the lightest version of that idea: a node that means "this memory, once these effects have
happened". It costs one node per ordering constraint and nothing in the generated code. It is also exactly
the tool we need for accumulators in [Chapter 5](05-loops.md), where `acc.after(init, k)` means "the
accumulator, after it was initialized, inside loop `k`". tinygrad uses the same op with the same name.

### SINK

A program with stores has no single "result value". Its result is a set of effects. `SINK` gathers them:
`sink(store1, store2, ...)` is the root of the graph, its dtype is `void`, and the linearizer and renderer
start from it. Anything not reachable from the `SINK` is dead code by definition, as the missing store above
showed.

## 4.4 A loopless program, end to end

The first week-2 test computes `c = a*b + 1` on four elements, one store per element:

@include tests/test_w02_render.py#test_elementwise_loopless

`run_c` renders the `SINK`, compiles the C, and calls it with the numpy arrays. Here is the C it produces
(we pass `name="elementwise"` to `render_c`; the prelude of helper functions is omitted):

```c
void elementwise(float* restrict data0, float* restrict data1, float* restrict data2) {
  float val1 = data0[0];
  float val2 = data1[0];
  data2[0] = ((val1*val2)+1.0f);
  float val3 = data0[1];
  float val4 = data1[1];
  data2[1] = ((val3*val4)+1.0f);
  float val5 = data0[2];
  float val6 = data1[2];
  data2[2] = ((val5*val6)+1.0f);
  float val7 = data0[3];
  float val8 = data1[3];
  data2[3] = ((val7*val8)+1.0f);
}
```

The rest of this chapter takes that pipeline apart: how the nodes got into this order (section 4.5), how
each became a line (4.6), how the C became machine code callable from Python (4.7 and 4.8), and which
mistakes the type system rejected along the way (4.9).

## 4.5 Ordering the graph

C is a list of statements. The graph is a DAG. *Linearizing* means choosing a list order in which every node
comes after all of its sources: a topological order. Any topological order is a correct program for a
loopless graph, because every dependency, including every ordering dependency added by `AFTER`, is an
edge.

`UOp.toposort` ([Chapter 1](01-uops.md)) already produces one: an iterative depth-first search that emits a
node after its children. Run on the elementwise program it visits the first store's operands, emits them,
emits the store, then moves on to the second store. That is why the C above is grouped by element.

The real `linearize` in `uopc/linearize.py` does more, because loops impose constraints that a plain
topological order ignores: a node that depends on loop variable `k` must sit *inside* loop `k`, and a node
that does not depend on `k` should sit outside it. For a loopless program all those constraints are empty
and `linearize` reduces to a topological order with ties broken by toposort position. The algorithm in full
is the subject of [Chapter 5](05-loops.md).

## 4.6 The C renderer

The renderer walks the linear order once and produces one C function. Here is the main loop:

@include uopc/render_c.py#CRenderer.render

It keeps a dict `r` from each UOp to the C *expression* that denotes it, and a list of output lines. Each op
either *defines a name* (a line like `float val1 = data0[0];`), or *is an expression* that later lines
paste into themselves (`r[u] = "(val1*val2)"`), or *emits a statement* (stores, loop headers). The cases
below go through the decisions in that loop.

### Names

| UOp | Name | Example |
|---|---|---|
| `BUFFER` slot `n` | `data{n}` | `data0` |
| `ALLOC` slot `n` | `acc{n}` (or `smem{n}` for shared) | `acc0` |
| `LOAD` | `val{counter}` | `val3` |
| multi-use ALU | `alu{counter}` | `alu2` |
| `RANGE` idx `n` | `ridx{n}` | `ridx1` |

Names come from a single counter, so they are unique within the function and *deterministic*: the same
graph always produces the same text, character for character. That matters more than it looks. The runtime
caches compiled code by a hash of the source text (section 4.8), so any nondeterminism in naming, such as
iterating over a Python `set` of nodes, would make identical kernels miss the cache.

### Expressions and statements: what gets inlined

Pure ALU ops used once become expressions and are pasted into their user. Ops used more than once get a
named variable. The decision is in `_assign`:

@include uopc/render_c.py#CRenderer._assign

Why not inline everything? Because the graph is a DAG, and a DAG can be exponentially smaller than the
tree you get by expanding it. Consider `v = v*v + v` applied four times:

```python
a, out = buf(0, 1), buf(1, 1)
v = a.index(0).load()
for _ in range(4): v = v * v + v
print(render_c(sink(out.index(0).store(v)), "chain"))
```

```c
void chain(float* restrict data0, float* restrict data1) {
  float val1 = data0[0];
  float alu2 = ((val1*val1)+val1);
  float alu3 = ((alu2*alu2)+alu2);
  float alu4 = ((alu3*alu3)+alu3);
  data1[0] = ((alu4*alu4)+alu4);
}
```

Every intermediate value is used three times. Fully inlined, the final expression would contain
3<sup>4</sup> = 81 copies of `val1`, and twenty levels would be 3.5 billion. gcc would eventually CSE it
back, after we spent gigabytes printing it. Naming every value used more than once keeps the C linear in
the size of the graph. Inlining the single-use ones keeps it readable.

**Loads are never inlined**, even when used once: a `LOAD` always becomes `float valN = ...;`. The reason is
ordering. The line for a load sits at the load's position in the linear order, which the linearizer chose to
respect every `AFTER`. If we inlined the load into its user, the read would move to wherever the user is,
possibly past a store to the same address. Naming the load pins the read to its place. (gcc is free to
reorder it again, but only where the C semantics allow, and `restrict` tells it exactly where that is.)

`CONST`, `INDEX` and `AFTER` produce expressions and never lines: a constant is a literal, an index is
`data0[expr]`, and `AFTER` is just its first source's expression.

### Constants

@include uopc/render_c.py#cconst

```text
>>> from uopc.render_c import cconst
>>> [cconst(v, dtypes.f32) for v in (1.0, 0.1, -0.0, 1e-45, float("inf"), float("nan"), 3.4028234663852886e38)]
['1.0f', '0.1f', '-0.0f', '1e-45f', 'INFINITY', 'NAN', '3.4028234663852886e+38f']
>>> [cconst(v, dtypes.i32) for v in (0, -5, 2**31 - 1, -2**31)], cconst(True, dtypes.bool)
(['0', '-5', '2147483647', '(-2147483647-1)'], '1')
```

Each case is there for a reason:

- **The `f` suffix.** In C, `1.0` is a `double`. `x * 1.0` with `float x` promotes `x` to double, multiplies
  in double precision, and rounds back. That is slower, defeats single-precision vectorization, and is not
  the f32 semantics that `exec_alu` promises. `1.0f` keeps the arithmetic in float.
- **`repr` for the digits.** Python's `repr` of a float is the shortest decimal string that round-trips to
  the same double. Any value that is exactly representable in float32 (which a correctly folded f32
  constant is) therefore prints as a decimal that C parses back to exactly that float. `%.6g` style
  formatting would silently lose bits.
- **`INFINITY` and `NAN`** come from `<math.h>`. `repr` would print `inf`, which is not C.
- **`(-2147483647-1)`.** C has no negative integer literals. `-2147483648` is the unary minus applied to
  `2147483648`, which does not fit in an `int`, so it has type `long` and the expression is a `long`. The
  standard idiom (it is how `<limits.h>` defines `INT_MIN`) builds the value inside `int` range.

### Division

The prelude defines two helpers:

@include uopc/render_c.py#PRELUDE

and `_binop` chooses between them and C's operators:

@include uopc/render_c.py#_binop

C's `a / b` truncates toward zero. Floor division differs exactly when the division is inexact and the
operands have opposite signs, and in that case the floor is one less than the truncated quotient. That is
`floordiv_i32`. For the remainder, C's `%` has the sign of the dividend while floor modulo has the sign of
the divisor. They differ when the remainder is nonzero and the signs disagree, and then adding `b` fixes it.
That is `floormod_i32`.

The fast path is the common case. If [Chapter 3](03-symbolic.md)'s bounds analysis proves the numerator is
non-negative and the divisor is a positive constant, truncation and floor agree, so the renderer emits plain
`/` and `%`, which gcc turns into shifts and masks for powers of two and into multiply-by-reciprocal
sequences otherwise. Index arithmetic almost always takes this path, because loop variables start at 0. The
test `test_floordiv_rendering_negative` exercises the slow path, with operands loaded from memory whose
bounds are unknown:

@include tests/test_w02_render.py#test_floordiv_rendering_negative

```c
void divmod(int* restrict data0, int* restrict data1) {
  int val1 = data0[0];
  int val2 = data0[1];
  data1[0] = floordiv_i32(val1,val2);
  data1[1] = floormod_i32(val1,val2);
}
```

The helpers are `static inline`, so gcc inlines them, and in the fast path they cost nothing because they
are not called.

### Casts, select and math functions

@include uopc/render_c.py#_unop

Casts are C casts, `RECIP` is `1.0f/x`, and the transcendental ops call `<math.h>`'s float versions
(`exp2f`, `log2f`, `sqrtf`, `sinf`). Calling `exp2` on a float would compute in double. `WHERE` becomes the
ternary operator. The `test_casts_and_where` program renders as:

```c
void casts(float* restrict data0, int* restrict data1) {
  float val1 = data0[0];
  data1[0] = ((val1<0.0f)?-1:((int)(val1)));
  float val2 = data0[1];
  data1[1] = ((val2<0.0f)?-1:((int)(val2)));
  float val3 = data0[2];
  data1[2] = ((val3<0.0f)?-1:((int)(val3)));
}
```

`(int)(3.7f)` is `3`: C's cast truncates toward zero, matching `exec_alu`'s `int(v)`.

### Scratch memory

An `ALLOC` of one element is rendered as a plain scalar variable (`float acc0;`), and an `INDEX` into it
renders as just `acc0`. Larger allocations become arrays:

@include uopc/render_c.py#CRenderer._scalar

The scalar case is not cosmetic. A local array is memory as far as C is concerned, and although gcc can
often promote small arrays to registers, a scalar is guaranteed to be eligible for a register. Accumulators
in [Chapter 5](05-loops.md) are size-1 allocs that get updated on every iteration of a hot loop, and they
must live in registers. For a 4-element scratch array written by four stores, `after(*ws)` names all four
writes as dependencies:

```python
t4 = UOp(Ops.ALLOC, (), ParamArg(1, dtypes.f32, (4,)))
ws = [t4.index(k).store(a.index(k % 2).load() * float(k)) for k in range(4)]
t4a = t4.after(*ws)
print(render_c(sink(out.index(0).store(t4a.index(0).load() + t4a.index(3).load()))))
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  float acc1[4];
  float val1 = data0[0];
  acc1[0] = (val1*0.0f);
  float val2 = data0[1];
  acc1[1] = (val2*1.0f);
  acc1[2] = (val1*2.0f);
  acc1[3] = (val2*3.0f);
  float val3 = acc1[0];
  float val4 = acc1[3];
  data1[0] = (val3+val4);
}
```

Notice `val1` used for both `acc1[0]` and `acc1[2]`. `a.index(0).load()` and `a.index(2 % 2).load()` are the
same hash-consed node, so the input is read once: CSE for free, correct because nobody writes `data0`.
Notice also `val1*0.0f`: nothing simplified this graph before rendering, and the simplifier would not have
removed it anyway, since `x*0.0` is not `0.0` for floats.

### The signature, and `restrict`

The function takes one pointer per `BUFFER` reachable from the `SINK`, sorted by slot:
`void kernel(float* restrict data0, float* restrict data1)`. The `restrict` qualifier is a promise to the C
compiler that, for the lifetime of the call, the memory reached through `data0` is not reached through any
other pointer. Without it, gcc must assume that a store to `data1[i]` might change `data0[j]`, so it has to
reload after every store and cannot vectorize freely. With it, loads can be hoisted, reordered and combined.
In return, *we* must never pass the same array as two different buffers. Our IR has no way to express
aliasing buffers, so the promise is easy to keep, but it is a promise. Breaking it is undefined behavior.

## 4.7 The C ABI in one page

To call the compiled function from Python we need to agree with it, at the machine level, on how arguments
are passed. That agreement is the *ABI* (application binary interface). On x86-64 Linux it is the System V
AMD64 ABI. The parts we use:

1. **Integer and pointer arguments** go in registers `rdi, rsi, rdx, rcx, r8, r9`, in that order (more go
   on the stack). Our kernels take only pointers, one per buffer.
2. **A pointer is just an address**, 64 bits. The callee does not know the array's length, its dtype, or
   whether it is contiguous. `data0[3]` on a `float*` means "the 4 bytes at address `data0 + 12`".
3. **Return values** go in `rax` (or `xmm0` for floats). Our kernels return `void`.

You can see all three in the machine code gcc produced for a tiny kernel:

```c
void k(float* restrict data0, float* restrict data1) {
  float val1 = data0[0];
  float val2 = data0[3];
  data1[1] = ((val1*2.0f)+val2);
}
```

```text
    1100:	endbr64
    1104:	vmovss xmm0,DWORD PTR [rdi]
    1108:	vmovss xmm1,DWORD PTR [rdi+0xc]
    110d:	vfmadd132ss xmm0,xmm1,DWORD PTR [rip+0xeea]        # 2000 <_fini+0xee4>
    1116:	vmovss DWORD PTR [rsi+0x4],xmm0
    111b:	ret
```

`data0` arrived in `rdi` and `data1` in `rsi`. `data0[3]` is `[rdi+0xc]` (3 × 4 bytes), and `data1[1]` is
`[rsi+0x4]`. The multiply and add became a single fused multiply-add, `vfmadd132ss`, because we compile with
`-ffp-contract=fast` (section 4.8). The constant `2.0f` lives in a read-only data section, addressed
relative to the instruction pointer (`rip`), which is what `-fPIC` asks for. (`objdump -d -M intel`
on the cached `.so`; `nm -D` lists the exported symbol: `0000000000001100 T k`.)

On the Python side, `ctypes` does the marshalling. A numpy array is a Python object with a header (dtype,
shape, strides) and a pointer to a data block. `arr.ctypes.data` is that pointer as an integer, and
`ctypes.c_void_p(...)` passes it as a 64-bit pointer argument. The header is left behind: the C function
sees only the address.

@diagram 04-abi

Four consequences follow, and the runtime handles only the first:

- **Contiguity.** A numpy view like `x[::2]` has a stride of 8 bytes, but the C code assumes 4. The runtime
  checks `c_contiguous` and refuses.
- **Dtype.** Nothing checks that a `float*` parameter receives float32 data. Pass a float64 array and the
  kernel reinterprets its bytes (Pitfalls).
- **Length.** Nothing checks that the array is long enough. Reading past the end is undefined behavior.
- **Lifetime.** The pointer is valid only while the numpy array is alive and not resized. Our calls are
  synchronous, so the array outlives the call. An asynchronous runtime (GPUs,
  [Chapter 12](12-gpus.md)) has to keep references explicitly.

## 4.8 The runtime

`uopc/runtime.py` is 66 lines. It has three jobs: compile C to a shared object, cache that by content, and
call it.

### Compiler flags

@include uopc/runtime.py#CFLAGS

| Flag | Effect | Why |
|---|---|---|
| `-O3` | Full optimization, including auto-vectorization | We want gcc's best code; we only generate simple C |
| `-march=native` | Use every instruction this CPU has (AVX2, FMA, ...) | Kernels run on the machine that compiled them |
| `-fopenmp` | Enable `#pragma omp` | `THREAD` axes in [Chapter 10](10-cpu-performance.md) |
| `-ffp-contract=fast` | Allow `a*b+c` to become one FMA instruction | Faster, and more accurate (one rounding instead of two) |
| `-fassociative-math` | Allow reordering float additions | Required to vectorize float reductions (note 20) |
| `-fno-signed-zeros` | Treat `-0.0` like `+0.0` | Required by `-fassociative-math`; matches `x+0.0 -> x` in [Chapter 3](03-symbolic.md) |
| `-fno-trapping-math` | Assume float ops never trap | Required by `-fassociative-math` |
| `-shared -fPIC` | Build a position-independent shared library | So it can be loaded into the running Python process |
| `-lm` | Link the math library | `exp2f`, `sqrtf`, ... |

Notice what is *not* there: `-ffast-math`. That flag also assumes there are no NaNs and no infinities,
which would break `max` reductions initialized to `-INFINITY` and any code that relies on `nan`
propagating. `-fassociative-math` and its two prerequisites are the part of fast-math that we actually want.
The FMA contraction means the C program and `exec_alu` can differ in the last bit for `a*b+c`. Folding
happens on constants, which gcc would fold the same way, so this does not bite in practice, but it is
another place where "bit-exact" has an asterisk. `CFLAGS` can be overridden from the environment.

### Compiling, with a content-addressed cache

@include uopc/runtime.py#compile_c

The cache key is a SHA-256 of the source text, the compiler, and the flags. The `.so` is stored under
`~/.cache/uopc/<key>.so`. The design is *content addressing*: the name of the artifact is a function of
everything that determines it, so there is no invalidation logic at all. Change one character of the kernel,
or one flag, and the key changes. The same kernel compiled by two processes, or on two runs a week apart,
gets the same key. The cost is that the renderer must be deterministic (section 4.6).

Atomicity matters because several processes (pytest workers, say) may compile the same kernel at the same
time. Each compiles into its own temporary directory and then `os.replace`s the result into place.
`os.replace` is atomic on POSIX *within one filesystem*, so a reader sees either no file or a complete one,
never a half-written `.so`. The temporary directory is created *inside* the cache directory for exactly
that reason. The first version used the system temp dir, and `os.replace` from `/tmp` (a tmpfs) to
`~/.cache` (the disk) failed: it was a cross-filesystem rename (note 10).

@diagram 04-compile-run

A compile error is reported with gcc's message and a numbered listing of the generated source, since the
generated source is the thing you need to read:

```text
gcc failed:
~/.cache/uopc/tmpmugz9v_0/k.c: In function ‘k’:
~/.cache/uopc/tmpmugz9v_0/k.c:2:14: error: expected ‘;’ before ‘}’ token
    2 |   a[0] = 1.0f
      |              ^
      |              ;
    3 | }
      | ~             

--- source ---
   1 void k(float* a) {
   2   a[0] = 1.0f
   3 }
```

(The cache path is shortened to `~/.cache/uopc`; the run used a scratch cache directory.)

### Calling

@include uopc/runtime.py#Program

`ctypes.CDLL` calls `dlopen` on the shared object, which maps it into the process. `getattr(lib, name)`
looks up the exported symbol. Setting `restype = None` tells ctypes the function returns `void` (the default
would read `rax` as an `int`). No `argtypes` are declared. Instead every argument is wrapped in `c_void_p`,
which ctypes passes as a 64-bit pointer. That is everything the System V ABI needs for a function of `n`
pointers.

A session, against an empty cache:

```text
>>> import time, numpy as np
>>> from uopc.runtime import compile_c, Program
>>> src = "void axpy(float* restrict x, float* restrict y) { for (int i = 0; i < 4; i++) y[i] = 2.0f*x[i] + y[i]; }\n"
>>> t0 = time.perf_counter(); so = compile_c(src); print(f"{(time.perf_counter()-t0)*1e3:.1f} ms", so.name)
60.1 ms e50750a3fe3c54975390ccb4.so
>>> t0 = time.perf_counter(); so = compile_c(src); print(f"{(time.perf_counter()-t0)*1e3:.1f} ms", so.name)
0.1 ms e50750a3fe3c54975390ccb4.so
>>> p = Program(src, "axpy")
>>> x, y = np.arange(4, dtype=np.float32), np.ones(4, np.float32)
>>> p(x, y); y
array([1., 3., 5., 7.], dtype=float32)
>>> p(x[::2], y[:2])
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "uopc/runtime.py", line 55, in __call__
    if not b.flags.c_contiguous: raise ValueError("buffers must be contiguous")
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: buffers must be contiguous
```

A cold compile is tens of milliseconds (gcc has to start, parse, optimize, link), and a warm one is a
`stat` call. For a GPT training run that compiles a few hundred kernels, the difference between the first
run and every later run is the difference between seconds and nothing.

### Gluing it together

@include uopc/runtime.py#run_c

`run_c` is the week-2 entry point: render, find the buffer slots in the same order as the renderer's
signature (sorted by slot, reachable from the `SINK`), compile, and call. Both sides compute the parameter
list from the same graph with the same sort, so they cannot disagree. Return the source, so tests can
inspect it.

## 4.9 Type systems for shape, dtype and addrspace

The syllabus devotes a lecture to "type systems for shape, dtype, addrspace". By now we have all three, and
it is worth being precise about what each one catches. All three are checked at construction time, in
`_infer` ([Chapter 1](01-uops.md)), so a malformed node never exists. The error points at the line of Python
that built it, not at some later pass that tripped over it. Tracebacks are trimmed to their last line here:

```text
>>> from uopc.uop import UOpError
>>> a, ai = buf(0, 4), buf(1, 4, dtypes.i32)
>>> (UOp.const(1.0) + 2.0).load()
uopc.uop.UOpError: LOAD needs memory
>>> ai.index(0).store(UOp.const(1.0))
uopc.uop.UOpError: STORE of dtypes.f32 into dtypes.i32
>>> a.index(UOp.const(1.0))
uopc.uop.UOpError: INDEX indices are i32 scalars
>>> a.index(0).load() + ai.index(0).load()
uopc.uop.UOpError: Ops.ADD: dtype mismatch dtypes.f32 vs dtypes.i32 (use CAST)
>>> a.index(0).load() // a.index(1).load()
uopc.uop.UOpError: Ops.FLOORDIV is integer only
>>> UOp(Ops.BUFFER, (), ParamArg(0, dtypes.f32, (2, 2)))
uopc.uop.UOpError: BUFFER is flat
>>> a.index(0, 1)
uopc.uop.UOpError: INDEX: 2 indices into shape (4,)
>>> a.index(7).load()
UOp(Ops.LOAD, src=(UOp(Ops.INDEX, src=(UOp(Ops.BUFFER, arg=ParamArg(slot=0, dtype=dtypes.f32, shape=(4,), addrspace=None)), UOp(Ops.CONST, arg=7),)),))
```

| Property | Catches | Example |
|---|---|---|
| dtype | Mixing ints and floats without a cast; storing the wrong type; float indices; integer-only ops on floats | `STORE of dtypes.f32 into dtypes.i32` |
| shape | Elementwise ops on mismatched shapes; too many indices; non-flat memory | `INDEX: 2 indices into shape (4,)` |
| addrspace | Loading from or storing to a value; computing with an address as if it were a value | `LOAD needs memory` |

The last line of the session is the important negative result: `a.index(7)` on a 4-element buffer is
accepted. Our types describe *what kind* of thing a node is, not *which values* it can take. Several real
bugs are invisible to them:

- **Out-of-bounds indices.** The index is an expression with bounds, and [Chapter 3](03-symbolic.md)
  computes them. A render-time check `0 <= vmin(idx)` and `vmax(idx) < n` would catch most out-of-bounds
  accesses statically (exercise 4). A type system that tracks value ranges is a *refinement type* system,
  and that is roughly what this check is.
- **Reading before writing.** The missing-`AFTER` bug of section 4.3 type-checks perfectly. Detecting it
  needs a dataflow analysis: is there a path from the `ALLOC` to the `LOAD` that does not pass through a
  store?
- **Aliasing.** Passing one numpy array as two `restrict` buffers is invisible to the IR.
- **Runtime marshalling.** The numpy array's dtype and length are not checked against the `BUFFER`.

Each check you add catches bugs earlier and costs code. The course grades "can it be simpler?", so add them
when a bug teaches you that you need them, and write the bug down.

## 4.10 One renderer, several targets

`CRenderer` is a class, not a function, because the CUDA backend of [Chapter 12](12-gpus.md) is a
subclass. The differences between C dialects are mostly a handful of strings, so they are class attributes:
`kernel_prefix` (`"void"` here, `extern "C" __global__ void __launch_bounds__(...)` for CUDA), `buffer_prefix`, `shared_prefix`
for GPU shared memory, `barrier` for `__syncthreads()`, and `restrict` (CUDA spells it `__restrict__`). The
methods `header`, `range_name`, `render_range` and `is_loop` are the structural hooks. Everything else, the
walk over the linear order, naming, inlining, constants and division, is shared. When you write a renderer
for another C-like target (OpenCL, Metal, HIP, or plain C with different intrinsics), start by listing which
of these strings and hooks change. If the list is short, subclass. If you find yourself overriding
`render` itself, the abstraction is in the wrong place.

## 4.11 Implementing week 2

A build order in which every step can be tested on its own:

1. **Memory ops in the IR.** `BUFFER`, `ALLOC`, `INDEX`, `LOAD`, `STORE`, `AFTER`, `SINK` in `Ops`, and
   their rules in `_infer`, including `addrspace`. Test the error messages of section 4.9 first: they are the
   cheapest tests you will ever write.
2. **A topological order.** For loopless programs, `toposort` from the `SINK` is enough. Print it with `%n`
   references, as in section 4.3, and check by eye that every store comes before the `AFTER` that names it.
3. **The renderer, one op at a time.** Start with `BUFFER`, `CONST`, `INDEX`, `LOAD`, `STORE` and `ADD`.
   Compile the output by hand with `gcc -c` before writing any runtime code. Then add `MUL`, comparisons,
   `WHERE`, casts, `RECIP` and the math functions, then `ALLOC` and `AFTER`, then the floor-division helpers
   with the `vmin` fast path.
4. **The runtime.** `compile_c` without the cache first, then add content addressing, then the atomic rename.
   Call with ctypes and compare with numpy.
5. **Differential tests.** Run random operands through every ALU op, compiled versus `exec_alu`, as in
   [Chapter 3](03-symbolic.md). This is the test that finds the bugs you did not think of.
6. **The week-2 tests.** `tests/test_w02_render.py`: elementwise, `ALLOC` with `AFTER`, negative floor
   division, casts and `WHERE`.

Keep the renderer boring. It should be a direct transcription of the linear order, with every decision about
*what* to compute made earlier, in the graph. Optimizations belong in rewrites, where they can be tested on
the IR, not in string manipulation.

## Pitfalls

**Forgetting `AFTER`.** Loading from scratch memory without `.after(store)` gives a load that does not
depend on the store, so the store is dead and the load reads garbage (section 4.3). The C compiles, the
test may even pass by luck, and gcc with `-Wall` may or may not warn about the uninitialized variable. The
fix is the invariant: a read of anything this program writes goes through `AFTER` naming the write.

**Rendering `FLOORDIV` as `/` (note 4).** It is correct for non-negative operands only. If you take the
shortcut without the `vmin` check, everything passes until the first negative index (padding, shifted
windows, any `x - c`). `test_floordiv_rendering_negative` is the regression test.

**Float literals without `f`.** `x * 0.5` with `float x` is computed in double. It is correct-ish, slower,
blocks single-precision vectorization, and differs from `exec_alu`'s f32 semantics in rounding. Always
suffix.

**`-2147483648` is not an int.** In C, it is `-(2147483648)`, and `2147483648` is a `long`. Use
`(-2147483647-1)`.

**Wrong dtype through ctypes.** The runtime does not check numpy dtypes. Passing a float64 array to a
`float*` kernel reinterprets the bytes:

```text
>>> y64 = np.ones(4)
>>> p(x, y64); y64
array([3.84000000e+02, 1.14688016e+05, 1.00000000e+00, 1.00000000e+00])
```

The kernel wrote four floats (16 bytes) over the first two doubles. Nothing crashed. That listing calls a
`Program` directly, which takes raw arrays and cannot know better. `run_c` knows each `BUFFER`'s
`ParamArg`, and after this pitfall was found it checks dtype and length before calling:

```text
TypeError: buffer slot 0: kernel wants 4 x f32, got 4 x float64
```

(`test_run_c_checks_buffer_dtypes`.) Exercise 5 asks you to add the same check to your runtime.

**Cross-filesystem `os.replace` (note 10).** Compiling in `/tmp` and renaming into `~/.cache` fails when
they are different filesystems, and even where a fallback copy works, it is not atomic. Create the temporary
directory inside the cache directory.

**Nondeterministic rendering.** If the order of lines or the choice of names depends on Python `set` or
`dict` iteration over objects hashed by `id`, two runs produce different source text for the same graph,
and the content-addressed cache misses every time. Iterate in linearized order and number names with a
counter.

**Inlining everything.** Rendering every node as an expression produces C whose size is exponential in the
depth of shared subexpressions. Name anything used more than once.

**Inlining loads.** Moving a load into its user can move it past a store to the same address. Name every
load at its position in the linear order.

## Exercises

1. ★ **Predict the C.** Without running it, write the C that `render_c` produces for
   `sink(out.index(0).store((a.index(0).load() + 1.0) * (a.index(0).load() + 1.0)))` with `a`, `out` as
   4-element f32 buffers. Which nodes get names? *Hint: count uses after hash consing.*

2. ★ **Find the bug.** A student writes a two-element dot product with a scratch accumulator:
   `acc = ALLOC f32[1]`, `w0 = acc[0].store(a[0]*b[0])`, `w1 = acc[0].store(acc[0].load() + a[1]*b[1])`,
   `out[0].store(acc.after(w1)[0].load())`. The result is wrong. Why, and what is the one-word fix?

3. ★★ **Dot product by hand.** Write a loopless program that computes the dot product of two 8-element
   vectors into `out[0]`, using a one-element `ALLOC` updated by a chain of eight stores. Check it against
   numpy with `run_c`, and look at the C.

4. ★★ **Static bounds checking.** Add a check to your renderer: for every `INDEX` into a `BUFFER` or
   `ALLOC` of size `n`, raise if `vmin(idx) < 0` or `vmax(idx) >= n`. Which test programs from weeks 2
   and 3 does it accept? Construct a program it wrongly rejects. (*Hint: the dependency problem of
   [Chapter 3](03-symbolic.md).*)

5. ★★ **Safer calls.** Make your runtime check each numpy array's dtype and size against the corresponding
   `BUFFER`'s `ParamArg` before calling (as `run_c` does). What does it cost per call? The kernelizer's hot
   path (`realize`) skips the check because it allocates every array itself: is that a sound argument?

6. ★★ **Why loads are named.** Construct a graph (with `AFTER`) where inlining a single-use `LOAD` into its
   user would read the wrong value. *Hint: you need a store between the load's position and its user's.*

7. ★★ **Disassemble.** Compile `test_elementwise_loopless`'s kernel, find the `.so` in the cache, and read
   it with `objdump -d -M intel`. Did gcc vectorize the four elements into one `vmulps`/`vaddps`? Remove
   `restrict` from the source by hand, recompile with `compile_c`, and compare.

8. ★★★ **A second backend.** Write `render_py(sink)` that emits a Python function operating on numpy
   arrays (one statement per UOp, same naming scheme), and use it as an oracle: run every week-2 test
   program through both backends and compare outputs. What do you have to do about floor division? About
   f32 rounding?

9. ★ **Constants.** For each value, predict `cconst(v, dtypes.f32)`: `2.5`, `1e10`, `-float("inf")`,
   `0.30000001192092896`. Why is `repr` safe for the last one?

<details>
<summary>Solutions to ★ exercises</summary>

**1.** `a.index(0).load()` is one node, used by both `+` nodes, but the two `+ 1.0` nodes are *also* the same
node (same op, same sources), so the multiply's two sources are one node used twice. The C is:

```c
void kernel(float* restrict data0, float* restrict data1) {
  float val1 = data0[0];
  float alu2 = (val1+1.0f);
  data1[0] = (alu2*alu2);
}
```

**2.** `w1` loads `acc[0]` without `AFTER(w0)`, so its load does not depend on `w0`, nothing depends on `w0`,
and `w0` is dead. `w1` reads an uninitialized accumulator. The rendered C says so plainly (note that
`data0[0]` and `data1[0]` are never read):

```c
void kernel(float* restrict data0, float* restrict data1, float* restrict data2) {
  float acc0;
  float val1 = acc0;
  float val2 = data0[1];
  float val3 = data1[1];
  acc0 = (val1+(val2*val3));
  float val4 = acc0;
  data2[0] = val4;
}
```

Fix: load through `acc.after(w0)` inside `w1`.

**9.** `2.5f`, `10000000000.0f`, `-INFINITY`, `0.30000001192092896f`. The last value is exactly a float32
(it is `f32(0.3)`), and `repr` prints the shortest decimal that identifies the double. That decimal is
closer to this float than to any other float, so C's float parser returns exactly this float.

</details>

## Quiz

1. Why does hash consing make `AFTER` necessary?
2. What does `AFTER(mem, dep)` render as in C?
3. Why is a single-use `LOAD` given a variable name, while a single-use `ADD` is inlined?
4. When does the renderer emit plain `/` for `FLOORDIV`?
5. What does `restrict` promise, and who must keep the promise?
6. In which register does the second buffer pointer arrive on x86-64 Linux?
7. Why is the compiled-kernel cache keyed on the source text and the flags, and not on, say, the UOp graph?
8. Name two bugs that type-check fine in our IR.

<details>
<summary>Answers</summary>

1. Hash consing merges structurally equal nodes, so "load `tmp[0]`" before and after a store would be the
   same node. `AFTER(tmp, store)` is a different node, so loads built on it are different nodes, and the
   store becomes a dependency.
2. Nothing. It renders as its first source; its only effect is on the linear order.
3. A load must happen at its position in the linear order, which respects stores; inlining could move it
   past a store. A pure ALU op has no such constraint, and inlining it keeps the C readable.
4. When bounds analysis proves the numerator non-negative (`vmin >= 0`) and the divisor is a positive
   constant. Then truncation and floor agree.
5. That memory reached through that pointer is not reached through any other pointer during the call. The
   caller (our runtime) must keep it; the compiler exploits it.
6. `rsi` (the first is in `rdi`).
7. The source text plus compiler and flags fully determine the `.so`, and that key is stable across
   processes and runs. A graph key would have to account for renderer changes too. (A graph-level cache
   exists as well, one level up, in [Chapter 7](07-kernelize.md).)
8. Any two of: out-of-bounds constant or computed indices; reading scratch memory before writing it (missing
   `AFTER`); passing aliased buffers; passing a numpy array of the wrong dtype or length.

</details>

## Further reading

- Randal E. Bryant and David R. O'Hallaron, *Computer Systems: A Programmer's Perspective*, 3rd ed.:
  chapter 3 (machine-level programs, x86-64 calling conventions) and chapter 7 (linking, shared libraries,
  position-independent code).
- Michael Matz, Jan Hubička, Andreas Jaeger, Mark Mitchell (eds.), *System V Application Binary Interface,
  AMD64 Architecture Processor Supplement*.
- Ulrich Drepper, "How To Write Shared Libraries" (2011).
- The Python documentation for `ctypes`, and numpy's documentation for `ndarray.ctypes`.
- The GCC manual, "Options That Control Optimization": `-ffast-math` and its component flags.
- ISO/IEC 9899 (the C standard), §6.7.3.1 "Formal definition of restrict".
- Cliff Click and Michael Paleczny, "A Simple Graph-Based Intermediate Representation", IR '95 (sea of
  nodes, memory as a value).
- LLVM documentation, "MemorySSA".
- tinygrad, `tinygrad/renderer/cstyle.py` (the C-style renderer this chapter's design follows) and
  `tinygrad/runtime/ops_cpu.py`.
