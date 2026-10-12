# 1. The UOp: A Graph IR

A compiler cannot work on your program directly. Python source is text; a running PyTorch model is a
tangle of objects, C++ and CUDA; neither can be *inspected, transformed and printed* in a uniform way.
Before a compiler can optimize anything it needs its own representation of programs: a data structure
in which "multiply these two things, then add that" is a value you can look at, compare, take apart
and rebuild. That data structure is the compiler's *intermediate representation*, its IR, and its
design decides how hard every later chapter will be. This chapter designs the IR for the whole book:
one tiny class, `UOp`, three fields wide, which by week 10 will describe a GPT.

**What you'll build**

- `UOp(op, src, arg)`: an immutable graph node, and the `Ops` enum of operations.
- *Hash consing*: building the same node twice returns the same object, so structural equality is `is`.
- *Recursive properties*: `dtype`, `shape` and `addrspace` inferred at construction time; a type
  checker that rejects malformed graphs with useful errors.
- Week-1 semantics: `CONST`, `STACK`, `INDEX`, ALU ops, `PARAM` and `CALL`.
- Graph utilities: iterative `toposort`, `substitute`.
- The `.uir` text format: a parser and a printer that round-trip exactly.

**Where this fits.** This is the first chapter of Part I. The [Preface](00-preface.md) shows where the
IR sits in the pipeline. [Chapter 2](02-rewriting.md) builds the rewrite engine that transforms these
graphs (and uses it to *run* them); [Chapter 3](03-symbolic.md) teaches it algebra. Every later chapter
adds ops to the enum you meet here, but never a second kind of node.

## 1.1 What an IR is for

A compiler is a pipeline of translations. In a C compiler, text becomes tokens, tokens become an
abstract syntax tree, the tree becomes a lower-level IR (LLVM IR, say), the IR is optimized in dozens of
passes, and finally becomes machine code. Each arrow in that chain is easier to write than a single
arrow from text to machine code would be, *because each IR is designed to make particular questions
easy to answer*:

- An abstract syntax tree answers "what did the programmer write?" It keeps parentheses, variable
  names and the order of statements.
- A control-flow graph of SSA instructions (LLVM's IR, GCC's GIMPLE) answers "what values flow where,
  and in which order can things happen?" It forgets names and nesting, and every value is assigned
  exactly once.
- Machine code answers "what does the CPU do?" and nothing else.

An ML compiler has an unusual source language. The "program" is a computation on tensors that the user
assembled at runtime by calling Python functions: `x @ w + b`, `softmax`, `cross_entropy`. There is no
text to parse; the frontend (our `Tensor` class, [Chapter 9](09-tensor-mnist.md)) builds the IR
directly as the user calls methods. And the programs are, overwhelmingly, *dataflow*: values are
computed from other values, nothing is overwritten, there are no `if` statements that depend on data
(a ReLU is a `max`, not a branch) and no loops whose trip count depends on data.

Dataflow programs have a natural representation: a graph whose nodes are operations and whose edges
say "this operation consumes that value". Take the expression `(a + b) * (a + b) + 1.0`:

@diagram 01-uop-dag

Three properties make this graph a good IR:

1. **It is a DAG, not a tree.** `a + b` is computed once and used twice. A tree (like the AST of the
   expression as written) would contain two copies of `a + b` and you would need a separate
   optimization, common subexpression elimination (CSE), to notice they are the same.
2. **It has no order beyond dependencies.** Whether `a` or `b` is "computed first" is not a question
   the graph asks. That freedom is what lets later passes fuse, reorder, and parallelize.
3. **Every node is a pure value.** A node means "the result of this op on these inputs", always,
   everywhere. Pure values can be duplicated, shared, moved and compared freely. (Week 2 will add
   memory and side effects, and we will need explicit ordering edges to keep this property; see
   [Chapter 4](04-memory-and-c.md).)

Production ML compilers agree: XLA's HLO, TVM's Relay, PyTorch's FX graphs, MLIR's tensor dialects
and tinygrad's UOps are all dataflow graphs. Where they differ is in how many *kinds* of nodes they
have, and that is our first design decision.

### One node type, not fifty

The obvious object-oriented design gives each operation its own class: `Add(lhs, rhs)`,
`Const(value)`, `Reshape(x, shape)`, each with its own fields and a visitor method. It feels tidy and
type-safe. It is the wrong choice for this compiler, for three reasons:

- **Pattern matching.** Most of the compiler is rewrite rules: "an ADD whose second source is a CONST 0
  becomes its first source". If every node has the same shape (an op, a tuple of sources, an argument)
  one generic matcher handles every rule ([Chapter 2](02-rewriting.md)). With fifty classes you write
  fifty matchers, or reflection.
- **Generic traversals.** `toposort`, `substitute`, hashing, printing, pickling and the rewrite engine
  never need to know what an op means. They work on `(op, src, arg)` and are each a dozen lines.
- **Hash consing.** We are about to make every node unique by its structure. That needs a single
  key-building function and a single cache.

So a UOp is a triple:

```
UOp(op: Ops, src: tuple[UOp, ...], arg: Any)
```

`op` says what it is, `src` are the inputs (in order), and `arg` holds any non-graph data: the value
of a constant, the target dtype of a cast, the new shape of a reshape. That is the entire node. The
alternative, a node per op class, is what you find in XLA's C++; tinygrad, which `uopc` follows, uses
the triple.

## 1.2 The `Ops` enum

Here are all the operations the compiler will ever have, grouped by the week they appear:

@include uopc/uop.py#Ops

Only the first block matters this week. Notice how *small* it is. There is no subtraction, no
negation, no division, no greater-than, no equality, no ReLU. Every one of those is expressible from
what is there:

| You want | You build | Why it is enough |
|---|---|---|
| `x - y` | `x + y * -1` | |
| `-x` | `x * -1` | |
| `x / y` (float) | `x * RECIP(y)` | |
| `x > y` | `CMPLT(y, x)` | swap the operands |
| `x == y` | `CMPNE(CMPNE(x, y), True)` | "not not-equal" |
| `relu(x)` | `WHERE(CMPLT(0, x), x, 0)` | |
| `x // y`, `x % y` (int) | `FLOORDIV`, `FLOORMOD` | primitive: these can't be built from the others |

The fewer ops there are, the fewer rules every later pass needs. Every rewrite rule, every renderer
case, every gradient rule ([Chapter 8](08-autodiff.md)) is written once per op. Seven arithmetic ops
instead of fifteen halves that work. The cost is that patterns must recognize the *encoded* forms
(a rule about subtraction has to look for `x + y*-1`), which [Chapter 3](03-symbolic.md) handles.

The week-1 ops, precisely:

| Op | `src` | `arg` | Result |
|---|---|---|---|
| `CONST` | `()` | a Python `bool`, `int` or `float` | a scalar; dtype from the Python type |
| `STACK` | `(x0, x1, ..., xn-1)`, all same shape and dtype | `None` | shape `(n, *x0.shape)` |
| `INDEX` | `(x, i0, ..., ik-1)`, indices are `i32` scalars | `None` | `x` with its first `k` axes removed |
| `CAST` | `(x,)` | target `DType` | `x` converted, same shape |
| `RECIP` | `(x,)`, float | `None` | `1/x` |
| `ADD`, `MUL` | `(a, b)`, same shape and dtype | `None` | same shape and dtype |
| `CMPLT` | `(a, b)`, same shape and dtype | `None` | `bool`, same shape |
| `FLOORDIV`, `FLOORMOD` | `(a, b)`, both `i32` | `None` | `i32`, *floor* semantics (Python's, not C's) |
| `WHERE` | `(cond, a, b)`, `cond` is `bool` | `None` | `a` where `cond` else `b`, elementwise |
| `PARAM` | `()` | `ParamArg(slot, dtype, shape)` | the `slot`-th input of the enclosing `CALL` |
| `CALL` | `(body, arg0, ..., argn-1)` | `CallArg(name)` | the body with its PARAMs bound to the args |

Why floor division instead of C's truncating division is a subtle and important decision, made
in [Chapter 3](03-symbolic.md). Python sets that the rest of the compiler uses to talk about groups
of ops are defined right after the enum:

@include uopc/uop.py:59-63

## 1.3 Data types and address spaces

A dtype is a tiny frozen record: a name, a size in bytes, the C type to print, and the
`struct`/numpy format character.

@include uopc/dtype.py#DType

@include uopc/dtype.py#dtypes

Only four matter for most of the book: `void` (the type of things that are not values, like a store),
`bool`, `i32` and `f32`. `f16` arrives in week 8 for tensor cores. That is deliberately fewer than
any real framework. There is no `i64`, so the largest tensor we can index has 2³¹−1 elements, and no
`f64`, so there are no accidental double-precision slow paths. Adding dtypes later is easy *if* every
place that depends on them goes through `DType`'s fields, which is why the renderer asks for
`dtype.ctype` instead of switching on names.

A Python literal's dtype is fixed by `from_py`: `bool` before `int` (in Python `True` is an `int`!),
then `int`, then `float`. This is why `UOp.const(1)`, `UOp.const(1.0)` and `UOp.const(True)` are three
different nodes with three different dtypes, which is about to become a real bug.

The third recursive property, `addrspace`, says *where* a value lives:

@include uopc/dtype.py#AddrSpace

All week-1 values are `ALU`: plain values that a C compiler would keep in registers. Memory arrives in
week 2 ([Chapter 4](04-memory-and-c.md)), shared memory in week 7. We carry the field from day one so
the type checker has a place to put the rule "you can only LOAD from memory".

## 1.4 The UOp class and hash consing

Here is the core of the class:

@include uopc/uop.py#UOp.__new__

Construction is a cache lookup. The key is `(op, src, _argkey(arg))`. If a live UOp with that key
exists, you get *that object back*; otherwise a new one is made, its dtype, shape and address space are
inferred (and checked), and it is put in the cache. This technique is called **hash consing**, after
Lisp's `cons`: it goes back to Ershov's 1958 work on expression evaluation, and functional language
implementations have used it to share structure ever since.

@diagram 01-hash-consing

The consequence is an invariant that the rest of the compiler leans on everywhere:

> **Key idea:** two UOps are structurally equal if and only if they are the *same Python object*.
> Equality is `is`, hashing is `id`, and both are O(1) no matter how big the graphs are.

Watch it happen. Two separate constructions of `x + y`:

```python
>>> x = UOp(Ops.PARAM, (), ParamArg(0, dtypes.f32))
>>> y = UOp(Ops.PARAM, (), ParamArg(1, dtypes.f32))
>>> a = x + y; b = x + y
>>> a is b, id(a) == id(b)
(True, True)
```

Note what this does *not* require: a recursive comparison. `src` is a tuple of UOps, and since every
UOp in it is already unique, comparing two `src` tuples compares a few pointers. Hashing `src` hashes a
few pointers. The cache key of a node with a million-node subgraph below it costs the same to build as
the key of a leaf. This works because of the order of construction: sources always exist before
the node that uses them, so by induction every source is already canonical.

### What hash consing buys

**Free CSE.** Common subexpression elimination is a standard compiler pass: find two computations of
the same thing, keep one. With hash consing there is nothing to find; duplicates can't exist. In the
DAG above, `s * s` has `s.src[0] is s.src[1]`, and no pass ever had to discover it.

**O(1) memoization.** Any function of a node can be cached in a `dict` keyed by the node. The rewrite
engine ([Chapter 2](02-rewriting.md)) memoizes its work this way, `simplify` is memoized globally
([Chapter 3](03-symbolic.md)), and the kernel cache ([Chapter 7](07-kernelize.md)) keys whole compiled
programs on graph structure.

**Cheap equality checks in tests and rules.** `parse(to_uir(g)) is g` is a complete round-trip test.
`simplify(x * 1) is x` checks a rewrite. Rules can say "the same `x` appears twice" by checking
identity.

**Acyclicity for free.** To build a node you must pass its sources, which must already exist. You
cannot create a node that is its own ancestor, because it would have to exist before itself. Every UOp
graph is a DAG by construction, and nothing ever has to check for cycles.

### What it costs

**Immutability is mandatory.** If you could change a UOp's `src` after construction, its cache key
would lie, and every node that had been hash-consed against it would silently change meaning. UOps
are treated as immutable: to "modify" a node, build a new one. That is what `replace` does:

@include uopc/uop.py#UOp.replace

**Every construction pays for a lookup.** A tuple, a hash and a dict probe per node. That is a few
hundred nanoseconds of Python, which is why compile-time performance shows up in later chapters (the
GPT graph is large); the payoff (no CSE pass, O(1) everything else) is much larger.

**Things that must be different must look different.** Two input buffers with the same shape and dtype
would be the same node. That is why every `PARAM` and `BUFFER` carries a `slot` number in its arg:
it is part of the identity.

### Why weak references

The cache is a `weakref.WeakValueDictionary`: it does not keep its values alive. When the last real
reference to a UOp disappears, the UOp is garbage collected and its entry vanishes from the cache.
Without that, every intermediate node ever created (and a training loop creates millions) would live
forever. You can watch the cache shrink:

```python
>>> before = len(UOp._cache)
>>> z = UOp.const(12345) + UOp.const(54321)
>>> mid = len(UOp._cache)
>>> del z; gc.collect()
>>> print(before, mid, len(UOp._cache))
12 15 12
```

Three nodes (two constants and the add) were created and three were reclaimed. The constants were
kept alive by the ADD's `src` tuple until the ADD itself died. One corollary: you must never key a
long-lived dictionary on `id(u)`, because after `u` dies its `id` can be reused by a different UOp.
Key on the UOp itself, which holds a strong reference.

### Equality and hashing are inherited from `object`

Look at what the class *doesn't* define: `__eq__` and `__hash__`. Python's defaults compare and hash
by identity, which, thanks to hash consing, is exactly structural equality. The comment in the source
says why this matters:

@include uopc/uop.py:119-124

The first version of `uopc` defined `__eq__` and `__hash__` in Python. Both were correct, and
compiling the GPT took 163 seconds for 12 steps, because every `dict` lookup in every pass called a
Python function. Switching to the C-implemented identity versions (and memoizing `simplify`) took it
to 47 seconds. The enum classes do the same: `Ops` sets `__hash__ = object.__hash__`, because Python's
`Enum` defines a slower `__hash__` of its own.

`__reduce__` makes pickling (and `copy.deepcopy`) go back through `UOp(...)`, so an unpickled graph is
re-hash-consed into the live cache:

```python
>>> pickle.loads(pickle.dumps(e)) is e, copy.deepcopy(e) is e
(True, True)
```

## 1.5 `_argkey`: when Python's equality lies

The cache key cannot be `(op, src, arg)` directly. This was the first real bug in `uopc`. Python
says:

```python
>>> 1 == 1.0 == True, hash(1), hash(1.0), hash(True)
(True, 1, 1, 1)
>>> 0.0 == -0.0, hash(0.0) == hash(-0.0)
(True, True)
>>> d = {}; d[(Ops.CONST, (), 1)] = "i32 one"
>>> d.get((Ops.CONST, (), True)), d.get((Ops.CONST, (), 1.0))
('i32 one', 'i32 one')
```

A cache keyed on the raw arg merged `CONST 1` (i32), `CONST 1.0` (f32) and `CONST True` (bool) into
whichever one was built first. After that, `UOp.const(1.0)` could return a node whose dtype was `i32`,
and the type checker would complain about a dtype mismatch in code that was obviously correct, or
worse, accept a graph whose constants had the wrong type. Zero has the same problem with sign:
`0.0` and `-0.0` are equal and hash equal, but `1/0.0` is `inf` and `1/-0.0` is `-inf`.

The fix makes the type part of the key, and uses `repr` for floats, which distinguishes `-0.0`:

@include uopc/uop.py#_argkey

```python
>>> _argkey(1), _argkey(1.0), _argkey(True), _argkey(-0.0), _argkey((1, 2.0))
((<class 'int'>, 1), (<class 'float'>, '1.0'), (<class 'bool'>, True), (<class 'float'>, '-0.0'), (<class 'tuple'>, ((<class 'int'>, 1), (<class 'float'>, '2.0'))))
>>> UOp.const(1) is UOp.const(1.0), UOp.const(0.0) is UOp.const(-0.0)
(False, False)
```

`repr` also fixes a third, subtler problem: NaN. `nan != nan`, so with raw floats in the key, every
`UOp.const(float('nan'))` would miss the cache (dict lookup compares keys with `==` after the identity
shortcut) and create a new node. With `repr`, all NaNs share the key `'nan'`:

```python
>>> UOp.const(float('nan')) is UOp.const(float('nan'))
True
```

Tuples are recursed into because movement-op args are tuples of numbers. Every other arg type (a
`DType`, a `ParamArg`, an `Ops` member) is a frozen dataclass or enum whose equality is already exact.

> **Key idea:** hash consing is only as correct as the equality on its keys. Python's numeric tower
> deliberately makes `1`, `1.0` and `True` equal; a compiler must not.

## 1.6 Recursive properties: the type checker

`dtype`, `shape` and `addrspace` are not passed to the constructor. They are *recursive properties*:
computed from the op, the arg and the sources' own properties, once, when the node is built. The
function that computes them is the type checker of the IR, and every malformed UOp dies inside it:

@include uopc/uop.py#_infer

Read it as a set of typing rules, one per op. In the notation of a type system, the ADD rule is

```
a : (d, s)    b : (d, s)
------------------------
   ADD(a, b) : (d, s)
```

"if both operands have dtype `d` and shape `s`, the sum does too". The code says the same thing, with
an error message for each way the premises can fail. CMPLT is identical except that the conclusion is
`bool`. FLOORDIV adds the premise `d = i32`. RECIP requires a float. WHERE requires a `bool` condition
and agreeing branches. There is no implicit conversion anywhere: `UOp.const(1) + UOp.const(1.0)` is an
error, not a silent cast. Implicit promotion is the frontend's business ([Chapter 9](09-tensor-mnist.md));
the IR stays explicit.

Here is every week-1 error, produced by actually building the bad node:

```
i32 + f32            UOpError: Ops.ADD: dtype mismatch dtypes.i32 vs dtypes.f32 (use CAST)
shape mismatch       UOpError: Ops.ADD: shape mismatch (2,) vs (3,) (broadcast with EXPAND)
ragged stack         UOpError: STACK srcs must agree in shape and dtype
floordiv float       UOpError: Ops.FLOORDIV is integer only
recip int            UOpError: Ops.RECIP needs a float, got dtypes.i32
where cond           UOpError: WHERE condition must be bool
index float          UOpError: INDEX indices are i32 scalars
too many idx         UOpError: INDEX: 2 indices into shape (3,)
call missing param   UOpError: CALL needs params 0..0, body has [0, 1]
call wrong type      UOpError: CALL arg 0: dtypes.f32() != param dtypes.i32()
empty stack          UOpError: STACK of nothing
```

Two messages suggest the fix (`use CAST`, `broadcast with EXPAND`). That costs nothing to write and
saves students hours.

### Why check at construction time

The alternative is a separate *verifier* pass that walks a finished graph and checks it (MLIR works
this way: ops have `verify()` methods that run between passes). Checking at construction is better
for us for three reasons:

1. **The stack trace points at the bug.** When a rewrite rule builds an ill-typed node, the
   exception is raised inside that rule, with the rule's line in the traceback. A verifier would report
   it after the pass, with no idea which of forty rules produced it.
2. **Ill-typed nodes never exist.** The cache insert is the last line of `__new__`, after `_infer`. A
   failed construction leaves nothing behind, so no other code can ever observe a malformed UOp.
3. **Properties are computed once.** `u.shape` is a field read, not a recursive computation. Passes
   ask for shapes constantly.

The cost is that every construction pays for the check. It is a handful of comparisons; it has never
shown up in a profile.

Note that the type checker does *not* stop a rewrite rule from replacing a node with a node of a
different type. `_infer` checks that each node is well-formed given its sources, not that a
replacement means the same thing. [Chapter 2](02-rewriting.md) shows a real rule that turned a `bool`
into an `i32` and how to catch it.

## 1.7 Building graphs

Writing `UOp(Ops.ADD, (a, b))` everywhere would be unreadable, so `UOp` overloads Python's operators.
This is the sugar, and it is worth reading closely because it fixes how the encoded ops look:

@include uopc/uop.py:166-206

Three details matter.

**Constants adopt the dtype and shape of the other operand.** `x + 1` builds `UOp.const(1, x.dtype)`
via `_c`/`const_like`, so `f32 + 1` means `f32 + 1.0`. For a shaped `x`, the constant is also
broadcast to `x`'s shape with RESHAPE and EXPAND (week-4 ops, but `const_like` needs them already). The
first version did not broadcast, and `-x` on any shaped UOp failed type checking, because `x * -1.0`
multiplied a `(3,)` by a `()`.

**There is no subtraction, negation or greater-than.** Each is lowered to the canonical form on the
spot. Here are the actual graphs, printed with `to_uir` (the format of section 1.10):

```
x - y:
%0 = param : slot=0 dtype=i32
%1 = param : slot=1 dtype=i32
%2 = mul %1, -1
%3 = add %0, %2

x > 3:
%0 = param : slot=0 dtype=i32
%1 = cmplt 3, %0

x.eq(y):
%0 = param : slot=0 dtype=i32
%1 = param : slot=1 dtype=i32
%2 = cmpne %0, %1
%3 = cmpne %2, true

v * 2.0:
%0 = param : slot=0 dtype=f32 shape=(3,)
%1 = reshape 2.0, (1,)
%2 = expand %1, (3,)
%3 = mul %0, %2
```

**`<` builds a node; it does not compare.** `a < b` returns a `CMPLT` UOp. This is why UOps cannot
use Python's default ordering, why `sorted(uops)` is meaningless, and why `__eq__` had to stay
identity: if `==` built a CMPNE node, `u in some_list` would always be true (a UOp is truthy). The
methods `ne` and `eq` are named methods precisely so that `==` keeps its Python meaning.

## 1.8 STACK and INDEX: arrays without memory

Week 1 has no memory, no buffers, no loops. How do you write a program over arrays? With two ops that
build and take apart *values* that happen to have shape.

`STACK(x0, ..., xn-1)` makes a new leading axis: it stacks n same-shaped values into one value of
shape `(n, *x0.shape)`. It is `np.stack(xs, axis=0)`. A 2×3 matrix is a STACK of two STACKs of three
scalars:

```python
>>> m = stack(stack(1, 2, 3), stack(4, 5, 6))
>>> m.shape, m.dtype
((2, 3), dtypes.i32)
```

`INDEX(x, i0, ..., ik-1)` is the inverse: it pops axes off the *left*. With one index it selects a row;
with as many indices as axes it selects a scalar.

```python
>>> m.index(UOp.const(1)).shape, m.index(UOp.const(1), UOp.const(2)).shape
((3,), ())
```

The indices are themselves UOps (i32 scalars), so they need not be constants. Indexing by a PARAM is a
perfectly good graph whose value is unknown until the PARAM is bound:

```
%0 = stack 1, 2, 3
%1 = stack 4, 5, 6
%2 = stack %0, %1
%3 = param : slot=0 dtype=i32
%4 = index %2, %3, 2
```

Why pop axes from the left and not allow arbitrary axes? Because that is what STACK builds: the
leftmost axis is the one STACK added last, so `INDEX(STACK(a, b), i)` is `a` or `b`, a rule
[Chapter 2](02-rewriting.md) uses to evaluate programs. Arbitrary axis selection is a composition of
a PERMUTE (week 4) and a left INDEX.

Elementwise ops work on shaped values exactly as they do on scalars: `ADD` of two `(3,)` values is a
`(3,)` value. Shapes must match exactly; there is no implicit broadcasting in the IR (the text
format and the Tensor frontend add it as sugar, by inserting RESHAPE and EXPAND).

In week 1, the syllabus restricts STACK's inputs to CONSTs and other STACKs (literals, in effect).
Week 4 lifts that restriction, and STACK becomes a general "concatenate along a new axis" op that
rangeify has to lower ([Chapter 6](06-rangeify.md)).

## 1.9 PARAM and CALL: functions as graphs

A *function* in this IR is just a graph that mentions PARAMs, wrapped in a CALL that supplies their
values. `ParamArg` and `CallArg` are frozen dataclasses:

@include uopc/uop.py#ParamArg

@include uopc/uop.py#CallArg

`CALL(body, a0, a1, ...)` means "the value of `body` with PARAM slot `i` replaced by `ai`". The
square function applied to 7:

```
%0 = param : slot=0 dtype=i32
%1 = mul %0, %0
%2 = call %1, 7 : name=sq
```

PARAMs are identified by *position* (`slot`), not by name, much like de Bruijn indices in the lambda
calculus. That has a consequence worth stating now: since PARAMs are hash consed, **`PARAM slot=0
dtype=i32` is the same object in every function body that uses it**. A nested CALL's body and the
enclosing body share the node. This is fine as long as every pass that cares about PARAMs knows which
body it is in, and the helper that finds the PARAMs of a body is careful not to look into nested
bodies:

@include uopc/uop.py#call_params

`stack.extend(u.src[1:] if u.op is Ops.CALL else u.src)`: when it reaches a nested CALL, it visits its
*arguments* (which belong to the outer scope) but not its *body* (`src[0]`, the inner scope). The
type checker uses `call_params` to check a CALL: the body must use exactly slots `0..n-1`, and each
argument must match its PARAM's dtype and shape. [Chapter 2](02-rewriting.md) tells the story of the bug
that made this function necessary.

Why have CALL at all in week 1, when there is nothing yet to call? Partly because the syllabus
asks for it (weeks 7–8 use calls as the unit of GPU kernels), and partly because it is the smallest
example of *scope*, a concept every later chapter needs: loops in week 3 are scopes, and kernels in
week 5 are bodies with parameters.

## 1.10 Walking the graph

Nearly every pass starts with "visit every node, sources before users". That order is a
*topological sort*, and here is the one every pass uses:

@include uopc/uop.py#UOp.toposort

It is a depth-first search with an explicit stack instead of recursion. Each node is pushed twice:
first as `(u, False)` meaning "expand me", then, after its sources, as `(u, True)` meaning "my sources
are done, emit me". The `seen` set ensures each node is expanded once, however many users it has. For
our running example the order is:

```
0 PARAM
1 CONST 1.0
2 ADD
3 MUL
```

(for `e = (x + 1.0) * (x + 1.0)`). Sources come out in left-to-right order, so the result is
deterministic, which matters: generated C should not change between runs of the same program, or the
compile cache ([Chapter 4](04-memory-and-c.md)) would miss.

**Why not recursion?** The obvious recursive DFS is four lines. It also crashes on a chain of 20,000
ADDs, which Python's default recursion limit of 1,000 frames cannot hold. Deep chains are not
exotic: unrolled reductions, long training graphs and the test `test_deep_graph_no_recursion_error`
all produce them. Raising the recursion limit just moves the crash to the C stack. Every traversal in
`uopc` is iterative.

### `substitute`: replace nodes everywhere

With a topological order, "replace these nodes, rebuild everything above them" is a single loop:

@include uopc/uop.py#UOp.substitute

Each node is rebuilt from the already-rebuilt versions of its sources, and the `src == u.src` check
means unaffected subgraphs are reused, not copied. Because rebuilt nodes go through `UOp.__new__`,
the result is automatically hash consed: if a substitution makes two subgraphs identical, they become
the same node.

```python
>>> to_uir(e.substitute({x: UOp.const(3.0)}))
%0 = add 3.0, 1.0
%1 = mul %0, %0
```

There is a sibling, `UOp.wrap`, that rebuilds bottom-up and wraps selected nodes with a function. It
exists because of a real bug in kernel fusion and is explained in [Chapter 7](07-kernelize.md).

## 1.11 The `.uir` text format

Graphs need a text form: to write test programs by hand, to print what a pass did, to exchange
programs between students' compilers and the course's oracle. The syllabus specifies one, and its
example is the first program of the course:

```
# gemm_256x256.uir
%0 = buffer 65536 : dtype=f32 slot=0
%1 = buffer 65536 : dtype=f32 slot=1
%2 = reshape %0, (256, 256, 1)
%3 = reshape %1, (256, 1, 256)
%4 = mul %2, %3
%5 = reduce %4 : op=add pop=1
```

(Despite its file name, this program does not compute a matrix product. Finding out what it *does*
compute is a worked example in [Chapter 6](06-rangeify.md).)

Each line defines one node, in SSA style: a name, an op, the sources, and after a colon the attributes
that make up the arg. The grammar, from the module docstring:

@include uopc/uir.py:11-23

### The parser

`parse` reads line by line, strips comments, and keeps an environment from names to UOps:

@include uopc/uir.py#parse

Every error is re-raised with the line number and the offending line, so a typo in a 50-line program
is found in seconds. The work of one line happens in `_parse_line`:

@include uopc/uir.py#_parse_line

Split `%name = op operands : attrs` into its pieces; parse `key=value` attributes with a regular
expression; for ops whose arg is positional (the size of a buffer, the value of a constant, the shape
of a reshape) peel that literal off the operand list; look up every `%name` operand in the
environment. Then `_build` turns op, sources, literal and attributes into a UOp:

@include uopc/uir.py#_build

The `match` dispatches on the few ops whose arg has structure; everything else is `UOp(op,
tuple(src))`, with the type checker doing the validation. The parser never checks types itself. It
does not need to: if `add` gets an i32 and an f32, `UOp.__new__` raises the same error it raises
everywhere else.

### Sugar: broadcasting and literal typing

Two conveniences make hand-written programs bearable. First, a bare literal operand adopts the dtype
of its neighbor: in `mul %f, 2`, the `2` becomes `2.0`. Second, binary ops broadcast numpy-style:
missing leading axes are added with RESHAPE and size-1 axes are EXPANDed.

@include uopc/uir.py#_broadcast

Both are expansions into ordinary UOps; the IR itself still has neither. Here is a program that uses
both, and what it means:

```python
g = parse("""
%row = stack 1.0, 2.0, 3.0
%m   = stack %row, %row      # shape (2, 3)
%b   = stack 10.0, 20.0, 30.0
%s   = add %m, %b            # (2,3) + (3,): broadcast
%t   = mul %s, 2             # bare 2 adopts f32
""")
```

```
(2, 3) dtypes.f32
%0 = stack 1.0, 2.0, 3.0
%1 = stack %0, %0
%2 = stack 10.0, 20.0, 30.0
%3 = reshape %2, (1, 3)
%4 = expand %3, (2, 3)
%5 = add %1, %4
%6 = reshape 2.0, (1, 1)
%7 = expand %6, (2, 3)
%8 = mul %5, %7
```

Names in the input are arbitrary (`%row`, `%m`); the printer renumbers them.

### The printer and round-tripping

@include uopc/uir.py#to_uir

The printer walks `toposort()` and gives each node the next number. Constants are printed *inline*
as literals at their use sites rather than as their own lines, which keeps programs readable
(`add %3, 1.0` rather than a separate `%4 = const 1.0`), unless the whole program is a single
constant.

The property that ties parser and printer together is the **round trip**: printing a graph and parsing
the result gives back the same graph. With hash consing, "the same graph" means the same object, so
the test is one line:

```python
>>> parse(to_uir(g)) is g
True
```

That `is` checks every node, every arg and every edge of the graph at once. The week-1 test
`test_uir_roundtrip_and_syllabus_example` asserts it for the syllabus program. Inline constants make
the round trip slightly delicate: a printed literal must parse back to *exactly* the same CONST,
including dtype (`2` vs `2.0`), sign of zero (`-0.0`), infinities (`inf`) and bools (`true`). Exercise
6 asks you to find a case where it fails.

Parse errors look like this:

```
ParseError: line 1: '%0 = add %1, 2': undefined %1
ParseError: line 3: '%2 = add %0, %1': can't broadcast (2,) with (3,)
ParseError: line 1: 'x = const 1': statement must start with %name =
```

## 1.12 What can this language compute?

The syllabus opens week 1 with a lecture on Turing completeness, and it is worth asking what we have
built. A week-1 program is a finite DAG of CONSTs, STACKs, INDEXes, ALU ops, PARAMs and CALLs. Then:

- **Every program terminates.** Evaluating a DAG visits each node once. There is no loop, and no
  recursion: a CALL's body is a graph that already exists when the CALL is built, so a function cannot
  mention itself (that would be a cycle, and hash-consed graphs cannot have cycles).
- **The amount of work is bounded by the size of the program.** A program with N nodes does at most
  N operations (plus the work hidden in STACK sizes, which are also written in the program).
- **So it is not Turing complete.** It cannot compute any function whose running time grows faster
  than its program size: no "loop until convergence", no general recursion. In circuit-complexity
  terms, a week-1 program is an arithmetic circuit.

Week 3 adds `RANGE`, a loop with a trip count fixed when the loop starts. Bounded loops
make the language much more expressive (a 3-line program can now do a billion multiply-adds), but
every program still terminates. This is exactly the classic *LOOP language* of Meyer and Ritchie
(1967), whose programs compute precisely the primitive recursive functions: everything you would
ordinarily call "computable" except what needs unbounded search. The language never gets a `while`.

This is a feature, not a limitation. The things we want to compile (a forward pass, a backward pass,
an optimizer step) are fixed-size computations; their trip counts are tensor shapes, known before the
computation runs. A language in which every program terminates and every loop bound is known is a
language a compiler can analyze completely: it can count FLOPs, compute memory traffic, prove index
bounds ([Chapter 3](03-symbolic.md)) and pick tile sizes. The Turing-complete part (the training loop,
data loading, early stopping) stays in Python, the host language, which calls the compiler once per
fixed-shape step. Every ML compiler you will meet makes this same split, with more or less dynamism
at the boundary ([Chapter 15](15-beyond.md) discusses symbolic shapes, the main exception).

## 1.13 Putting it together: implementing week 1's IR

If you are building your own compiler, here is the order that works, with the test from
`tests/test_w01_uops.py` that checks each step:

1. `DType`, `dtypes`, `AddrSpace`. Then `Ops`, and a `UOp` class with `__new__` that hash conses with
   an `_argkey`ed key. Test: `test_hash_consing`.
2. `_infer` for CONST, STACK, INDEX, ALU, PARAM, CALL; operator sugar. Test:
   `test_shape_dtype_inference`, `test_call_type_checks`.
3. Iterative `toposort` and `substitute`.
4. The `.uir` parser and printer. Tests: `test_uir_roundtrip_and_syllabus_example` (this one needs
   BUFFER, RESHAPE, MUL and REDUCE to *construct*, though nothing evaluates them yet).

Here are the first two tests, which together pin down most of this chapter:

@include tests/test_w01_uops.py#test_hash_consing

@include tests/test_w01_uops.py#test_shape_dtype_inference

The remaining week-1 tests (evaluation, rewriting, symbolic) need [Chapter 2](02-rewriting.md) and
[Chapter 3](03-symbolic.md).

## Pitfalls

**1. Hash-consing collisions from Python equality.** (Section 1.5.) A cache keyed on raw args merged
`CONST 1`, `CONST 1.0` and `CONST True`, and `0.0` with `-0.0`. The symptom was baffling dtype
mismatch errors (`dtypes.i32 vs dtypes.f32`) on lines that built only float constants, and they
depended on *which constant had been built first in the process*, so tests passed alone and failed in
a suite. Fix: `_argkey` puts the Python type in the key and uses `repr` for floats. Lesson: any time
you key a cache on user values, ask what `==` means for them.

**2. Recursion depth.** A chain of 20,000 ADDs blew Python's recursion limit in a recursive traversal.
The symptom is a `RecursionError` deep inside a pass, only on big graphs, i.e. only in week 9.
Fix: every traversal (`toposort`, `call_params`, `graph_rewrite`, `inline_call`) uses an explicit
stack. The test `test_deep_graph_no_recursion_error` keeps it that way.

**3. `const_like` must broadcast.** The first `const_like` returned a scalar. `x * 2` worked for
scalar `x`, and every shaped program failed at `-x` with `Ops.MUL: shape mismatch (3,) vs ()`.
Constants that stand in for "the same value everywhere" must take the shape of the thing they combine
with.

**4. Printing a DAG as a tree.** `repr(u)` prints the graph as a *tree*: shared subgraphs are printed
once per use. For `x` squared ten times (`t = t * t`, 11 nodes), the repr is 105,449 characters, and
for twenty squarings it would be about 100 million. If your debug print hangs, this is why. Use
`to_uir`, which prints each node once:

```
%0 = param : slot=0 dtype=f32
%1 = mul %0, %0
%2 = mul %1, %1
...
%10 = mul %9, %9
```

The same exponential blow-up applies to *any* recursive function on UOps without a memo table, which
is the subject of [Chapter 2](02-rewriting.md)'s first experiment.

**5. Mutating a UOp.** `UOp` has a `__dict__` (passes cache things like bounds and buffers on nodes),
so Python will let you write `u.src = (...)` or `u.arg = 3`. Doing so corrupts the cache: the node is
still filed under its old key, and every other node that was hash consed to it changes meaning. Never
assign to `op`, `src`, `arg` or the inferred properties after construction; build a new node with
`replace`.

**6. `==` on UOps.** Because `<` builds a node, it is tempting to expect `==` to as well, or to think
`u == v` compares structure in some special way. It is plain identity, which *is* structural equality
here. But `u < v` in an `if` is always true (a UOp object is truthy), so `if a < b:` on UOps is a bug
that never raises.

## Exercises

1. ★ **Predict the output.** For `x = PARAM(slot=0, f32)`, how many nodes does
   `(x * x) + (x * x) * (x * x)` contain? What does `to_uir` print? Check your answer.

2. ★ **Implement** `UOp.__sub__` without looking at the source, so that `x - 2` on an i32 PARAM prints
   as `mul`/`add` lines. Why does `-1` need to be `-1` for i32 and `-1.0` for f32?

3. ★ **Find the bug.** A student's cache uses `key = (op, src, arg)` and adds `and type(arg)` to the
   key as `(op, src, arg, type(arg))`. Which of the three collisions from section 1.5 does this fix,
   and which does it not?

4. ★★ **A recursive toposort.** Write the four-line recursive version of `toposort`, check that it
   produces the same order as the iterative one on random graphs, and find the smallest chain length
   that crashes it on your machine. Then explain why `sys.setrecursionlimit(10**6)` is not a fix.

5. ★★ **Extend the type checker.** Add a `SUB` op properly: a new enum member, an `_infer` rule, and
   make `x - y` build it. List every other file in `uopc/` you would have to change for SUB to be a
   first-class op (renderer, symbolic rules, autodiff, ...). This is the cost section 1.2 was arguing
   about.

6. ★★ **Find the bug: a round trip that failed.** An earlier version of `_build` typed bare literals
   like the *first* UOp operand. Build `c = PARAM(0, i32) < PARAM(1, i32)` and `w = c.where(1, 0)` (an
   i32 WHERE with constant branches). With that version, what dtype would the literal `1` in
   `where %c, 1, 0` get, why does `parse(to_uir(w)) is w` fail, and why does `where %c, %x, 0` raise a
   UOpError? Read the current `_build` to see the one-line fix, then write a round-trip property test
   (random WHEREs with literal and non-literal branches) that would have caught the original.

7. ★★ **Property-based round-trip test.** Write a generator of random well-typed week-1 graphs (random
   CONSTs of each dtype including `-0.0`, `inf`, `nan`; STACKs; ALU ops; INDEX with constant indices),
   and assert `parse(to_uir(g)) is g` for 10,000 of them. Count how many distinct bugs it finds in
   your implementation.

8. ★★ **Memory.** Build a 1,000,000-node chain, record `len(UOp._cache)`, drop the reference, call
   `gc.collect()`, and check the cache again. Now replace the `WeakValueDictionary` with a plain
   `dict` in your copy and repeat. Why might a long training run with a plain dict look fine for an
   hour and then die?

9. ★★★ **Hash consing in C.** Implement the same node type in C: a struct with an op, a source array
   and an arg union, a global hash table keyed on (op, source pointers, arg bits), and a `mk_node`
   function. How do you handle `-0.0` and NaN in the arg? How would you free nodes (reference counting
   is the usual answer; why is it safe here, when it is not safe for general graphs)?

10. ★★★ **What can't it compute?** Prove that every week-1 program without CALL evaluates in at most
    `N` ALU operations, where `N` is its number of nodes (count STACK elements carefully). Then write a
    week-1 program of size O(n) that computes `2^(2^n)` mod 2³² (hint: squaring). What does this say
    about "bounded by program size" when constants wrap around?

<details><summary>Solutions to ★ exercises</summary>

1. `x * x` is one node, used three times; `(x*x) * (x*x)` is a MUL with both sources the same node;
   the root is an ADD. With the PARAM that is 4 nodes:

   ```
   %0 = param : slot=0 dtype=f32
   %1 = mul %0, %0
   %2 = mul %1, %1
   %3 = add %1, %2
   ```

2. `def __sub__(self, x): return self + -self._c(x)` with `__neg__` multiplying by `-1` or `-1.0`.
   The constant must have the same dtype as `self`, or the MUL fails the type check
   (`Ops.MUL: dtype mismatch`); `const_like` would also convert it for you.

3. Adding `type(arg)` distinguishes `1`, `1.0` and `True` (different types), but `0.0` and `-0.0`
   are both `float`, compare equal and hash equal, so they still collide. NaN constants still never
   hit the cache. And a tuple arg like `(1, 2)` vs `(1.0, 2)` (both `tuple`) still collides; that is
   why `_argkey` recurses into tuples.

</details>

## Quiz

1. Name two things hash consing gives a compiler for free, and one thing it forces on it.
2. Why is `UOp(Ops.CONST, (), 1) is UOp(Ops.CONST, (), 1.0)` false, and what would go wrong if it were
   true?
3. What is the shape of `INDEX(STACK(STACK(1,2,3), STACK(4,5,6)), PARAM(slot=0, i32))`?
4. Why can a UOp graph never contain a cycle? Which line of code guarantees it?
5. Write `x >= y` for i32 UOps using only week-1 ops.
6. Why is `toposort` iterative?
7. In a CALL nested inside another CALL's body, which PARAMs does `call_params` of the outer body
   return, and why does it skip `src[0]` of the inner CALL but not `src[1:]`?
8. Is the week-1 language Turing complete? Is the week-3 language (with bounded RANGE loops)?

<details><summary>Answers</summary>

1. Free: O(1) structural equality (`is`), common subexpression elimination, O(1) memoization keyed by
   node, acyclicity. Forced: immutability (and anything that must be distinct must differ in its
   arg, like PARAM slots).
2. `_argkey` includes the Python type, so the keys differ. If they were merged, one node would have one
   dtype, and asking for an f32 constant could return an i32 node (or vice versa), producing spurious
   type errors or wrongly typed programs depending on construction order.
3. `(3,)`: one index pops the leading axis of `(2, 3)`.
4. A node's sources must exist before the node is constructed and can never be changed afterward, so a
   node cannot be its own (transitive) source. The guarantee is that `src` is fixed in `UOp.__new__`
   (`ret.op, ret.src, ret.arg = op, src, arg`) and never reassigned.
5. `CMPNE(CMPLT(x, y), True)`, i.e. "not (x < y)".
6. Real graphs have dependency chains tens of thousands of nodes deep; a recursive DFS exceeds
   Python's recursion limit (and raising it risks overflowing the C stack).
7. Only the PARAMs belonging to the outer body: those reachable without entering a nested CALL's body.
   The inner CALL's arguments (`src[1:]`) are expressions in the *outer* scope, so their PARAMs are the
   outer function's; its body (`src[0]`) has its own scope.
8. No. Every week-1 program terminates in time bounded by its size. Week 3 adds bounded loops: still
   total, still not Turing complete (it is the LOOP language, computing primitive recursive functions).

</details>

## Further reading

- Jean-Christophe Filliâtre and Sylvain Conchon, "Type-Safe Modular Hash-Consing", ML Workshop 2006.
  A careful treatment of hash consing with weak tables, the same design as `UOp._cache`.
- Andrei P. Ershov, "On programming of arithmetic operations", *Communications of the ACM* 1(8), 1958.
  The origin of hashing expressions to share common subexpressions.
- Cliff Click and Michael Paleczny, "A Simple Graph-Based Intermediate Representation", IR Workshop
  1995. The "sea of nodes" IR used in HotSpot's C2 compiler: a dataflow graph with hash-consed nodes
  ("global value numbering" by construction), and explicit control edges where needed.
- Chris Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation", CGO
  2021. The opposite design point: an extensible IR with many op kinds, regions and verifiers.
- Albert R. Meyer and Dennis M. Ritchie, "The complexity of loop programs", Proc. ACM National
  Meeting 1967. LOOP programs and the primitive recursive functions.
- tinygrad, `tinygrad/uop/ops.py` (<https://github.com/tinygrad/tinygrad>): the production ancestor of
  `UOp`, with many more ops and the same `(op, dtype, src, arg)` core.
