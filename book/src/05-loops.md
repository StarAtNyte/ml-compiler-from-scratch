# 5. Loops and the linearizer

A 1024x1024 matrix multiply is a billion multiply-adds. Written in the loopless language of
[Chapter 4](04-memory-and-c.md) it would be a billion `STORE` nodes and a C file measured in gigabytes.
Every real kernel is a loop nest, so the IR needs loops. Adding them to the IR is easy: two ops, `RANGE` and
`END`. Getting them back *out* of the IR is not. The graph says which values depend on which loop
variables, but it does not say which loop is nested inside which, which statements go inside a loop and
which go before it, or which of two sibling loops runs first. The *linearizer* has to decide all of that,
and decide it correctly for every program. It took four designs to get right. This chapter writes GEMM and
convolution by hand in the loop IR, then builds the linearizer, including the three designs that failed and
the exact way each one failed.

**What you'll build**

- `RANGE` and `END`: loop variables as ordinary i32 values, and a node that closes loops.
- Hand-written GEMM and 1-D convolution in the loop IR, with accumulators built from `ALLOC` + `AFTER`.
- `scopes`: the set of loops each node must sit inside.
- `loop_targets`: scopes closed under "encloses", which turns them into paths in a loop tree.
- `linearize`: a list scheduler with one priority heap per scope, which opens and closes loops, orders
  sibling loops by their data dependencies, and places accumulators so each thread gets its own.

**Where this fits.** [Chapter 4](04-memory-and-c.md) linearized loopless programs with a topological sort
and rendered them to C. This chapter replaces that sort with the real `linearize` from
`uopc/linearize.py`, and the renderer gains `for` loops. [Chapter 6](06-rangeify.md) will generate exactly
these loop programs automatically from shapes and movement ops, so everything here is also what rangeify's
output looks like.

## 5.1 RANGE and END

@include uopc/uop.py#RangeArg

@include uopc/uop.py#range_

`RANGE` has one source, the loop's end (an i32 value, usually a constant), and an arg `RangeArg(idx, kind)`.
Its *value* is the loop counter: a `RANGE` with end 4 is an i32 that takes the values 0, 1, 2, 3. That is
the whole trick. A loop variable is a value like any other, so `i*8 + j` is ordinary arithmetic, the
simplifier of [Chapter 3](03-symbolic.md) works on it unchanged, and bounds analysis knows `i` lies in
`[0, end-1]`.

`idx` gives each loop an identity and a nesting order: **loops nest in increasing `idx` order**. If loops
0 and 1 are ever open at the same time, loop 1 is inside loop 0. `kind` is an `AxisType`. In this chapter
every loop is a plain `LOOP`, plus one `THREAD` in section 5.7. The others are for later chapters:

@include uopc/uop.py#AxisType

`END` closes loops. Its sources are `(passthrough, *ranges)`: `x.end(i, j)` is a node that means "`x`, with
loops `i` and `j` finished". It has dtype `void`, and it is how a loop's body becomes a dependency of
whatever comes after the loop. A store inside a loop over `k` is *one* node in the graph, but it
executes `K` times. `store.end(k)` is the event "all `K` of them have happened".

Because UOps are hash consed, a `RANGE` *is* its `(end, idx, kind)`. `range_(8, 0)` called twice gives the
same loop. That is what you want when you mean the same loop, and a bug when you meant two (see Pitfalls).

Here is the first test of week 3 that uses loops, a 2-D elementwise kernel that adds the first element of
each row to every element of the row:

@include tests/test_w03_loops.py#test_loop_invariant_code_is_hoisted

```c
void add_row_head(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 8; ridx0++) {
    int alu1 = (ridx0*8);
    float val2 = data0[alu1];
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      int alu3 = (alu1+ridx1);
      float val4 = data0[alu3];
      data1[alu3] = (val2+val4);
    }
  }
}
```

The renderer turns a `RANGE` into a `for` header and an `END` into closing braces (one per range it
closes). The interesting part is *where* everything else went. `row = A[i*N]` depends on `i` only, so it was
computed once per row, outside the `j` loop. `i*8` is computed once per row too and reused by both indices.
Nobody asked for this hoisting. It falls out of the linearizer's rules, and the test checks it.

## 5.2 Scope: which loops a node must sit inside

A node has to be inside loop `r` if it uses the value of `r`, directly or through any chain of sources. It
can leave loop `r` once something has `END`ed `r`. That gives three rules, the *scope* of a node:

```text
scope(RANGE r)       = scope(end) | {r}
scope(END x, *rs)    = scope(x) - rs
scope(anything else) = union of scope(srcs)
```

@include uopc/linearize.py#scopes

One pass over the topological order computes every scope, since a node's scope depends only on its sources'.
For the row-add kernel: `i` has scope `{i}`, `i*8` has `{i}`, the load `A[i*8]` has `{i}`, `i*8 + j` has
`{i, j}`, and the final store has `{i, j}`. The `END(store, i, j)` has scope `{}`, and so does the `SINK`.

Scope is a *lower bound* on placement: a node with scope `{i}` must be inside loop `i`. Could it go
deeper, inside `j` too? It would compute the same value, just eight times as often. So the natural rule is
"as shallow as possible": place each node exactly inside the loops of its scope. That is what made `A[i*8]`
land outside the `j` loop. Section 5.5 shows that this natural rule is not quite right, and why.

## 5.3 Writing GEMM by hand

`C = A @ B` with `A` of shape `(N, K)`, `B` of shape `(K, M)`:

```text
for i in range(N):
  for j in range(M):
    acc = 0
    for k in range(K):
      acc = acc + A[i*K + k] * B[k*M + j]
    C[i*M + j] = acc
```

The accumulator is the interesting part. It is mutable state that is reset for every `(i, j)`, updated `K`
times, and read once after the `k` loop. In our IR mutable state is memory, so `acc` is a one-element
`ALLOC`, and every "after" in that description becomes an `AFTER`. Here is the week-3 test's GEMM:

@include tests/test_w03_loops.py#gemm

Line by line:

1. `acc = ALLOC(f32[1]).after(i, j)`. The accumulator, *inside loops `i` and `j`*. `AFTER` with ranges as
   dependencies is how you put memory into a scope: `scope(acc) = {i, j}`, so everything built on `acc`
   happens per `(i, j)`.
2. `init = acc[0].store(0.0)`. The reset. Its scope is `{i, j}`, so it runs once per output element, before
   anything that depends on it.
3. `acck = acc.after(init, k)`. The accumulator, after the reset, inside loop `k`. Scope `{i, j, k}`.
4. `upd = acck[0].store(acck[0].load() + A[i*K + k] * B[k*M + j])`. The update. It loads and stores through
   `acck`, so it is ordered after `init` and sits inside `k`.
5. `upd.end(k)`. The event "all `K` updates have happened". Scope `{i, j}`.
6. `acc.after(upd.end(k))[0].load()`. The final value, read after the `k` loop. Scope `{i, j}`.
7. `C[i*M + j].store(...)`, then `.end(i, j)` closes the outer loops, and `sink(...)` makes it a program.

@diagram 05-gemm-ir

The test runs it against numpy and checks that there are exactly three loops:

@include tests/test_w03_loops.py#test_gemm_loops

Here is the C for `gemm(4, 3, 5)`:

```c
void gemm(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 5; ridx2++) {
        float val1 = acc0;
        float val2 = data0[((ridx0*5)+ridx2)];
        float val3 = data1[((ridx2*3)+ridx1)];
        acc0 = (val1+(val2*val3));
      }
      float val4 = acc0;
      data2[((ridx0*3)+ridx1)] = val4;
    }
  }
}
```

Every line traces back to a node. `float acc0;` is the `ALLOC`, declared inside `j` (section 5.7 explains
why there). `acc0 = 0.0f;` is `init`. The four lines inside `ridx2` are the update: load, two input loads,
store. `float val4 = acc0;` is the load through `acc.after(upd.end(k))`, after the closing brace that `END(k)`
produced. The accumulator is a scalar, so gcc keeps it in a register for the whole `k` loop.

The convolution test has the same shape, with one output loop instead of two and a sliding input index
`o + k`:

@include tests/test_w03_loops.py#test_conv1d_loops

```c
void conv1d(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      float val1 = acc0;
      float val2 = data0[(ridx0+ridx1)];
      float val3 = data1[ridx1];
      acc0 = (val1+(val2*val3));
    }
    float val4 = acc0;
    data2[ridx0] = val4;
  }
}
```

Once you have written one reduction in this style, every reduction is the same five lines: allocate in the
output loops, initialize, accumulate in the reduce loop, end it, read back. [Chapter 6](06-rangeify.md)'s
`REDUCE` lowering generates exactly these five lines.

## 5.4 What the linearizer must do

Given a `SINK`, produce a list of UOps such that:

1. **Topological:** every node comes after its sources (this includes every `AFTER` ordering edge).
2. **Scoped:** every node is emitted while all loops of its scope are open.
3. **Nested:** loops open and close like a stack. A `RANGE` pushes, an `END` pops the innermost ones.
4. **Shallow:** a node is not put inside loops it does not need, or the work is multiplied for nothing (and
   an accumulator reset placed too deep destroys the reduction).
5. **Ordered siblings:** if two loops are both inside `i` and neither inside the other, the one whose
   results the other reads must run first.

Constraint 2 says "at least these loops". Constraint 4 says "at most these loops". If you could always
satisfy both with equality, i.e. "emit each node with exactly its scope open", linearizing would be easy.
GEMM shows you can't. Here is every node of `gemm(4, 3, 5)` in final order, with its scope and the
*target* that the final linearizer actually places it at (the table comes from a scratch script that calls
`scopes` and `loop_targets` and prints sources as `%n`):

```text
%0  BUFFER                scope={}       target={}
%1  CONST                 scope={}       target={}
%2  CONST                 scope={}       target={}
%3  CONST                 scope={}       target={}
%4  CONST                 scope={}       target={}
%5  CONST                 scope={}       target={}
%6  BUFFER                scope={}       target={}
%7  BUFFER                scope={}       target={}
%8  RANGE  %1             scope={i}      target={}
%9  MUL    %8 %2          scope={i}      target={i}
%10 MUL    %8 %5          scope={i}      target={i}
%11 RANGE  %2             scope={j}      target={i}
%12 ADD    %9 %11         scope={i,j}    target={i,j}
%13 INDEX  %0 %12         scope={i,j}    target={i,j}
%14 ALLOC                 scope={}       target={}
%15 AFTER  %14 %8 %11     scope={i,j}    target={i,j}
%16 INDEX  %15 %3         scope={i,j}    target={i,j}
%17 STORE  %16 %4         scope={i,j}    target={i,j}
%18 RANGE  %5             scope={k}      target={i,j}
%19 AFTER  %15 %17 %18    scope={i,j,k}  target={i,j,k}
%20 INDEX  %19 %3         scope={i,j,k}  target={i,j,k}
%21 LOAD   %20            scope={i,j,k}  target={i,j,k}
%22 ADD    %10 %18        scope={i,k}    target={i,j,k}
%23 INDEX  %6 %22         scope={i,k}    target={i,j,k}
%24 LOAD   %23            scope={i,k}    target={i,j,k}
%25 MUL    %18 %2         scope={k}      target={i,j,k}
%26 ADD    %25 %11        scope={j,k}    target={i,j,k}
%27 INDEX  %7 %26         scope={j,k}    target={i,j,k}
%28 LOAD   %27            scope={j,k}    target={i,j,k}
%29 MUL    %24 %28        scope={i,j,k}  target={i,j,k}
%30 ADD    %21 %29        scope={i,j,k}  target={i,j,k}
%31 STORE  %20 %30        scope={i,j,k}  target={i,j,k}
%32 END    %31 %18        scope={i,j}    target={i,j}
%33 AFTER  %15 %32        scope={i,j}    target={i,j}
%34 INDEX  %33 %3         scope={i,j}    target={i,j}
%35 LOAD   %34            scope={i,j}    target={i,j}
%36 STORE  %13 %35        scope={i,j}    target={i,j}
%37 END    %36 %8 %11     scope={}       target={}
%38 SINK   %37            scope={}       target={}
```

Look at `%24`, the load `A[i*K + k]`. Its scope is `{i, k}`: it does not depend on `j`. But loop `k` only
exists inside loop `j` (the reduction runs once per output element), so there is no moment when `i` and
`k` are open and `j` is not. The load *must* be placed in `{i, j, k}`, one level deeper than its scope.
The same holds for `k*3` (`%25`, scope `{k}`) and `B[k*3 + j]` (`%28`, scope `{j, k}`). Scope is a set of
loops. What we need is a *position in the nest*, and those are not the same thing. Note also `%18`, the
`RANGE k` itself: its scope is `{k}`, but it must be opened while `i` and `j` are open. The difference
between the scope and the target columns is the whole story of this chapter.

## 5.5 Four designs, three failures

This section retells development note 8. The failed designs are worth studying: each is the first thing
a good programmer would try, and each fails on a program from this course. To make the failures concrete,
here is a deliberately simple scheduler (a scratch file, not in the repo) that emits, at each step, the
first ready node that is "legal here", preferring plain nodes, then opening loops, then closing them. Its
notion of "legal" is a parameter:

```python
from uopc.uop import Ops
from uopc.linearize import scopes

def naive_linearize(sink, fits):
  """First tries at a scheduler. Walk the toposort and emit the first ready node that is legal here."""
  topo, sc = sink.toposort(), scopes(sink)
  out, stack, done = [], [], set()
  def legal(u):
    if u.op is Ops.RANGE: return sc[u] - {u} <= set(stack)                         # its end is computable here
    if u.op is Ops.END: return list(u.src[1:]) == stack[len(stack) - len(u.src) + 1:]  # closes the innermost loops
    return fits(sc[u], set(stack))
  while len(out) < len(topo):
    ready = [u for u in topo if u not in done and all(s in done for s in u.src) and legal(u)]
    # plain nodes first, then open a loop, then close one
    u = min(ready, key=lambda u: (u.op is Ops.RANGE, u.op is Ops.END), default=None)
    if u is None:
      stuck = [(x.op.name, sorted(r.arg.idx for r in sc[x])) for x in topo if x not in done and all(s in done for s in x.src)]
      raise RuntimeError(f"deadlock: open loops {[r.arg.idx for r in stack]}, ready {stuck}")
    out.append(u); done.add(u)
    if u.op is Ops.RANGE: stack.append(u)
    if u.op is Ops.END: del stack[len(stack) - len(u.src) + 1:]
  return out

exact = lambda scope, open_loops: scope == open_loops    # attempt (a)
subset = lambda scope, open_loops: scope <= open_loops   # attempt (b)
```

### Attempt (a): emit a node when its scope equals the open loops

This is "as shallow as possible" taken literally. It fails on GEMM exactly as the table above predicts:

```text
--- attempt (a): scope == open loops, on GEMM
deadlock: open loops [0, 1, 2], ready [('ADD', [0, 2]), ('MUL', [2])]
```

With loops `i, j, k` open, the index `i*5 + k` (scope `{i, k}`) and `k*3` (scope `{k}`) are ready but never
legal: their scope never equals the stack. Nothing else can run until they do, so the scheduler is stuck.

### Attempt (b): emit a node when its scope is a subset of the open loops

The obvious fix: "inside at least these loops". GEMM now linearizes and runs. But legality alone no longer
determines placement. A node with scope `{i}` is legal at depth `{i}`, `{i, j}` and `{i, j, k}`, and *which*
it gets depends on what the scheduler happens to try first. Correctness now rests on scheduling luck.

Here is where the luck runs out. Subtract each row's sum from the row: `out[i, j] = x[i, j] - sum_k x[i, k]`.
The reduction loop `k` and the output loop `j` are *siblings* inside `i`. The program, with `j` given the
smaller `idx`:

```python
def sub_rowsum(N, M):
  X, OUT = buf(0, N * M), buf(1, N * M)
  i, j, k = range_(N, 0), range_(M, 1), range_(M, 2)
  acc = UOp(Ops.ALLOC, (), ParamArg(0, dtypes.f32, (1,))).after(i)
  init = acc.index(0).store(UOp.const(0.0))
  acck = acc.after(init, k)
  upd = acck.index(0).store(acck.index(0).load() + X.index(i * M + k).load())
  total = acc.after(upd.end(k)).index(0).load()
  return sink(OUT.index(i * M + j).store(X.index(i * M + j).load() - total).end(i, j))
```

Attempt (b) opens `j` first (it is ready, and nothing says not to), then finds that the store inside `j`
needs `total`, which needs the `k` loop, which is legal to open inside `j`. On a 2x3 matrix of ones:

```c
void sub_rowsum_b(float* restrict data0, float* restrict data1) {
  float acc0;
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    int alu1 = (ridx0*3);
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu2 = (alu1+ridx1);
      float val3 = data0[alu2];
      for (int ridx2 = 0; ridx2 < 3; ridx2++) {
        float val4 = acc0;
        float val5 = data0[(alu1+ridx2)];
        acc0 = (val4+val5);
      }
      float val6 = acc0;
      data1[alu2] = (val3+(val6*-1.0f));
    }
  }
}
```

```text
attempt (b): [[-2.0, -5.0, -8.0], [-2.0, -5.0, -8.0]]
uopc:        [[-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]]
```

Every row sum should be 3, so every output should be `1 - 3 = -2`. Attempt (b) nests the reduction inside
the output loop, so the accumulator, reset once per row, keeps accumulating across `j`: the sums come out
3, 6, 9. (With `max` instead of `sum` the bug would hide, since `max` is idempotent. That is how bugs like
this survive testing.) Even if the reset were moved inside `j`, the program would recompute the row sum
`M` times, turning O(N·M) work into O(N·M²).

### Attempt (c): order sibling loops by data dependency

The fix for (b) seemed clear: decide loop order up front. If any node that needs loop `A` depends on any
node that needs loop `B`, then `B` must be opened (and finished) before `A`. In `sub_rowsum` the output store
(needs `j`) depends on the update (needs `k`), so `k` comes first. Good.

But "depends on any node that needs `B`" is far too broad. In GEMM, the load `B[k*M + j]` depends on
`k*M`, a pure function of `k`, and the update depends on `A[i*K + k]`. By this rule, everything inside the
`i` loop that touches `k` "depends on loop `k`", and `k` was forced outermost, deadlocking or producing
nonsense nests. Data dependency on a loop *variable* says a node is *inside* the loop. Only a dependency on
the loop's `END`, its finished results, says the loop must come *before*.

### Attempt (d): scopes as paths in the loop tree

The final design separates the two questions that (a) to (c) kept mixing up.

**Nesting** comes from scopes. If two ranges `a < b` (by `idx`) ever appear together in some scope, then
`b` is nested inside `a`, and that holds *everywhere*: `b` is a single loop, it has one position in the
nest, and that position is inside `a`. So every node that uses `b` must also sit inside `a`, whether or not
it mentions `a`. Define `enc(b)`, the loops enclosing `b`, as every smaller range that `b` shares any scope
with, and *close* each scope under `enc`. The closed scope, sorted by `idx`, is the node's *target*: a path
from the root of the loop tree, and therefore exactly the stack of open loops when the node should run.

For GEMM, scope `{i, j, k}` appears (the update), so `enc(j) = {i}` and `enc(k) = {i, j}`. The load
`A[i*K + k]` has scope `{i, k}`, which closes to `{i, j, k}`, the target in the table. In `sub_rowsum`, `j`
and `k` never share a scope, so neither encloses the other: they are siblings in the tree, both children of
`i`.

@diagram 05-loop-tree

**Sibling order** comes only from `END`s. When the scheduler must open one of several sibling loops, it
looks for pending nodes that need loop `A` and that (transitively, through pending nodes) read the `END`
of a sibling candidate `B`. Those block `A`: `B` goes first. Reading `k`'s *variable* never blocks
anything. Reading `k`'s *results*, which is only possible through `END(k)`, does.

With nesting fixed by the tree and siblings ordered by `END`s, "emit a node exactly when the open loops
equal its target" is now always possible. That is attempt (a)'s rule again, applied to targets instead of
scopes. The deadlock is gone because targets are, by construction, positions that exist in the nest.

## 5.6 The final linearizer

The module docstring is a summary of everything above:

@include uopc/linearize.py:1-25

### Targets

@include uopc/linearize.py#_key

@include uopc/linearize.py#loop_targets

`enc` is built from the distinct scopes: for each scope, sorted by `idx`, each range is enclosed by all the
ranges before it in that scope. `close` is a transitive closure (if `k` is inside `j` and `j` is inside `i`,
then `k` is inside `i`) with an explicit worklist. Every node's target is its closed scope.

`RANGE` nodes are special: the target of loop `r` is where `r` is *opened*, which is its parent's path, so
it is the closure of its enclosing loops (plus its end's scope, if the end is itself loop-dependent), without
`r` itself. In the GEMM table, `RANGE k` has target `{i, j}`: open `k` when `i` and `j` are the open loops.

### The scheduler

@include uopc/linearize.py#linearize

This is *list scheduling*, the standard algorithm compilers use to order instructions in a basic block:
keep a set of *ready* nodes (all sources emitted), repeatedly pick one by priority, emit it, and update
readiness of its users. The `waiting` dict counts unemitted distinct sources per node, so readiness updates
are O(1) per edge. Two refinements make it a *loop* scheduler.

**One heap per target.** Ready ordinary nodes go into `heaps[target]`, a min-heap ordered by toposort
position. The heap for the current stack of open loops holds exactly the nodes that may run *now*. Toposort
order as the priority keeps the output close to the order in which the program was written, which keeps
the C readable and deterministic. Ready `RANGE`s and `END`s go into a separate set, since they change the
stack.

**Three steps, in priority order.** Each iteration looks at the current stack `cur`:

1. **Emit.** If anything is ready at exactly `cur`, emit the first one. Doing this before opening loops is
   what makes hoisting happen: everything that can run at the current depth runs before the scheduler goes
   deeper.
2. **Open.** Otherwise, find which loop the pending work needs next. For every pending non-`RANGE` node
   whose target extends `cur`, `need[u]` is the next loop on its path. Candidates are those that are ready
   and whose own target is `cur` (they open here). Then the sibling rule: walk the pending nodes in
   topological order, propagating `ended[u]`, the set of candidate loops whose `END` `u` transitively reads.
   A node that needs loop `A` but has read the `END` of a different candidate `B` blocks `A`. Open the
   unblocked candidate with the smallest `idx`.
3. **Close.** Otherwise, if a ready `END` closes exactly the innermost loops of the stack (and its own
   target is what remains), emit it and pop.

If none applies, the program cannot be scheduled, and the error lists the open loops and the first few
stuck nodes with their targets.

@diagram 05-schedule

### Watching it run

The renderer sees only the final list, but the list records every decision. A scratch script replays
`linearize(gemm(4, 3, 5))`, classifying each node by the step that must have produced it (a `RANGE` is an
*open*, an `END` a *close*, anything else an *emit*) and tracking the stack. Constants and buffers, all
emitted at the start with an empty stack, are left out:

```text
2 open   %8  RANGE  %1            open after: (i)
1 emit   %9  MUL    %8 %2         open after: (i)
1 emit   %10 MUL    %8 %5         open after: (i)
2 open   %11 RANGE  %2            open after: (i, j)
1 emit   %12 ADD    %9 %11        open after: (i, j)
1 emit   %13 INDEX  %0 %12        open after: (i, j)
1 emit   %14 ALLOC                open after: (i, j)
1 emit   %15 AFTER  %14 %8 %11    open after: (i, j)
1 emit   %16 INDEX  %15 %3        open after: (i, j)
1 emit   %17 STORE  %16 %4        open after: (i, j)
2 open   %18 RANGE  %5            open after: (i, j, k)
1 emit   %19 AFTER  %15 %17 %18   open after: (i, j, k)
1 emit   %20 INDEX  %19 %3        open after: (i, j, k)
1 emit   %21 LOAD   %20           open after: (i, j, k)
1 emit   %22 ADD    %10 %18       open after: (i, j, k)
1 emit   %23 INDEX  %6 %22        open after: (i, j, k)
1 emit   %24 LOAD   %23           open after: (i, j, k)
1 emit   %25 MUL    %18 %2        open after: (i, j, k)
1 emit   %26 ADD    %25 %11       open after: (i, j, k)
1 emit   %27 INDEX  %7 %26        open after: (i, j, k)
1 emit   %28 LOAD   %27           open after: (i, j, k)
1 emit   %29 MUL    %24 %28       open after: (i, j, k)
1 emit   %30 ADD    %21 %29       open after: (i, j, k)
1 emit   %31 STORE  %20 %30       open after: (i, j, k)
3 close  %32 END    %31 %18       open after: (i, j)
1 emit   %33 AFTER  %15 %32       open after: (i, j)
1 emit   %34 INDEX  %33 %3        open after: (i, j)
1 emit   %35 LOAD   %34           open after: (i, j)
1 emit   %36 STORE  %13 %35       open after: (i, j)
3 close  %37 END    %36 %8 %11    open after: ()
1 emit   %38 SINK   %37           open after: ()
```

Read it as a conversation between the three steps. With only `i` open, step 1 finds `i*3` and `i*5`
(`%9`, `%10`) and emits them; those are the hoisted row offsets. Nothing else has target `(i)`, so step 2
opens `j`, the only loop anything needs next. Inside `(i, j)`, step 1 drains everything with that target:
the output address, the `ALLOC` (placed here by the common-prefix rule), the accumulator view and the reset
store. Only when that heap is empty does step 2 open `k`. That ordering is what puts `acc0 = 0.0f;` before
the `k` loop and not inside it. Inside `(i, j, k)` the body runs in toposort order. The update store makes
`END(k)` ready, step 1 has nothing left at `(i, j, k)`, and step 3 closes `k`. Back at `(i, j)`, the
read-back and the store to `C` become ready and are emitted, then `END(i, j)` closes two loops at once, and
the `SINK` is last.

Here is the sibling rule on the row-max version of the program above, `out[i, j] = x[i, j] - max_k x[i, k]`.
Again `j` (idx 1) has a smaller `idx` than `k` (idx 2), so "smallest `idx` first" alone would open `j`:

```python
def sub_rowmax(N, M):
  X, OUT = buf(0, N * M), buf(1, N * M)
  i, j, k = range_(N, 0), range_(M, 1), range_(M, 2)     # j (output) has a *smaller* idx than k (reduce)
  acc = UOp(Ops.ALLOC, (), ParamArg(0, dtypes.f32, (1,))).after(i)
  init = acc.index(0).store(UOp.const(-math.inf))
  acck = acc.after(init, k)
  upd = acck.index(0).store(acck.index(0).load().maximum(X.index(i * M + k).load()))
  m = acc.after(upd.end(k)).index(0).load()            # the row max, readable only after the k loop
  return sink(OUT.index(i * M + j).store(X.index(i * M + j).load() - m).end(i, j))
```

```c
void sub_rowmax(float* restrict data0, float* restrict data1) {
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

With `i` open, both `j` and `k` are candidates. The final store needs `j` and reads `m`, which reads
`END(..., k)`, so `ended[store] = {k}` and `j` is blocked. The scheduler opens `k`, runs the reduction,
closes it, emits the read-back `val4 = acc0` (target `{i}`, step 1), and only then opens `j`. The test
against numpy passes with error `0.0`. Rangeify produces this structure for every softmax and layer norm,
and [Chapter 11](11-fusion-flash.md) depends on it.

@diagram 05-sibling-loops

### Cost

Each node enters and leaves a heap once: O(n log n). Step 2 scans the pending nodes, O(n), but it runs only
when a loop is about to be opened, which happens once per `RANGE`, and kernels have few loops. For the kernels in this course linearization takes milliseconds. It is not where compile time
goes ([Chapter 14](14-gpt.md) profiles that).

## 5.7 Where accumulators live, and threads

An `ALLOC` has no sources, so its scope is `{}`, and by the rules so far it would be declared at the top of
the function. For a single-threaded program that is fine: the reset `acc0 = 0.0f;` sits in the right place
regardless of where the variable was declared. With threads it is a bug. The linearizer therefore overrides
the `ALLOC`'s target: it is declared in the *deepest scope common to all its users*, the longest common
prefix of their targets (development note 9). In GEMM the only user of the `ALLOC` is `acc = ALLOC.after(i, j)`, with
target `{i, j}`, so `float acc0;` appears inside the `j` loop.

Why it matters: in [Chapter 10](10-cpu-performance.md) the outer loop becomes a `THREAD` axis, rendered as
an OpenMP parallel loop. Mark `i` as `AxisType.THREAD` in the GEMM and the C becomes:

```c
void gemm_omp(float* restrict data0, float* restrict data1, float* restrict data2) {
  #pragma omp parallel for
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    int alu1 = (ridx0*64);
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      float acc0;
      acc0 = 0.0f;
      for (int ridx2 = 0; ridx2 < 64; ridx2++) {
        float val2 = acc0;
        float val3 = data0[(alu1+ridx2)];
        float val4 = data1[((ridx2*64)+ridx1)];
        acc0 = (val2+(val3*val4));
      }
      float val5 = acc0;
      data2[(alu1+ridx1)] = val5;
    }
  }
}
```

In OpenMP, variables declared inside the parallel loop's body are private to each thread, and variables
declared outside it are shared. Declared inside the `j` loop, every thread has its own `acc0`. Declared at
function scope, all 16 threads would update one variable at once: a data race. A scratch experiment moves
the declaration to function scope by editing the generated C, and runs both versions on a 256x256x256 GEMM:

```text
-O3 -march=native -fopenmp   inside j loop   max error 4.96e-05
-O3 -march=native -fopenmp   function scope  max error 4.96e-05
-O0 -fopenmp                 inside j loop   max error 4.96e-05
-O0 -fopenmp                 function scope  max error 85.8
```

At `-O3` the race is invisible, because gcc keeps the accumulator in a register for the whole `k` loop and
the shared variable is barely touched. At `-O0` every access goes to memory and the threads trample each
other. A data race is undefined behavior whether or not today's optimizer hides it, so the declaration has
to be in the right place by construction, not by luck.

(GPU shared memory is the opposite case: an `ALLOC` in `SHARED` space must be visible to the whole thread
block, so the linearizer pins it to the empty scope, the top of the kernel. See
[Chapter 13](13-shared-memory-tensor-cores.md) and development note 24.)

## 5.8 Loop-invariant code motion, for free

Classic compilers have a pass called *loop-invariant code motion* (LICM): find computations inside a loop
whose operands do not change across iterations and move them before the loop. Our linearizer never runs
such a pass, yet it hoists. Placement by target *is* LICM: a node is emitted at the shallowest depth its
target allows, and step 1 runs before step 2, so everything that can run before a loop opens does.

Predict the C for this kernel before reading it:

```python
X, O = buf(0, 4 * 8), buf(1, 4 * 8)
i, j = range_(4, 0), range_(8, 1)
s = sink(O.index(i * 8 + j).store(X.index(i * 8 + j).load() * X.index(j).load() + X.index(i).load()).end(i, j))
```

```c
void predict(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val1 = data0[ridx0];
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      int alu2 = ((ridx0*8)+ridx1);
      float val3 = data0[alu2];
      float val4 = data0[ridx1];
      data1[alu2] = ((val3*val4)+val1);
    }
  }
}
```

`X[i]` (scope `{i}`) is hoisted out of `j`. `X[j]` (scope `{j}`) is not hoisted out of `i`, even though it
does not depend on `i`: `j` is nested inside `i`, so the closure puts it at `{i, j}`. Getting it out of the
`i` loop would mean computing all of `X[0..7]` first into a temporary, which is a *different program*
(more memory, one more loop), not a placement decision. That is the job of the optimizer in
[Chapter 10](10-cpu-performance.md), not the linearizer. The same limit explains why `A[i*K + k]` in GEMM is
reloaded for every `j`.

Two things make this cheap trick robust. Hash consing means a value used in several places is one node,
placed once. And the simplifier of [Chapter 3](03-symbolic.md) removes spurious dependencies before the
linearizer sees them: an index like `((i*8 + j)//8)` that *mentions* `j` but equals `i` would otherwise
pin a node, or an accumulator, one loop too deep (development note 12, [Chapter 6](06-rangeify.md)).

## 5.9 Rendering loops

The renderer from [Chapter 4](04-memory-and-c.md) needs only three additions for loops, all visible in
`CRenderer.render`: a `RANGE` prints a loop header and increases the indentation depth, an `END` prints one
closing brace per range it closes, and every name in the loop body is declared inside the braces, which C
scopes correctly for free. The header itself is a hook:

@include uopc/render_c.py#CRenderer.render_range

@include uopc/render_c.py#CRenderer.is_loop

A `THREAD` range gets an OpenMP pragma. Everything else is a plain counted loop with the range's end
rendered as an expression, so a symbolic end ([Chapter 15](15-beyond.md)) needs no change here. `is_loop`
exists for the GPU renderer of [Chapter 12](12-gpus.md): there, a `GLOBAL` or `LOCAL` range is not a loop at
all but a hardware index (`blockIdx.x`, `threadIdx.x`), so it opens no braces and its `END` closes none. On
the CPU every range is a loop, including GPU-kind ranges, which is how the C backend doubles as an emulator
for GPU kernels.

Because the linearizer guarantees proper nesting, the renderer never has to think about structure. It
prints the list in order and keeps a depth counter. Every structural decision was made in one place.

## 5.10 Why a flat graph, and not structured loops?

There is an obvious alternative design. MLIR's `scf.for`, LLVM's loop nests, Halide's loop trees and TVM's
TIR all represent a loop as a *region*: an op that owns a nested body, the way a `for` statement owns its
block. With regions, nesting is explicit and there is nothing to linearize. Why does this course (and
tinygrad) use a flat graph with `RANGE` and `END` instead?

**Rewrites see through loops.** In a flat graph `i*8 + j` is an ordinary expression whose leaves happen to
be `RANGE`s. The simplifier, CSE through hash consing, and every pattern rule work across loop boundaries
with no special cases. With regions, an expression inside a loop body and the same expression outside it
are different objects in different blocks, and "hoist this out of the loop" is a transformation someone
has to write.

**Placement is derived, not stored.** In a region-based IR every rewrite that changes what an expression
depends on must also decide where the result lives. In the flat graph, rangeify ([Chapter 6](06-rangeify.md))
and the optimizer ([Chapter 10](10-cpu-performance.md)) can rewrite indices, split ranges and unroll loops
without ever thinking about placement. The linearizer recomputes the best placement at the end.

**The cost** is this chapter. Structure has to be recovered, which needs scopes, the closure under
"encloses", the sibling rule and the `ALLOC` placement rule, plus error reporting for programs that have no
valid nesting. Region-based IRs pay instead with verifier rules ("a value defined inside a region cannot be
used outside it") and with passes that move code between regions. Neither choice is free. The flat graph
fits a compiler whose main activity is rewriting index expressions, which is what an ML compiler mostly does.

## 5.11 Implementing week 3

A build order that keeps every step testable:

1. **RANGE and END in `_infer`.** `RANGE` is i32 with an i32 end, `END` is void. Add `range_` and `.end()`.
   Bounds of a `RANGE` go into `_bounds` ([Chapter 3](03-symbolic.md)).
2. **`scopes`.** Write it and print the scope column for GEMM. It must match the table in section 5.4.
3. **Targets.** Write `enc` and `close`. Print the target column. Every target must be a prefix-closed path
   (sorted by `idx`), and `A[i*K + k]` must be at `(i, j, k)`.
4. **A scheduler without siblings.** Steps 1, 2 (smallest `idx`, no blocking) and 3. The row-add and GEMM
   kernels and both week-2 tests (which have no loops) should pass. Check the hoisting test.
5. **`ALLOC` placement.** The deepest common prefix of the users' targets. GEMM's `float acc0;` must be
   inside `j`.
6. **The sibling rule.** Write `sub_rowsum` with `j` given the smaller `idx` and watch it fail, then add the
   `ended` propagation until it passes. Test with `sum`, not `max`.
7. **Errors.** Make every way of getting stuck raise `LinearizeError` with the open loops and the stuck nodes.
8. **The renderer.** Loop headers, braces and indentation. Compile and run against numpy, always.

At each step, print the linearized list next to the generated C. Nearly every linearizer bug is visible in
one of those two printouts before it is visible in a wrong number.

## Pitfalls

**Exact scope matching deadlocks (note 8a).** A node's scope is a set; its position is a path. `A[i*K + k]`
has scope `{i, k}`, and no moment in GEMM has exactly `{i, k}` open. Close scopes under "encloses" first.

**Subset matching relies on luck (note 8b).** "Scope is a subset of the open loops" makes placement depend
on scheduler tie-breaking. It nested a reduction inside its sibling and gave `[-2, -5, -8]` for
`[-2, -2, -2]` (section 5.5).

**Ordering siblings by any data dependency (note 8c).** Reading a loop's *variable* means being inside it.
Only reading through its `END` means coming after it. Mixing the two forced `k` outermost because of `k*N`.

**Accumulators declared too shallow (note 9).** An `ALLOC` at function scope is shared by every OpenMP
thread. The race may be invisible at `-O3` and catastrophic at `-O0` (section 5.7). Declare in the deepest
scope common to the users.

**Accumulator in the wrong scope.** Write `acc = ALLOC(...).after(i)` instead of `.after(i, j)` in GEMM, and
the reset runs once per row, not once per output element:

```c
void gemm_bad(float* restrict data0, float* restrict data1, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    float acc0;
    acc0 = 0.0f;
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      for (int ridx2 = 0; ridx2 < 4; ridx2++) {
        float val1 = acc0;
        float val2 = data0[((ridx0*4)+ridx2)];
        float val3 = data1[((ridx2*3)+ridx1)];
        acc0 = (val1+(val2*val3));
      }
      float val4 = acc0;
      data2[((ridx0*3)+ridx1)] = val4;
    }
  }
}
```

```text
[[ 4.  8. 12.]
 [ 4.  8. 12.]]
```

With all-ones inputs of shape `(2, 4) @ (4, 3)` every entry should be 4. The linearizer did exactly what it
was told: the reset depends only on `i`. The `AFTER` ranges on an accumulator *are* its reset frequency.

**Forgetting `END`.** A store inside a loop that is never `END`ed leaves the loop open. The `SINK` then
sits inside the loop, the linearizer finishes with a loop still on its stack, and the renderer emits a
function with a missing brace:

```c
void noend(float* restrict data0, float* restrict data2) {
  for (int ridx0 = 0; ridx0 < 8; ridx0++) {
    float val1 = data0[ridx0];
    data2[ridx0] = val1;
}
```

gcc reported `error: expected declaration or statement at end of input`: the first sign of the bug was a
C syntax error, far from its cause. `linearize` now checks that its loop stack is empty when it finishes:

```text
LinearizeError: RANGE without END: loops [0] are never closed
```

(`test_missing_end_is_an_error`.)

**One RANGE where you meant two.** Two sibling loops written as `range_(8, 0)` twice are the *same* hash-consed
node. The first `END` closes it, and the second `END` can never be scheduled:

```text
LinearizeError: can't schedule; open loops [], stuck at [(Ops.END, []), (Ops.SINK, [])]
```

Give every distinct loop its own `idx`. With `range_(8, 0)` and `range_(8, 1)` the same program renders as
two sibling loops.

**Ending only some of the loops.** `store.end(i)` when the store's scope is `{i, j}` leaves `j` open with no
`END` for it, and the scheduler stops:

```text
LinearizeError: can't schedule; open loops [0, 1], stuck at [(Ops.END, [0, 1]), (Ops.SINK, [0, 1])]
```

**Max hides accumulator bugs.** A `max` reduction that accidentally accumulates across extra iterations
often still gives the right answer, because `max` is idempotent. Test reductions with `sum` too.

## Exercises

1. ★ **Transpose by hand.** Write a loop program that stores the transpose of an `N x M` buffer into an
   `M x N` buffer, run it with `run_c` for `N=3, M=5`, and compare with numpy.

2. ★ **Predict the loops.** For `out[i*8 + j] = x[i*8 + j] * w[j] + b[i]` with `i` of 4 and `j` of 8
   (idx 0 and 1), which loads are inside the `j` loop? Then check against the `predict` kernel in section 5.8.

3. ★★ **Conv2d.** Write a single-channel valid 2-D convolution `out[y, x] = sum_{ky, kx} img[y+ky, x+kx] *
   w[ky, kx]` in the loop IR with four ranges, test it against a numpy reference, and read the C. Which loads
   are hoisted, and where is the accumulator declared?

4. ★★ **Row softmax by hand.** Write `softmax` over the rows of an `N x M` matrix as three sibling loops
   inside the row loop: a max reduction, a sum of `exp(x - max)`, and the normalized output. Use `EXP2` with
   `exp(x) = exp2(x * log2(e))`. Check that the generated C has the loops in the right order whatever `idx`
   values you give the three sibling ranges.

5. ★★ **Find the bug.** A student's GEMM has no `init`: the update is
   `acck[0].store(acck[0].load() + A[i*K+k] * B[k*M+j])` with `acck = ALLOC.after(i, j, k)`, and the read-back
   is `acc.after(upd.end(k))[0].load()`. Predict the generated C. Why might the test still pass on some runs?

6. ★★ **GEMM with k in the middle.** Write the `i, k, j` loop order: for each `i` and `k`, broadcast
   `A[i, k]` and update a whole row of `C`. You need a row accumulator of `M` elements (an `ALLOC` of size `M`)
   or direct updates of `C` through `AFTER`. Compare the generated C with the `i, j, k` version.

7. ★★★ **Implement it.** Implement `scopes`, `loop_targets` and `linearize` yourself from the rules in
   sections 5.2 to 5.6, without looking at `uopc/linearize.py`, until `tests/test_w03_loops.py`, the week-2
   tests and the `sub_rowsum`/`sub_rowmax` programs pass with your linearizer.

8. ★★★ **Better errors.** `linearize` now names the loops left open by a forgotten `END`. Extend the message
   to name the *nodes* that sit in those loops, then detect the "same `RANGE` closed twice" case and suggest
   using distinct `idx` values.

9. ★★ **Where is the accumulator?** Rewrite GEMM so that the `ALLOC` has two users: `acc.after(i, j)` for
   the reset and the read-back, and a separate `acc.after(i, j, init, k)` for the update. Compute by hand the
   targets of both users and their longest common prefix, and predict where `float acc0;` is declared. Then
   make `i` a `THREAD` axis and check the C.

<details markdown="1">
<summary>Solutions to the ★ exercises</summary>

**1.**

```python
N, M = 3, 5
A, B = buf(0, N * M), buf(1, M * N)
i, j = range_(N, 0), range_(M, 1)
s = sink(B.index(j * N + i).store(A.index(i * M + j).load()).end(i, j))
a = np.arange(N * M, dtype=np.float32).reshape(N, M); b = np.zeros((M, N), np.float32)
print(run_c(s, {0: a, 1: b}, "transpose")); print((b == a.T).all())
```

```c
void transpose(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      float val1 = data0[((ridx0*5)+ridx1)];
      data1[((ridx1*3)+ridx0)] = val1;
    }
  }
}
```

and `True`. (Reads are contiguous and writes are strided. [Chapter 10](10-cpu-performance.md) discusses
when that matters.)

**2.** `x[i*8 + j]` and `w[j]` are inside `j`. `b[i]` has scope `{i}` and is hoisted out of `j`. `w[j]`
does not depend on `i` but is still inside `i`, because loop `j` is nested inside `i`. The `predict` kernel
has exactly this shape, with `data0` playing all three roles.

</details>

## Quiz

1. What is the scope of `END(store, k)` if `store` has scope `{i, j, k}`?
2. Why does `A[i*K + k]` in GEMM end up inside the `j` loop although it does not depend on `j`?
3. Two ranges never appear in a common scope. What is their relationship in the loop tree?
4. In the final linearizer, what is the *only* kind of dependency that decides which of two sibling loops
   runs first?
5. Why does step 1 (emit at the current depth) come before step 2 (open a loop)? What optimization does that
   ordering give you?
6. Where is an `ALLOC` of `LOCAL` space declared, and why does it matter for OpenMP?
7. What happens if you write `range_(8, 0)` for two loops that should be siblings?
8. Why can't the linearizer hoist `X[j]` out of the `i` loop in section 5.8's kernel?

<details markdown="1">
<summary>Answers</summary>

1. `{i, j}`: `END` removes the ranges it closes.
2. Loop `k` is nested inside `j` (they share the update's scope `{i, j, k}`), so every node using `k` sits
   inside `j`. Closing the scope `{i, k}` under "encloses" gives `{i, j, k}`.
3. Neither encloses the other; they are siblings (or in unrelated subtrees).
4. A pending node that needs one loop and reads, through an `END`, the results of the other.
5. Everything that can run at the current depth runs before going deeper, which is loop-invariant code
   motion for free (and it emits accumulator resets before the loops that use them).
6. In the deepest scope common to all its users (the longest common prefix of their targets). Variables
   declared inside an OpenMP parallel loop are private per thread; declared outside, they are shared and
   race.
7. Hash consing makes them one `RANGE` node, so they are one loop. The second `END` can never be
   scheduled and linearization fails.
8. `j` is nested inside `i`, so any node using `j` is inside `i`. Computing `X[0..7]` once before the `i`
   loop would need a temporary buffer and an extra loop, which is a change to the program, not a placement.

</details>

## Further reading

- Keith D. Cooper and Linda Torczon, *Engineering a Compiler*, 2nd ed., chapter 12 (instruction scheduling,
  list scheduling) and chapter 10 (loop-invariant code motion).
- Steven S. Muchnick, *Advanced Compiler Design and Implementation* (1997), chapter 17 (code scheduling).
- Randy Allen and Ken Kennedy, *Optimizing Compilers for Modern Architectures: A Dependence-based Approach*
  (2001): loop nests, dependence, and why legal loop orders are a dependence question.
- Jonathan Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and
  Recomputation in Image Processing Pipelines", PLDI 2013: separating *what* is computed from *where* in the
  loop nest it is computed.
- OpenMP Architecture Review Board, *OpenMP Application Programming Interface* specification, "Data-sharing
  attribute rules".
- tinygrad, `tinygrad/codegen/` (the linearizer and the `RANGE`/`END` design this chapter follows).
