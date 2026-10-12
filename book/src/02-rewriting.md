# 2. Term Rewriting and the Rewrite Engine

Almost every pass in this compiler has the same shape: *find a piece of the graph that looks like
this; replace it with that.* "An ADD of something and zero is that something." "A multiply of two
constants is their product." "Indexing into a STACK at a known position is that element." "A CALL is
its body with the arguments plugged in." Written one at a time, each of those is a tiny function.
Written as a *system*, a set of rules plus an engine that applies them until nothing changes, they
are a constant folder, an algebraic simplifier, an interpreter, and later the lowering from shapes to
loops. This chapter builds that engine: a pattern language, a matcher, and a bottom-up rewriter that
reaches a fixed point without recursion and without ever doing the same work twice. Then it uses the
engine to *run* week-1 programs, with no interpreter at all.

**What you'll build**

- `UPat`: patterns that match UOps by op, dtype, arg and sources, bind names, and handle commutative
  ops by trying both operand orders.
- `PatternMatcher`: an ordered list of `(pattern, function)` rules, indexed by op.
- `graph_rewrite`: a memoized, iterative, bottom-up rewriter that runs to a fixed point.
- `evaluate`: rules that push ALU ops through STACK, index into STACKs, compute movement ops on
  constants, and inline CALLs, so that `run(program)` reduces any closed week-1 program to a value.

**Where this fits.** [Chapter 1](01-uops.md) gave us the graphs. This chapter gives us the tool for
transforming them. [Chapter 3](03-symbolic.md) writes the forty-odd algebra rules that the engine
runs, and every chapter after that (rangeify, layout, GPU lowering) is mostly new rule sets for the
same engine.

## 2.1 Term rewriting in one page

A **term** is a tree built from function symbols and variables: `add(mul(x, 3), 0)`. A **rewrite
rule** is a pair `l → r` of terms where every variable in `r` also appears in `l`:

```
add(x, 0) → x
mul(x, 1) → x
mul(x, 0) → 0
```

A rule *applies* to a term `t` if some subterm of `t` is an instance of `l` (you can substitute terms
for `l`'s variables to make it equal); that subterm is called a **redex**. Applying the rule replaces
the redex with the correspondingly instantiated `r`. A **term rewriting system** (TRS) is a set of such
rules. A term with no redex is in **normal form**.

```
add(mul(add(y, 0), 1), 0)
  → add(mul(y, 1), 0)        by add(x, 0) → x   on the inner add
  → add(y, 0)                by mul(x, 1) → x
  → y                        by add(x, 0) → x
```

Two questions decide whether a TRS is a well-behaved program:

**Termination.** Does every sequence of rewrites stop? The three rules above obviously terminate:
each one makes the term strictly smaller. In general one proves termination by finding a *measure*
(term size, or a fancier ordering) that every rule strictly decreases. A single bad rule breaks it:
`add(x, y) → add(y, x)` (commutativity) can be applied forever.

**Confluence.** If a term can be rewritten in two different ways, do the two paths always end at the
same normal form? If yes, the *order* in which rules are applied doesn't matter for the final answer,
and the TRS defines a function. If no, the answer depends on the strategy. Newman's lemma gives a
practical test: a terminating system is confluent iff it is *locally* confluent, which can be checked
by looking at every pair of rules whose left sides overlap (the *critical pairs*) and checking that
both results can be rewritten to a common term.

For instance, take `mul(x, 0) → 0` and the distribution rule `mul(add(a, b), c) → add(mul(a, c), mul(b, c))`.
They overlap on `mul(add(a, b), 0)`. One path gives `0` immediately. The other gives
`add(mul(a, 0), mul(b, 0))`, then `add(0, 0)`, then (with a constant folding rule) `0`. The critical
pair is joinable: no conflict.

Now a real conflict from `uopc` itself. The constant folder evaluates `add(-0.0, 0.0)` in IEEE
arithmetic, which gives `0.0`. The identity rule `add(x, 0.0) → x` gives `-0.0`. Both rules are in
the real `symbolic` rule set, and the result depends on which one is tried first:

```python
>>> e = UOp.const(-0.0) + UOp.const(0.0)
>>> graph_rewrite(e, symbolic).arg                     # folding is listed first
0.0
>>> rev = PatternMatcher(symbolic.rules[1:] + symbolic.rules[:1])
>>> graph_rewrite(e, rev).arg                          # identity first
-0.0
```

So `symbolic` is not confluent, and we know it: the comment on that rule in `uopc/symbolic.py` says
`x+0.0 is not exact for x=-0.0; we accept that`. The practical stance of every production rewriter
(LLVM's InstCombine, tinygrad's symbolic, GCC's match.pd) is the same: rule *order* is part of the
specification, the engine applies rules in a fixed, deterministic order, and non-confluence is
tolerated where the results are equally correct. What is not tolerated is *unsoundness* (a rule that
changes the value) or *non-termination* (which the engine detects; see section 2.5).

> **Key idea:** a rewrite system is a program whose instructions are equations. Soundness is each
> rule's job. Termination and determinism are the engine's job. Confluence is nice to have.

### Rewriting graphs, not trees

Our terms are DAGs: `s * s` contains one `s`, not two. Rewriting on shared graphs (*term graph
rewriting*) has one big advantage: rewrite a shared subterm once and every user sees the result.
On trees, a subterm used `k` times is rewritten `k` times, and nested sharing makes that exponential.
Hash consing gives us term graph rewriting for free: if the engine memoizes "node → rewritten node",
each distinct node is processed once. Section 2.4 measures the difference.

### Strategy

A *strategy* says which redex to rewrite next. The two classic choices are **innermost** (rewrite
arguments before the term containing them; this is call-by-value) and **outermost** (try the top first;
call-by-need, roughly). `uopc` uses innermost, bottom-up: every source is fully rewritten before its
user is looked at. Rules can then assume their operands are already in normal form, which makes them
much simpler to write: a rule for `(x + c1) + c2` never needs to worry that `x + c1` might itself
still be reducible.

## 2.2 Patterns: `UPat`

A rule's left-hand side is a *pattern*: a tree that describes a family of UOps. Here is the pattern
class:

@include uopc/rewrite.py#UPat.__init__

A `UPat` constrains a UOp on up to five things, and every constraint is optional:

| Field | Matches if... | Example |
|---|---|---|
| `op` | the UOp's op is in this set | `UPat({Ops.ADD, Ops.MUL})` |
| `dtype` | the UOp's dtype is in this set | `UPat(dtype=dtypes.i32)` |
| `arg` | `u.arg == arg` | `UPat(Ops.CONST, arg=0)` |
| `src` | the sources match these sub-patterns | `UPat(Ops.ADD, src=(p, q))` |
| `name` | always; binds the UOp to this name | `UPat(name="x")` |

A pattern with no constraints at all matches any UOp; with a name it is a *variable*. Two helpers make
the common cases short: `UPat.var("x")` (any UOp, bound to `x`) and `UPat.cvar("c")` (any CONST, bound
to `c`).

@include uopc/rewrite.py#UPat.var

@include uopc/rewrite.py#UPat.cvar

### Matching

@include uopc/rewrite.py#UPat.match

`match` returns a *list* of bindings: every way the pattern matches. Each binding is a dict from names
to UOps. An empty list means no match. Three features need explaining.

**Consistent names.** If a name is already bound, it must be bound to the *same node* (`is`, which by
hash consing means structurally equal). So `UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("a")))` matches
`x + x` and not `x + y`:

```python
>>> same = UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("a")))
>>> same.match(x + y, {}), same.match(x + x, {})
([], [{'a': UOp(Ops.PARAM, arg=ParamArg(slot=0, dtype=dtypes.i32, shape=(), addrspace=None))}])
```

In a tree-based rewriter this check would be a recursive structural comparison. Here it is one
pointer comparison, no matter how large the subterm.

**Tuple vs list sources.** A `tuple` of source patterns matches in order. A `list` matches in *any*
order: `match` tries every permutation (`itertools.permutations`). That is how commutative ops are
handled. A rule for `x + c` should match both `x + 3` and `3 + x`, and rather than writing it twice, we
write the sources as a list:

```python
>>> pat = UPat(Ops.ADD, src=[UPat.var("a"), UPat.var("b")])
>>> pat.match(x + y, {})
[{'a': UOp(Ops.PARAM, arg=ParamArg(slot=0, ...)), 'b': UOp(Ops.PARAM, arg=ParamArg(slot=1, ...))},
 {'b': UOp(Ops.PARAM, arg=ParamArg(slot=0, ...)), 'a': UOp(Ops.PARAM, arg=ParamArg(slot=1, ...))}]
```

(output abbreviated) Two bindings, one per order. With a tuple, only the first. Permutations cost
`n!` for `n` sources, which is fine because every commutative op is binary.

**`allow_any_len`.** Normally the number of sources must equal the number of sub-patterns. With
`allow_any_len=True` the pattern only constrains a *prefix* of the sources. INDEX has a variable number
of indices, and the evaluator uses this to match "INDEX of a STACK with at least one index".

Matching is a *join* over sources: start with the bindings so far, match the first source in each of
them, feed every resulting binding to the second source, and so on (`stores = [m for st in stores
for m in p.match(s, st)]`). The early `break` stops as soon as no binding survives.

### Pattern sugar

Rules are easier to read when the pattern looks like the math. `UPat` overloads arithmetic to build
patterns:

@include uopc/rewrite.py:128-137

A non-UPat operand becomes a CONST pattern with that arg, and commutative ops get *list* sources
automatically. So `UPat.var("x") + UPat.cvar("c")` is an ADD pattern that matches a constant on
either side:

```python
>>> c = UPat.var("x") + UPat.cvar("c")
>>> c.src, type(c.src)
([UPat(None, name=x), UPat({Ops.CONST}, name=c)], <class 'list'>)
>>> c.match(x + 3, {}), c.match(UOp.const(3) + x, {})
([{'x': UOp(Ops.PARAM, ...), 'c': UOp(Ops.CONST, arg=3)}], [{'c': UOp(Ops.CONST, arg=3), 'x': UOp(Ops.PARAM, ...)}])
```

(PARAM reprs abbreviated.)

`UPat.var("x") // 2` is non-commutative, so its sources are a tuple: `(UPat(None, name=x),
UPat({Ops.CONST}, name=None))`. The rule `x + 0 → x` from section 2.1 is written, in full:

```python
(UPat.var("x") + 0, lambda x: x)
```

## 2.3 Rules and the `PatternMatcher`

A rule is a pattern and a Python function. When the pattern matches, the function is called with the
bindings as keyword arguments, and returns either a replacement UOp or `None`, meaning "on closer
inspection, this rule does not apply". The `None` escape hatch is important: many conditions cannot be
expressed as patterns ("`c` is positive", "the bounds of `x` fit in `[0, c)`"), so they go in the
function.

@include uopc/rewrite.py#PatternMatcher

**Order is priority.** `rewrite` tries rules in the order given and returns the first non-`None`
result. Section 2.1 showed that this order is observable. A rule returning the very node it was
given is treated as "no change" (`ret is not u`): with hash consing, a rule that rebuilds an identical
node returns the identical object, so this check is exact and costs nothing.

**Indexing by op.** Trying all 44 rules of `symbolic + evaluate` on every node would be wasteful: a
rule whose pattern is an ADD can't match a MUL. The constructor files each rule under every op its
root pattern can match (all ops, for a pattern without an op), and `rewrite` only tries that list.
For the real rule set:

```python
>>> pm = symbolic + evaluate
>>> len(pm.rules)
44
>>> {op.name: len(pm.by_op.get(op, [])) for op in Ops if pm.by_op.get(op)}
{'CALL': 1, 'INDEX': 3, 'CAST': 4, 'RECIP': 3, 'ADD': 14, 'MUL': 10, 'CMPLT': 6, 'FLOORDIV': 8,
 'FLOORMOD': 7, 'WHERE': 7, 'CMPNE': 6, 'MAX': 5, 'EXP2': 3, 'LOG2': 3, 'SQRT': 3, 'SIN': 3,
 'RESHAPE': 1, 'EXPAND': 1, 'PERMUTE': 1, 'SHRINK': 1, 'REDUCE': 1}
```

CONST, PARAM and STACK nodes, which are most leaves, get zero candidates and are skipped immediately.
(tinygrad goes further and compiles each pattern into a specialized Python function; that is
Exercise 10.)

**Composition.** `pm1 + pm2` is a new matcher with `pm1`'s rules first. `uopc` builds its rule sets
this way: `symbolic = fold + PatternMatcher([...])`, `run` uses `symbolic + evaluate`.

**Context.** `graph_rewrite(..., ctx=something)` passes `ctx=something` to rule functions as an extra
keyword argument. Rangeify ([Chapter 6](06-rangeify.md)) uses it to carry per-pass state. Note the
implementation: when `ctx` is not `None`, *every* rule in the matcher receives it. A rule set that
needs context and a rule set that doesn't (like `symbolic`, whose lambdas take only their pattern
variables) cannot be mixed in one call; see Pitfall 5.

## 2.4 `graph_rewrite`: the engine

The job: given a root UOp and a matcher, return the root with every node rewritten to normal form,
bottom-up.

### First attempt: recursion

The most direct implementation is four lines:

```python
def rewrite_rec(u, pm):
  n = u.replace(src=tuple(rewrite_rec(s, pm) for s in u.src))   # normalize sources first
  r = pm.rewrite(n)
  return n if r is None else rewrite_rec(r, pm)                   # a rule fired: normalize the result
```

It is correct, and it is the right mental model. It has two fatal problems.

**It walks trees, not DAGs.** Nothing remembers that a node was already rewritten, so a node with
`k` users is rewritten `k` times, recursively. For `x` squared `d` times (`t = t * t`, a DAG of `d + 1`
nodes) we can count rule lookups:

```
depth  5:   6 nodes   recursive tree walk       63 rule lookups   graph_rewrite   6
depth 10:  11 nodes   recursive tree walk     2047 rule lookups   graph_rewrite  11
depth 15:  16 nodes   recursive tree walk    65535 rule lookups   graph_rewrite  16
depth 20:  21 nodes   recursive tree walk  2097151 rule lookups   graph_rewrite  21
```

`2^(d+1) - 1` versus `d + 1`. Real graphs are full of sharing (every weight is used by the forward
and backward pass, every index expression is shared by every load), so this is not academic.

**It recurses.** On the 20,000-ADD chain from `test_deep_graph_no_recursion_error`:

```
RecursionError: maximum recursion depth exceeded
```

### The real engine

@include uopc/rewrite.py#graph_rewrite

Two data structures: `done`, the memo table mapping each original node to its normal form, and an
explicit `stack` of work items `(node, stage, aux)`. Each node goes through up to three stages:

- **Stage 0, "schedule my sources".** Push myself back as stage 1, then push every source that is not
  already done as stage 0. Because the stack is LIFO, all the sources (and their sources, and so on)
  are finished before the stage-1 item comes back up. The `reversed` makes sources come off the stack
  left to right, so the traversal order matches `toposort`.
- **Stage 1, "my sources are done".** Rebuild the node from its sources' normal forms (`n`, which is
  just `u` if nothing below changed). Try the rules on `n`.
  - No rule applies: `n` is the normal form. Record `done[u] = n`.
  - A rule returned `m`, and `m` has already been normalized: `done[u] = done[m]`.
  - Otherwise `m` is new and might itself be a redex, or contain new redexes. Push `(u, 2, m)` and then
    `(m, 0, None)`: normalize `m` completely, then come back.
- **Stage 2, "the thing I rewrote to is done".** `done[u] = done[aux]`.

@diagram 02-fixpoint

The first line of the loop skips work items for nodes that are already finished: a node with two
users is pushed twice but processed once. Stage-2 items are exempt, because they carry the final
answer for their node and must always record it.

### Watching it run

To see the traversal, subclass `PatternMatcher` so that `rewrite` prints every call. This is a
debugging trick worth keeping around:

```python
def show(u):
  if u.op is Ops.CONST: return repr(u.arg)
  if u.op is Ops.PARAM: return f"p{u.arg.slot}"
  return f"{u.op.name.lower()}({', '.join(show(s) for s in u.src)})"

class Trace(PatternMatcher):
  def rewrite(self, u, ctx=None):
    r = super().rewrite(u, ctx)
    print(f"  visit {show(u):28s} -> {show(r) if r is not None else '.'}")
    return r

p0 = UOp(Ops.PARAM, (), ParamArg(0, dtypes.i32))
e = (p0 + 2) * 3 + (UOp.const(1) + UOp.const(4))
out = graph_rewrite(e, Trace(symbolic.rules))
```

```
program: add(mul(add(p0, 2), 3), add(1, 4))
  visit p0                           -> .
  visit 2                            -> .
  visit add(p0, 2)                   -> .
  visit 3                            -> .
  visit mul(add(p0, 2), 3)           -> add(mul(p0, 3), 6)
  visit mul(p0, 3)                   -> .
  visit 6                            -> .
  visit add(mul(p0, 3), 6)           -> .
  visit 1                            -> .
  visit 4                            -> .
  visit add(1, 4)                    -> 5
  visit 5                            -> .
  visit add(add(mul(p0, 3), 6), 5)   -> add(mul(p0, 3), 11)
  visit 11                           -> .
  visit add(mul(p0, 3), 11)          -> .
result: add(mul(p0, 3), 11)
```

@diagram 02-bottom-up

Read the trace against the algorithm:

- Leaves first, left to right: `p0`, `2`, then their user `add(p0, 2)`, which no rule changes.
- `mul(add(p0, 2), 3)` matches the rule `(x + c1) * c2 → x*c2 + c1*c2` and becomes `add(mul(p0, 3), 6)`.
  That is a *new* node, so it goes back through stage 0: its new pieces `mul(p0, 3)` and `6` are
  visited, then the new ADD itself. Nothing fires, so it is the normal form of the old MUL.
  Note that `p0` and `3` are not visited again: they are already in `done`.
- The right subtree folds `add(1, 4)` to `5`.
- The root is rebuilt from the normal forms of its sources: `add(add(mul(p0, 3), 6), 5)`. That is not
  the node we started with, and *now* the reassociation rule `(x + c1) + c2 → x + (c1+c2)` matches,
  giving `add(mul(p0, 3), 11)`. It is re-normalized (only `11` is new) and recorded as the answer for
  the root.

The root's rule could only fire *after* its sources were rewritten: the original root was
`add(mul(...), add(1, 4))`, which matches nothing. This is why bottom-up order matters.

### Why results are rewritten again

Stage 2 exists because a rule's output is built by ordinary Python code (`x * c2 + c1.arg * c2.arg`)
from ordinary constructors, and nothing guarantees that the pieces it builds are in normal form. Here
is a "one-pass" rewriter that applies rules bottom-up but never looks at a rule's result again:

```python
def one_pass(root, pm):
  done = {}
  for u in root.toposort():
    n = u.replace(src=tuple(done[s] for s in u.src))
    done[u] = pm.rewrite(n) or n
  return done[root]
```

On `(p0*2 + p1) * 3`, the distribution rule turns the MUL into `(p0*2)*3 + p1*3`, and the one-pass
rewriter leaves `(p0*2)*3` unsimplified:

```
one_pass:                           graph_rewrite:
%0 = param : slot=0 dtype=i32       %0 = param : slot=0 dtype=i32
%1 = mul %0, 2                      %1 = mul %0, 6
%2 = mul %1, 3                      %2 = param : slot=1 dtype=i32
%3 = param : slot=1 dtype=i32       %3 = mul %2, 3
%4 = mul %3, 3                      %4 = add %1, %3
%5 = add %2, %4
```

For evaluation (section 2.6) the failure is total: every rule that pushes an operation into a STACK
creates new, unevaluated operations. One pass over `a * b + a` with `a = [1, 2, 3]`,
`b = [10, 20, 30]` leaves:

```
%0 = mul 1, 10
%1 = add %0, 1
...
%6 = stack %1, %3, %5
```

### Memoization, and why it is sound

`done` maps an *original* node to its normal form, and the check `if u in done` skips any node
reached a second time. That is only correct if rewriting is a *pure function of the node*: the same
node must always rewrite to the same result. Two things make that true. Hash consing means "the same
node" is a well-defined, O(1)-checkable notion. And rules are pure functions of their bindings (they do
not consult global state that changes during the rewrite). A rule that, say, counted how many times it
fired and behaved differently the tenth time would break memoization silently. If a rule needs state,
it belongs in `ctx`, and the result is still memoized per call of `graph_rewrite`.

The `elif m in done` shortcut is the memo table paying off again: rewriting `mul(x, 1)` to `x`, where
`x` was normalized long ago, finishes in one step.

### Termination, detected

If the rules loop, so does the engine. The guard is `max_steps`: one step per stage-1 visit, and an
exception after ten million. Here is a commutativity rule, the textbook non-terminating rule:

```python
>>> swap = PatternMatcher([(UPat(Ops.ADD, src=(UPat.var("x"), UPat.var("y"))), lambda x, y: UOp(Ops.ADD, (y, x)))])
>>> graph_rewrite(a + b, swap, max_steps=1000)
RuntimeError: graph_rewrite did not converge (a rule loops?)
```

`a + b` rewrites to `b + a`, which is a new node, which rewrites to `a + b`, which is *not yet in
`done`* (it is still waiting in stage 2), and so on. A hard limit is crude, but it turns a hang into an
error with a hint, which is the difference between a five-minute bug and a lost afternoon.

How do real rule sets avoid loops? By making every rule decrease *something*. The canonicalization
rule in `symbolic` moves constants to the right of commutative ops (`c + y → y + c`), but only when
`y` is not itself a constant, so applying it twice is impossible. Reassociation moves constants
*outward*, toward each other, where they fold. [Chapter 3](03-symbolic.md) discusses termination of
the symbolic rules in detail.

### Costs

Every node is visited once in stage 0 and once in stage 1, plus one visit per node created by a rule.
Each stage-1 visit tries the rules indexed under the node's op, and each try is a pattern match whose
cost depends on the pattern's size. For the 44-rule set, a stage-1 visit costs on the order of 100 µs
of Python, which is perfectly adequate for graphs of tens of thousands of nodes and the reason the
compiler caches whole compiled plans in [Chapter 7](07-kernelize.md) rather than trying to make
rewriting faster.

## 2.5 Writing your own rules

The week-1 test for the engine is a custom matcher with a single rule, strength reduction of
`x * 2` into `x + x`:

@include tests/test_w01_uops.py#test_custom_pattern_matcher

Three things to notice: the dtype constraint on the constant (`dtype=dtypes.i32`, because `2.0` is a
different rule for floats), the list sources (so `2 * x` also matches), and the function returning
`None` when `c` is not 2, rather than encoding "arg is 2" in the pattern. (It could: `UPat(Ops.CONST,
arg=2, name="c")`. Both are fine.)

### A worked example: `where(c, 1, 0)` is a cast

A comparison used as a 0/1 integer often appears as `cond.where(1, 0)`. In C, `(int)cond` is the same
thing and cheaper. Here is the rule:

```python
b2i = PatternMatcher([(UPat(Ops.WHERE, src=(UPat.var("c"), UPat(Ops.CONST, arg=1), UPat(Ops.CONST, arg=0))),
                       lambda c: c.cast(dtypes.i32))])
```

```
%0 = param : slot=0 dtype=i32            %0 = param : slot=0 dtype=i32
%1 = param : slot=1 dtype=i32            %1 = param : slot=1 dtype=i32
%2 = cmplt %0, %1                        %2 = cmplt %0, %1
%3 = where %2, 1, 0                      %3 = cast %2 : dtype=i32
```

It works, and it has a bug. Apply it to a *bool* WHERE, `c.where(True, False)`:

```python
>>> wb = c.where(True, False)
>>> graph_rewrite(wb, b2i).dtype
dtypes.i32          # was dtypes.bool
```

The pattern's `arg=1` is compared with `==`, and `True == 1`, `False == 0`. The same Python numeric
tower that caused the hash-consing bug of [Chapter 1](01-uops.md) is back, this time in pattern
matching: `UPat(Ops.CONST, arg=0)` matches `0`, `0.0`, `-0.0` and `False`. The rewrite silently
changed a node's dtype, and since every node above it was rebuilt from the replacement, the type
checker would only complain if some user now saw mismatched operands, possibly far away, possibly
never. The fix is to constrain the dtype of the node being replaced:

```python
UPat(Ops.WHERE, dtype=dtypes.i32, src=(UPat.var("c"), UPat(Ops.CONST, arg=1), UPat(Ops.CONST, arg=0)))
```

> **Key idea:** a rewrite must preserve the dtype and shape of the node it replaces. The engine does
> not check this. Every rule author must, and a debug assertion in your engine is cheap insurance
> (Exercise 4).

### A checklist for every rule

1. **Is it sound?** For all values of the variables, do both sides compute the same thing, *in the
   target's arithmetic*? (Int32 wraps; floats round; `x * 0 → 0` is false for `x = inf` or NaN, which
   is why `symbolic` restricts it to ints.) Test it by evaluating both sides on random inputs.
2. **Does it preserve dtype and shape?**
3. **Does it terminate together with the other rules?** Name the measure it decreases.
4. **Can its pattern be more specific?** Every condition you move from the function into the pattern
   is checked faster and documents itself.
5. **Is the order right?** If another rule matches the same nodes, which one should win?

### Rules with context

A matcher that binds PARAMs to values (a *partial evaluator*) needs to know the values. That is what
`ctx` is for:

```python
bind = PatternMatcher([(UPat(Ops.PARAM, name="p"),
                        lambda p, ctx: UOp.const(ctx[p.arg.slot], p.dtype) if p.arg.slot in ctx else None)])
e = (p0 + 2) * p1 + p0 * 0
graph_rewrite(e, bind, ctx={0: 5})
```

```
%0 = add 5, 2
%1 = param : slot=1 dtype=i32
%2 = mul %0, %1
%3 = mul 5, 0
%4 = add %2, %3
```

Combining it with `symbolic` in one call does not work, because `ctx` is passed to *every* rule:

```
TypeError: _fold() got an unexpected keyword argument 'ctx'
```

Run two passes instead: `simplify(graph_rewrite(e, bind, ctx={0: 5}))` gives

```
%0 = param : slot=1 dtype=i32
%1 = mul %0, 7
```

## 2.6 Running programs by rewriting

Now the payoff. The syllabus says: *the graph rewrite engine should be able to "run" these programs via
const folding*. A closed week-1 program (no free PARAMs) is a DAG of constants, STACKs, INDEXes, ALU
ops and CALLs. If we have rules that eliminate every non-constant op, the normal form of a program is
its value. Rewriting *is* the interpreter.

The rules come in two groups. `fold`, the constant folder, evaluates any scalar ALU op whose sources
are all CONSTs, using `exec_alu`, the reference semantics of every ALU op:

@include uopc/symbolic.py#_fold

`exec_alu` (int32 wraparound, f32 rounding, floor division, C-style casts, refusing to fold undefined
behavior) is the subject of [Chapter 3](03-symbolic.md). Here we only need to know that for
constant scalar inputs it returns the same constant the C program would compute.

The second group, `evaluate`, eliminates everything else:

@include uopc/symbolic.py#evaluate

and `run` is one line:

@include uopc/symbolic.py#run

along with a helper that converts the resulting CONST/STACK tree into Python lists:

@include uopc/symbolic.py#to_python

### Pushing ALU ops through STACK

`fold` only handles scalars. `[1, 2, 3] * [10, 20, 30]` is a MUL of two STACKs. The rule: an
elementwise op of STACKs is a STACK of elementwise ops.

```
f(STACK(a0, a1, ...), STACK(b0, b1, ...))  →  STACK(f(a0, b0), f(a1, b1), ...)
```

@include uopc/symbolic.py#_push_stack

@include uopc/symbolic.py#_elem

The rule fires for any ALU op with a shape and at least one STACK source. Sources that are not
STACKs are split with INDEX (`_elem`), so `STACK(...) * PARAM(shape=(3,))` becomes a STACK of
`INDEX(param, i) * ...`, which stays symbolic: the rule works even on programs that are not fully
known. The rule produces new scalar ALU nodes, which the engine (stage 2) immediately rewrites, and
`fold` finishes them. Here is `a * b + a` with `a = [1, 2, 3]`, `b = [10, 20, 30]`, tracing only the
rules that fired:

```
%0 = stack 1, 2, 3
%1 = stack 10, 20, 30
%2 = mul %0, %1
%3 = add %2, %0

  mul(stack(1, 2, 3), stack(10, 20, 30))   => stack(mul(1, 10), mul(2, 20), mul(3, 30))
  mul(1, 10)                               => 10
  mul(2, 20)                               => 40
  mul(3, 30)                               => 90
  add(stack(10, 40, 90), stack(1, 2, 3))   => stack(add(10, 1), add(40, 2), add(90, 3))
  add(10, 1)                               => 11
  add(40, 2)                               => 42
  add(90, 3)                               => 93
[11, 42, 93]
```

A WHERE works the same way; its condition is a STACK of bools produced by pushing CMPLT through:

```
  cmplt(stack(1, 2, 3), stack(2, 2, 2))    => stack(cmplt(1, 2), cmplt(2, 2), cmplt(3, 2))
  cmplt(1, 2)                              => True
  cmplt(2, 2)                              => False
  cmplt(3, 2)                              => False
  where(stack(True, False, False), stack(1, 2, 3), stack(10, 20, 30)) => stack(where(True, 1, 10), where(False, 2, 20), where(False, 3, 30))
  where(True, 1, 10)                       => 1
  where(False, 2, 20)                      => 20
  where(False, 3, 30)                      => 30
[1, 20, 30]
```

### INDEX of STACK

Three INDEX rules complete the picture. `INDEX(x)` with no indices is `x`. `INDEX(INDEX(x, i), j)` is
`INDEX(x, i, j)`, a normal form with one INDEX. And the one that does the work: an INDEX of a STACK at
a constant position is that element (indexed further by the remaining indices, if any).

@include uopc/symbolic.py#_index_stack

```
  index(stack(stack(1, 2, 3), stack(10, 20, 30)), 1) => stack(10, 20, 30)
  index(stack(10, 20, 30), 2)              => 30
30
```

An out-of-range constant index is an error in the program, so the rule *raises* instead of returning
`None` (which would leave the INDEX in place and make `to_python` fail later with a less useful
message):

```
IndexError: index 5 out of bounds for STACK of 3
```

### Movement ops on constant data

`run` can also evaluate RESHAPE, EXPAND, PERMUTE, SHRINK and REDUCE, the week-4 ops, as long as their
input is a fully known CONST/STACK tree. The approach is the bluntest one possible: flatten the input
to a Python list, compute each output element's source position with ravel/unravel index arithmetic,
and rebuild a STACK tree.

@include uopc/symbolic.py#_eval_movement

The helpers are worth reading, because ravel/unravel is the core of rangeify in
[Chapter 6](06-rangeify.md), where the same arithmetic is done *symbolically* on loop variables
instead of on Python ints:

@include uopc/symbolic.py#_unravel

@include uopc/symbolic.py#_ravel

Here is the second week-1 evaluation test, run with a trace. Note how the `.uir` sugar's `mul %1, 2`
became a broadcast constant (RESHAPE + EXPAND), which the movement rules evaluate before the MUL can
be pushed through:

```
%0 = stack 1.0, 2.0, 3.0
%1 = stack %0, %0
%2 = reshape 2.0, (1, 1)
%3 = expand %2, (2, 3)
%4 = mul %1, %3
%5 = permute %4, (1, 0)
%6 = reduce %5 : op=add pop=1

  reshape(2.0, (1, 1))
      => stack(stack(2.0))
  expand(stack(stack(2.0)), (2, 3))
      => stack(stack(2.0, 2.0, 2.0), stack(2.0, 2.0, 2.0))
  mul(stack(stack(1.0, 2.0, 3.0), stack(1.0, 2.0, 3.0)), stack(stack(2.0, 2.0, 2.0), stack(2.0, 2.0, 2.0)))
      => stack(mul(stack(1.0, 2.0, 3.0), stack(2.0, 2.0, 2.0)), mul(stack(1.0, 2.0, 3.0), stack(2.0, 2.0, 2.0)))
  mul(stack(1.0, 2.0, 3.0), stack(2.0, 2.0, 2.0))
      => stack(mul(1.0, 2.0), mul(2.0, 2.0), mul(3.0, 2.0))
  mul(1.0, 2.0)
      => 2.0
  mul(2.0, 2.0)
      => 4.0
  mul(3.0, 2.0)
      => 6.0
  permute(stack(stack(2.0, 4.0, 6.0), stack(2.0, 4.0, 6.0)), (1, 0))
      => stack(stack(2.0, 2.0), stack(4.0, 4.0), stack(6.0, 6.0))
  reduce(stack(stack(2.0, 2.0), stack(4.0, 4.0), stack(6.0, 6.0)), add, 1)
      => stack(4.0, 8.0, 12.0)
[4.0, 8.0, 12.0]
```

Look at the third rule firing. The outer MUL became a STACK of *two* row MULs, but only *one* of them
appears in the trace. The two rows are identical, so hash consing made them the same node, and the
memo table rewrote it once. This is term graph rewriting doing its job without anyone asking.

### CALL inlining is beta reduction

The last evaluation rule inlines function calls. In the lambda calculus, applying a function to an
argument is *beta reduction*: `(λx. x * x) 7 → 7 * 7`, the body with the parameter replaced by the
argument. A CALL is exactly a lambda application, with PARAM slots in place of variable names:

```
%0 = param : slot=0 dtype=i32
%1 = mul %0, %0
%2 = call %1, 7 : name=sq                  run(...) → 49
```

The rule is `(UPat(Ops.CALL, name="call"), inline_call)`:

@include uopc/symbolic.py#inline_call

It builds a map from each of the body's PARAMs to the corresponding argument, then rebuilds the body
bottom-up, substituting. It is `UOp.substitute` from [Chapter 1](01-uops.md) with one crucial
difference, in two places: when it meets a nested CALL, it *does not descend into its body*
(`u.src[1:] if u.op is Ops.CALL`), and when rebuilding a nested CALL, it keeps `u.src[0]` unchanged.
The PARAM-finding helper `call_params` has the same restriction.

@diagram 02-beta-reduction

### The scoping bug

Why the restriction? Recall from [Chapter 1](01-uops.md) that PARAMs are hash consed: `PARAM slot=0
dtype=i32` is *one object*, shared by every function body that uses slot 0. Consider three nested
functions:

```
f(5)  where  f(y) = g(y + 1),  g(z) = sq(z) * 2,  sq(w) = w * w
```

The expected answer is `g(6) = sq(6) * 2 = 72`. As UOps, all three bodies use slot 0, so all three
"different" parameters `y`, `z`, `w` are the same node `p0`:

```
f[g[mul(sq[mul(p0, p0)](p0), 2)](add(p0, 1))](5)
```

(written as `name[body](args)`). Now inline `g` with a naive substitution that replaces `p0`
everywhere reachable from `g`'s body:

```python
def naive_inline(call): return call.src[0].substitute({p0: call.src[1]})
```

```
naive  inline g: mul(sq[mul(add(p0, 1), add(p0, 1))](add(p0, 1)), 2)
scoped inline g: mul(sq[mul(p0, p0)](add(p0, 1)), 2)
```

The naive version reached into `sq`'s body and replaced *its* parameter, which belongs to a different
scope. `sq`'s body is now `(p0+1)*(p0+1)` where `p0` is `sq`'s own parameter, so `sq(z)` computes
`(z+1)²`. The program still type checks (the body still uses slot 0), and it silently computes the
wrong answer:

```
run: 72   by hand: g(6) = sq(6)*2 = 72
naive result: 98
```

`98 = sq'(7) * 2 = 7 * 7 * 2`: the argument was incremented twice. In other cases the type checker
does catch it: if the argument of the outer call is a constant, the naive substitution removes the
inner body's PARAM entirely and rebuilding the inner CALL fails with
`UOpError: CALL needs params 0..0, body has []`. That is the lucky case.

This is the classic *variable capture* problem of substitution in the lambda calculus, in a slightly
unusual form. Textbook capture happens when a substituted term mentions a variable that a binder
inside the body also binds; the cure is to rename the inner binder (α-conversion). Here the
"variables" are positional slots and hash consing has merged all the same-numbered slots, so
renaming is not available. The cure is to respect scopes: a CALL's body is a binder, and
substitution must not cross it.

Note that `run` would get the right answer even with the naive inliner, because bottom-up rewriting
reaches the innermost CALL first: `sq(p0)` is inlined into `g`'s body (as `p0 * p0`, which happens to
be correct since its argument is `p0` itself) before `g` is inlined. But `inline_call` is a library
function, and any pass that calls it directly on an outer CALL (a top-down inliner, a pass that inlines
only some functions) would hit the bug. Correctness should not depend on the caller's traversal order.

### Interpreter versus compiler

`run` is a complete interpreter for week-1 programs, written in about 80 lines of rules. How good an
interpreter is it?

```
n=   10  run()    17.35 ms   first [0, 12, 26]
n=  100  run()    47.65 ms   first [0, 102, 206]
n= 1000  run()   141.11 ms   first [0, 1002, 2006]
```

(`a * b + a` for vectors of length `n`.) About 140 µs per element: every element of every
intermediate value is a hash-consed node, every operation is a pattern match. numpy does all 1000
elements in 1.76 µs (`python -m timeit`). Nobody should run models this way.

But the interpreter has two jobs where speed is irrelevant. First, it is an **oracle**: a second,
independent implementation of the semantics. When the C backend of [Chapter 4](04-memory-and-c.md)
produces an answer, `run` can check it on small inputs; the course's graders use exactly this kind of
reference evaluator. Second, it shows that the *same machinery* that optimizes a program can execute
it. The line between "simplify" and "evaluate" is just how much is known: `symbolic` rewrites `x * 1`
to `x` because the rule is true for every `x`; `evaluate` rewrites `2 * 3` to `6` because everything
is known. A program with some PARAMs bound and some not is *partially evaluated* by the same rules
(section 2.5's `bind` example). Partial evaluation, taken seriously, is a way of deriving compilers
from interpreters (the Futamura projections); this book takes the more conventional route of
generating C, but keep the idea in mind when [Chapter 6](06-rangeify.md) turns shape computations
into loops using these same rules.

## Pitfalls

**1. CALL scoping (NOTES 2).** Section 2.6 in full. Because PARAMs are hash consed, the PARAM of an
inner function body is the same object as the PARAM of the outer one, and naive substitution while
inlining the outer call rewrote the inner body too. Symptoms: either a `UOpError` about missing
params when rebuilding the inner CALL (lucky), or a silently wrong value (98 instead of 72). Fix:
`call_params` and `inline_call` never descend into a nested CALL's `src[0]`. Bottom-up rewriting also
happens to inline inner calls first, which hid the bug in `run` until a test built nested calls by
hand.

**2. Recursion depth (NOTES 3).** The first `graph_rewrite` was the four-line recursive version. A
20,000-node ADD chain raised `RecursionError`. Fix: the explicit stack with three stages.
`test_deep_graph_no_recursion_error` is the regression test.

**3. Forgetting the memo table.** A recursive rewriter without `done` is exponential in the depth of
sharing (`2097151` rule lookups instead of `21` for a 21-node graph). It works on every small test
and hangs on the first real model. Memoize on the node; hash consing makes it O(1).

**4. Patterns compare args with `==`.** `UPat(Ops.CONST, arg=0)` matches `0`, `0.0`, `-0.0` and `False`;
`arg=1` matches `True`. A rule meant for i32 constants fires on bools and changes a node's dtype
(section 2.5's `where(c, 1, 0)` example). Add a `dtype=` constraint whenever a rule's correctness
depends on the type, and check that replacements preserve dtype and shape.

**5. `ctx` goes to every rule.** When `graph_rewrite` is called with a context, every rule function in
the matcher receives a `ctx=` keyword argument. Combining a context-using rule set with `symbolic`
fails with `TypeError: _fold() got an unexpected keyword argument 'ctx'`. Either run separate passes,
or give context-free rules a `ctx=None` parameter.

**6. Rules that undo each other.** `(a + b) * c → a*c + b*c` (distribute) and `a*c + b*c → (a+b) * c`
(factor) are each sound and together loop forever. The engine reports `graph_rewrite did not converge
(a rule loops?)` after `max_steps`. When it does, print the last few rewrites (a `Trace` matcher)
and look for a cycle. The fix is always to pick one direction as canonical.

**7. Order-dependent results.** Rule order is priority. Moving a rule (or concatenating matchers in
the other order) can change results, as `-0.0 + 0.0` showed. Treat the order of a rule list as part
of its meaning, and put the most specific rules first.

## Exercises

1. ★ **Predict the trace.** Using the `Trace` matcher from section 2.4 with `symbolic.rules`, predict
   every line printed for `graph_rewrite((p0 * 1) + (p0 * 1), Trace(symbolic.rules))` where `p0` is
   an i32 PARAM. Remember hash consing. Then run it.

2. ★ **Your first rule.** Write a matcher with one rule that rewrites `MAX(x, x) → x` and a test for
   it. (`symbolic` already has this rule; write it yourself, then compare.) Why is the source pattern a
   tuple and not a list, even though MAX is commutative?

3. ★ **Find the bug.** A student writes the rule
   `(UPat.var("x") * 0, lambda x: x.const_like(0))` for all dtypes. Give a float value of `x` for which
   it is unsound, and explain how `symbolic` avoids the problem.

4. ★★ **Type-preserving rewrites.** Add an optional check to your `graph_rewrite`: when a rule returns
   `m` for node `n`, assert `m.dtype == n.dtype and m.shape == n.shape`, with an error message naming
   the rule (hint: `PatternMatcher.rewrite` knows which rule fired). Run the whole test suite with the
   check on. Then catch the `where(c, 1, 0)` bug from section 2.5 with it.

5. ★★ **Top-down rewriting.** Write `graph_rewrite_topdown`, which tries rules on a node *before*
   rewriting its sources (and again after). Find a rule set and a program where it produces a
   different normal form than bottom-up, and one where it is much faster (hint: a rule that discards
   a large subtree, like `WHERE(True, x, y) → x`).

6. ★★ **Soundness by testing.** Write a function `check_rule(pat, fn, n=1000)` that generates random
   i32 inputs, builds random instances of the pattern, applies the rule, and compares `run(before)`
   and `run(after)` after binding PARAMs to random constants. Run it on every rule in `symbolic` (you
   will need to synthesize instances from patterns). Did you find any rule that is unsound for some
   int32 values? (Think about overflow.)

7. ★★ **Implement `inline_call` yourself**, then write a test with three levels of nesting where the
   outer argument mentions the outer PARAM (like section 2.6's `f`, `g`, `sq`), that fails for the
   naive version and passes for yours. Call `inline_call` directly on the middle CALL, not via `run`.

8. ★★ **A rewrite tracer.** Add a `DEBUG_REWRITE=1` environment variable to your engine that prints
   each rule that fires as `rule-name: before -> after` using `to_uir`-style one-line output. Use it to
   explain every step of `simplify(((x * 4) + 8) // 4)` for an i32 PARAM `x`.

9. ★★★ **Equality saturation.** A greedy rewriter commits to the first rule that fires, which is why
   order matters. *Equality saturation* (the `egg` library) instead applies all rules in all ways,
   recording equivalences in an e-graph, and extracts the best term at the end. Implement a tiny
   e-graph (union-find over hash-consed nodes, congruence closure by rebuilding) for the rules
   `x+0→x`, `x*1→x`, `x*2→x+x`, `(a+b)+c→a+(b+c)`, commutativity, and extract the smallest term.
   Show that commutativity, which loops in `graph_rewrite`, is harmless here.

10. ★★★ **Compile your patterns.** `UPat.match` interprets the pattern tree on every call. Write a
    function that turns a `UPat` into Python source code for a specialized matcher (nested `if`s on
    `u.op`, `u.src[0].op`, ..., with the permutations unrolled), `exec` it, and use it in
    `PatternMatcher`. Measure `simplify` on a large index expression before and after. (tinygrad
    compiles its patterns too; find its pattern compiler in `tinygrad/uop/` after you've tried.)

<details><summary>Solutions to ★ exercises</summary>

1. `p0 * 1` is built once (hash consing), so the root is `add(m, m)` with `m = mul(p0, 1)`:

   ```
     visit p0                           -> .
     visit 1                            -> .
     visit mul(p0, 1)                   -> p0
     visit add(p0, p0)                  -> mul(p0, 2)
     visit 2                            -> .
     visit mul(p0, 2)                   -> .
   ```

   The identity rule turns `m` into `p0`, which is already done, so it finishes immediately. The root
   is rebuilt as `add(p0, p0)` and the rule `xi + xi → x * 2` fires. The second use of `m` is never
   visited: it is the same node, already in `done`.

2. ```python
   pm = PatternMatcher([(UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("x"))), lambda x: x)])
   p = UOp(Ops.PARAM, (), ParamArg(0, dtypes.f32))
   assert graph_rewrite(p.maximum(p), pm) is p
   ```
   Both sub-patterns are the same variable, so both orders bind the same thing; a list would just try
   the same match twice.

3. `x = inf` (or NaN): `inf * 0` is NaN in IEEE arithmetic, not 0. `symbolic` restricts the rule to
   integer and bool dtypes: `(UPat.var("x", dtype=ints) * 0, ...)`, and `exec_alu` returns NaN for
   `inf * 0` when folding constants.

</details>

## Quiz

1. What is a normal form? What two properties of a rewrite system together guarantee that every term
   has exactly one?
2. Why does `graph_rewrite` rewrite a node's sources before trying rules on the node?
3. What does stage 2 of `graph_rewrite` do, and what goes wrong without it?
4. A rule returns a node that is structurally identical to its input. What does `PatternMatcher.rewrite`
   do, and why is the check O(1)?
5. Why is `done` (the memo table) correct? What property must every rule have for it to be?
6. What does `UPat(Ops.ADD, src=[UPat.var("x"), UPat.cvar("c")])` match that
   `UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c")))` does not?
7. In beta reduction of a CALL, why must substitution not enter a nested CALL's body? Why doesn't
   α-renaming (the usual fix in the lambda calculus) apply here?
8. `run` evaluates `[1,2,3] * [10,20,30]`. Name the two rules that fire, in order.

<details><summary>Answers</summary>

1. A term with no redex (no rule applies anywhere). Termination (some normal form is always reached)
   and confluence (all rewrite paths reach the same one).
2. So that rules see sources in normal form (simpler rules), and because many rules can only match
   after their sources are rewritten (the root of `(p0+2)*3 + (1+4)` matched nothing until `1+4`
   became `5`).
3. It records the normal form of a rule's *result* as the answer for the original node, after the
   result has been fully rewritten. Without it (one-pass rewriting), the new nodes a rule builds are
   never simplified: `(p0*2)*3` stays unfolded, and evaluation leaves `mul(1, 10)` nodes unevaluated.
4. It treats it as "no change" (`ret is not u`) and tries the next rule. With hash consing a
   structurally identical node *is* the same object, so the check is a pointer comparison.
5. Rewriting must be a pure function of the node: same node in, same normal form out. Rules must not
   depend on mutable global state; hash consing makes "same node" well-defined.
6. `3 + x` (constant on the left). A list source tries both orders.
7. PARAMs are hash consed, so the inner body's slot-0 PARAM is the same object as the outer body's;
   substituting into the inner body changes a different function's parameter. There are no names to
   rename (slots are positional, and all same-slot PARAMs are one node), so scoping must be respected
   structurally instead.
8. The STACK push-through rule (`_push_stack`) turns the MUL into a STACK of three scalar MULs, then
   `fold` (constant folding via `exec_alu`) evaluates each scalar MUL.

</details>

## Further reading

- Franz Baader and Tobias Nipkow, *Term Rewriting and All That*, Cambridge University Press, 1998. The
  standard textbook: termination orderings, confluence, critical pairs, Newman's lemma.
- Nachum Dershowitz and Jean-Pierre Jouannaud, "Rewrite Systems", in *Handbook of Theoretical Computer
  Science*, Vol. B, 1990. A dense survey of the theory.
- Eelco Visser, "A Survey of Rewriting Strategies in Program Transformation Systems", *Electronic
  Notes in Theoretical Computer Science* 57, 2001. Innermost, outermost, and programmable strategies
  in practical systems.
- Max Willsey et al., "egg: Fast and Extensible Equality Saturation", POPL 2021. The alternative to
  greedy rewriting (Exercise 9).
- Benjamin C. Pierce, *Types and Programming Languages*, MIT Press, 2002, chapters 5–6: the lambda
  calculus, substitution, variable capture, and de Bruijn indices.
- Neil D. Jones, Carsten K. Gomard and Peter Sestoft, *Partial Evaluation and Automatic Program
  Generation*, Prentice Hall, 1993. Interpreters, specializers, and the Futamura projections.
- tinygrad, `tinygrad/uop/ops.py` (<https://github.com/tinygrad/tinygrad>): the production
  `UPat`/`PatternMatcher`/`graph_rewrite`, from which `uopc`'s are simplified.
