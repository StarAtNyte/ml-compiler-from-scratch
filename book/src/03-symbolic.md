# 3. Constant folding and symbolic algebra

When rangeify ([Chapter 6](06-rangeify.md)) turns a reshape into loops, it writes index expressions such as
`((i*8 + j)//8) % 4`. That is just `i` with extra steps, and a C compiler will not always spot it: integer
division and modulo are expensive, and gcc has to prove a lot before it can delete them. Our own compiler
knows more than gcc does. It knows that `i` runs from 0 to 3 and `j` from 0 to 7, because it made those
loops. This chapter teaches the compiler to use that knowledge. We build three things: a precise
*reference semantics* for every ALU op, a *bounds analysis* that computes a range `[vmin, vmax]` for every
integer expression, and a *symbolic simplifier*, a rule set for the rewrite engine of
[Chapter 2](02-rewriting.md) that does high-school algebra on the IR. Together they turn
`((i*8+j)//8)%4` into `i`, decide `i < 4` is always true, and pick between C's fast `/` and a slower
floor-division helper. As a side effect they also fold constant programs all the way down to a value, which
is what the course means by "running a program by rewriting".

**What you'll build**

- `exec_alu`: the single source of truth for what `ADD`, `MUL`, `FLOORDIV`, `CAST` and friends compute,
  bit-exact with the C we will generate (int32 wraparound, float32 rounding, floor division).
- `fold`: one rewrite rule that evaluates ALU ops on constants.
- `bounds`/`vmin`/`vmax`: interval arithmetic over the IR, memoized on each node.
- `symbolic`: about forty rewrite rules for canonical form, identities, reassociation, distribution,
  division and modulo, recombination and comparisons, plus a catch-all that turns any expression whose
  bounds pin it to one value into that value.
- `simplify`: a memoized entry point that the rest of the compiler calls thousands of times.

**Where this fits.** [Chapter 1](01-uops.md) built the UOp and [Chapter 2](02-rewriting.md) built
`UPat`, `PatternMatcher` and `graph_rewrite`. This chapter is the first big *client* of the rewrite engine.
[Chapter 4](04-memory-and-c.md) uses `vmin` to choose how to render division, and
[Chapter 6](06-rangeify.md) calls `simplify` on every index it creates.

## 3.1 Why a compiler needs algebra

Start with the end product. Here is a kernel that copies a 4x8 array through a reshape to a flat 32-element
view and back. Rangeify's output index is mathematically `i*8+j`, but syntactically it is a tangle. We render
it twice, once raw and once after `simplify`:

```python
from uopc.uop import UOp, Ops, ParamArg, range_, sink
from uopc.dtype import dtypes
from uopc.symbolic import simplify
from uopc.render_c import render_c
def buf(slot, n, dtype=dtypes.f32): return UOp(Ops.BUFFER, (), ParamArg(slot, dtype, (n,)))
A, O = buf(0, 32), buf(1, 32)
i, j = range_(4, 0), range_(8, 1)
flat = i * 8 + j
idx = ((flat // 8) % 4) * 8 + flat % 8          # what a reshape (4,8)->(32,)->(4,8) produces
print(render_c(sink(O.index(flat).store(A.index(idx).load()).end(i, j))))
print(render_c(sink(O.index(flat).store(A.index(simplify(idx)).load()).end(i, j))))
```

The two function bodies (the shared prelude of helper functions is omitted from both):

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      int alu1 = ((ridx0*8)+ridx1);
      float val2 = data0[((((alu1/8)%4)*8)+(alu1%8))];
      data1[alu1] = val2;
    }
  }
}
```

```c
void kernel(float* restrict data0, float* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 8; ridx1++) {
      int alu1 = ((ridx0*8)+ridx1);
      float val2 = data0[alu1];
      data1[alu1] = val2;
    }
  }
}
```

On a toy this is cosmetic, since gcc can strength-reduce division by 8. On real kernels the stakes are higher.
Division by a non-power-of-two costs tens of cycles, and an index that *mentions* a loop variable it does
not really depend on changes where the compiler places loops and accumulators. Pitfall 12 in the development
notes, retold in [Chapter 6](06-rangeify.md), is a 250x slowdown caused by exactly that. Algebra is
how the compiler finds out what an expression *means*, as opposed to how it was written.

There is a second, older reason. The week-1 milestone asks you to "run" closed programs (only `CONST`,
`STACK`, `INDEX` and ALU ops) with nothing but rewriting. [Chapter 2](02-rewriting.md) built the `evaluate`
rules that push ALU ops through `STACK` and resolve `INDEX`. Once those rules expose a scalar op on two
constants, something has to *compute* it. That something is constant folding, and it needs a precise
answer to "what does `ADD` compute?"

## 3.2 Reference semantics: `exec_alu`

A compiler that folds `2 + 3` into `5` is making a promise: the program with the fold behaves exactly like
the program without it. Without the fold, the addition happens in compiled C on int32 or float32 values.
So the folder must compute what *C on this machine* would compute. Python's arithmetic is not that: Python
ints are unbounded, Python floats are 64-bit, and Python's `//` floors where C's `/` truncates.

`exec_alu` is the one place where the IR's arithmetic is defined:

@include uopc/symbolic.py#exec_alu

Every result passes through `truncate`, which brings a Python value into the exact value set of the dtype:

@include uopc/dtype.py#truncate

Read the two functions together, op by op, and compare with C:

| Op | `exec_alu` | C we generate | Notes |
|---|---|---|---|
| `ADD`, `MUL` on i32 | Python op, then wrap to 32 bits | `a+b`, `a*b` | Signed overflow is formally UB in C; see Pitfalls |
| `ADD`, `MUL` on f32 | f64 op, then round to f32 | `a+b` in `float` | One rounding of the exact result, as IEEE requires |
| `MUL` f32 `inf*0` | `nan` | `nan` | Python's `inf*0` is also nan, but the guard makes it explicit |
| `FLOORDIV`, `FLOORMOD` | Python `//`, `%` (floor) | `/`, `%` or helpers | Division by zero refuses to fold (`None`) |
| `CMPLT`, `CMPNE` | Python `<`, `!=` | `<`, `!=` | Result is `bool`, not truncated |
| `WHERE` | `b if c else d` | `c?b:d` | |
| `RECIP` | `1/x`, signed `inf` for 0 | `1.0f/x` | `1/-0.0 = -inf` in IEEE; Python would raise |
| `EXP2`, `LOG2`, `SQRT`, `SIN` | `math` functions with IEEE edge cases | `exp2f`, `log2f`, `sqrtf`, `sinf` | Python raises where IEEE returns inf or nan |
| `CAST` to i32 | `int(v)`, truncation toward 0 | `(int)v` | inf/nan to int is UB: refuse to fold |
| `CAST` to bool | `bool(v)` | `(_Bool)v` | |

Three design decisions are buried in that table.

**Return `None` rather than guess.** `exec_alu` returns `None` for division by zero and for casting a
non-finite float to int. `None` means "rule does not apply", so the folder simply leaves the op alone and
the runtime will do whatever the hardware does. Folding an undefined operation into a defined value would
make the program behave differently depending on whether the optimizer ran, which is the worst kind of
bug to chase.

**Round the result, every time.** Python computes `0.1 + 0.2` in double precision. C on `float` operands
computes it in single precision. `truncate(f32, x)` rounds through `struct.pack("f", x)`, which is how we
get the exact float32 result. For `+`, `*` and `/`, computing in double and then rounding to single gives
the correctly rounded single result: a double has more than twice the significand bits of a float
(53 vs 24), so double rounding cannot happen for one operation on float inputs.

**Floor, not truncation.** The IR's `FLOORDIV` and `FLOORMOD` are Python's operators: the quotient rounds
toward minus infinity and the remainder has the sign of the divisor. C's `/` rounds toward zero. That
choice costs the renderer some work ([Chapter 4](04-memory-and-c.md) emits a helper when it cannot prove
the operands are non-negative), and in exchange makes the division rules in section 3.5 true without
side conditions. Here is the semantics in action:

```text
>>> from uopc.uop import Ops
>>> from uopc.dtype import dtypes
>>> from uopc.symbolic import exec_alu
>>> exec_alu(Ops.FLOORDIV, dtypes.i32, [-7, 2]), exec_alu(Ops.FLOORMOD, dtypes.i32, [-7, 2])
(-4, 1)
>>> exec_alu(Ops.ADD, dtypes.i32, [2**31 - 1, 1])
-2147483648
>>> exec_alu(Ops.MUL, dtypes.i32, [65536, 65536])
0
>>> exec_alu(Ops.ADD, dtypes.f32, [0.1, 0.2]), 0.1 + 0.2
(0.30000001192092896, 0.30000000000000004)
>>> exec_alu(Ops.ADD, dtypes.f32, [16777216.0, 1.0]), 16777216.0 + 1.0
(16777216.0, 16777217.0)
>>> exec_alu(Ops.CAST, dtypes.i32, [-2.7]), exec_alu(Ops.CAST, dtypes.i32, [float("inf")])
(-2, None)
>>> exec_alu(Ops.MUL, dtypes.f32, [float("inf"), 0.0])
nan
>>> exec_alu(Ops.RECIP, dtypes.f32, [-0.0]), exec_alu(Ops.FLOORDIV, dtypes.i32, [5, 0])
(-inf, None)
>>> exec_alu(Ops.CMPLT, dtypes.bool, [3, 4])
True
```

`16777216.0` is 2<sup>24</sup>, the point where float32 runs out of integers: the next float above it is
`16777218.0`, so adding 1 rounds back down. Python's double happily says `16777217.0`. A folder that skipped
`truncate` would compute a number the generated C program can never produce.

The picture to keep in your head for floor versus truncation is a number line cut into buckets. Floor
division by 4 cuts it into buckets of exactly four integers each, everywhere. Truncating division has one
oversized bucket: everything from -3 to 3 lands on 0.

@diagram 03-floor-vs-trunc

### Checking the semantics against the machine

A reference semantics is only worth something if it matches the hardware, so test it against the
hardware. The script below (a scratch script, not part of the repo) builds a loopless kernel per op with
the tools of [Chapter 4](04-memory-and-c.md), runs it on 1000 random operand pairs, and compares every
output with `exec_alu`:

```python
import random, numpy as np
from uopc.uop import UOp, Ops, ParamArg, sink
from uopc.dtype import dtypes
from uopc.symbolic import exec_alu
from uopc.runtime import run_c, np_dtype

def buf(slot, n, dtype): return UOp(Ops.BUFFER, (), ParamArg(slot, dtype, (n,)))

N = 1000
random.seed(0)
cases = [(Ops.ADD, dtypes.i32), (Ops.MUL, dtypes.i32), (Ops.FLOORDIV, dtypes.i32),
         (Ops.FLOORMOD, dtypes.i32), (Ops.ADD, dtypes.f32), (Ops.MUL, dtypes.f32)]
for op, dt in cases:
  a, b = buf(0, N, dt), buf(1, N, dt)
  out = buf(2, N, dt)
  s = sink(*[out.index(i).store(a.index(i).load().alu(op, b.index(i).load())) for i in range(N)])
  if dt == dtypes.i32:
    A = np.array([random.randint(-2**31, 2**31 - 1) for _ in range(N)], np.int32)
    B = np.array([random.choice([random.randint(-9, 9) or 1, random.randint(-2**31, 2**31 - 1)]) for _ in range(N)], np.int32)
  else:
    A = np.array([random.uniform(-1e6, 1e6) for _ in range(N)], np.float32)
    B = np.array([random.uniform(-1e6, 1e6) for _ in range(N)], np.float32)
  O = np.zeros(N, np_dtype(dt))
  run_c(s, {0: A, 1: B, 2: O})
  ref = [exec_alu(op, dt, [x.item(), y.item()]) for x, y in zip(A, B)]
  bad = sum(r != o.item() for r, o in zip(ref, O))
  print(f"{op.name:9s} {dt.name}: {bad} mismatches / {N}")
```

```text
ADD       i32: 0 mismatches / 1000
MUL       i32: 0 mismatches / 1000
FLOORDIV  i32: 0 mismatches / 1000
FLOORMOD  i32: 0 mismatches / 1000
ADD       f32: 0 mismatches / 1000
MUL       f32: 0 mismatches / 1000
```

The integer operands are chosen so that most additions and multiplications overflow, so this also checks
the wraparound. This kind of *differential test*, where two implementations that should agree are run on
random inputs, is the cheapest strong test a compiler writer has. [Appendix B](17-appendix-testing.md)
returns to it.

## 3.3 Constant folding

With `exec_alu` in hand, constant folding is one rule:

@include uopc/symbolic.py#_fold

@include uopc/symbolic.py#fold

The pattern `UPat(ALU, name="x")` matches any ALU op (the `PatternMatcher` indexes it under every ALU op,
see [Chapter 2](02-rewriting.md)). The function checks that every source is a `CONST`, evaluates, and builds
the result with `UOp.const(r, x.dtype)`. Because UOps are hash consed, the result *is* the canonical node
for that constant, so folding `2+3` and writing `5` by hand give the very same object.

Two details matter:

- **Only scalars fold.** A shaped op like `STACK(1,2) + STACK(3,4)` is not handled here. The `evaluate`
  rules of [Chapter 2](02-rewriting.md) first push the ADD inside the STACK
  (`STACK(1+3, 2+4)`), and then this rule folds the scalar pieces. Keeping the folder scalar keeps it
  trivially correct.
- **Folding can refuse.** If `exec_alu` returns `None`, `_fold` returns `None` and the node stays.
  `graph_rewrite` treats `None` as "no rule fired" and moves on.

Since `graph_rewrite` is bottom-up and re-rewrites every rule result, folding the leaves of a deep constant
expression produces new constants, which make their parents foldable, and so on up to the root. That is all
the machinery needed to evaluate `(2+3)*(4+5)`, or the 20,000-deep chain in
`test_deep_graph_no_recursion_error`.

## 3.4 Bounds analysis: interval arithmetic

Folding only helps when the inputs are literally constant. Most index expressions contain loop variables,
which are not constant, but they are *bounded*: a `RANGE` with end 4 takes the values 0, 1, 2, 3 and
nothing else. If we know a range for every subexpression we can prove facts like "this division by 8 is
always 0" or "this comparison is always true".

### The idea

Represent what we know about an integer expression `e` as an interval `[lo, hi]` with the guarantee
`lo <= e <= hi` for every possible execution. Then define, for each operation, how intervals combine. This
is *interval arithmetic*, the simplest instance of what program analysis calls *abstract interpretation*:
run the program on abstract values (here, intervals) instead of concrete ones.

The one property that matters is **soundness**: the true value is always inside the interval. Tightness is
nice but optional. `[-2147483648, 2147483647]` is always a correct answer for an i32 expression, it just
says nothing. The rules:

- **Constant** `c`: `[c, c]`.
- **Range** with end `n`: `[0, n-1]`.
- **Addition** is monotone in both arguments, so the extremes of the sum come from the extremes of the
  inputs: `[a0, a1] + [b0, b1] = [a0+b0, a1+b1]`.
- **Multiplication** is not monotone when signs are mixed (`-3 * -3` is large), but it is *bilinear*, so
  its extremes over a box lie at the corners. Compute all four products `a0*b0, a0*b1, a1*b0, a1*b1` and
  take min and max.
- **Max** is monotone: `[max(a0,b0), max(a1,b1)]`.
- **Floor division by a positive constant** `c` is monotone (non-decreasing): `[a0//c, a1//c]`.
- **Floor modulo by a positive constant** `c` always lands in `[0, c-1]`. If the input already lies in
  `[0, c-1]` the modulo is the identity, so the input's interval is tighter and we return it.
- **Comparison** `a < b` is a boolean. Its interval is over `{False < True}`: if `a1 < b0` it is always
  true, `[True, True]`; if `a0 >= b1` always false; otherwise `[False, True]`, unknown.
- **WHERE** takes the true branch's interval if the condition is always true, the false branch's if always
  false, and the union (hull) otherwise.
- **CAST from bool** to int maps `False/True` to `0/1`.

Everything else (floats, shaped values, ops we have no rule for) gets the full range of its dtype.

### The code

@include uopc/symbolic.py#_bounds

The rules above are almost line for line. Notice the special cases: `RANGE` uses `max(0, vmax(end) - 1)`
so that a zero-trip loop still has a well-formed interval, and `MUL` gives up if a bound is infinite. That check is defensive: float expressions return the full range before reaching it, and `0 * inf` would be nan.

The public entry point adds two things:

@include uopc/symbolic.py#bounds

**Memoization on the node.** UOps are immutable and hash consed (see [Chapter 1](01-uops.md)), so the bounds
of a node never change and every structurally equal expression is the same object. The result can be cached
directly on the node, in its `__dict__`. Without the cache, `bounds` of a deep expression would recompute
shared subexpressions exponentially many times.

**Overflow means "give up".** Our intervals are computed with unbounded Python ints. If an i32 interval
escapes `[-2^31, 2^31-1]`, the real computation *might* wrap around, and then the true value could be
anywhere. Returning the full range is the only sound answer. This one line is why everything else in the
analysis can ignore wraparound.

```text
>>> from uopc.uop import range_
>>> from uopc.symbolic import vmin, vmax, bounds
>>> i, j = range_(4, 0), range_(8, 1)
>>> bounds(i), bounds(j)
((0, 3), (0, 7))
>>> bounds(i * 8), bounds(i * 8 + j)
((0, 24), (0, 31))
>>> bounds((i * 8 + j) // 8), bounds((i * 8 + j) % 8)
((0, 3), (0, 7))
>>> bounds((i + -2) * (j + -3))
(-8, 6)
>>> bounds((i + -2) // 3), bounds((i + -2) % 3)
((-1, 0), (0, 2))
>>> bounds(i < 4), bounds(i < 2)
((True, True), (False, True))
>>> bounds(i + i * -1)
(-3, 3)
```

Check one by hand. `(i-2)*(j-3)` has factors in `[-2, 1]` and `[-3, 4]`. The corners are `6, -8, -3, 4`,
so `[-8, 6]`. Both extremes are attained (`i=0, j=7` and `i=0, j=0`), so the interval is exact here.

The last line shows the classic weakness, called the *dependency problem*. `i - i` is always 0, but interval
arithmetic treats the two occurrences of `i` as independent variables, each anywhere in `[0, 3]`, and
concludes `[-3, 3]`. The interval is sound, just loose. The cure is to simplify algebraically *first*
(`x + x*-1 -> 0` is one of our rules) and ask for bounds afterwards. Rules and bounds work as a team.

The figure shows bounds flowing up the expression tree of `(i*8 + j) // 8`. Each node's interval is computed
from its children's, exactly the order in which `graph_rewrite` visits them.

@diagram 03-interval

### What bounds are used for

Bounds analysis shows up all over the compiler:

1. **Deciding comparisons.** `i < 4` with `i` in `[0, 3]` is `True`. Masks and guards that rangeify
   generates for padding collapse when they are always true.
2. **Deleting division and modulo.** If `0 <= x < c` then `x // c == 0` and `x % c == x`.
3. **Collapsing anything constant.** If `vmin(e) == vmax(e)`, `e` is a constant (section 3.5).
4. **Choosing C operators.** The renderer emits plain `/` and `%` only if `vmin(numerator) >= 0`, otherwise
   a floor-division helper ([Chapter 4](04-memory-and-c.md)).
5. **Placing loops.** Bounds-driven simplification removes spurious mentions of loop variables, which
   changes where the linearizer ([Chapter 5](05-loops.md)) puts code.

## 3.5 The symbolic rule set

Now the algebra. `symbolic` is `fold` plus one more `PatternMatcher`. We read it in groups. The pattern
variables are module-level `UPat`s, reused by many rules:

@include uopc/symbolic.py:138-141

`xi` matches only i32 values. That one restriction carries a lot of weight, as we will see: most of
algebra is only true for integers.

To show what each group does we use a ten-line printer and a helper that prints an expression next to its
simplified form. Neither is part of `uopc`, they are just for this chapter (`x` and `y` are i32 `PARAM`s,
`f` is an f32 `PARAM`, `i` and `j` are ranges with ends 4 and 8):

```python
from uopc.uop import UOp, Ops, ParamArg, range_
from uopc.dtype import dtypes
from uopc.symbolic import simplify

SYM = {Ops.ADD: "+", Ops.MUL: "*", Ops.FLOORDIV: "//", Ops.FLOORMOD: "%", Ops.CMPLT: "<", Ops.CMPNE: "!="}
def fmt(u: UOp) -> str:
  if u.op is Ops.CONST: return repr(u.arg)
  if u.op is Ops.RANGE: return "ijklmn"[u.arg.idx]
  if u.op is Ops.PARAM: return "xyf"[u.arg.slot]
  if u.op in SYM: return f"({fmt(u.src[0])}{SYM[u.op]}{fmt(u.src[1])})"
  if u.op is Ops.MAX: return f"max({fmt(u.src[0])}, {fmt(u.src[1])})"
  if u.op is Ops.WHERE: return f"({fmt(u.src[0])} ? {fmt(u.src[1])} : {fmt(u.src[2])})"
  return f"{u.op.name.lower()}({', '.join(fmt(s) for s in u.src)})"

x = UOp(Ops.PARAM, (), ParamArg(0, dtypes.i32)); y = UOp(Ops.PARAM, (), ParamArg(1, dtypes.i32))
f = UOp(Ops.PARAM, (), ParamArg(2, dtypes.f32))
i, j = range_(4, 0), range_(8, 1)
def show(e): print(f"{fmt(e):34s} -> {fmt(simplify(e))}")
```

### Canonical form: constants on the right

@include uopc/symbolic.py:143-146

`3 + x` and `x + 3` are the same value. If both forms could occur, every rule that wants "something plus a
constant" would need two patterns, and rules that match two levels deep would need four. Instead, the first
rule picks a *canonical form*: in commutative ops, the constant goes on the right. Every later rule can then
assume `x + c`, never `c + x`.

`UPat` can match commutative ops in any order (a `list` src tries all permutations, see
[Chapter 2](02-rewriting.md)), and the sugar `x + 0` builds exactly such a list pattern. So the canonical
form is not strictly needed for correctness. It is needed for *meeting*: two constants can only fold if
some rule brings them next to each other, and that is much easier when every constant is known to sit on
the right. The guard `y.op is not Ops.CONST` stops the rule from swapping two constants forever (that case
belongs to `fold`).

### Identities

@include uopc/symbolic.py:147-161

These are the rules you learned before algebra: `x+0 = x`, `x*1 = x`, `x*0 = 0`, `x//1 = x`, `x%1 = 0`,
`max(x,x) = x`, a `WHERE` with equal branches is that branch, a `WHERE` with a constant condition is one of
its branches, and a `WHERE` nested inside a `WHERE` on the same condition can skip the inner test. A cast
to the same dtype is a no-op. `b != False` is `b`. `x < x` and `x != x` are false *for integers*.
`x + x` becomes `x*2`, and `x + x*-1` (which is how `x - x` is spelled, since `__sub__` builds
`x + (-x)` and `-x` is `x * -1`) becomes 0.

```text
(3+x)                              -> (x+3)
(2*x)                              -> (x*2)
(x*1)                              -> x
(x*0)                              -> 0
(f*0.0)                            -> (f*0.0)
(f+0.0)                            -> f
(x//1)                             -> x
(x%1)                              -> 0
((x<y) ? x : x)                    -> x
((x<y) ? ((x<y) ? x : 5) : y)      -> ((x<y) ? x : y)
(x+x)                              -> (x*2)
(x+(x*-1))                         -> 0
```

Look at the floats. `f*0.0` does *not* simplify, because in IEEE arithmetic `inf*0 = nan`, `nan*0 = nan`
and `-1.0*0.0 = -0.0`. The rule `x*0 -> 0` is guarded by `dtype=ints`. In the same spirit, `x < x` is only
false for integers: `nan < nan` is false too, but `nan != nan` is *true*, so `x != x -> False` would be
wrong for floats.

`f + 0.0 -> f` *does* fire, and it is not quite exact: `-0.0 + 0.0` is `+0.0` in IEEE, so the rewrite
turns a `+0.0` result into `-0.0` when `f` is `-0.0`. The comment in the source says "we accept that". Our
compiler also tells gcc `-fno-signed-zeros` ([Chapter 4](04-memory-and-c.md)), so the C side makes the same
trade. This is a policy choice, and every ML compiler makes it. Write such choices down.

### Reassociation, and why only for integers

@include uopc/symbolic.py:162-165

Pull constants outward until they meet: `(x+3)+4 -> x+7`, `(x*4)*3 -> x*12`, and `(x+3)*4 -> x*4 + 12`.
For i32 these are true even with wraparound. Integers modulo 2<sup>32</sup> form a ring, and associativity
and distributivity hold in every ring.

Float addition is not associative. With f32 semantics from `exec_alu`:

```text
>>> A = lambda a, b: exec_alu(Ops.ADD, dtypes.f32, [a, b])
>>> A(A(1.0, 1e8), -1e8), A(1.0, A(1e8, -1e8))
(0.0, 1.0)
```

Near 10<sup>8</sup> the spacing between float32 values is 8, so `1.0 + 1e8` rounds back to `1e8` and the 1 is
lost. Reassociating would change the answer from 0 to 1. Hence `xi`, not `x`, in every reassociation rule:

```text
((x+3)+4)                          -> (x+7)
((x*4)*3)                          -> (x*12)
((x+3)*4)                          -> ((x*4)+12)
((x+1)+y)                          -> ((x+y)+1)
((x+y)*4)                          -> ((x*4)+(y*4))
((x*3)+(x*5))                      -> (x*8)
((f+1.0)+2.0)                      -> ((f+1.0)+2.0)
```

(ML compilers do reassociate floats when they vectorize reductions, but they do it deliberately in one
place, with a flag: see `-fassociative-math` in [Chapter 10](10-cpu-performance.md). The simplifier is not
that place.)

### Distribution and collecting like terms

@include uopc/symbolic.py:166-170

The first rule distributes a constant over a sum: `(a+b)*c -> a*c + b*c`. Usually distributing makes
expressions *bigger*, so why do it? Because the division and modulo rules below look at the individual
terms of a sum. `((x+y)*4) // 4` can only be seen as "a multiple of 4, divided by 4" after the product is
spread over the terms. Development note 14 records the cost of not having this rule: GPT attention kernels
kept chains like `((x/4096)%4)` in their inner loops.

The second rule floats a constant out of a sum, `(x+c) + y -> (x+y) + c`, so that the constant reaches the
outermost addition, where it can meet another constant. The last two collect like terms:
`x*3 + x*5 -> x*8` and `x*3 + x -> x*4`.

### Division and modulo

@include uopc/symbolic.py:171-178

This is the heart of index simplification. The first three rules handle nested division and modulo by
constants:

- `(x // c1) // c2 = x // (c1*c2)` for positive `c1, c2`. Write `x = q*c1*c2 + r` with
  `0 <= r < c1*c2`. Then `x // c1 = q*c2 + r // c1`, where `0 <= r // c1 < c2`, and dividing that by `c2`
  gives `q`.
- `(x % c1) % c2 = x % c2` when `c2` divides `c1`. Taking the remainder by `c1` subtracts a multiple of
  `c1`, which is a multiple of `c2`, and that does not change the remainder mod `c2`.
- `(x % c1) // c2 = (x // c2) % (c1 // c2)` when `c2` divides `c1`. This is "the middle digit": think
  of `x` in mixed radix with digit sizes `c2` and `c1/c2`. This one is exercise 6.

The next two call `_div_sum`, which splits a sum into terms, and the last two use bounds to delete division
outright. The helpers:

@include uopc/symbolic.py#_divides

@include uopc/symbolic.py#_split_terms

@include uopc/symbolic.py#_div_sum

`_divides(u, c)` returns `u/c` if every term of `u` is visibly a multiple of `c` (a constant multiple, or a
product with a constant multiple), and `None` otherwise. `_div_sum` partitions the terms of `x` into those
divisible by `c` (`quot`) and the rest. The rules it implements are:

```text
(a*c + b) // c  =  a + b // c
(a*c + b) %  c  =  b % c
```

**Why these are always true for floor division.** Floor division is defined by
`x = (x // c)*c + x % c` with `0 <= x % c < c` (for `c > 0`). Let `q = b // c` and `r = b % c`, so
`b = q*c + r` with `0 <= r < c`. Then `a*c + b = (a + q)*c + r`, and since `0 <= r < c`, uniqueness of
floor division says `(a*c + b) // c = a + q` and `(a*c + b) % c = r`. Nothing was assumed about the sign of
`a` or `b`.

**Why they are false for truncating division.** Take `a = 1, b = -1, c = 4`. Then `a*c + b = 3`, and
`3 / 4 = 0` in C. But `a + b/c = 1 + (-1/4) = 1 + 0 = 1`, because C truncates `-0.25` to `0`. The rule is
only true for truncation when `b >= 0`, so a compiler with C semantics would need a bounds check before
every application. Look at the number-line figure again: truncation's oversized bucket around zero is
exactly what breaks "shift by a whole bucket". This is development note 4: floor semantics make the IR's
algebra *unconditional*, and the renderer pays a small price instead.

```text
((x*4)//4)                         -> x
(((x*6)+4)%2)                      -> 0
(((x*6)+4)//2)                     -> ((x*3)+2)
(((x*4)+y)//4)                     -> (x+(y//4))
(((x*4)+y)%4)                      -> (y%4)
(((i*8)+j)//2)                     -> ((i*4)+(j//2))
((x//4)//2)                        -> (x//8)
((x%12)%4)                         -> (x%4)
((x%12)//4)                        -> ((x//4)%3)
(j//8)                             -> 0
(j%8)                              -> j
```

The last two lines are the bounds-based rules: `j` lies in `[0, 7]`, so `j // 8 = 0` and `j % 8 = j`.
Unlike `_div_sum`, these need bounds, and they are what makes index simplification work on loop variables.

### Recombination

@include uopc/symbolic.py:179-181

The definition of floor division, read backwards: `(x//c)*c + x%c = x`. Unraveling a flat index into
digits and raveling it back produces exactly this shape, so it appears whenever a reshape round-trips. The
second rule handles the same pattern with an extra term on the left. The `list` src means the two addends
may come in either order.

```text
(((x//4)*4)+(x%4))                 -> x
(((((i*8)+j)//8)*8)+(((i*8)+j)%8)) -> ((i*8)+j)
```

### Comparisons

@include uopc/symbolic.py:182-184

@include uopc/symbolic.py#_lt_mul

Move constants across a comparison: `x + c1 < c2` is `x < c2 - c1` (true over the integers; we are again
trusting there is no overflow). For multiplication, `x*c1 < c2` with `c1 > 0` is `x < ceil(c2/c1)`. Proof:
`x*c1 < c2` holds iff `x < c2/c1` over the reals, and for an integer `x` that is iff
`x < ceil(c2/c1)`. The expression `-(-c2 // c1)` is the standard ceiling-division idiom with floor
division. Negative `c1` would flip the inequality, so `_lt_mul` declines.

```text
((x+3)<10)                         -> (x<7)
((x*4)<10)                         -> (x<3)
((x*-4)<10)                        -> ((x*-4)<10)
(i<4)                              -> True
((i+j)<11)                         -> True
(i!=5)                             -> True
cast((i<5))                        -> 1
```

The last four did not come from comparison rules at all. They came from the final rule.

### The catch-all: `_const_bounds`

@include uopc/symbolic.py:185-187

@include uopc/symbolic.py#_const_bounds

Any integer or boolean scalar ALU expression whose interval has collapsed to a point *is* that point.
`i + j < 11` has `i + j` in `[0, 10]`, so the comparison is `[True, True]`, so it is `True`. This one rule
subsumes dozens of special cases. It also makes the analysis compose: once a subexpression becomes a
constant, the rules above it can fire.

It is last on purpose. `PatternMatcher.rewrite` tries rules in list order and returns the first result that
is not `None`. Cheap structural rules go first, and the comparatively expensive bounds query only runs when
nothing else applied.

### A rewrite, step by step

How do the rules cooperate? Wrap the matcher to print every rule that fires (a scratch subclass, not in the
repo) and simplify `((i*8 + j)//8) % 4`:

```python
from uopc.rewrite import PatternMatcher, graph_rewrite
from uopc.symbolic import symbolic

class Traced(PatternMatcher):
  def rewrite(self, u, ctx=None):
    r = super().rewrite(u, ctx)
    if r is not None: print(f"  {fmt(u):28s} => {fmt(r)}")
    return r

e = ((i * 8 + j) // 8) % 4
print(fmt(e))
print("=", fmt(graph_rewrite(e, Traced(symbolic.rules))))
```

```text
((((i*8)+j)//8)%4)
  (((i*8)+j)//8)               => ((i*1)+(j//8))
  (i*1)                        => i
  (j//8)                       => 0
  (i+0)                        => i
  (i%4)                        => i
= i
```

Bottom-up, `i*8+j` has nothing to do. At `//8`, `_div_sum` sees the term `i*8` divisible by 8 and the
remainder `j`, and produces `i*1 + j//8`. That result is itself rewritten (that is the "re-rewrite" step of
`graph_rewrite`): `i*1 -> i` by an identity, `j//8 -> 0` by bounds, `i+0 -> i`. Finally `i % 4 -> i`
because `i` lies in `[0, 3]`. No single rule knows how to simplify the whole expression, but five small
rules, each obviously correct, do it together.

The same machinery inverts any mixed-radix ravel. Take a `(2, 3, 4)` array with indices `i, j, k`, ravel
them into a flat index, and pull each digit back out, then reshape the array to `(4, 6)`:

```python
i, j, k = range_(2, 0), range_(3, 1), range_(4, 2)
flat = (i * 3 + j) * 4 + k
for name, e in [("flat", flat), ("k", flat % 4), ("j", (flat // 4) % 3), ("i", flat // 12),
                ("row of (4,6)", flat // 6), ("col of (4,6)", flat % 6)]:
  print(f"{name:12s} {fmt(e):30s} -> {fmt(simplify(e))}")
```

```text
flat         ((((i*3)+j)*4)+k)              -> (((i*12)+(j*4))+k)
k            (((((i*3)+j)*4)+k)%4)          -> k
j            ((((((i*3)+j)*4)+k)//4)%3)     -> j
i            (((((i*3)+j)*4)+k)//12)        -> i
row of (4,6) (((((i*3)+j)*4)+k)//6)         -> ((i*2)+(((j*4)+k)//6))
col of (4,6) (((((i*3)+j)*4)+k)%6)          -> (((j*4)+k)%6)
```

The digits come back exactly. The reshape gives the honest answer, `i*2 + (j*4+k)//6` for the row and
`(j*4+k)%6` for the column, which no rule should simplify further since `j*4+k` really does straddle
rows of 6.

## 3.6 Termination and confluence

A term rewriting system ([Chapter 2](02-rewriting.md)) has two classic worries: does it stop
(*termination*), and does the answer depend on which rule fires first (*confluence*)?

**Termination.** Most of our rules shrink the expression or keep its size and push constants rightward and
outward. You can make this precise with a *measure*, a number that strictly decreases with every rewrite:
node count first, then the number of constants that are not in canonical position. The distribution rule is
the troublemaker, since `(a+b)*c -> a*c + b*c` grows the term. It still terminates because it pushes
multiplications strictly *below* additions, and no rule pushes them back up.

Add the reverse rule, `a*c + b*c -> (a+b)*c` ("factoring"), and that argument dies:

```python
xi, yi, c = UPat.var("x", dtype=dtypes.i32), UPat.var("y", dtype=dtypes.i32), UPat.cvar("c")
factor = PatternMatcher([(UPat(Ops.ADD, src=[xi * c, yi * c]), lambda x, y, c: (x + y) * c)])
graph_rewrite(x * 4 + y * 4, symbolic + factor, max_steps=100_000)
```

```text
RuntimeError: graph_rewrite did not converge (a rule loops?)
```

Distribution and factoring undo each other forever. `graph_rewrite` counts steps and raises instead of
hanging, which turns a silent infinite loop into a clear error. When you add a rule, ask what it makes
smaller.

**Confluence.** Our system is not confluent in general. Different rule orders can reach different normal
forms that are equal as functions but not as trees, for instance `x*4 + y*4` and `(x+y)*4` if both forms
could be reached. We do not need confluence for *correctness*, since every rule is an equivalence, so any
normal form computes the right value. We need it for *quality* and for caching: hash consing and the
kernel cache of [Chapter 7](07-kernelize.md) work best when equal expressions end up as the same tree. The
canonical-form rule and the fixed rule order buy most of that in practice.

## 3.7 Memoized `simplify`

The rest of the compiler does not call `graph_rewrite(u, symbolic)` directly. It calls `simplify`:

@include uopc/symbolic.py#simplify

Rewriting is a pure function of the input node, and nodes are hash consed, so the result can be cached in
a dict keyed by node. The result also maps to itself, since a normal form simplifies to itself. The cache
is cleared wholesale when it grows past 200,000 entries, a crude but adequate bound on memory.

Why bother? Rangeify simplifies every index it builds, and the same subexpressions recur across thousands of
nodes and many kernels. Development note 22 records that compiling the GPT graph for 12 steps took 163 s
before two fixes, identity-based `__eq__`/`__hash__` and memoizing `simplify`, and 47 s after. A
micro-benchmark: ravel six loop indices into one flat index, unravel it again, and combine the digits into a
46-node expression.

```python
import time
from uopc.uop import range_
from uopc.symbolic import simplify
from uopc.rewrite import graph_rewrite
from uopc.symbolic import symbolic
rs = [range_(n, i) for i, n in enumerate((2, 3, 4, 5, 6, 7))]
flat = 0
for r, n in zip(rs, (2, 3, 4, 5, 6, 7)): flat = flat * n + r       # ravel 6 indices
idxs = []
for n in (7, 6, 5, 4, 3, 2): idxs.append(flat % n); flat = flat // n   # unravel again
e = idxs[0]
for t in idxs[1:]: e = e * 10 + t
for name, f in [("graph_rewrite", lambda: graph_rewrite(e, symbolic)), ("simplify #1", lambda: simplify(e)), ("simplify #2", lambda: simplify(e))]:
  t0 = time.perf_counter(); r = f(); print(f"{name:14s} {(time.perf_counter()-t0)*1e3:8.3f} ms  nodes {len(e.toposort())} -> {len(r.toposort())}")
```

```text
graph_rewrite     8.400 ms  nodes 46 -> 27
simplify #1       7.329 ms  nodes 46 -> 27
simplify #2       0.003 ms  nodes 46 -> 27
```

(The first call to `simplify` is a little faster than the raw `graph_rewrite` before it only because the
bounds caches on the nodes are already warm.)

## 3.8 Testing an algebra

Every rule above is "obviously correct", and so was every buggy rule ever written. The good test for a
simplifier is a *property test*: generate random expressions, simplify them, and check on random inputs
that the simplified expression computes the same value. While we are at it, check that `bounds` is sound,
that is, every concrete value lies inside the computed interval. A scratch version:

```python
import random, time
from uopc.uop import UOp, Ops, range_
from uopc.symbolic import simplify, vmin, vmax

def ev(u, env):
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.RANGE: return env[u]
  a, b = (ev(s, env) for s in u.src)
  return {Ops.ADD: a + b, Ops.MUL: a * b, Ops.FLOORDIV: a // b if b else 0, Ops.FLOORMOD: a % b if b else 0,
          Ops.CMPLT: a < b, Ops.MAX: max(a, b)}[u.op]

rs = [range_(n, i) for i, n in enumerate((3, 5, 8, 16))]
def gen(d):
  if d == 0 or random.random() < 0.25: return random.choice(rs) if random.random() < 0.6 else UOp.const(random.randint(-6, 6))
  op = random.choice(["+", "*", "//", "%", "+", "*"])
  a = gen(d - 1)
  if op in ("//", "%"): return a // random.randint(1, 9) if op == "//" else a % random.randint(1, 9)
  b = gen(d - 1) if op == "+" else UOp.const(random.randint(-4, 6))
  return a + b if op == "+" else a * b

random.seed(1)
n_exprs, wrong, unsound, smaller = 3000, 0, 0, 0
t0 = time.perf_counter()
for _ in range(n_exprs):
  e = gen(5); s = simplify(e)
  smaller += len(s.toposort()) < len(e.toposort())
  for _ in range(20):
    env = {r: random.randrange(r.src[0].arg) for r in rs}
    v = ev(e, env)
    wrong += v != ev(s, env)
    unsound += not (vmin(e) <= v <= vmax(e))
print(f"{n_exprs} random expressions, {n_exprs*20} evaluations: {wrong} wrong, {unsound} out of bounds, "
      f"{smaller} got smaller ({time.perf_counter()-t0:.1f} s)")
```

```text
3000 random expressions, 60000 evaluations: 0 wrong, 0 out of bounds, 1880 got smaller (2.7 s)
```

Three seconds of runtime checks every rule in combination with every other rule, which no hand-written
test can match. Note what the generator *avoids*: divisors are positive constants and values stay small, so
nothing overflows. That is the domain the rules are designed for (see Pitfalls). A generator that produced
huge constants would find counterexamples, and that is a feature: it tells you exactly where your
assumptions are. Exercise 5 asks you to build a stronger version.

## 3.9 Relation to computer algebra systems

What we built is a tiny *computer algebra system* (CAS). Real ones such as Mathematica, SymPy and Maxima
face the same questions on a larger scale:

- **Normal forms.** A CAS tries to put expressions in a canonical form such that two expressions are equal
  iff their normal forms are identical. For polynomials over the integers this is achievable: expand
  everything and sort the monomials. Once you add floor division and modulo, no simple complete normal form
  exists. Our rules are a *heuristic* simplifier: sound, but not complete.
- **Decision procedures.** Deciding "is this expression always equal to that one" for integer arithmetic
  with `+`, constant multiplication, comparisons and division by constants is *Presburger arithmetic*,
  which is decidable but expensive. SMT solvers such as Z3 decide it routinely. Polyhedral compilers
  (isl, used in Polly and PPCG) use exactly this theory for loop analysis. We trade completeness for speed:
  `simplify` must run thousands of times per compile.
- **Incompleteness, concretely.** Here are true identities our rules miss:

```text
((x*2)//4)                         -> ((x*2)//4)
max(i, 0)                          -> max(i, 0)
((x+1)*(x+1))                      -> ((x+1)*(x+1))
```

`(x*2)//4` is `x//2` (divide numerator and divisor by their gcd). `max(i, 0)` is `i` since `vmin(i) = 0`.
`(x+1)*(x+1)` is not wrong as written, but there is no product-of-sums expansion. Each is a reasonable rule
to add if a real kernel needs it (exercises 2 and 7). The discipline is: add rules when a real kernel needs
them, test each with the property test, and keep the set small enough to reason about termination.

tinygrad's `symbolic.py` is the industrial-strength version of this file and a good next read. It has the
same structure (folding, bounds, div/mod rules, a `_min_max` per node) and many more rules.

## 3.10 The payoff in rangeify

The chain from this chapter to performance is direct. [Chapter 6](06-rangeify.md) lowers movement ops by
pushing index expressions through them. A `RESHAPE` from `(N, N)` to `(N*N,)` and back produces
`((i*N + j)//N) % N` for the row, which is `i` after `simplify`, but syntactically it mentions `j`. The reduce
lowering places an accumulator in the loops its index *mentions*. With the raw index, the max and sum
reductions of fused attention were nested inside the `j` loop, making an O(N<sup>2</sup>) computation
O(N<sup>3</sup>). Simplifying indices at every reshape took fused attention at N=512 from 1787 ms to 7 ms.
That is development note 12, told in full in [Chapter 6](06-rangeify.md) and
[Chapter 11](11-fusion-flash.md).

> **Key idea:** a simplifier is not about pretty output. It decides what an expression *depends on*, and
> dependencies decide loop structure. One wrong dependency can cost a factor of N.

## Pitfalls

**Floor versus truncation (note 4).** The IR's `FLOORDIV` floors and C's `/` truncates. If your renderer
emits `/` for every `FLOORDIV`, everything works until an index goes negative (a convolution with padding,
a shifted window). Then you read the wrong element, silently. The fix is in the renderer: `/` only when
`vmin(numerator) >= 0` and the divisor is a positive constant, otherwise `floordiv_i32`. The test
`test_floordiv_rendering_negative` exists for this.

**Folding in the wrong precision (note 5).** Computing `16777216.0 + 1.0` in Python's double gives
`16777217.0`, a value no float32 program can produce. Every folded result must be rounded to its dtype.

**Rounding the result but not the inputs.** A sharper version of the same bug survived in `uopc` until a
chapter author found it while writing this section. `UOp.const(16777217.0)` stores the double
`16777217.0` as the arg of an f32 constant. C rounds the literal `16777217.0f` to `16777216.0f` at compile
time, but the first `exec_alu` added the unrounded double. With that version:

```python
out = UOp(Ops.BUFFER, (), ParamArg(0, dtypes.f32, (1,)))
e = UOp.const(16777217.0) + UOp.const(2.0)
print("folded:", simplify(e).arg)
O = np.zeros(1, np.float32); run_c(sink(out.index(0).store(e)), {0: O}); print("C:", O[0])
```

```text
folded: 16777220.0
C: 1.6777218e+07
```

The folder said `16777217 + 2 = 16777220` (round the exact `16777219` to even), while the C program
computes `16777216 + 2 = 16777218`. The fix is one line: `truncate` the inputs as well as the output. The
current `exec_alu` does exactly that (its first line), so the same snippet now prints `folded: 16777218.0`,
and `test_folding_matches_c_semantics` keeps it that way. (Rounding constants at creation would also work,
but would change every float literal the renderer prints.)

**Casting inf or nan to int (note 6).** In C this is undefined behavior. A folder that "helpfully" turns
`(int)INFINITY` into `INT_MAX`, or into whatever Python's `int()` raises, makes your program's behavior
depend on whether optimization ran. `exec_alu` returns `None` and leaves the cast to runtime.

**Float reassociation (note 7).** `(x+1.0)+2.0 -> x+3.0` looks harmless and is not exact. Every
reassociation and distribution rule in `symbolic` is restricted to `xi` (i32). The one float identity we
keep, `x+0.0 -> x`, is wrong for `x = -0.0`, and the code says so in a comment.

**Distribution is load-bearing (note 14).** Without `(a+b)*c -> a*c + b*c`, `_div_sum` cannot see the
terms of a product of a sum, and `((x+y)*4096 + z) // 4096` style chains survive into inner loops.

**Constants outside the dtype.** The folder wraps results to 32 bits, but the first reassociation rules
built new constants with plain Python arithmetic. `(x*65536)*65536` became `x*4294967296`, an i32 `CONST`
whose value does not fit in an i32. The C literal `4294967296` silently has type `long`, so the C arithmetic
was done in 64 bits. Index expressions never get near this range, which is why nothing had broken. The fix
wraps every constant the rules create with `_i32` (a `truncate` to int32): because int32 arithmetic is
arithmetic mod 2<sup>32</sup>, `x*65536*65536` *is* `x*0`, and `simplify` now returns `0`.

**The rules assume no overflow.** `(a*c + b)//c = a + b//c` is a theorem about integers, not about int32. If
`a*c` wraps, the two sides differ. `bounds` gives up on intervals that might wrap, but `_div_sum` does not
consult bounds. Index arithmetic in this course stays far below 2<sup>31</sup>, and GPU compilers make the
same assumption. Know that you are making it.

**`INT_MIN // -1`.** `exec_alu` computes `-2147483648 // -1 = 2147483648`, which wraps to `-2147483648`. In C
on x86 the same division traps:

```c
#include <stdio.h>
int main(){ volatile int a=-2147483647-1, b=-1; printf("%d\n", a/b); }
```

```text
$ bash -c 'gcc -O2 m.c -o m && ./m'
/bin/bash: line 1: 1062784 Floating point exception   (core dumped) bash -c 'gcc -O2 m.c -o m && ./m'
```

Signed overflow and this case are undefined behavior in C, which `exec_alu` models as wraparound. The index
arithmetic the compiler generates never divides by a negative constant, so it never hits this. Still, a
folder must never be *more defined* than the target: `exec_alu` now returns `None` for this one case and
leaves the trap to run time.

**Recompute on every call (note 22).** Calling `graph_rewrite(u, symbolic)` directly from rangeify
recomputes the same simplifications for every kernel. Memoize on the hash-consed node.

**UPat `arg` matching uses `==`.** The pattern `x + 0` stores `arg=0`, and `UPat.match` compares with
`u.arg != self.arg`. Since `0 == 0.0 == False` in Python, `x + 0` also matches `x + 0.0` and `x + False`. It
is harmless for these identities (all three are identities), but it is the same trap as hash-consing
collision note 1 in [Chapter 1](01-uops.md). If you write a rule whose meaning depends on the constant's
type, also match on `dtype`.

## Exercises

1. ★ **Predict the output.** Without running anything, write down `simplify(e)` for each `e`, with `x` an
   i32 `PARAM` and `i = range_(4, 0)`, `j = range_(8, 1)`: (a) `(x*6 + 4)//2`, (b) `(x % 12)//4`,
   (c) `((i*8 + j)//2) % 4`, (d) `(i*8 + j) < 32`, (e) `(x + 5)*3 + -15`.
   *Hint: apply `_div_sum` by hand; use bounds for the last steps.*

2. ★ **Add a rule.** `max(i, 0)` is `i` when `vmin(i) >= 0`, but `symbolic` leaves it alone. Write a
   `PatternMatcher` with one rule for `MAX(x, CONST c)` that returns `x` when `vmin(x) >= c`, and check
   that `graph_rewrite(i.maximum(0), symbolic + yours)` returns `i`.

3. ★★ **Tighter bounds.** `_bounds` returns the full range for `FLOORDIV` by a negative constant. Derive the
   interval for `[a0, a1] // c` with `c < 0` (is the function increasing or decreasing?) and for `MAX` with a
   `WHERE` operand. Add both and run the property test of section 3.8 to check soundness.

4. ★★ **Find the bugs.** In your *own* compiler (the reference one has been fixed, see Pitfalls):
   (a) run `simplify((x*65536)*65536)` and look at the constant's value. What C code would it produce?
   (b) Fold `UOp.const(16777217.0) + UOp.const(2.0)` and compare with the C program that computes it.
   Fix both with `truncate`, and add a regression test for each (compare `test_folding_matches_c_semantics`).
   *Hint: every place that builds a `CONST` from arithmetic on `.arg` values is suspect.*

5. ★★ **Property-test your simplifier.** Extend the fuzzer of section 3.8 to generate `WHERE`, `CMPLT`,
   `MAX` and negative divisors, and to check that `simplify(simplify(e)) is simplify(e)` (normal forms are
   fixed points). Run it on your own week-1 implementation.

6. ★★ **Prove it.** Show that `(x % c1) // c2 = (x // c2) % (c1 // c2)` for every integer `x` when
   `c2 > 0` and `c2` divides `c1`. *Hint: write `x = q*c1 + r` and `r = s*c2 + t`.*

7. ★★★ **GCD division.** Generalize `_div_sum` so that `(x*2)//4 -> x//2` and `(x*6 + 4)//8 -> (x*3 + 2)//4`.
   *Hint: divide every term and the divisor by the gcd of all the constant factors. When is this valid
   for floor division? (Always, if every term is a multiple of the gcd.)*

8. ★★ **Floats.** For each of `x*0.0`, `x - x`, `x < x`, `x != x` and `(x + 1.0) - 1.0` (with `x` an
   f32), give a value of `x` for which the "obvious" integer simplification is wrong.

9. ★★★ **Termination measure.** Define a measure on expressions that strictly decreases under every rule
   in `symbolic` except canonicalization, and argue separately that canonicalization can fire at most
   once per node. Then explain exactly why adding factoring (section 3.6) breaks your measure.

10. ★ **Fold the edge cases.** For each input, predict what `exec_alu` returns, then run it:
    `(Ops.RECIP, f32, [0.0])`, `(Ops.LOG2, f32, [0.0])`, `(Ops.SQRT, f32, [-1.0])`,
    `(Ops.CAST, bool, [0.5])`, `(Ops.FLOORMOD, i32, [7, -2])`.

<details markdown="1">
<summary>Solutions to the ★ exercises</summary>

**1.** (a) `(x*3)+2`: both terms are divisible by 2. (b) `(x//4)%3`, the middle digit rule.
(c) `((i*8)+j)//2` becomes `(i*4)+(j//2)`, then `% 4` keeps the remainder of the sum mod 4. `i*4` is a
multiple of 4, so it is dropped and `(j//2)%4` remains. `j//2` lies in `[0, 3]`, so the modulo is the
identity: the answer is `j//2`. (d) `True`, since `i*8+j` lies in `[0, 31]`. (e) `(x+5)*3` becomes
`x*3 + 15`, then `+ -15` reassociates to `x*3 + 0`, then `x*3`.

**2.**

```text
>>> from uopc.rewrite import UPat, PatternMatcher, graph_rewrite
>>> from uopc.symbolic import symbolic, vmin
>>> mx = PatternMatcher([(UPat(Ops.MAX, src=(UPat.var("x"), UPat.cvar("c"))), lambda x, c: x if vmin(x) >= c.arg else None)])
>>> fmt(graph_rewrite(i.maximum(0), symbolic + mx)), fmt(graph_rewrite((i + -2).maximum(0), symbolic + mx))
('i', 'max((i+-2), 0)')
```

The second expression, `max(i-2, 0)`, is correctly left alone: `i-2` can be negative.

**10.** `inf` (IEEE: `1/+0 = +inf`), `-inf`, `nan`, `True`, and `-1` (floor modulo takes the sign of the
divisor: `7 = -4*-2 + -1`).

</details>

## Quiz

1. Why does `exec_alu` return `None` for `CAST` of `inf` to i32 instead of some fixed value?
2. Give integers `a, b, c` for which `(a*c + b)/c != a + b/c` with C's truncating division.
3. What interval does `bounds` compute for `(i + -2) * (i + -2)` with `i = range_(4, 0)`? Is it tight?
   Why not?
4. Why is `x*0 -> 0` restricted to integer dtypes?
5. Why is `_const_bounds` the *last* rule in `symbolic`?
6. What does `bounds` return for an i32 expression whose computed interval exceeds 2<sup>31</sup>-1, and
   why is that the only sound choice?
7. `simplify` caches results in a dict keyed by node. What property of UOps makes this correct?
8. Name two places outside `symbolic.py` where `vmin`/`vmax` change what code the compiler emits.

<details markdown="1">
<summary>Answers</summary>

1. Casting a non-finite float to int is undefined behavior in C. Any fixed value would make the folded
   program disagree with the unfolded one on some machine. Refusing keeps both identical.
2. `a = 1, b = -1, c = 4`: `3/4 = 0` but `1 + (-1/4) = 1 + 0 = 1`.
3. The factor is in `[-2, 1]`, and the corners give `4, -2, -2, 1`, so `[-2, 4]`. The true range of a square
   is `[0, 4]`. The dependency problem: interval arithmetic treats the two factors as independent.
4. Floats: `inf*0` and `nan*0` are nan, and `-1.0*0.0` is `-0.0`.
5. It is the most expensive rule (it queries bounds) and the most general. Cheaper structural rules should
   get the first chance, and `PatternMatcher` returns the first rule that fires.
6. The full i32 range. The real computation may wrap around, after which the value can be anything in the
   type, so no tighter interval is guaranteed.
7. Hash consing makes structurally equal nodes the same object, and nodes are immutable. So the
   simplification of a node is a pure function of its identity.
8. The C renderer (`/` versus `floordiv_i32`, [Chapter 4](04-memory-and-c.md)), and rangeify / the
   linearizer, through simplified indices that no longer mention spurious loop variables
   ([Chapters 5](05-loops.md) and [6](06-rangeify.md)).

</details>

## Further reading

- tinygrad, `tinygrad/uop/symbolic.py` and `tinygrad/uop/ops.py` (`vmin`/`vmax`, `_min_max`): the production
  version of everything in this chapter.
- Ramon E. Moore, *Interval Analysis* (1966), and Moore, Kearfott and Cloud, *Introduction to Interval
  Analysis* (SIAM, 2009).
- Patrick Cousot and Radhia Cousot, "Abstract Interpretation: A Unified Lattice Model for Static Analysis of
  Programs by Construction or Approximation of Fixpoints", POPL 1977.
- Franz Baader and Tobias Nipkow, *Term Rewriting and All That* (Cambridge, 1998): termination orders,
  confluence, critical pairs.
- David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM Computing
  Surveys, 1991.
- Raymond T. Boute, "The Euclidean Definition of the Functions div and mod", ACM TOPLAS, 1992: floor,
  truncation and Euclidean division compared.
- Henry S. Warren, *Hacker's Delight*, 2nd ed., chapter 9 (integer division) and chapter 10 (division by
  constants).
- Leonardo de Moura and Nikolaj Bjørner, "Z3: An Efficient SMT Solver", TACAS 2008. Try proving your
  rewrite rules with Z3's Python API.
