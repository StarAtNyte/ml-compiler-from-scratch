# 31. Optimization with Executable-Path Reasoning

Constant folding evaluates an operation with known arguments. Conditional constant propagation also
reasons about which control-flow edges can execute. That difference lets it ignore an unreachable
phi input while remaining conservative about unknown runtime values.

## 31.1 Three lattice states, not two

Each SSA value is initially UNDEFINED, a known constant, or OVERDEFINED. UNDEFINED means analysis
has not established information yet; it does not mean an uninitialized source variable is legal.
The frontend already rejected those. OVERDEFINED means the value cannot be represented as one
constant by this analysis. Parameters begin OVERDEFINED.

The join rules are:

| Join | Result |
|---|---|
| UNDEFINED and x | x |
| Constant c and the same c | c |
| Different constants | OVERDEFINED |
| OVERDEFINED and any established value | OVERDEFINED |

A value's state only moves upward. It can become a constant, then become OVERDEFINED when another
path contributes a different value; it never becomes a constant again after that. Executable edges
also only accumulate. These monotonic properties give a finite fixed-point computation.

## 31.2 Control flow participates in the proof

Start with the entry executable and no executable edges. Evaluate instructions in executable blocks.
An unconditional jump enables its edge. A constant condition enables only its selected edge. An
OVERDEFINED condition enables both edges. An UNDEFINED condition enables neither yet; later scans
may establish more information and enable the correct edges.

For a phi, join operands only from executable incoming edges. Consider:

```python
def f(n):
    flag = 1
    if flag:
        x = 5
    else:
        x = n
    return x * 2
```

Joining every syntactic incoming operand would combine 5 with unknown n and lose the constant.
Executable-edge reasoning proves the else edge unavailable, so x is 5 and the result is 10.
After rewriting, the dead arm is pruned and the unused phi disappears.

Now replace `flag=1` with `flag=n`. Both branches become executable. The phi is OVERDEFINED;
returning 10 unconditionally would be wrong. Tests need both cases to distinguish valid path reasoning
from an optimizer that simply follows the first path it encounters.

## 31.3 Loops require revisiting optimistic facts

For `x=0; while x<n: x+=1; return x`, the header phi first sees the entry value 0. If the body can
execute, its backedge eventually contributes 1, then the phi becomes OVERDEFINED. The analysis must
revisit instructions and branch decisions affected by that change. Assuming the initially observed
zero is permanent would miscompile every multi-iteration case.

This implementation scans to a fixed point for readability. A production sparse implementation
would keep value-user and executable-block worklists, evaluating only affected nodes. The lattice,
edge policy, and correctness conditions remain the same; the scan is not a claim about compile-time
scalability.

## 31.4 Rewrite only after the analysis converges

Once facts stabilize, substitute constants into operands and pure results, convert constant branches
to jumps, remove unreachable blocks, and repair phi edge maps. Keep trapping loads and stores in
order. The result is reverified before any later pass consumes it.

An unreachable bounds error can disappear: that instruction cannot execute. A reachable load whose
result is unused must remain: its failure is still observable. This distinction is precisely why
“dead result” and “dead instruction” are different ideas.

@include minicc/optimize.py

The transfer function deliberately declines identities that require more reasoning. For instance,
an unknown integer multiplied by zero can be simplified in this language, but basic constant
propagation leaves it unknown. Missing a valid optimization is different from making an invalid
one. Add identities separately with their semantic domain and counterexample tests.

## 31.5 Dead-code elimination is backward reachability

Seed a live-value set with return and branch operands plus the operands needed by effectful
instructions. Follow each live definition backward through its dependencies; a phi contributes its
incoming values. Remove pure instructions and phis whose results are not reached. Preserve all
reachable loads and stores, even when their scalar results are dead.

A cycle of phis and arithmetic with no connection to an observable root can be removed. Conversely,
a value that computes an index for a store remains live even if no scalar return depends on it.
Thinking of DCE as graph reachability handles both without relying on source-line intuition.

## 31.6 Experiments that can falsify the optimizer

The test suite compares pre-optimization and post-optimization execution against trusted source
oracles, then checks separately specified cases that differ from Python:

- Wrapping arithmetic at signed boundaries and in large products.
- Unknown branch conditions, not only constant conditions.
- Loops with zero, one, and multiple iterations.
- Stores through aliased buffers followed by loads through the other name.
- An unused invalid load, which must still raise an error.
- A store before a trapping load, which remains visible after failure.
- A later store after that load, which must not execute.

Run these directly:

```bash
.venv/bin/python -m pytest tests/test_minicc.py -q
```

**Extension:** add copy propagation, algebraic simplification, global value numbering, and loop-invariant
code motion one at a time. For load motion, first implement an effect/alias model capable of proving
that intervening writes and traps permit the move. Do not reuse pure-expression legality rules for
memory operations. For each new pass, verify the output and retain a minimized test that would fail
if its key precondition were omitted.
