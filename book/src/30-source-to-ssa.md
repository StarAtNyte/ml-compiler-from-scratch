# 30. Building a Source Compiler: Contracts, CFGs, and SSA Construction

The analysis exercise in Chapter 21 began with a hand-written graph. This chapter builds the graph
from source. The implementation is `minicc`, a separate compiler whose purpose is to expose control
flow, effects, verification, and backend decisions that tensor expressions can hide. It complements
`uopc`; it does not replace its autodiff or GPU pipeline.

Run the complete example before reading the implementation:

```bash
.venv/bin/python -m minicc
.venv/bin/python -m pytest tests/test_minicc.py -q
```

The example compiles a loop with a conditional accumulation and obtains 87 through an SSA interpreter,
a register-constrained machine, and generated native C. Agreement across these paths is useful, but
not sufficient by itself: they share the IR. The tests also execute trusted source programs with
Python as an independent oracle over a small range that avoids overflow. Overflow and memory errors
have separate tests against the compiler's explicitly different semantics.

## 30.1 Define the language before optimizing it

The source syntax is a restricted Python function, parsed by Python's `ast` module. This delegates
lexing and parsing, not execution or language semantics. No source function is evaluated by the
compiler. Function calls, comprehensions, imports, general objects, exceptions, chained comparisons,
short-circuit `and`/`or`, and unsupported statements are rejected on the lowered path.

The supported language includes scalar variables, integer constants, `+`, `-`, `*`, comparisons,
`not`, assignment, `if`, `while`, `break`, `continue`, and explicit `return`. Buffer parameters use
an annotation naming `buffer`; scalar parameters may be unannotated or annotated `int`. Buffers are
rank-one indexed storage. An integer result is required on every structurally reachable exit.
Unreachable statements after a terminator need not be lowered; this is not a general Python validator.

Every scalar value is signed i32. Arithmetic wraps modulo `2^32` and is reinterpreted as signed.
Comparisons operate on those signed values and produce 0 or 1. A condition is true when nonzero.
Input scalar integers and constants are normalized to that domain. For example, `2147483647+1`
is `-2147483648`. A compiler that tests only tiny positive values could accidentally implement
mathematical integers or undefined C overflow without noticing.

Buffer indices must be in `[0,length)`. Negative indexing does not have Python's meaning. A failed
access reports a bounds error; earlier stores remain visible, and later instructions do not execute.
Two buffer arguments may overlap. The compiler therefore cannot assume that storing through a leaves
loads through b unchanged. No generated `restrict` promise is made.

These choices define optimization legality. Removing an unused add is valid. Removing an unused
load is not generally valid, because the load can fail. Reordering a store before a potentially
failing load can change the state visible after an error. Treat trapping and memory effects as
part of the program, rather than incidental implementation details.

## 30.2 An inspectable intermediate representation

A function owns parameters, named blocks, and an entry label. An instruction has an operation,
an optional result, and operands that are either integer literals or value names. Stores have no
scalar result. A terminator is a jump, conditional branch, or return.

A block's instruction order preserves memory effects. The representation does not yet have memory
SSA or general alias analysis: it establishes safety by declining to reorder accesses. That is a
useful baseline against which a future alias analysis must prove additional transformations.

@include minicc/ir.py

**Trace:** lower `x=a[0]; b[0]=2; return x`. The return uses the loaded scalar x even if a and b alias.
Change it to `b[0]=2; return a[0]`; now the aliasing case returns the updated value. An expression-level
common-subexpression rule must not identify the two loads across an intervening possibly aliased store.

## 30.3 Constructing blocks from structured syntax

An assignment emits its expression's operations followed by a copy into the mutable variable name.
Temporary names are compiler-generated and cannot collide with source identifiers. Right-hand-side
expressions are lowered before the assignment target, preserving the chosen evaluation order.

An `if` creates then, else, and merge blocks. Both arms end by jumping to the merge unless they
already returned or transferred control. If both arms return, the merge is unreachable and is removed.
A `while` creates a header, body, and exit. The current block jumps to the header; the header computes
the condition and branches; a normal body completion jumps back to the header.

A stack of `(header,exit)` pairs implements nested `continue` and `break`: they target the innermost
loop. After emitting any terminator, the frontend stops appending statements to that path. This
prevents an apparent assignment after a return from becoming an executed side effect.

@include minicc/frontend.py

**Exercise:** draw the graph for a loop whose body contains an early `continue`, a nested loop, and
an early return. Identify every backedge and the destination of each `break`. Compare your graph with
`format_ir(lower(source))`, then execute the mutable CFG before introducing SSA.

## 30.4 Definite assignment is a must analysis

Consider a variable x assigned only in one arm of an unknown branch, then used at the merge. A
may-have-been-assigned analysis is insufficient. Every predecessor must establish x. For each block,
intersect predecessor OUT sets, then add definitions performed by the block:

```text
IN(entry) = parameters
IN(b) = intersection(OUT(p) for p in predecessors(b))
OUT(b) = IN(b) union DEF(b)
```

Initialize non-entry sets to the universe of names and iterate downward to a fixed point. Using the
empty set would compute the wrong extremal solution for some cycles. After convergence, scan each
block in order, checking every use before adding its instruction's result to the defined set.

A loop that assigns x only in its body cannot establish x after the loop, because the loop may
execute zero times. An assignment `x=x+1` cannot establish the old x required by its own right-hand
side. The tests exercise both errors. This conservative analysis runs before branch optimization;
it can reject source that is safe only because a later constant-propagation pass could prove a
branch always executes. That restriction is explicit rather than a hidden inconsistency.

## 30.5 Dominators, immediate dominators, and frontiers

The fixed-point dominator sets from Chapter 21 let us form a tree. For each non-entry block b,
its immediate dominator is the strict dominator nearest to b. Strict dominators form a chain;
in this implementation, the one with the deepest dominator set is the nearest.

The dominance frontier of d contains blocks where paths from the region dominated by d meet a
path not strictly dominated by d. At a diamond merge, each arm has the merge in its frontier.
A loop header can belong to its own frontier: it dominates the backedge source but does not strictly
dominate itself. That self-frontier is how a loop-carried assignment discovers its header phi.

For each join b and predecessor p, walk from p up the immediate-dominator tree until reaching
`idom(b)`, placing b in each visited node's frontier. Analyze reachable blocks only, and require a
separate entry without predecessors to make the initial parameter definitions unambiguous.

@include minicc/analysis.py

The same module verifies block structure, instruction arity, buffer/scalar use, literal range,
unique SSA definitions, phi edge sets, and dominance of ordinary and edge uses. Verification is
part of the pipeline. A malformed program should fail close to the pass that created it, rather
than survive until a native crash.

## 30.6 Phi placement needs an iterated frontier

For a variable with definitions in blocks D, put phis at their dominance frontiers. A newly inserted
phi is itself a definition and may require another phi in a further frontier. Continue until no
new placement is needed. This is the iterated dominance frontier.

Placing every possible phi is correct but noisy. Pruned SSA inserts one only when the original
variable is live into the candidate block. If both arms assign a dead variable that is never read,
the merge needs no phi. Compute pre-SSA liveness before placement; there are no phi edge uses yet.

For the running example, i and total receive header phis joining initial values with backedge
values. The conditional update of total receives a second phi at the branch merge. The result
of that merge feeds total's loop-header phi. Placing phis only at the first visible diamond would
miss the loop-carried value.

## 30.7 Renaming with scoped stacks

Maintain one stack of current SSA versions per mutable variable. Parameters seed their stacks.
Walking the dominator tree, first assign names to block phis, then rewrite instruction operands
using current stack tops and push fresh names for definitions. After the terminator, fill every
successor phi's incoming operand for this edge from the current stacks.

Recursively process dominated children, then pop exactly the definitions introduced in this block.
Without popping, a name from one branch leaks into its sibling. Filling phi edges at the predecessor
also handles loop backedges: the header phi name already exists when the body is renamed.

@include minicc/ssa.py

**Acceptance:** inspect the generated loop phis, run zero-trip and multi-trip loops, and mutate a
phi edge or an ordinary use to demonstrate verifier rejection. Then generate small trusted programs
and compare mutable execution, SSA execution, and a source oracle. The next chapter adds transformations
on this verified representation.
