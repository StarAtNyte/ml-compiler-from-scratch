# 21. Control Flow, SSA, Analysis, and Lowering

The UOp graph makes pure tensor expressions easy to share and rewrite. A general compiler also needs
to represent branches, loops, mutable memory, and facts that hold only on particular paths. Build a
small scalar compiler beside `uopc` to learn these ideas without complicating its tensor IR immediately.

## 21.1 From syntax to a control-flow graph

Start with integer expressions, variables, assignment, `if`, `while`, and `return`. A lexer turns text
into tokens. A parser groups tokens into an abstract syntax tree, respecting precedence; `a+b*c`
means `a+(b*c)`. Name resolution binds each variable use to a declaration, and type checking rejects
invalid operations before lowering.

A basic block is a sequence of instructions with a single entry and a terminating transfer of control.
Lower `if p: x=a+1; else: x=a-1; return x*2` into:

```text
entry: branch p, then, else
then:  x_then = a + 1; jump merge
else:  x_else = a - 1; jump merge
merge: x = phi [x_then, then], [x_else, else]; return x * 2
```

The phi selects the value from the predecessor that actually executed. It is not a normal operation
that evaluates both branches. Verify that every block ends with a terminator, every target exists,
and every phi has an incoming value for each predecessor.

**Lab:** implement an interpreter and a CFG dumper. Test nested branches, zero-trip loops, and a
loop-carried accumulator. Define whether uninitialized variables are rejected and test the rule.

## 21.2 Dominance and SSA

A block d dominates b if every path from entry to b passes through d. On reachable blocks, initialize
`Dom(entry)={entry}` and other dominator sets to all reachable blocks. Iterate:

```text
Dom(b) = {b} union intersection(Dom(p) for p in predecessors(b))
```

until nothing changes. Unreachable blocks need separate handling. In static single assignment form,
each name has one definition, and ordinary uses must be dominated by their definition. Phi operands
are uses on incoming edges, so their dominance check is edge-specific.

For the example above, `then` does not dominate `merge`: execution can arrive through `else`. Merely
using `x_then` in merge would be invalid. A loop's phi joins an initial value from the preheader with
an updated value from the backedge.

**Lab:** compute dominators, place phis for a small mutable-variable language, and rename definitions.
Convert an accumulator loop and verify its results against the source interpreter. Add an intentionally
invalid use and ensure the verifier rejects it. Explain where the tensor DAG avoids these issues.

## 21.3 Dataflow and effects

Liveness answers whether a computed value may be used later. For each block:

```text
OUT[b] = union(IN[s] for s in successors(b))
IN[b]  = USE[b] union (OUT[b] - DEF[b])
```

Solve backward to a fixed point. Count a value as USE only if it is used before a definition in that
block. In SSA, account for phi uses on predecessor edges. Liveness later determines when a runtime
can reuse a buffer or a backend can reuse a register.

Dead-code elimination removes an unused pure add. It cannot remove a store just because the store
has no ordinary result. Memory effects require ordering and alias reasoning. In
`a[0]=1; b[0]=2; return a[0]`, the return is 2 when a and b alias and 1 when they do not.
Hoisting the load before the second store is valid only with proof that the relevant addresses do
not overlap. `restrict` in generated C is a promise the compiler must actually uphold.

**Lab:** implement liveness and effect-aware dead-code elimination. Test both aliased and disjoint
inputs. State which transformations require no-alias assumptions and ensure the runtime enforces them.

## 21.4 Loops, legality, and machine code

Loop interchange and tiling change execution order. Before choosing a profitable schedule, prove
that it preserves dependencies. `a[i]=a[i-1]+1` carries a dependence across iterations and cannot be
parallelized as independent work. A reduction may be parallelizable under a declared floating-point
policy, but its rounding can change.

Machine lowering introduces another set of constraints: legal instruction types, register classes,
calling conventions, and finite registers. Instruction selection maps IR operations to target
instructions. Register allocation maps live values to registers; excess live values spill to memory.
Unrolling can reduce loop overhead while increasing live values and spills. C rendering delegates
these tasks to the host compiler; a direct backend must handle them itself.

**Lab:** build an interference graph from live scalar values and implement a small register allocator
with a fixed register budget. Report spills before and after unrolling. Separately inspect compiler
assembly for a generated C kernel and identify loads, arithmetic, branches, and vector instructions.

## 21.5 LLVM and MLIR bridge

Use the official [LLVM frontend tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)
for parsing, LLVM IR generation, control flow, and JIT integration. Port one expression and one loop
from your scalar language. Ask LLVM's verifier to check the module before optimization.

Use the [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) to study operations, dialects,
shape inference, rewrites, and progressive lowering. Translate a small tensor expression into an
appropriate structured representation, then lower toward loops and a backend. Record the IR after
every pass. Pin the LLVM/MLIR revision because APIs and pass names evolve.

A tensor value describes logical elements; a memory reference adds storage, layout, and aliasing.
Bufferization must decide when an output may reuse existing storage. A use of the old value after
an update can force a copy. Compare that decision with Chapter 7's materialization boundaries.

**Exit gate:** provide a verified CFG/SSA compiler, an effect-safe optimization, and a runnable
LLVM or MLIR example. Given a wrong output, identify the first pass whose output changes semantics.

## 21.6 Worked lab: why phi operands live on edges

Run the executable analysis from the repository root:

```bash
.venv/bin/python -m labs.ssa
```

The diamond has a parameter x, a branch condition p, a left result a, a right result b, and a merge
phi z. The entry dominates both branches and the merge. Neither branch dominates the merge, which
is why the merge cannot simply use a. The phi is the only place the two branch results meet.

The liveness calculation should produce `OUT(left)={a}` and `OUT(right)={b}`. If both a and b appear
live out of both branches, the analysis has treated phi arguments as ordinary merge-block uses.
That creates impossible interference and can force unnecessary register pressure. At the merge
itself, z is already defined by its phi, so z is not live into the block.

Here is the complete implementation. It accepts already constructed SSA and verifies it; it does
not yet parse source or insert phis automatically. Unreachable blocks are excluded from dominance
and ordinary-use checks. Phi incoming edges still match the complete predecessor set.

@include labs/ssa.py

**Experiment:** change the merge's ordinary use from z to a. The verifier must report that a's
definition does not dominate its use. Then replace one phi incoming edge with a nonexistent branch.
That must fail before liveness analysis. Finally construct a loop with an initial parameter and a
backedge value, and show why a header phi may legally use the later-written backedge value.

**Next implementation:** add AST lowering, dominance frontiers, phi placement, and renaming. Validate
against a source interpreter. The executable lab supplies analysis and verification, not that entire
frontend. Use `tests/test_advanced_labs.py` as a starting example of independent semantic checks.


## 21.7 Continue into a complete executable pipeline

The isolated analysis above is now extended by [Chapters 30–32](30-source-to-ssa.md). They implement
a source frontend, CFG construction, definite assignment, pruned SSA construction, SCCP, effect-aware
dead-code removal, phi destruction, register allocation with executed spills, and a bounds-checked
native C backend. Run `.venv/bin/python -m minicc` to execute the pipeline. The earlier “next
implementation” assignment can now be studied against an actual reference implementation.
