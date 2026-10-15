# minicc: source to SSA, optimization, registers, and native C

This is an executable teaching compiler alongside `uopc`, not a replacement for its tensor/autodiff
pipeline. Python's parser supplies syntax; the compiler independently defines i32 arithmetic,
control flow, buffer effects, SSA, optimization, and code generation.

```bash
python -m minicc
python -m pytest tests/test_minicc.py -q
```

```python
from minicc import compile_ir, compile_native, execute
from minicc.ir import format_ir
from minicc.machine import allocate, destroy_phis, execute_allocated

source = """
def total(n: int) -> int:
    i = 0
    acc = 0
    while i < n:
        acc += i
        i += 1
    return acc
"""
function = compile_ir(source)
print(format_ir(function))
assert execute(function, 10) == 45
assert compile_native(source)(10) == 45
machine = allocate(destroy_phis(function), registers=2)
assert execute_allocated(machine, 10)[0] == 45
```

Read book Chapters 30–32 for the complete pipeline and implementation. Supported syntax is scalar
assignment, addition/subtraction/multiplication, single comparisons, unary minus/not, if/else, while,
break/continue, return, and indexed buffer reads/writes. Calls, arbitrary Python objects, division,
short-circuit boolean operators, chained comparisons, and other syntax are unsupported. Definite
assignment is conservative over structural CFG paths before optimization.

Scalars are wrapping signed i32. Comparisons produce integer 0/1. Buffers use an explicit `buffer`
parameter annotation and checked nonnegative indices. Aliasing is permitted, effects remain ordered,
and traps preserve prior writes. Native execution requires gcc and contiguous writable int32 NumPy
buffers. Native calls do not have the interpreters' step limit; use bounded trusted workloads or a
separate process timeout.

The register-machine backend executes allocated locations and counts spill traffic, but permits
spill operands directly. It does not emit machine bytes or model CPU/GPU timing. The C backend
delegates physical instruction selection and allocation to gcc. SCCP uses a fixed-point scan rather
than production worklists. The compiler has no persistent executable cache, closures, general alias
analysis, or memory SSA. These are explicit extension points.
