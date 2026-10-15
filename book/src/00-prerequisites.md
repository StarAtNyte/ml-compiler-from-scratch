# Foundations: From Programming to Tensor Compilers

Start here if the original course prerequisites are unfamiliar. The course's “no formal prerequisites”
means no enrollment requirement; it does not mean no background knowledge. Work through these labs
before Chapter 1. Experienced readers may skip a lab after passing its exit check.

This book has two sequences. Chapters 1–15 build the compact `uopc` compiler. Chapters 20–29 extend
the learning path into compiler analysis, GPU performance, dynamic execution, runtime engineering,
and independent research. Those later labs are assignments to implement, not claims that all their
features already exist in `uopc`. [The syllabus contract](20-syllabus-contract.md) maps the supplied
course specification to chapters, tests, and remaining hardware evidence.

## F.1 Programming and debugging

If programming itself is new, first learn variables, functions, loops, arrays, classes, exceptions,
and tests. Write a command-line calculator, then an interpreter for nested expressions such as
`("add", ("mul", 2, 3), 4)`. The interpreter should return 10. Extend it with named variables and
report an undefined variable as an error. This is the execution model Chapter 1 later represents as a graph.

Python objects have identity as well as value. Two equal expressions can be separate objects;
hash consing deliberately makes structurally identical immutable expressions share an object.
Implement an immutable expression class with `(operation, operands)` as its key. Never mutate an
object after using its structure as a dictionary key.

**Lab:** build the interpreter, add ten tests, deliberately introduce a wrong multiplication result,
and use a debugger to locate it. Commit the working version with Git, break it, and restore it.
**Exit check:** explain identity versus equality, recursion versus iteration, and why a shared graph
node must not be modified in place.

## F.2 C, memory, and the operating system

A pointer is an address interpreted with an element type. For `float *p`, `p + 3` advances three
floats, not three bytes. In a row-major `M × N` matrix, element `(i,j)` lives at `p[i*N+j]`.
The array's shape is metadata; the allocation itself is a sequence of bytes.

A local array's lifetime ends when its function returns. A heap allocation persists until released.
A shared library exports compiled functions which Python can call through a foreign function interface.
The ABI determines how pointers and scalar arguments are passed; mismatching a function signature can
corrupt memory even when the source code looks plausible.

**Lab:** implement vector addition and rectangular matrix multiplication in C. Compile with warnings
and debug symbols; run memory checks with address and undefined-behavior sanitizers when supported.
Call vector addition from Python using `ctypes`, declaring the argument types. Compare every result
with a simple Python reference. Exercise lengths 1, 7, and 257 and matrix dimensions `(3,5,7)`.
**Exit check:** draw the memory layout of a 2×3 array and explain why returning a pointer to a local
array is invalid. Know how to inspect a crash and separate a compiler error from a runtime error.

## F.3 Linear algebra and tensor indexing

Matrix multiplication is a reduction:

```text
C[i,j] = sum over k of A[i,k] * B[k,j]
A: (M,K), B: (K,N), C: (M,N)
```

There are approximately `2*M*N*K` floating-point operations when multiply and add count separately.
Broadcasting adds repeated logical values without requiring repeated physical storage. A vector of
length N broadcast across M rows can use address `j` for every row i: its row stride is zero.
A transpose swaps logical axes and their strides; it need not copy the data. A reshape is a view
only when the existing layout supports the requested indexing; arbitrary transposes cannot always
be flattened into a contiguous view.

**Lab:** write `offset(index, strides)` and use it to read contiguous, transposed, and broadcast
views. Implement a reduction with explicit loops. For `A=[[1,2],[3,4]]` and `B=[[5,6],[7,8]]`,
obtain `[[19,22],[43,50]]`. Derive each address rather than calling a matrix library.
**Exit check:** distinguish shape, stride, element count, and byte size; derive the gradient shape
for a broadcast addition.

## F.4 Floating point and differentiation

Floating-point arithmetic rounds after operations. Consequently `(a+b)+c` can differ from `a+(b+c)`.
A compiler that reassociates reductions needs an explicit numerical policy. Use both absolute and
relative tolerances when comparing results; a fixed relative tolerance is unhelpful near zero.

For a scalar loss `L` and `z=x*y`, reverse mode accumulates `dL/dx += dL/dz*y` and
`dL/dy += dL/dz*x`. Accumulation matters because a value can feed several consumers. For
`L=x*x+x` at `x=3`, the derivative is 7, not 3 or 4.

**Lab:** implement scalar reverse-mode autodiff with addition and multiplication. Compare its
derivative with `(L(x+h)-L(x-h))/(2*h)` in double precision for several h. Explain why extremely
small h causes cancellation. Extend the engine to a sum and a broadcast operation.
**Exit check:** derive a linear layer's input and weight gradients and explain why a gradient check
should avoid a nondifferentiable ReLU point.

## F.5 Measuring a machine

A cache stores recently accessed memory in lines. Contiguous traversal usually reuses each fetched
line; a large stride may fetch many bytes to consume one element. Threads share some resources and
compete for bandwidth. More threads do not guarantee a speedup.

**Lab:** time row-first and column-first array traversals in compiled C, consuming the result so the
compiler cannot eliminate the loop. Warm up first, repeat measurements, and record the machine,
compiler flags, data sizes, and timing distribution. Separate compilation from execution.
**Exit check:** predict whether doubling arithmetic while preserving bytes transferred should double
runtime for a bandwidth-limited kernel. Explain the prediction using the roofline model in Chapter 10.

## F.6 Study order and equipment

Use the following order; move on by demonstrated ability rather than a fixed calendar.

| Stage | Material | Required artifact |
|---|---|---|
| Foundations | This chapter | C kernels, tensor views, scalar autodiff |
| Compiler construction | Chapters 1–9 | Tested tensor compiler and MNIST training |
| Performance and models | Chapters 10–15 | Benchmark notebook, GPU kernels, GPT training |
| Compiler depth | Chapter 21 | CFG analysis and an LLVM/MLIR lowering experiment |
| Hardware depth | Chapter 22 | Real GPU profiles and a GEMM optimization ladder |
| Dynamic systems | Chapters 23–24 | Guarded execution, memory planner, precision experiments |
| Numerical and backend depth | Chapters 26–28 | Error analysis, schedule search, irregular operations, target port |
| Implemented internals | Chapters 30–34 | Source-to-machine compiler and dynamic tensor runtime |
| Independent work | Chapters 25 and 29 | Reviewed capstone, external codebase change, frontier specialization |

A CPU machine is sufficient for the foundations and compiler construction. GPU indexing can be
studied with the emulator, but GPU performance and synchronization milestones require real hardware.
Multi-device milestones require two devices or processes, with simulated exercises clearly labeled.
Do not report an emulator's timing as GPU performance. Record tool versions and GPU architecture
because supported instructions and tooling vary.
