# Compilers for Machine Learning — archived course syllabus

This file preserves the version originally used for Chapters 1–15. The subsequently supplied syllabus
moves kernel fusion/fission into week 4, restricts that week's shapes to rank below two, and moves
general movement ops and autodiff into week 5. It also explicitly requests dtype broadcasting.
Use the [syllabus contract](src/20-syllabus-contract.md) for the requirement-by-requirement mapping
and documented implementation differences. Start the expanded self-study path with
[Foundations](src/00-prerequisites.md) and continue through Chapters 20–29 after the core book.


A hands-on one semester course where students build their own compiler from scratch, starting from elementwise programs and ending with training SOTA LLMs on GPUs. This course aggressively builds on the previous week, and is an exercise in slop management. If you let any slop in early, it will compound and you will not finish the class.

Course description: This course covers the design and implementation of a modern machine learning compiler, and examines the interaction between IR design, hardware capabilities, and the structure of machine learning programs. Topics covered include term rewriting, code generation, movement operators, kernel fusion, memory hierarchies, GPU architecture, automatic differentiation, and flash attention. It is a project course, providing experience with performance-oriented programming, managing a codebase that grows all semester, and working in 1 or 2 person teams, culminating in a compiler capable of training modern LLMs.

Prerequisites: no formal prerequisites, but designed for someone deeply familiar with 15-213 or equivalent, highly skilled in their implementation language, capable of managing long programming projects, and comfortable with C and some GPU programming. Frequent in-person quizzes.

The syllabus outlines the spec, not the implementation. No starter code. A universal, simple interchange format for UOps:

```
# gemm_256x256.uir
%0 = buffer 65536 : dtype=f32 slot=0
%1 = buffer 65536 : dtype=f32 slot=1
%2 = reshape %0, (256, 256, 1)
%3 = reshape %1, (256, 1, 256)
%4 = mul %2, %3
%5 = reduce %4 : op=add pop=1
```

Grading: 55% compiler (weeks 1-10: 5% tests due Mondays judged by an open binary oracle; 20% correctness due Wednesdays — all student + teacher tests + basic performance gates; 15% style — "did you write slop? can I read your code? can it be simpler?"; 15% performance, weeks 6-10, on a curve), 25% final project (weeks 11-14), 20% in-person quizzes (at least 20).

## Week 1: UOps & Rewrites — `uops` `elementwise ops` `symbolic` `rewrites` `renderer`
The UOp is hash consed: `UOp(op, src:tuple[UOp, ...], arg)`. Recursive properties: `dtype(void,bool,i32,f32)`, `shape:tuple[int, ...]`.
Ops: PARAM(ParamArg) input to the call; CALL(CallArg) src[0] body, src[1:] params; CONST (type from arg); STACK (new leading axis, same-shaped srcs; until week 4 inputs only STACK/CONST); INDEX (shape (n,) -> (), pops axes off the left with srcs); CAST/RECIP; ADD/MUL/CMPLT; FLOORDIV/FLOORMOD; WHERE. The graph rewrite engine should be able to "run" these programs via const folding.
Lectures: course intro, turing completeness, the UOp, the format · ALU, bottom-up greedy rewrite engines, const folding · call/param, symbolic simplification, computer algebra systems.

## Week 2: Rendering & Runtime & Memory
Recursive property `addrspace(GLOBAL, ALU)`. Ops: BUFFER(ParamArg) global persistent buffer (never in call); ALLOC(ParamArg) call-scope storage (`float a[10]`); LOAD/STORE (load 1 src, store 2); AFTER(buf, <store>). Render loopless C functions acting on memory and immediates.
Lectures: toposort and rendering, C refresher · memory management and runtime · type systems for shape, dtype, addrspace.

## Week 3: Loops — `loops` `gemms` `convs`
RANGE (src is the end, arg is the name), END (srcs are passthrough, range). Write gemms and convs in this language.

## Week 4: Movement Ops + Rangeify — `movement ops` `broadcasting` `reduction` `rangeify`
STACK now allows arbitrary inputs; RESHAPE/EXPAND/PERMUTE/SHRINK; REDUCE (axis count / op). Rangeify removes shapes from all ops. PAD and FLIP are composite ops.

## Week 5: Kernel fission and fusion & Autodiff
Compile *any* model, split into kernels to avoid repeated compute, compile to C. STAGE: a temp buffer you can index into. Autodiff only on pure shape graphs (not STORE, CALL, AFTER, RANGE, END). Train your first MNIST.

## Week 6: Memory Hierarchies & Multithreading & Fast Kernels — `FLOPS counting` `upcasting` `fast GEMMs` `flash attention`
SOTA-competitive C for GEMMs, reduces, convs on CPU. Ranges have axis type LOOP, UPCAST, or THREAD.

## Weeks 7–8: GPUs & Hardware Accelerators — `parallel` `GPUs` `hardware accelerators` `tensor cores`
Kernels, GPUs, axis mappings, tensor cores. Decent torch-competitive CUDA/HIP.

## Weeks 9–10: Real Models — `real models` `symbolic`
Implement a real LLM and train models.

## Weeks 11–End: Student Projects
Implement a paper, port to strange hardware, mostly anything.

| Weeks | Topics | Milestone |
|---|---|---|
| 1–2 | UOps, elementwise ops, symbolic, rewrites | Compile & simplify elementwise programs to C |
| 3–4 | Loops, movement ops, reduction, rangeify | Compile any model to C (slow, but correct) |
| 5–6 | Memory hierarchies, upcasting, GEMMs, convs | SOTA-competitive CPU code |
| 7–8 | Call, GPUs, accelerators, tensor cores | Torch-competitive CUDA/HIP code |
| 9–10 | Real models, autodiff | Train a real LLM |
| 11+ | Student projects | Choose a project and extend your compiler |
