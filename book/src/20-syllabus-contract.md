# 20. The Syllabus Contract and Completion Gates

The original course is ambitious: build a compiler, optimize it, and train modern language models in
one semester. This self-study edition uses that specification as its core and adds prerequisite and
advanced labs. Reading a chapter, passing a correctness test, and demonstrating competitive performance
are three different achievements. Keep an evidence ledger for each.

The source specification is the supplied [Compilers for Machine Learning syllabus](https://gist.github.com/geohot/4768597d9dc536446ee2d5de1f29e89d).
The table below follows its week 4–5 ordering. The existing narrative teaches general movement ops
before kernelization; this is a teaching-order difference. When following the course strictly,
restrict the week-4 student interface to rank below two and unlock general shapes in week 5.
`uopc` itself is a cumulative reference implementation and does not enforce weekly feature gates.

The syllabus is a minimum. [Chapters 26–29](29-frontier-lab.md) extend the track into numerical
analysis, compiler-driven scheduling, direct backends, irregular workloads, hardware codesign,
production reliability, distributed training, and research. The final chapter defines specialization
paths and portfolio gates for frontier-lab engineering roles.

## 20.1 Requirement-to-evidence map

| Syllabus requirement | Teaching | Evidence to produce |
|---|---|---|
| Hash-consed UOps; recursive dtype and shape | Chapters 1–3 | Identity, type-error, and shape inference tests |
| PARAM/CALL with scoped parameters | Chapters 1–2 | Nested-call evaluation without scope capture |
| CONST/STACK/INDEX; CAST/RECIPROCAL; ADD/MUL/CMPLT; FLOORDIV/FLOORMOD; WHERE | Chapters 1–3, Appendix A | Constant evaluation, negative division, casts, and selection tests |
| dtype broadcasting | Section 20.2 | Explicit promotion contract and mixed-type cases |
| GLOBAL/ALU; BUFFER/ALLOC/LOAD/STORE/AFTER | Chapter 4 | Typed memory operations and ordered load/store tests |
| C rendering, compilation, runtime, memory management | Chapter 4 | Execute a generated shared library and compare outputs |
| RANGE/END; GEMM and convolution loops | Chapters 5–6 | Rectangular GEMM and convolution versus independent references |
| Rank < 2 week-4 subset; REDUCE/STAGE; implicit loads | Chapters 6–7; section 20.3 | Vector reduction, explicit materialization, recomputation accounting |
| STACK/general movement ops; rank >= 2; PAD/FLIP composites | Chapters 6 and 9 | Index maps and noncontiguous/view composition tests |
| EXP2/LOG2 and extensible elementwise ops | Chapters 3, 8–9; Appendix A | Evaluation, rendering, and derivative tests |
| Autodiff only on pure shape graphs | Chapter 8 | Shared-use accumulation and unsupported-effect rejection |
| MNIST training | Chapter 9 | Reproducible loss and held-out accuracy |
| FLOPs, memory hierarchy, threads, upcasting | Chapter 10 | Optimization ablations and a machine roofline |
| CPU GEMM, reductions, convolution; FlashAttention | Chapters 10–11 and 22 | Separate workload tables; attention correctness and memory use |
| GPU kernels, axis mappings, accelerators, tensor cores | Chapters 12–13 and 22 | Real-device correctness, synchronization checks, measured performance |
| CUDA/HIP competitive with PyTorch | Sections 20.4 and 22.6 | Matched backend, precision, shapes, and timing protocol |
| Real LLM training; symbolic execution | Chapters 14–15 and 23 | Training record plus variable-shape execution and cache behavior |
| Implement a paper or hardware port | Chapters 15 and 25 | Reproduction or port with correctness and performance report |
| Frequent quizzes, correctness, style, performance | Appendix C and Chapter 25 | Closed-book explanations, independent tests, code review, benchmark ledger |

## 20.2 Dtype broadcasting is a semantic decision

The supplied syllabus requests dtype broadcasting without specifying its complete promotion table.
The reference IR instead requires equal operand dtypes for many arithmetic operations, with explicit
CAST nodes. Its tests intentionally reject `i32 + f32`. Tensor shape broadcasting is a separate rule.
This is a documented difference from the requested interface, not evidence that mixed arithmetic works.

To implement promotion, first write a table for every supported pair. A possible teaching policy is
`bool < i32 < f32`, promoting arithmetic operands to the higher type, while keeping comparisons boolean.
This policy is a project choice, not an inferred official oracle rule. Specify integer division,
boolean arithmetic, f16 interactions, and float-to-int conversion independently. Insert casts at a
single boundary so the low-level IR retains simple equal-type invariants. Test the table exhaustively,
including `i32(16777217)` promoted to f32, where representability changes the value.

**Gate:** either implement and test the course's agreed promotion policy or mark this requirement
incomplete. A shape-broadcasting test cannot satisfy the dtype requirement.

## 20.3 Materialization before general movement ops

In the week-4 subset, compile `sum(x*x)` and `sum(x*x) + sum(x*x)` for one-dimensional x. First inline
all work, then introduce a STAGE boundary and count loads, multiplies, and temporary bytes. A stage
can save repeated compute while adding an allocation and memory traffic. It is not always an improvement.

After this works, unlock general movement ops and derive matrix multiplication with broadcasted
products and a reduction. Follow Chapter 6's index maps to avoid materializing the full `(M,N,K)`
product. Verify that implicit loads lower to explicit memory reads before rendering.

**Gate:** show generated loops for both the restricted vector case and the general matrix case.
Explain where the temporary exists and why its store happens before any dependent load.

## 20.4 Competitive performance is a measured claim

“SOTA-competitive” and “torch-competitive” are course targets, not properties implied by correct code.
Choose comparison thresholds before tuning and publish every workload, including regressions.
A credible table includes small and large GEMMs, rectangular shapes, reductions, and convolutions;
one favorable square GEMM is insufficient. Match input storage, accumulation, numerical policy,
layout, device, and thread count. Compare steady-state device execution separately from end-to-end
latency including transfers and compilation.

The reference book's CPU measurements are examples from its recorded machine. GPU render tests and
emulation establish narrower properties. The workspace may include real-device tests in
`tests/test_w08_cuda.py`; their existence does not establish that they ran or passed on your hardware.
HIP support is a backend project; emitting CUDA does not demonstrate AMD execution.

**Gate:** retain raw timing samples, commands, versions, tolerances, profiler captures, and generated
kernels. Mark hardware-dependent results “not run” when hardware is unavailable.

## 20.5 A small GPT and a modern LLM milestone

Chapter 14's small character GPT teaches the training pipeline. To meet a larger language-model
milestone, specify the architecture, parameter count, tokenizer, data, context length, precision,
optimizer, checkpoint format, and available compute before starting. Verify a tiny version first.
Load compatible pretrained weights for a forward comparison if the project permits it; this checks
integration but does not replace training.

Scale gradually and record tokens per second, peak memory, compile time, loss, and validation quality.
Save optimizer state and random-number state as well as weights; compare a resumed run against an
uninterrupted run. A short training run on a larger architecture demonstrates execution, not
state-of-the-art model quality. Set the intended training and quality criterion explicitly.

**Gate:** reproduce the tiny training run, pass forward and gradient checks, demonstrate checkpoint
resume, then report the largest configuration actually trained. Never relabel the character GPT as
SOTA LLM training.

## 20.6 Completion ledger

For each row in section 20.1, record `explained`, `implemented`, `correctness verified`, and
`performance verified` separately, with a commit, command, and artifact. Chapters 21–29 provide the
advanced learning path; completing their labs requires additional implementation. The book teaches
the route and acceptance criteria without claiming the reference compiler already fulfills every target.
