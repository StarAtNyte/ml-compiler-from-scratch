# tensorjit: dynamic tensor graph to fused native execution

Compile a pure float32 graph with bounded dynamic dimensions, broadcasting, pointwise fusion,
materialized reductions/matmul, and an aligned temporary arena whose lifetimes come from the actual
fused kernel dependencies. Read book Chapter 33 for a worked implementation and measurements.

```bash
python -m tensorjit
python -m pytest tests/test_tensorjit.py -q
python -m examples.bench_tensorjit --sizes 16 32 64 --repeat 10
python -m examples.train_tensorjit
```

```python
import numpy as np
from tensorjit import Dim, input, CompiledGraph

batch, inner, width = Dim("B", 1, 32), Dim("K", 1, 64), Dim("N", 1, 64)
x = input("x", (batch, inner))
w = input("w", (inner, width))
bias = input("bias", (width,))
graph = CompiledGraph((((x @ w) + bias).relu().sum(1),))
result, = graph(x=np.ones((4, 7), np.float32), w=np.ones((7, 9), np.float32), bias=np.zeros(9, np.float32))
np.testing.assert_array_equal(result, np.full(4, 63, np.float32))
```

Requires gcc. Supported operations: add/multiply with trailing-axis broadcasting, ReLU, rank-two
transpose and matmul, single-axis sum, singleton insertion and explicit broadcast. Reverse-mode
`gradients(loss, inputs)` supports these pure operations and ReLU, with accumulated shared uses. Symbol names imply dimension equality; conflicting bounds
and unproved equalities fail. Inputs must be aligned contiguous float32 arrays. Dimensions are
positive, and zero-length axes, arbitrary strides, dtype promotion, arbitrary symbolic expressions,
GPU lowering, and higher derivatives through ReLU masks are not supported in this compiler.

ReLU follows `fmaxf(x,0)`, including numeric-zero selection for NaN. Sequential reductions differ
in rounding from optimized BLAS. Inputs are read-only; returned outputs have independent storage.
Calls have separate arenas and are serialized for consistent diagnostics. `max_bytes` accounts for
input/output payload plus the aligned arena extent; compiler storage, Python objects, and up to 63
base-alignment bytes are outside this budget. It is not a process memory limit.

`compilations` counts native library builds per object. `last_memory` reports the most recent
successful invocation. The benchmark includes guard/planning/allocation/dispatch costs and records
cold compilation separately. Naive matrix loops are intentionally a correctness baseline, not a
competitive GEMM implementation. Shape-specific planning and caching, optimized schedules, numerical
policies, remain implementation projects.


Chapter 34 explains the dynamic forward/backward training path. Shape-only gradient graphs can use
`compiled.run_with_bindings({"N": n}, **inputs)`; explicit dimensions are bounded and must agree with
any supplied input metadata. The training example uses NumPy for SGD updates and generated kernels
for loss and gradients, alternating batch sizes with one native library. It is a CPU integration
baseline, not large-model or GPU training evidence.
