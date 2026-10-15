import numpy as np
from tensorjit import CompiledGraph, Dim, input


def main():
  batch, inner, width = Dim("batch", 1, 32), Dim("inner", 1, 64), Dim("width", 1, 64)
  x, weights, bias = input("x", (batch, inner)), input("weights", (inner, width)), input("bias", (width,))
  output = ((x @ weights) + bias).relu().sum(axis=1)
  compiled = CompiledGraph((output,))
  rng = np.random.default_rng(0)
  for b, k, n in ((1, 7, 5), (4, 16, 9), (7, 31, 17)):
    arrays = {"x": rng.normal(size=(b, k)).astype(np.float32), "weights": rng.normal(size=(k, n)).astype(np.float32),
              "bias": rng.normal(size=n).astype(np.float32)}
    result, = compiled(**arrays)
    reference = np.maximum(arrays["x"] @ arrays["weights"] + arrays["bias"], 0).sum(axis=1)
    np.testing.assert_allclose(result, reference, atol=2e-5, rtol=2e-5)
    print(f"shape={b,k,n}; kernels={compiled.last_memory['kernel_count']}; workspace={compiled.last_memory['temporary_bytes']} bytes")
  print(f"compiled variants={compiled.compilations}; calls={compiled.calls}")


if __name__ == "__main__": main()
