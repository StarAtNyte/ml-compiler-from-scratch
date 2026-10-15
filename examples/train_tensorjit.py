"""Train a linear model through generated dynamic forward/backward kernels.

Run: python -m examples.train_tensorjit. Requires gcc; no PyTorch dependency.
Changing minibatch length reuses one compiled library. NumPy performs SGD updates.
"""
import numpy as np
from tensorjit import CompiledGraph, Dim, gradients, input


def train(steps=100, seed=7):
  if steps < 1: raise ValueError("steps must be positive")
  batch = Dim("B", 1, 64)
  x, y = input("x", (batch, 3)), input("y", (batch, 2))
  w, bias, scale = input("w", (3, 2)), input("bias", (2,)), input("scale", ())
  error = x @ w + bias - y
  loss = (error*error).sum(0).sum(0)*scale
  dw, db = gradients(loss, (w, bias))
  compiled = CompiledGraph((loss, dw, db))
  rng = np.random.default_rng(seed)
  true_w, true_bias = rng.normal(size=(3, 2)).astype(np.float32), rng.normal(size=2).astype(np.float32)
  weights, biases = np.zeros((3, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
  history = []
  for step in range(steps):
    length = (8, 16, 32)[step % 3]
    data = rng.normal(size=(length, 3)).astype(np.float32)
    target = data @ true_w + true_bias
    actual_loss, grad_w, grad_b = compiled(x=data, y=target, w=weights, bias=biases, scale=np.array(1/(length*2), dtype=np.float32))
    history.append(float(actual_loss))
    weights -= np.float32(0.15)*grad_w
    biases -= np.float32(0.15)*grad_b
  return {"initial_loss": history[0], "final_loss": history[-1], "parameter_max_error": float(np.abs(weights-true_w).max()),
          "bias_max_error": float(np.abs(biases-true_bias).max()), "compiled_variants": compiled.compilations,
          "calls": compiled.calls, "kernels": len(compiled.plan.kernels)}


def main():
  result = train()
  for key, value in result.items(): print(f"{key}: {value}")
  assert result["parameter_max_error"] < 1e-3 and result["bias_max_error"] < 1e-3


if __name__ == "__main__": main()
