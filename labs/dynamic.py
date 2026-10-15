"""Guarded C JIT: runtime length, shape bounds, and a layout fallback.

Run: python -m labs.dynamic. Requires gcc; uses the book's C runtime.
"""
import numpy as np
from uopc.runtime import Program


SOURCE = r"""
void square_plus_one(float *x, float *out, int *length) {
  for (int i = 0; i < length[0]; i++) out[i] = x[i]*x[i] + 1.0f;
}
"""


class GuardedJIT:
  def __init__(self, maximum=4096):
    if maximum < 1: raise ValueError("maximum must be positive")
    self.maximum, self.program = maximum, None
    self.compilations = self.fast_calls = self.fallback_calls = 0

  def __call__(self, x):
    if not isinstance(x, np.ndarray) or x.ndim != 1 or not np.issubdtype(x.dtype, np.floating):
      raise ValueError("expected a floating-point vector")
    if not 1 <= len(x) <= self.maximum: raise ValueError("length outside declared bounds")
    if x.dtype != np.float32 or not x.flags.c_contiguous or not x.flags.aligned:
      self.fallback_calls += 1
      return x*x + 1
    if self.program is None:
      self.program = Program(SOURCE, "square_plus_one")
      self.compilations += 1
    output = np.empty_like(x)
    self.program(x, output, np.array([len(x)], dtype=np.int32))
    self.fast_calls += 1
    return output


def main():
  jit = GuardedJIT(64)
  for n in (1, 7, 31, 64, 7):
    x = np.arange(n, dtype=np.float32)
    np.testing.assert_allclose(jit(x), x*x+1)
  x = np.arange(20, dtype=np.float32)[::2]
  np.testing.assert_allclose(jit(x), x*x+1)
  print(f"variants built={jit.compilations}, compiled calls={jit.fast_calls}, layout fallbacks={jit.fallback_calls}")


if __name__ == "__main__": main()
