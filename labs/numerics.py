"""Mergeable variance and per-channel int8 quantization. Run: python -m labs.numerics"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Moments:
  count: int = 0
  mean: float = 0.0
  m2: float = 0.0

  def add(self, value):
    count = self.count + 1
    delta = float(value) - self.mean
    mean = self.mean + delta / count
    return Moments(count, mean, self.m2 + delta * (float(value) - mean))

  def merge(self, other):
    if self.count == 0: return other
    if other.count == 0: return self
    count = self.count + other.count
    delta = other.mean - self.mean
    return Moments(count, self.mean + delta * other.count / count,
                   self.m2 + other.m2 + delta*delta * self.count * other.count / count)

  def variance(self):
    if self.count == 0: raise ValueError("empty population")
    return self.m2 / self.count


def moments(values):
  state = Moments()
  for value in values: state = state.add(value)
  return state


def quantize_weights(weights):
  """Symmetric per-column, ties-to-even; input must be a finite nonempty matrix."""
  weights = np.asarray(weights, dtype=np.float64)
  if weights.ndim != 2 or min(weights.shape) < 1 or not np.isfinite(weights).all():
    raise ValueError("expected finite nonempty weight matrix")
  maxima = np.max(np.abs(weights), axis=0)
  scale = np.where(maxima == 0, 1.0, maxima / 127)
  quantized = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
  return quantized, scale


def main():
  rng = np.random.default_rng(0)
  values = 1e8 + rng.normal(size=1000)
  state = moments(values[:500]).merge(moments(values[500:]))
  print(f"variance: merged={state.variance():.8f}, reference={np.var(values):.8f}")
  weights = rng.normal(size=(64, 8)); weights[:, 0] = 0
  q, scale = quantize_weights(weights)
  print(f"int8 reconstruction max error={np.abs(q*scale - weights).max():.6f}")


if __name__ == "__main__": main()
