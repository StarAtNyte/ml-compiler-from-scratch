"""Tiled online attention and recomputing backward, using NumPy as a CPU lab.

Single head, Q:(queries,dim), K:(keys,dim), V:(keys,value_dim).
Causal queries use absolute positions starting at query_offset. Fully masked rows
produce zero output, -inf log-normalizer, and zero gradients by explicit policy.
Run: python -m labs.attention
"""
import numpy as np


def _inputs(q, k, v, tile, query_offset):
  if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
    raise ValueError("expected compatible rank-two Q, K, V")
  if min(*q.shape, *k.shape, *v.shape) < 1 or tile < 1 or query_offset < 0:
    raise ValueError("dimensions and tile must be positive; query_offset nonnegative")
  if not all(np.issubdtype(x.dtype, np.floating) and np.isfinite(x).all() for x in (q, k, v)):
    raise ValueError("this lab requires finite floating-point inputs")


def _scores(q, k_tile, start, scale, causal, query_offset):
  scores = (q @ k_tile.T) * scale
  if causal:
    mask = np.arange(start, start + len(k_tile))[None, :] > (np.arange(len(q)) + query_offset)[:, None]
    scores = np.where(mask, -np.inf, scores)
  return scores


def online_attention(q, k, v, tile=32, causal=False, query_offset=0):
  _inputs(q, k, v, tile, query_offset)
  dtype = np.result_type(q, k, v, np.float32)
  q, k, v = (x.astype(dtype, copy=False) for x in (q, k, v))
  maximum = np.full(len(q), -np.inf, dtype=dtype)
  normalizer = np.zeros(len(q), dtype=dtype)
  output = np.zeros((len(q), v.shape[1]), dtype=dtype)
  scale = 1 / np.sqrt(q.shape[1])
  for start in range(0, len(k), tile):
    scores = _scores(q, k[start:start+tile], start, scale, causal, query_offset)
    next_maximum = np.maximum(maximum, scores.max(axis=1))
    # Zero is a harmless exponent reference for a still-fully-masked row.
    safe_maximum = np.where(np.isfinite(next_maximum), next_maximum, 0)
    alpha = np.exp(maximum - safe_maximum)
    probability = np.exp(scores - safe_maximum[:, None])
    output = alpha[:, None] * output + probability @ v[start:start+tile]
    normalizer = alpha * normalizer + probability.sum(axis=1)
    maximum = next_maximum
  output = np.divide(output, normalizer[:, None], out=np.zeros_like(output), where=normalizer[:, None] != 0)
  log_normalizer = np.full_like(maximum, -np.inf)
  valid = normalizer > 0
  log_normalizer[valid] = maximum[valid] + np.log(normalizer[valid])
  return output, log_normalizer


def attention_backward(q, k, v, output, log_normalizer, grad_output, tile=32, causal=False, query_offset=0):
  """Recompute probability tiles; never allocate a full queries-by-keys matrix."""
  _inputs(q, k, v, tile, query_offset)
  dtype = np.result_type(q, k, v, grad_output, np.float32)
  q, k, v = (x.astype(dtype, copy=False) for x in (q, k, v))
  if output.shape != (len(q), v.shape[1]) or grad_output.shape != output.shape or log_normalizer.shape != (len(q),):
    raise ValueError("invalid saved state or output gradient shape")
  dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
  correction = (grad_output * output).sum(axis=1)
  scale = 1 / np.sqrt(q.shape[1])
  valid = np.isfinite(log_normalizer)
  safe_log = np.where(valid, log_normalizer, 0)
  for start in range(0, len(k), tile):
    scores = _scores(q, k[start:start+tile], start, scale, causal, query_offset)
    probability = np.exp(scores - safe_log[:, None])
    probability[~valid] = 0
    dp = grad_output @ v[start:start+tile].T
    ds = probability * (dp - correction[:, None])
    dq += (ds @ k[start:start+tile]) * scale
    dk[start:start+tile] += (ds.T @ q) * scale
    dv[start:start+tile] += probability.T @ grad_output
  return dq, dk, dv


def main():
  rng = np.random.default_rng(0)
  q, k, v = (rng.normal(size=shape) for shape in ((7, 5), (11, 5), (11, 3)))
  output, log_normalizer = online_attention(q, k, v, tile=4, causal=True)
  gradients = attention_backward(q, k, v, output, log_normalizer, np.ones_like(output), tile=4, causal=True)
  print("output:", output.shape, "saved row statistics:", log_normalizer.shape)
  print("gradient shapes:", *(x.shape for x in gradients))
  print("CPU algorithm lab; no GPU performance claim")


if __name__ == "__main__": main()
