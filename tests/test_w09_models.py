"""Weeks 9-10: a tiny GPT trains (loss goes down) and its pieces match numpy."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gpt import GPT, Attention
from uopc.tensor import Tensor
from uopc.nn import LayerNorm, AdamW, parameters

def test_layernorm_matches_numpy():
  x = np.random.randn(4, 8).astype(np.float32)
  y = LayerNorm(8)(Tensor(x)).numpy()
  ref = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
  np.testing.assert_allclose(y, ref, atol=1e-5)

def test_attention_is_causal():
  Tensor.manual_seed(0)
  att = Attention(8, 2, 6)
  x = np.random.randn(1, 6, 8).astype(np.float32)
  y1 = att(Tensor(x)).numpy()
  x2 = x.copy(); x2[0, 4:] += 10                 # change the future
  y2 = att(Tensor(x2)).numpy()
  np.testing.assert_allclose(y1[0, :4], y2[0, :4], atol=1e-5)   # the past doesn't see it

def test_tiny_gpt_learns():
  Tensor.manual_seed(0)
  V, T = 11, 8
  model = GPT(V, T, 16, 2, 1)
  opt = AdamW(parameters(model), lr=1e-2)
  seq = np.arange(64) % V                         # a trivially predictable sequence
  x = Tensor(np.stack([seq[i:i + T] for i in range(4)]).astype(np.int32))
  y = Tensor(np.stack([seq[i + 1:i + T + 1] for i in range(4)]).astype(np.int32))
  losses = []
  for _ in range(30):
    loss = model(x).reshape(-1, V).cross_entropy(y.reshape(-1))
    opt.zero_grad(); loss.backward(opt.params); opt.step(loss)
    losses.append(loss.item())
  assert losses[-1] < 0.25 * losses[0]
