"""Week 5: layers and optimizers, written against the Tensor API only."""
from __future__ import annotations
import math
from uopc.tensor import Tensor


class Linear:
  def __init__(self, n_in: int, n_out: int, bias: bool = True):
    bound = 1 / math.sqrt(n_in)
    self.weight = Tensor.uniform(n_in, n_out, low=-bound, high=bound, requires_grad=True)
    self.bias = Tensor.zeros(n_out).realize() if bias else None
    if self.bias is not None: self.bias.requires_grad = True
  def __call__(self, x: Tensor) -> Tensor: return x.linear(self.weight, self.bias)


class Conv2d:
  def __init__(self, c_in: int, c_out: int, k: int, padding: int = 0, stride: int = 1):
    bound = 1 / math.sqrt(c_in * k * k)
    self.weight = Tensor.uniform(c_out, c_in, k, k, low=-bound, high=bound, requires_grad=True)
    self.bias = Tensor.zeros(c_out).realize()
    self.bias.requires_grad = True
    self.padding, self.stride = padding, stride
  def __call__(self, x: Tensor) -> Tensor: return x.conv2d(self.weight, self.bias, self.padding, self.stride)


class LayerNorm:
  def __init__(self, n: int, eps: float = 1e-5):
    self.weight, self.bias, self.eps = Tensor.ones(n).realize(), Tensor.zeros(n).realize(), eps
    self.weight.requires_grad = self.bias.requires_grad = True
  def __call__(self, x: Tensor) -> Tensor:
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    return xc * (xc.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight + self.bias


class Embedding:
  """A one-hot matmul: differentiable without a scatter op."""
  def __init__(self, vocab: int, dim: int):
    self.vocab, self.weight = vocab, Tensor.randn(vocab, dim) * 0.02
    self.weight = self.weight.realize(); self.weight.requires_grad = True
  def __call__(self, idx: Tensor) -> Tensor: return idx.one_hot(self.vocab) @ self.weight


def parameters(obj) -> list[Tensor]:
  """Every Tensor with requires_grad reachable through attributes, lists and dicts."""
  out, seen = [], set()
  def walk(o):
    if id(o) in seen: return
    seen.add(id(o))
    if isinstance(o, Tensor):
      if o.requires_grad: out.append(o)
    elif isinstance(o, (list, tuple)): [walk(x) for x in o]
    elif isinstance(o, dict): [walk(x) for x in o.values()]
    elif hasattr(o, "__dict__"): [walk(x) for x in vars(o).values()]
  walk(obj)
  return out


class Optimizer:
  def __init__(self, params: list[Tensor]): self.params = params
  def zero_grad(self):
    for p in self.params: p.grad = None


class SGD(Optimizer):
  def __init__(self, params, lr=0.01, momentum=0.0):
    super().__init__(params)
    self.lr, self.momentum = lr, momentum
    self.v = [Tensor.zeros(*p.shape).realize() for p in params] if momentum else []
  def step(self, *extra: Tensor):
    """Update every parameter. Extra tensors (e.g. the loss) are realized in the same batch."""
    news = []
    for i, p in enumerate(self.params):
      g = p.grad
      if self.momentum:
        self.v[i].assign(self.v[i] * self.momentum + g); g = self.v[i]
      news.append(p.assign(p.detach() - g * self.lr))
    Tensor.realize_all(*news, *self.v, *extra)


class AdamW(Optimizer):
  def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps, self.wd, self.t = lr, b1, b2, eps, wd, 0
    self.m = [Tensor.zeros(*p.shape).realize() for p in params]
    self.v = [Tensor.zeros(*p.shape).realize() for p in params]
  def step(self, *extra: Tensor):
    self.t += 1
    # bias corrections are python floats baked into the graph as constants: each step's
    # graph differs only in two CONSTs, so we pass them as tensors to keep the JIT cache warm
    c1 = Tensor([self.lr / (1 - self.b1 ** self.t)])
    c2 = Tensor([1 / (1 - self.b2 ** self.t)])
    news = []
    for p, m, v in zip(self.params, self.m, self.v):
      g = p.grad
      m.assign(m * self.b1 + g * (1 - self.b1))
      v.assign(v * self.b2 + g.square() * (1 - self.b2))
      upd = m * c1.reshape(*([1] * p.ndim)) / ((v * c2.reshape(*([1] * p.ndim))).sqrt() + self.eps)
      news.append(p.assign(p.detach() * (1 - self.lr * self.wd) - upd))
    Tensor.realize_all(*news, *self.m, *self.v, *extra)
