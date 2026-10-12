"""Week 5: the Tensor frontend, kernelization (fusion/fission), autodiff."""
import math
import numpy as np, pytest
from uopc.tensor import Tensor
from uopc.kernelize import make_plan
from uopc.nn import Linear, SGD, AdamW, parameters

rng = np.random.default_rng(0)
def r(*s): return rng.standard_normal(s).astype(np.float32)

@pytest.mark.parametrize("name,fn,ref", [
  ("add_broadcast", lambda a, b: a + b[0], lambda a, b: a + b[0]),
  ("mul_sub_div", lambda a, b: (a * b - 3) / (b.abs() + 1), lambda a, b: (a * b - 3) / (np.abs(b) + 1)),
  ("exp_log", lambda a, b: (a.exp() + 1).log(), lambda a, b: np.log(np.exp(a) + 1)),
  ("sum_axes", lambda a, b: a.sum((0, 2)), lambda a, b: a.sum((0, 2))),
  ("max_keepdim", lambda a, b: a.max(1, keepdim=True), lambda a, b: a.max(1, keepdims=True)),
  ("mean_var", lambda a, b: a.var(-1), lambda a, b: a.var(-1)),
  ("softmax", lambda a, b: a.softmax(-1), lambda a, b: np.exp(a - a.max(-1, keepdims=True)) / np.exp(a - a.max(-1, keepdims=True)).sum(-1, keepdims=True)),
  ("matmul_batched", lambda a, b: a @ b.transpose(), lambda a, b: a @ b.transpose(0, 2, 1)),
  ("where", lambda a, b: (a > b).where(a, b * 2), lambda a, b: np.where(a > b, a, b * 2)),
  ("getitem_cat", lambda a, b: Tensor.cat([a[:, 1:], b[:, :1]], 1), lambda a, b: np.concatenate([a[:, 1:], b[:, :1]], 1)),
  ("pad_flip", lambda a, b: a.pad(((0, 0), (1, 2), (0, 1))).flip(1), lambda a, b: np.pad(a, ((0, 0), (1, 2), (0, 1)))[:, ::-1]),
])
def test_ops(name, fn, ref):
  a, b = r(2, 3, 4), r(2, 3, 4)
  np.testing.assert_allclose(fn(Tensor(a), Tensor(b)).numpy(), ref(a, b), rtol=1e-4, atol=1e-5)

def numgrad(f, x, eps=1e-3):
  g = np.zeros_like(x, dtype=np.float64)
  for i in np.ndindex(x.shape):
    xp, xm = x.astype(np.float64), x.astype(np.float64)
    xp[i] += eps; xm[i] -= eps
    g[i] = (f(xp) - f(xm)) / (2 * eps)
  return g

@pytest.mark.parametrize("name,fn,npfn", [
  ("tanh_sum", lambda t: t.tanh().sum(), lambda x: np.tanh(x).sum()),
  ("softmax_xent", lambda t: t.log_softmax(-1)[:, 1].sum(), lambda x: (x - np.log(np.exp(x).sum(-1, keepdims=True)))[:, 1].sum()),
  ("max", lambda t: (t.max(-1) * t.sum(0).max()).sum(), lambda x: (x.max(-1) * x.sum(0).max()).sum()),
  ("pad_shrink", lambda t: t.pad(((1, 1), (0, 2)))[1:3, 1:].square().sum(),
   lambda x: (np.pad(x, ((1, 1), (0, 2)))[1:3, 1:] ** 2).sum()),
  ("sqrt_recip", lambda t: (t.square() + 1).sqrt().reciprocal().sum(), lambda x: (1 / np.sqrt(x ** 2 + 1)).sum()),
  ("matmul", lambda t: (t @ t.T).sin().sum(), lambda x: np.sin(x @ x.T).sum()),
])
def test_gradients(name, fn, npfn):
  x = r(3, 4)
  t = Tensor(x, requires_grad=True)
  fn(t).backward([t])
  np.testing.assert_allclose(t.grad.numpy(), numgrad(npfn, x), rtol=2e-2, atol=2e-3)

def test_kernelize_fuses_elementwise_and_splits_reductions():
  a = Tensor(r(64, 64))
  plan, _ = make_plan([((a * 2 + 1).relu() * 3).uop])
  assert len(plan.kernels) == 1                         # a chain of elementwise ops: one kernel
  plan, _ = make_plan([a.softmax(-1).uop])
  assert len(plan.kernels) == 3                         # max, sum, and the elementwise output

def test_training_reduces_loss():
  Tensor.manual_seed(1)
  X = r(64, 8); Y = (X.sum(1) > 0).astype(np.int32)
  l1, l2 = Linear(8, 16), Linear(16, 2)
  opt = AdamW(parameters([l1, l2]), lr=1e-2)
  losses = []
  for _ in range(40):
    loss = l2(l1(Tensor(X)).relu()).cross_entropy(Tensor(Y))
    opt.zero_grad(); loss.backward(opt.params); opt.step(loss)
    losses.append(loss.item())
  assert losses[-1] < 0.3 * losses[0]

def test_sgd_momentum_step():
  w = Tensor(np.array([1.0, -2.0], np.float32), requires_grad=True)
  opt = SGD([w], lr=0.1, momentum=0.9)
  for _ in range(2):
    loss = w.square().sum()
    opt.zero_grad(); loss.backward(opt.params); opt.step()
  # v1 = 2w0, w1 = w0 - .1*v1 = .8 w0;  v2 = .9 v1 + 2 w1 = 3.4 w0, w2 = .8 w0 - .34 w0 = .46 w0
  np.testing.assert_allclose(w.numpy(), np.array([0.46, -0.92]), rtol=1e-5)

def test_deleted_reads_do_not_shift_kernel_arguments():
  t = Tensor(np.array([1, 2, 3], np.int32))
  assert (t * 0).numpy().tolist() == [0, 0, 0]          # simplifies to a constant: t is no longer read...
  assert t.numpy().tolist() == [1, 2, 3]                # ...and must not be overwritten

def test_python_float_promotes_int_tensors():
  t = Tensor(np.array([1, 2, 3], np.int32))
  np.testing.assert_allclose((t * 0.5).numpy(), [0.5, 1.0, 1.5])
  assert t.mean().item() == 2.0

def test_item_before_backward_and_detach():
  w = Tensor(np.array([1.0, 2.0], np.float32), requires_grad=True)
  loss = (w * w).sum()
  loss.item()                                          # realizing the loss must keep its graph
  loss.backward([w])
  np.testing.assert_allclose(w.grad.numpy(), [2.0, 4.0])
  w.grad = None
  (w.detach() * w).sum().backward([w])                 # d/dw (c * w) with c = w held constant
  np.testing.assert_allclose(w.grad.numpy(), [1.0, 2.0])

def test_adamw_steps_reuse_one_plan():                             # NOTES 17
  from uopc import kernelize
  Tensor.manual_seed(0)
  lin = Linear(4, 2)
  opt = AdamW([lin.weight, lin.bias], lr=1e-2)
  x = Tensor(np.random.randn(8, 4).astype(np.float32))
  def step():
    loss = lin(x).square().mean()
    opt.zero_grad(); loss.backward(opt.params); opt.step(loss)
  step(); step()
  n = len(kernelize._plans)
  for _ in range(3): step()
  assert len(kernelize._plans) == n                                # no new plans: nothing recompiled
