"""Week 5 milestone: train an MLP on MNIST with our own compiler.

  python examples/mnist_mlp.py            # ~94% test accuracy in 600 steps (~15 ms/step)
  DEBUG=2 python examples/mnist_mlp.py    # print every kernel with its time
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from datasets import mnist
from uopc.tensor import Tensor
from uopc.nn import Linear, AdamW, parameters


class MLP:
  def __init__(self):
    self.l1, self.l2 = Linear(784, 128), Linear(128, 10)
  def __call__(self, x: Tensor) -> Tensor: return self.l2(self.l1(x).relu())


def main(steps=600, bs=128):
  Xtr, Ytr, Xte, Yte = mnist()
  Tensor.manual_seed(0)
  model = MLP()
  opt = AdamW(parameters(model), lr=1e-3)
  rng = np.random.default_rng(0)
  t0 = time.perf_counter()
  for step in range(steps):
    idx = rng.integers(0, len(Xtr), bs)
    x, y = Tensor(Xtr[idx]), Tensor(Ytr[idx])
    loss = model(x).cross_entropy(y)
    opt.zero_grad()
    loss.backward(opt.params)
    opt.step(loss)
    if step % 100 == 0 or step == steps - 1:
      print(f"step {step:4d}  loss {loss.item():.4f}  {(time.perf_counter() - t0) / (step + 1) * 1e3:.1f} ms/step")
  pred = np.concatenate([model(Tensor(Xte[i:i + 1000])).argmax(-1).numpy() for i in range(0, len(Xte), 1000)])
  acc = (pred == Yte).mean()
  print(f"test accuracy {acc * 100:.2f}%")
  return acc


if __name__ == "__main__": main()
