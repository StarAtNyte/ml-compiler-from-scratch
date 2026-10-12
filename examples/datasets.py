"""Tiny dataset loaders. Files are cached in ~/.cache/uopc/data."""
import gzip, urllib.request
from pathlib import Path
import numpy as np

DATA = Path.home() / ".cache" / "uopc" / "data"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def fetch(url: str, name: str) -> Path:
  DATA.mkdir(parents=True, exist_ok=True)
  p = DATA / name
  if not p.exists():
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, p)
  return p


def mnist():
  """(X_train (60000, 784) f32 in [0,1], Y_train (60000,) i32, X_test, Y_test)"""
  def images(n): return np.frombuffer(gzip.open(fetch(MNIST_URL + n, n)).read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255
  def labels(n): return np.frombuffer(gzip.open(fetch(MNIST_URL + n, n)).read(), np.uint8, offset=8).astype(np.int32)
  return (images("train-images-idx3-ubyte.gz"), labels("train-labels-idx1-ubyte.gz"),
          images("t10k-images-idx3-ubyte.gz"), labels("t10k-labels-idx1-ubyte.gz"))


def shakespeare() -> str:
  return fetch(SHAKESPEARE_URL, "tinyshakespeare.txt").read_text()
