import numpy as np
from uopc.uop import UOp, Ops, ParamArg, prod
from uopc.dtype import dtypes
from uopc.rangeify import lower
from uopc.runtime import run_c, np_dtype

def buf(slot, n, dtype=dtypes.f32): return UOp(Ops.BUFFER, (), ParamArg(slot, dtype, (n,)))

def run_shaped(root: UOp, inputs: list[np.ndarray]) -> np.ndarray:
  """Lower a shaped graph whose BUFFERs are slots 0..len(inputs)-1, run it, return the result."""
  n = max(1, prod(root.shape))
  out = buf(len(inputs), n, root.dtype)
  res = np.zeros(n, np_dtype(root.dtype))
  run_c(lower(root, out), {**dict(enumerate(inputs)), len(inputs): res})
  return res.reshape(root.shape)
