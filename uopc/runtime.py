"""Week 2: the runtime. Compile C with the system compiler, load it, call it.

Compiled shared objects are cached on disk by a hash of (source, flags), so a
program is only compiled once per machine.
"""
from __future__ import annotations
import ctypes, hashlib, os, subprocess, tempfile
from pathlib import Path
import numpy as np
from uopc.dtype import DType, dtypes

CC = os.environ.get("CC", "gcc")
# -fassociative-math lets gcc vectorize reductions (it reorders float adds, like every ML compiler does).
# NaN/inf semantics are kept. Set CFLAGS to override, e.g. CFLAGS="-O2 -fopenmp" for strict IEEE order.
CFLAGS = os.environ.get("CFLAGS", "-O3 -march=native -fopenmp -ffp-contract=fast -fassociative-math "
                                  "-fno-signed-zeros -fno-trapping-math").split()
CACHE = Path(os.environ.get("UOPC_CACHE", Path.home() / ".cache" / "uopc"))

NP = {dtypes.f32: np.float32, dtypes.i32: np.int32, dtypes.bool: np.bool_, dtypes.f16: np.float16}

def np_dtype(d: DType): return NP[d]


class CompileError(Exception): pass


def compile_c(src: str, flags: list[str] | None = None) -> Path:
  flags = CFLAGS if flags is None else flags
  key = hashlib.sha256((src + "\0" + " ".join([CC, *flags])).encode()).hexdigest()[:24]
  CACHE.mkdir(parents=True, exist_ok=True)
  so = CACHE / f"{key}.so"
  if so.exists(): return so
  with tempfile.TemporaryDirectory(dir=CACHE) as td:   # same filesystem, so os.replace is atomic
    c = Path(td) / "k.c"
    c.write_text(src)
    tmp = Path(td) / "k.so"
    p = subprocess.run([CC, *flags, "-shared", "-fPIC", "-o", str(tmp), str(c), "-lm"], capture_output=True, text=True)
    if p.returncode != 0:
      numbered = "\n".join(f"{i+1:4d} {l}" for i, l in enumerate(src.splitlines()))
      raise CompileError(f"{CC} failed:\n{p.stderr}\n--- source ---\n{numbered}")
    os.replace(tmp, so)   # atomic: concurrent compiles of the same kernel are fine
  return so


class Program:
  """A compiled C function taking one pointer per buffer slot."""
  def __init__(self, src: str, name: str = "kernel", flags: list[str] | None = None):
    self.src, self.name = src, name
    self.lib = ctypes.CDLL(str(compile_c(src, flags)))
    self.fn = getattr(self.lib, name)
    self.fn.restype = None

  def __call__(self, *bufs: np.ndarray):
    for b in bufs:
      if not b.flags.c_contiguous: raise ValueError("buffers must be contiguous")
    self.fn(*(ctypes.c_void_p(b.ctypes.data) for b in bufs))


def run_c(sink, buffers: dict[int, np.ndarray], name: str = "kernel") -> str:
  """Render, compile and run a SINK of stores. buffers maps slot -> numpy array. Returns the C source."""
  from uopc.render_c import render_c
  from uopc.uop import Ops
  src = render_c(sink, name)
  bufs = sorted({u for u in sink.toposort() if u.op is Ops.BUFFER}, key=lambda u: u.arg.slot)
  for b in bufs:
    a = buffers[b.arg.slot]
    if a.dtype != np_dtype(b.dtype) or a.size < b.arg.shape[0]:
      raise TypeError(f"buffer slot {b.arg.slot}: kernel wants {b.arg.shape[0]} x {b.dtype.name}, got {a.size} x {a.dtype}")
  Program(src, name)(*(buffers[b.arg.slot] for b in bufs))
  return src
