"""Measure dynamic tensor compilation, fusion, and workspace reuse.

Run from root: python -m examples.bench_tensorjit --sizes 16 32 64 --repeat 10
Use --output path.json to retain raw samples and the environment. CPU only.
The NumPy reference is an end-to-end expression, not a matched low-level GEMM.
"""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from tensorjit import CompiledGraph, Dim, input


def samples(call, repeat):
  call()  # Warm up; NumPy may initialize its BLAS thread pool.
  result = []
  for _ in range(repeat):
    start = time.perf_counter()
    call()
    result.append((time.perf_counter()-start)*1000)
  return result


def benchmark(sizes, repeat):
  if repeat < 1 or not sizes or min(sizes) < 1 or max(sizes) > 512:
    raise ValueError("positive repeats and sizes in [1,512] required")
  n = Dim("N", 1, 512)
  x, a, b, c, bias = (input(name, shape) for name, shape in
                       (("x", (n, n)), ("a", (n, n)), ("b", (n, n)), ("c", (n, n)), ("bias", (n,))))
  output = ((((x @ a) @ b) @ c) + bias).relu().sum(1)
  variants, compile_ms = {}, {}
  for fuse in (False, True):
    key = "fused" if fuse else "unfused"
    start = time.perf_counter()
    variants[key] = CompiledGraph((output,), fuse=fuse)
    compile_ms[key] = (time.perf_counter()-start)*1000
  report = {"platform": platform.platform(), "processor": platform.processor(), "numpy": np.__version__,
            "compiler": subprocess.run(["gcc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "compile_ms": compile_ms, "cases": []}
  rng = np.random.default_rng(17)
  for size in sizes:
    arrays = {name: rng.normal(size=(size, size)).astype(np.float32)*np.float32(1/np.sqrt(size)) for name in ("x", "a", "b", "c")}
    arrays["bias"] = rng.normal(size=size).astype(np.float32)
    def reference(): return np.fmax(((arrays["x"] @ arrays["a"]) @ arrays["b"]) @ arrays["c"] + arrays["bias"], 0).sum(1)
    expected = reference()
    case = {"shape": [size, size], "numpy_ms": samples(reference, repeat)}
    for key, compiled in variants.items():
      actual, = compiled(**arrays)
      np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
      timing = samples(lambda: compiled(**arrays), repeat)
      case[key] = {"samples_ms": timing, "median_ms": float(np.median(timing)),
                   "kernels": compiled.last_memory["kernel_count"],
                   "temporary_bytes": compiled.last_memory["temporary_bytes"],
                   "unreused_temporary_bytes": compiled.last_memory["unreused_temporary_bytes"],
                   "max_absolute_error": float(np.max(np.abs(actual-expected)))}
    report["cases"].append(case)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
  parser.add_argument("--repeat", type=int, default=10)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = benchmark(args.sizes, args.repeat)
  content = json.dumps(report, indent=2) + "\n"
  if args.output: args.output.write_text(content)
  print(content, end="")


if __name__ == "__main__": main()
