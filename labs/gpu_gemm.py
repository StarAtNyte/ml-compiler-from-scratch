"""Real-device GEMM lab: shared tiles, register tiles, tails, and event timing.

Run: python -m labs.gpu_gemm --shape 256 256 256 --repeat 10 --batch 20
Needs a working CUDA driver and NVRTC discoverable by uopc.cuda (toolkit setup).
This module never substitutes CPU emulation for a GPU result.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from uopc.cuda import CUDA

SOURCE_PATH = Path(__file__).parent / "kernels" / "gemm.cuh"


def enqueue(driver, function, grid, block, pointers):
  args = (ctypes.c_void_p * len(pointers))(*(ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in pointers))
  driver._check(driver.cu.cuLaunchKernel(function, *grid, *block, 0, None, args, None))


def event_samples(driver, function, grid, block, pointers, repeat, batch):
  start, stop = ctypes.c_void_p(), ctypes.c_void_p()
  driver._check(driver.cu.cuEventCreate(ctypes.byref(start), 0))
  try:
    driver._check(driver.cu.cuEventCreate(ctypes.byref(stop), 0))
    try:
      for _ in range(5): enqueue(driver, function, grid, block, pointers)
      driver._check(driver.cu.cuCtxSynchronize())
      samples = []
      for _ in range(repeat):
        driver._check(driver.cu.cuEventRecord(start, None))
        for _ in range(batch): enqueue(driver, function, grid, block, pointers)
        driver._check(driver.cu.cuEventRecord(stop, None))
        driver._check(driver.cu.cuEventSynchronize(stop))
        elapsed = ctypes.c_float()
        driver._check(driver.cu.cuEventElapsedTime(ctypes.byref(elapsed), start, stop))
        samples.append(elapsed.value / batch)
      return samples
    finally: driver._check(driver.cu.cuEventDestroy_v2(stop))
  finally: driver._check(driver.cu.cuEventDestroy_v2(start))


def benchmark(shape=(256, 256, 256), repeat=10, batch=20):
  if len(shape) != 3 or min(shape) < 1 or max(shape) > 2048 or repeat < 1 or batch < 1:
    raise ValueError("positive M,N,K <=2048, repeat, and batch required")
  m, n, k = map(int, shape)
  driver = CUDA()
  device_name = ctypes.create_string_buffer(256)
  driver._check(driver.cu.cuDeviceGetName(device_name, len(device_name), driver.dev))
  driver_version, rtc_major, rtc_minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
  driver._check(driver.cu.cuDriverGetVersion(ctypes.byref(driver_version)))
  if driver.rtc.nvrtcVersion(ctypes.byref(rtc_major), ctypes.byref(rtc_minor)):
    raise RuntimeError("could not query NVRTC version")
  rng = np.random.default_rng(11)
  a, b = rng.normal(size=(m, k)).astype(np.float32), rng.normal(size=(k, n)).astype(np.float32)
  # A double-precision CPU oracle checks the quantized float32 input values.
  expected = a.astype(np.float64) @ b.astype(np.float64)
  output = np.empty((m, n), dtype=np.float32)
  dimensions = np.array([m, n, k], dtype=np.int32)
  source = SOURCE_PATH.read_text()
  pointers = []
  results = {}
  try:
    for array in (a, b, output, dimensions): pointers.append(driver.to_device(array))
    for name, block, grid in (("gemm_naive", (16, 16, 1), ((n+15)//16, (m+15)//16, 1)),
                              ("gemm_tiled", (32, 8, 1), ((n+63)//64, (m+63)//64, 1))):
      output.fill(np.nan)
      driver._check(driver.cu.cuMemcpyHtoD_v2(pointers[2], output.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(output.nbytes)))
      started = time.perf_counter()
      function = driver.function(source, name)
      compile_ms = (time.perf_counter()-started)*1000
      enqueue(driver, function, grid, block, pointers)
      driver._check(driver.cu.cuCtxSynchronize())
      driver.to_host(pointers[2], output)
      np.testing.assert_allclose(output, expected, atol=3e-4, rtol=3e-4)
      timings = event_samples(driver, function, grid, block, pointers, repeat, batch)
      median = float(np.median(timings))
      results[name] = {"samples_ms": timings, "median_ms": median, "compile_ms": compile_ms,
                       "gflops": 2*m*n*k/(median*1e6), "max_absolute_error": float(np.abs(output-expected).max()),
                       "block": block, "grid": grid}
  finally:
    for pointer in pointers: driver.free(pointer)
  return {"device": device_name.value.decode(), "architecture": driver.arch, "driver_version": driver_version.value,
          "nvrtc_version": [rtc_major.value, rtc_minor.value], "shape": [m, n, k],
          "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "repeat": repeat, "batch": batch,
          "precision": "float32 inputs/accumulators; runtime NVRTC fast-math policy; no tensor cores",
          "timing": "CUDA event interval divided by batch; includes any host submission gaps; excludes compilation and transfers",
          "results": results}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256], metavar=("M", "N", "K"))
  parser.add_argument("--repeat", type=int, default=10)
  parser.add_argument("--batch", type=int, default=20)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  content = json.dumps(benchmark(args.shape, args.repeat, args.batch), indent=2)+"\n"
  if args.output: args.output.write_text(content)
  print(content, end="")


if __name__ == "__main__": main()
