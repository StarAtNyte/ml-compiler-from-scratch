"""Run the actual register-tiled kernels when a driver and NVRTC are available."""
import pytest


def available():
  try:
    from uopc.cuda import CUDA, CUDAError
    CUDA()
    return True
  except (OSError, CUDAError): return False


@pytest.mark.skipif(not available(), reason="CUDA driver/NVRTC unavailable")
@pytest.mark.parametrize("shape", [(1, 1, 1), (31, 33, 17), (63, 65, 33), (128, 96, 80)])
def test_naive_and_register_tiled_kernels_on_actual_gpu(shape):
  from labs.gpu_gemm import benchmark
  report = benchmark(shape, repeat=2, batch=2)
  assert set(report["results"]) == {"gemm_naive", "gemm_tiled"}
  # benchmark checks every element against an independent double-precision oracle.
  assert all(result["median_ms"] > 0 for result in report["results"].values())
