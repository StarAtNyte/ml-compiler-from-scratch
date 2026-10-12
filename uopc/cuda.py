"""Weeks 7-8: a minimal CUDA runtime over ctypes: NVRTC to compile, the driver API to run.

NOTE: this file needs an NVIDIA GPU, a working driver (libcuda.so) and NVRTC
(libnvrtc.so, from the CUDA toolkit). The reference machine for this book had
neither working, so this module is exercised only by `tests/test_w07_gpu.py`
when CUDA is available; elsewhere GPU kernels are tested with gpu.emulate().
"""
from __future__ import annotations
import ctypes, ctypes.util
import numpy as np


def _lib(*names):
  for n in names:
    path = ctypes.util.find_library(n)
    if path: return ctypes.CDLL(path)
  raise OSError(f"none of {names} found")


class CUDAError(Exception): pass


class CUDA:
  def __init__(self, device: int = 0):
    self.cu = _lib("cuda")
    self.rtc = _lib("nvrtc")
    self._check(self.cu.cuInit(0))
    self.dev = ctypes.c_int()
    self._check(self.cu.cuDeviceGet(ctypes.byref(self.dev), device))
    self.ctx = ctypes.c_void_p()
    self._check(self.cu.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.dev))
    major, minor = ctypes.c_int(), ctypes.c_int()
    self._check(self.cu.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), self.dev))
    self.arch = f"sm_{major.value}{minor.value}"

  def _check(self, status):
    if status != 0: raise CUDAError(f"CUDA driver error {status}")

  def compile(self, src: str) -> bytes:
    prog = ctypes.c_void_p()
    if self.rtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"kernel.cu", 0, None, None):
      raise CUDAError("nvrtcCreateProgram failed")
    opts = [f"--gpu-architecture=compute_{self.arch[3:]}".encode(), b"--use_fast_math"]
    status = self.rtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    if status != 0:
      n = ctypes.c_size_t()
      self.rtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
      log = ctypes.create_string_buffer(n.value)
      self.rtc.nvrtcGetProgramLog(prog, log)
      raise CUDAError(f"nvrtc: {log.value.decode()}\n{src}")
    n = ctypes.c_size_t()
    self.rtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    self.rtc.nvrtcGetPTX(prog, ptx)
    return ptx.raw

  def alloc(self, nbytes: int) -> ctypes.c_uint64:
    p = ctypes.c_uint64()
    self._check(self.cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)))
    return p

  def to_device(self, a: np.ndarray) -> ctypes.c_uint64:
    p = self.alloc(a.nbytes)
    self._check(self.cu.cuMemcpyHtoD_v2(p, a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes)))
    return p

  def to_host(self, p: ctypes.c_uint64, a: np.ndarray):
    self._check(self.cu.cuMemcpyDtoH_v2(a.ctypes.data_as(ctypes.c_void_p), p, ctypes.c_size_t(a.nbytes)))

  def free(self, p): self.cu.cuMemFree_v2(p)

  def function(self, src: str, name: str):
    mod = ctypes.c_void_p()
    self._check(self.cu.cuModuleLoadData(ctypes.byref(mod), self.compile(src)))
    fn = ctypes.c_void_p()
    self._check(self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()))
    return fn

  def launch(self, fn, grid, block, *ptrs, shared: int = 0):
    args = (ctypes.c_void_p * len(ptrs))(*(ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ptrs))
    self._check(self.cu.cuLaunchKernel(fn, *grid, *block, shared, None, args, None))
    self._check(self.cu.cuCtxSynchronize())


def run_cuda(src: str, name: str, grid, block, *arrays: np.ndarray):
  """Copy arrays to the GPU, run, copy every array back (they may be outputs)."""
  cu = CUDA()
  ptrs = [cu.to_device(a) for a in arrays]
  cu.launch(cu.function(src, name), grid, block, *ptrs)
  for p, a in zip(ptrs, arrays): cu.to_host(p, a); cu.free(p)
