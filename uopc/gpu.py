"""Weeks 7-8: GPUs.

A GPU kernel is the body of a loop nest whose outer loops the hardware runs for
you: GLOBAL ranges become blockIdx, LOCAL ranges become threadIdx. Everything
else (reduce loops, upcasts, accumulators) is exactly what we already have.

  * CUDARenderer   -- renders the same linearized UOps as CUDA C
  * gpu_opt        -- picks blocks/threads/upcasts for a kernel
  * emulate        -- renders a GPU kernel with the *C* renderer, where GLOBAL and
                      LOCAL ranges are ordinary loops. Same index math, runs on a CPU,
                      so every GPU kernel is testable without a GPU.
  * smem_gemm_ir   -- a tiled GEMM written by hand in the loop IR, using SHARED memory
                      and BARRIER: the core trick of every fast GPU GEMM
  * wmma_gemm      -- tensor cores: pattern-match a GEMM and emit a WMMA kernel
"""
from __future__ import annotations
from uopc.uop import UOp, Ops, ParamArg, AxisType, prod, sink, range_
from uopc.dtype import dtypes, AddrSpace
from uopc.render_c import CRenderer
from uopc.rangeify import lower, Ctx, Opts

CUDA_PRELUDE = """#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
__device__ inline int floordiv_i32(int a, int b) { int q = a / b; return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q; }
__device__ inline int floormod_i32(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
"""


class CUDARenderer(CRenderer):
  buffer_prefix = ""
  shared_prefix = "__shared__ "
  barrier = "__syncthreads()"
  restrict = "__restrict__"

  def render(self, sink: UOp, name: str = "kernel") -> str:
    rngs = [u for u in sink.toposort() if u.op is Ops.RANGE]
    # the last (fastest varying) axis gets .x, so neighbouring threads touch neighbouring memory
    self.glob = [u for u in sorted(rngs, key=lambda r: r.arg, reverse=True) if u.arg.kind is AxisType.GLOBAL]
    self.loc = [u for u in sorted(rngs, key=lambda r: r.arg, reverse=True) if u.arg.kind is AxisType.LOCAL]
    assert len(self.glob) <= 3 and len(self.loc) <= 3, "fold axes: CUDA has 3 grid and 3 block dimensions"
    self.kernel_prefix = f'extern "C" __global__ void __launch_bounds__({max(1, prod(launch_dims(sink)[1]))})'
    # dtypes.f16.ctype is gcc's _Float16; CUDA spells it `half` (from cuda_fp16.h)
    return super().render(sink, name).replace("_Float16", "half")

  def header(self, sink, lin) -> str:
    return ("#include <cuda_fp16.h>\n" if any(u.dtype == dtypes.f16 for u in lin) else "") + CUDA_PRELUDE

  def is_loop(self, u: UOp) -> bool: return u.arg.kind not in (AxisType.GLOBAL, AxisType.LOCAL)

  def range_name(self, u: UOp) -> str:
    if u.arg.kind is AxisType.GLOBAL: return f"gidx{self.glob.index(u)}"
    if u.arg.kind is AxisType.LOCAL: return f"lidx{self.loc.index(u)}"
    return super().range_name(u)

  def render_range(self, u: UOp, name: str, end: str) -> list[str]:
    if u.arg.kind is AxisType.GLOBAL: return [f"int {name} = blockIdx.{'xyz'[self.glob.index(u)]}; /* {end} */"]
    if u.arg.kind is AxisType.LOCAL: return [f"int {name} = threadIdx.{'xyz'[self.loc.index(u)]}; /* {end} */"]
    return [f"for (int {name} = 0; {name} < {end}; {name}++) {{"]


def launch_dims(sink: UOp) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """(grid, block) sizes from the GLOBAL and LOCAL ranges, in x, y, z order."""
  rngs = sorted((u for u in sink.toposort() if u.op is Ops.RANGE), key=lambda r: r.arg, reverse=True)
  g = tuple(r.src[0].arg for r in rngs if r.arg.kind is AxisType.GLOBAL)
  l = tuple(r.src[0].arg for r in rngs if r.arg.kind is AxisType.LOCAL)
  return (g + (1,) * (3 - len(g)), l + (1,) * (3 - len(l)))


def render_cuda(sink: UOp, name: str = "kernel") -> str: return CUDARenderer().render(sink, name)


# ---------------------------------------------------------------------------
# picking a launch configuration

def gpu_opt_choice(body: UOp, max_threads: int = 256) -> Opts:
  """Up to 3 output axes on the grid. Threads on the last axes (coalesced stores), 2x2 upcast for reduces."""
  shape = body.shape
  o = Opts(gpu=True)
  reduces = any(u.op is Ops.REDUCE for u in body.toposort())
  axes = [k for k, s in enumerate(shape) if s > 1][-3:]
  if len(axes) < len(shape) - shape.count(1):
    raise NotImplementedError("more than 3 non-trivial output axes: reshape first")
  budget = max_threads
  for k in reversed(axes):
    if reduces and shape[k] % 2 == 0 and shape[k] >= 32: o.upcast[k] = 2
    rest = shape[k] // o.upcast.get(k, 1)
    t = 1
    while t * 2 <= min(budget, 16 if len(axes) > 1 else budget) and rest % (t * 2) == 0: t *= 2
    if t > 1: o.local[k] = t; budget //= t
  return o


def gpu_lower(body: UOp, out: UOp, opts: Opts | None = None) -> UOp:
  return lower(body, out, Ctx(opts=opts or gpu_opt_choice(body)))


def emulate(sink: UOp, name: str = "kernel") -> str:
  """The C source of a GPU kernel where blockIdx/threadIdx are loops. Run it with runtime.Program."""
  if any(u.op is Ops.BARRIER for u in sink.toposort()):
    raise NotImplementedError("emulate: kernels with BARRIER need all threads to run in lockstep; "
                              "running threads one after another (what loops do) would be wrong")
  return CRenderer().render(sink, name)


# ---------------------------------------------------------------------------
# shared memory: a tiled GEMM in the loop IR

def smem_gemm_ir(N: int, T: int = 16) -> UOp:
  """C = A @ B, N x N, with T x T thread blocks and T x T tiles staged in shared memory.

  Each block computes one T x T tile of C. For each of the N/T steps along K, the
  block cooperatively copies a T x T tile of A and of B into shared memory (one
  element per thread), waits at a barrier, multiplies the tiles out of shared
  memory, and waits again before the next copy overwrites them. Global memory
  traffic drops by a factor of T.
  """
  f32 = dtypes.f32
  A, B, C = (UOp(Ops.BUFFER, (), ParamArg(s, f32, (N * N,))) for s in range(3))
  sA = UOp(Ops.ALLOC, (), ParamArg(0, f32, (T * T,), AddrSpace.SHARED))
  sB = UOp(Ops.ALLOC, (), ParamArg(1, f32, (T * T,), AddrSpace.SHARED))
  acc = UOp(Ops.ALLOC, (), ParamArg(2, f32, (1,)))
  by, bx = range_(N // T, 0, AxisType.GLOBAL), range_(N // T, 1, AxisType.GLOBAL)
  ty, tx = range_(T, 2, AxisType.LOCAL), range_(T, 3, AxisType.LOCAL)
  row, col = by * T + ty, bx * T + tx
  acc0 = acc.after(by, bx, ty, tx)
  init = acc0.index(0).store(UOp.const(0.0))
  kt = range_(N // T, 4)                                     # step along K, one tile at a time
  ld_a = sA.after(kt).index(ty * T + tx).store(A.index(row * N + kt * T + tx).load())
  ld_b = sB.after(kt).index(ty * T + tx).store(B.index((kt * T + ty) * N + col).load())
  sync1 = UOp(Ops.BARRIER, (ld_a, ld_b))
  k = range_(T, 5)
  accl = acc0.after(init, kt, k)
  prod_ = sA.after(sync1).index(ty * T + k).load() * sB.after(sync1).index(k * T + tx).load()
  inner = accl.index(0).store(accl.index(0).load() + prod_).end(k)
  sync2 = UOp(Ops.BARRIER, (inner,))                        # don't overwrite tiles others still read
  loop = sync2.end(kt)
  out = C.index(row * N + col).store(acc0.after(loop).index(0).load())
  return sink(out.end(by, bx, ty, tx))


# ---------------------------------------------------------------------------
# tensor cores: pattern-match a GEMM, emit a WMMA kernel

def match_gemm(body: UOp) -> tuple[int, int, int] | None:
  """If body is (M,N) = sum_K A(M,K) * B(K,N) over contiguous f16/f32 buffers, return (M, N, K)."""
  if body.op is not Ops.REDUCE or body.arg != (Ops.ADD, 1) or body.src[0].op is not Ops.MUL: return None
  if len(body.src[0].shape) != 3 or not body.dtype.is_float: return None
  M, N, K = body.src[0].shape
  # the two operands must be broadcast along *different* axes: A over N (axis 1), B over M (axis 0).
  # Without this check, x*x reduced over its last axis (a sum of squares) "is" a GEMM.
  a, b = body.src[0].src
  def bcast_axes(x: UOp) -> set[int]:
    return {i for i, (s, t) in enumerate(zip(x.src[0].shape, x.shape)) if s == 1 and t > 1} if x.op is Ops.EXPAND else set()
  if {frozenset(bcast_axes(a)), frozenset(bcast_axes(b))} != {frozenset({1}), frozenset({0})}: return None
  return (M, N, K) if all(x % 16 == 0 for x in (M, N, K)) else None


def wmma_gemm(M: int, N: int, K: int) -> str:
  """C(f32, MxN) = A(f16, MxK) @ B(f16, KxN), one 16x16 output tile per warp, via nvcuda::wmma.

  A warp (32 threads) cooperatively owns three "fragments": 16x16 tiles of A, B and C
  spread across its registers in a hardware-defined layout. mma_sync does a
  16x16x16 matrix multiply-accumulate in one instruction on the tensor cores.
  """
  return f"""#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void wmma_gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C) {{
  int tile_m = blockIdx.y, tile_n = blockIdx.x;          // one warp per 16x16 tile of C
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  for (int k = 0; k < {K}; k += 16) {{
    wmma::load_matrix_sync(a, A + tile_m * 16 * {K} + k, {K});
    wmma::load_matrix_sync(b, B + k * {N} + tile_n * 16, {N});
    wmma::mma_sync(c, a, b, c);
  }}
  wmma::store_matrix_sync(C + tile_m * 16 * {N} + tile_n * 16, c, {N}, wmma::mem_row_major);
}}
/* launch: grid = ({N // 16}, {M // 16}), block = (32, 1, 1) */
"""
