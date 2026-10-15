// Two float32 GEMMs with identical row-major input/output contracts.
// No tensor cores or reduced-precision input conversion. All partial tiles are masked.
extern "C" __global__ void gemm_naive(const float *a, const float *b, float *c,
                                    const int *dimensions) {
  const int m = dimensions[0], n = dimensions[1], k = dimensions[2];
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; inner++)
    accumulator += a[(long long)row*k + inner] * b[(long long)inner*n + col];
  c[(long long)row*n + col] = accumulator;
}

// Output block: 64x64; reduction chunk: 16; thread block: (32,8).
// Each thread owns 8x2 accumulator values. Adjacent lanes own adjacent columns.
extern "C" __global__ void gemm_tiled(const float *a, const float *b, float *c,
                                    const int *dimensions) {
  const int m = dimensions[0], n = dimensions[1], k = dimensions[2];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int lane = ty*32 + tx;
  const int row_base = blockIdx.y*64, col_base = blockIdx.x*64;
  __shared__ float a_tile[64][16];
  __shared__ float b_tile[16][64];
  float accumulator[8][2] = {};
  for (int reduction_base = 0; reduction_base < k; reduction_base += 16) {
    // 256 threads cooperatively load each 1024-element tile. Invalid elements
    // become zero. No thread exits before either block-wide barrier.
    for (int linear = lane; linear < 64*16; linear += 256) {
      const int row = linear / 16, inner = linear % 16;
      a_tile[row][inner] = row_base+row < m && reduction_base+inner < k
        ? a[(long long)(row_base+row)*k + reduction_base+inner] : 0.0f;
    }
    for (int linear = lane; linear < 16*64; linear += 256) {
      const int inner = linear / 64, col = linear % 64;
      b_tile[inner][col] = reduction_base+inner < k && col_base+col < n
        ? b[(long long)(reduction_base+inner)*n + col_base+col] : 0.0f;
    }
    __syncthreads();  // Every staged element is visible before any consumer.
    #pragma unroll
    for (int inner = 0; inner < 16; inner++) {
      float a_fragment[8], b_fragment[2];
      #pragma unroll
      for (int i = 0; i < 8; i++) a_fragment[i] = a_tile[ty+i*8][inner];
      #pragma unroll
      for (int j = 0; j < 2; j++) b_fragment[j] = b_tile[inner][tx+j*32];
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) accumulator[i][j] += a_fragment[i]*b_fragment[j];
      }
    }
    __syncthreads();  // No producer overwrites a tile while another warp reads it.
  }
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int row = row_base + ty+i*8, col = col_base + tx+j*32;
      if (row < m && col < n) c[(long long)row*n+col] = accumulator[i][j];
    }
  }
}
