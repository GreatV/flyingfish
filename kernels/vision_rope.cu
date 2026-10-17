// Vision 2D RoPE: 18 frequencies per axis over the full 72-dim head, theta = 1e4.
// Model definition (vision rotary embedding, dim = head_dim // 2 = 36):
//       inv_freq[k] = 1 / theta**(k/18),  k = 0..17
//       rope[p] = concat(row_pos[p] * inv_freq, col_pos[p] * inv_freq)   -> [36]
// emb = cat((rope, rope), -1); q,k rotate in fp32 and round back to bf16. rotate_half pairs
// dim i with dim i+36, both carrying the same angle:
//
//       out[i]    = x[i]   *c[i] - x[i+36]*s[i]
//       out[i+36] = x[i+36]*c[i] + x[i]   *s[i]        i = 0..35
//
// with c[i] = cos(pos_axis(i) * inv_freq[i mod 18]) and pos_axis(i) = row for
// i < 18, col for i >= 18.
// Positions follow the merge-block packing order, not a row-major grid.
// Not in-place: q_out/k_out must not alias q/k.
//
// Launch: grid(ceil(rows*HQ / 256)), block 256. One thread per (row, head).
#include <cuda_bf16.h>
#include <stdint.h>

namespace vision_rope {

using bf16 = __nv_bfloat16;

constexpr int HQ     = 16;      // heads sharing one row's angles
constexpr int D      = 72;      // head_dim
constexpr int HALF   = D / 2;   // 36: rotate_half's split
constexpr int NFREQ  = 18;      // frequencies per axis
constexpr int MERGE  = 2;       // spatial_merge_size
constexpr float THETA = 1.0e4f;

static_assert(D == 2 * HALF, "rotate_half splits the head in half");
static_assert(HALF == 2 * NFREQ, "two axes of NFREQ frequencies fill half the head");

__device__ __forceinline__ float2 load_bf16x2(const bf16* p) {
  const __nv_bfloat162 t = *reinterpret_cast<const __nv_bfloat162*>(p);
  return make_float2(__low2float(t), __high2float(t));
}
__device__ __forceinline__ void store_bf16x2(bf16* p, float lo, float hi) {
  const __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  *reinterpret_cast<__nv_bfloat162*>(p) = t;
}

// Device-side geometry: a block's grid must describe exactly its rows.
__device__ __forceinline__ void guard_blocks(const int* block_rows, const int* grid_h,
                                             const int* grid_w, int nblocks) {
  if (nblocks <= 0) __trap();
  if (block_rows[0] != 0) __trap();
  for (int b = 0; b < nblocks; ++b) {
    const int rows = block_rows[b + 1] - block_rows[b];
    if (rows <= 0) __trap();
    if (grid_h[b] % MERGE != 0 || grid_w[b] % MERGE != 0) __trap();
    if (grid_h[b] * grid_w[b] != rows) __trap();
  }
}

__device__ __forceinline__ void rope_body(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                          bf16* __restrict__ q_out, bf16* __restrict__ k_out,
                                          const int* __restrict__ block_rows,
                                          const int* __restrict__ grid_h,
                                          const int* __restrict__ grid_w, int nblocks, int rows) {
  __shared__ float s_inv[NFREQ];
  if (threadIdx.x < NFREQ)
    s_inv[threadIdx.x] = 1.0f / powf(THETA, (float)threadIdx.x / (float)NFREQ);
  __syncthreads();

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= rows * HQ) return;
  const int row = tid / HQ;
  const int head = tid % HQ;

  int b = 0;
  while (b + 1 < nblocks && block_rows[b + 1] <= row) ++b;
  const int p = row - block_rows[b];
  const int nblk_w = grid_w[b] / MERGE;

  // Merge-block packing order: blocks row-major, then the m x m patch inside a block row-major.
  const int blk = p / (MERGE * MERGE);
  const int inb = p % (MERGE * MERGE);
  const int pos_row = (blk / nblk_w) * MERGE + inb / MERGE;
  const int pos_col = (blk % nblk_w) * MERGE + inb % MERGE;
  // Row position feeds dims [0, 18), column position dims [18, 36).
  const float axis_lo = (float)pos_row, axis_hi = (float)pos_col;

  float c[HALF], s[HALF];
#pragma unroll
  for (int i = 0; i < HALF; ++i) {
    const float pos = (i < NFREQ) ? axis_lo : axis_hi;
    const float ang = pos * s_inv[i % NFREQ];
    c[i] = cosf(ang);
    s[i] = sinf(ang);
  }

  const size_t base = ((size_t)head * (size_t)rows + (size_t)row) * D;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const bf16* src = pass ? k : q;
    bf16* dst = pass ? k_out : q_out;
#pragma unroll
    for (int i = 0; i < HALF; i += 2) {
      const float2 xa = load_bf16x2(src + base + i);          // dims i, i+1
      const float2 xb = load_bf16x2(src + base + i + HALF);   // dims i+36, i+37
      // i and i+1 share an axis: the (17,53) pair lands in the i == 16 step.
      store_bf16x2(dst + base + i, xa.x * c[i] - xb.x * s[i], xa.y * c[i + 1] - xb.y * s[i + 1]);
      store_bf16x2(dst + base + i + HALF, xb.x * c[i] + xa.x * s[i], xb.y * c[i + 1] + xa.y * s[i + 1]);
    }
  }
}

}  // namespace vision_rope

extern "C" __global__ void __launch_bounds__(256) vision_rope_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ k_out,
    const int* __restrict__ block_rows, const int* __restrict__ grid_h,
    const int* __restrict__ grid_w, int nblocks, int rows) {
  using namespace vision_rope;
  if (rows <= 0) __trap();
  guard_blocks(block_rows, grid_h, grid_w, nblocks);
  rope_body(q, k, q_out, k_out, block_rows, grid_h, grid_w, nblocks, rows);
}
