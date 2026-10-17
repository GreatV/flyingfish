// MoE combine: one deterministic combine per row over per-(row, choice)
// BF16 contributions plus the sigmoid-gated shared expert, with an
// exactly-once completeness check per (row, choice).
//   contributions: [rows*top_k, hidden] BF16, slot a = row*top_k + choice,
//     already weighted (down output x router weight, BF16 boundary).
//   shared_out: [rows, hidden] BF16; scale: [rows] BF16 (sigmoid gate).
//   y[row] = BF16( sum over the row's final-active choices, ascending
//     expert id, FP32, of contributions[a] + BF16(scale*shared_out[row]) ).
// __trap() on top_k outside [1,32], done[a] != final_active[a] (a choice
// must be contributed exactly once), or row out of range.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// scale[row] = BF16(sigmoid(BF16(dot(x_row, shared_expert_gate)))).
extern "C" __global__ void moe_shared_scale(const bf16* __restrict__ x,
                                             const bf16* __restrict__ seg,
                                             bf16* __restrict__ scale,
                                             int rows, long long hidden,
                                             long long x_stride) {
  if (blockDim.x != 256) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const bf16* __restrict__ xr = x + (long long)row * x_stride;
  float part = 0.0f;
  for (long long c = tid; c < hidden; c += blockDim.x)
    part += __bfloat162float(xr[c]) * __bfloat162float(seg[c]);
  for (int off = 16; off > 0; off >>= 1)
    part += __shfl_down_sync(0xffffffffu, part, off);
  __shared__ float red[8];
  if ((tid & 31) == 0) red[tid >> 5] = part;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x >> 5) ? red[tid] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (tid == 0) {
      const float sig =
          1.0f / (1.0f + expf(-__bfloat162float(__float2bfloat16(v))));
      scale[row] = __float2bfloat16(sig);
    }
  }
}

extern "C" __global__ void moe_combine(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ scale, const int* __restrict__ ids,
    const unsigned* __restrict__ done, const unsigned* __restrict__ final_active,
    bf16* __restrict__ y, int rows, int top_k, long long hidden) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  int live = 0;
  for (int c = 0; c < top_k; ++c) {
    const unsigned want = final_active[a0 + c];
    if (done[a0 + c] != want) __trap();
    if (want == 0u) continue;
    int p = live;
    while (p > 0 && ids[a0 + order[p - 1]] > ids[a0 + c]) {
      order[p] = order[p - 1];
      --p;
    }
    order[p] = c;
    ++live;
  }
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  for (long long d = tid; d < hidden; d += blockDim.x) {
    const bf16 gated = __hmul(scale[row], sh[d]);
    float acc = __bfloat162float(gated);
    for (int k = 0; k < live; ++k)
      acc += __bfloat162float(contributions[(long long)(a0 + order[k]) * hidden + d]);
    yr[d] = __float2bfloat16(acc);
  }
}
