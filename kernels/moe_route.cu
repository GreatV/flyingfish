// MoE router: FP32 softmax over BF16 expert logits, deterministic top-k
// (descending, exact ties by lower expert id), FP32 renormalised weights
// stored as BF16.
//   moe_route_bf16(x, router, ids, weights, rows, experts, top_k, hidden):
//     x [rows, hidden] bf16 row-major, router W [experts, hidden] bf16
//     row-major; each expert logit is an FP32 accumulation of BF16 products
//     over uint4 (8-element) loads, rounded to BF16 (F.linear boundary).
//   moe_route_topk(logits, ids, weights, rows, experts, top_k):
//     logits [rows, experts] bf16 row-major, pre-computed by a BF16 GEMM with
//     FP32 accumulation; no rounding is applied here (already BF16).
//   ids [rows, top_k] i32, weights [rows, top_k] bf16 for both entries.
// One block per row; thread e < experts owns expert e. Softmax runs in FP32
// over the BF16 logits. top_k selection rounds each take the block maximum
// under the comparator (higher probability, then lower expert id); a selected
// thread retires with probability -1.0f, below every softmax output. The
// top-k probabilities are renormalised in FP32 accumulated in selection order
// and stored as BF16.
// __trap() guards (both entries unless named): grid beyond rows, blockDim.x
// not a warp multiple or below experts, experts outside [1,1024], top_k
// outside [1,32] or >= experts. moe_route_bf16 additionally: hidden < 8 or
// % 8 != 0, x/router base pointers not 16-byte aligned; moe_route_topk:
// logits base pointer not 16-byte aligned.
// Static shared memory: two 32-word reduction stages plus 32 selected
// (id, probability) slots; blockDim.x <= 1024.
// Launch: grid = rows, block = experts rounded up to a warp multiple.
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

__device__ void moe_route_select(
    float logit, int* __restrict__ ids, bf16* __restrict__ weights,
    int experts, int top_k) {
  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int nw = nthr >> 5;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = blockIdx.x;
  const int e = tid;
  const bool live = e < experts;

  __shared__ float s_red[32];
  __shared__ int s_rid[32];
  __shared__ float s_sel_p[32];
  __shared__ int s_sel_id[32];
  __shared__ float s_tsum;

  float v = logit;
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  if (lane == 0) s_red[warp] = v;
  __syncthreads();
  v = warp == 0 && lane < nw ? s_red[lane] : -CUDART_INF_F;
  if (warp == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) s_red[0] = v;
  }
  __syncthreads();
  const float max_logit = s_red[0];
  // max_logit reads s_red[0]; the sum stage scatters into s_red[warp].
  __syncthreads();

  const float ev = live ? expf(logit - max_logit) : 0.0f;
  v = ev;
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) s_red[warp] = v;
  __syncthreads();
  v = warp == 0 && lane < nw ? s_red[lane] : 0.0f;
  if (warp == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) s_red[0] = v;
  }
  __syncthreads();
  const float prob = ev / s_red[0];

  if (tid == 0) s_tsum = 0.0f;
  __syncthreads();
  bool taken = false;
  for (int r = 0; r < top_k; ++r) {
    // comparator: higher probability wins, exact ties go to the lower id
    v = !taken && live ? prob : -1.0f;
    int id = live ? e : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, v, off);
      const int oi = __shfl_down_sync(0xffffffffu, id, off);
      if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
    }
    if (lane == 0) { s_red[warp] = v; s_rid[warp] = id; }
    __syncthreads();
    if (warp == 0) {
      v = lane < nw ? s_red[lane] : -CUDART_INF_F;
      id = lane < nw ? s_rid[lane] : 0x7fffffff;
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, id, off);
        if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
      }
      if (lane == 0) { s_red[0] = v; s_rid[0] = id; }
    }
    __syncthreads();
    if (live && !taken && prob == s_red[0] && e == s_rid[0]) {
      s_sel_id[r] = e;
      s_sel_p[r] = prob;
      s_tsum += prob;
      taken = true;
    }
    __syncthreads();
  }

  if (tid < top_k) {
    ids[(size_t)row * top_k + tid] = s_sel_id[tid];
    weights[(size_t)row * top_k + tid] =
        __float2bfloat16(s_sel_p[tid] / s_tsum);
  }
}

extern "C" __global__ void moe_route_bf16(
    const bf16* __restrict__ x, const bf16* __restrict__ router,
    int* __restrict__ ids, bf16* __restrict__ weights,
    int rows, int experts, int top_k, int hidden) {
  if ((int)blockIdx.x >= rows || experts < 1 || experts > 1024 || top_k < 1 ||
      top_k > 32 || top_k >= experts || hidden < 8 || (hidden & 7) != 0 ||
      (blockDim.x & 31) != 0 || (int)blockDim.x < experts ||
      (reinterpret_cast<uintptr_t>(x) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(router) & 15) != 0)
    __trap();

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int e = tid;
  float logit = -CUDART_INF_F;
  if (e < experts) {
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * hidden);
    const uint4* wr =
        reinterpret_cast<const uint4*>(router + (size_t)e * hidden);
    const int words = hidden >> 3;
    float acc = 0.0f;
#pragma unroll 4
    for (int w = 0; w < words; ++w) {
      const uint4 a = xr[w];
      const uint4 b = wr[w];
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 fa = __bfloat1622float2(ha[j]);
        const float2 fb = __bfloat1622float2(hb[j]);
        acc = fmaf(fa.x, fb.x, acc);
        acc = fmaf(fa.y, fb.y, acc);
      }
    }
    logit = __bfloat162float(__float2bfloat16(acc));
  }
  moe_route_select(logit, ids, weights, experts, top_k);
}

extern "C" __global__ void moe_route_topk(
    const bf16* __restrict__ logits, int* __restrict__ ids,
    bf16* __restrict__ weights, int rows, int experts, int top_k) {
  if ((int)blockIdx.x >= rows || experts < 1 || experts > 1024 || top_k < 1 ||
      top_k > 32 || top_k >= experts || (blockDim.x & 31) != 0 ||
      (int)blockDim.x < experts ||
      (reinterpret_cast<uintptr_t>(logits) & 15) != 0)
    __trap();

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  float logit = -CUDART_INF_F;
  if (tid < experts)
    logit = __bfloat162float(logits[(size_t)row * experts + tid]);
  moe_route_select(logit, ids, weights, experts, top_k);
}
