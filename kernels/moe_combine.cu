// MoE combine: one deterministic combine per row over per-(row, choice)
// BF16 contributions plus either a sigmoid-gated or ungated shared expert, with
// an exactly-once completeness check per (row, choice).
//   contributions: [rows*top_k, hidden] BF16, slot a = row*top_k + choice,
//     already weighted (down output x router weight, BF16 boundary).
//   shared_out: [rows, hidden] BF16; scale: [rows] BF16 (sigmoid gate).
//   routed[row] = BF16(FP32 sum of final-active contributions, ascending expert id).
//   y[row] = BF16(routed[row] + BF16(scale*shared_out[row])).
// __trap() on top_k outside [1,32], done[a] != final_active[a] (a choice
// must be contributed exactly once), or row out of range.
#include "gpu_compat.cuh"

#if defined(__HIP__)
#define __hadd __hadd_rn
#define __hmul __hmul_rn
#endif

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
    part += __shfl_down_sync(FF_WARP_MASK, part, off);
  __shared__ float red[8];
  if ((tid & 31) == 0) red[tid >> 5] = part;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x >> 5) ? red[tid] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(FF_WARP_MASK, v, off);
    if (tid == 0) {
      const float sig =
          1.0f / (1.0f + expf(-__bfloat162float(__float2bfloat16(v))));
      scale[row] = __float2bfloat16(sig);
    }
  }
}

__device__ __forceinline__ int moe_order_routed_row(
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active, int a0, int top_k,
    int (&order)[32]) {
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
  return live;
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
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  for (long long d = tid; d < hidden; d += blockDim.x) {
    const bf16 gated = __hmul(scale[row], sh[d]);
    float acc = 0.0f;
    for (int k = 0; k < live; ++k)
      acc += __bfloat162float(contributions[(long long)(a0 + order[k]) * hidden + d]);
    yr[d] = __hadd(__float2bfloat16(acc), gated);
  }
}

extern "C" __global__ void moe_combine_ungated(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active,
    bf16* __restrict__ y, int rows, int top_k, long long hidden) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  for (long long d = tid; d < hidden; d += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < live; ++k)
      acc += __bfloat162float(contributions[(long long)(a0 + order[k]) * hidden + d]);
    yr[d] = __hadd(__float2bfloat16(acc), sh[d]);
  }
}

// One 16-byte group of eight BF16 values.
union Bf16x8 {
  float4 v;
  bf16 h[8];
};

// scale[row] = BF16(sigmoid(BF16(dot(x_row, seg)))) with the thread stride,
// FP32 products and warp/shared reduction of moe_shared_scale; the value is
// broadcast to every thread of the block through shared memory.
__device__ __forceinline__ bf16 moe_row_scale(const bf16* __restrict__ xr,
                                               const bf16* __restrict__ seg,
                                               long long hidden) {
  const int tid = threadIdx.x;
  __shared__ float red[8];
  __shared__ float scale;
  float part = 0.0f;
  for (long long c = tid; c < hidden; c += blockDim.x)
    part += __bfloat162float(xr[c]) * __bfloat162float(seg[c]);
  for (int off = 16; off > 0; off >>= 1)
    part += __shfl_down_sync(FF_WARP_MASK, part, off);
  if ((tid & 31) == 0) red[tid >> 5] = part;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x >> 5) ? red[tid] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(FF_WARP_MASK, v, off);
    if (tid == 0) {
      const float sig =
          1.0f / (1.0f + expf(-__bfloat162float(__float2bfloat16(v))));
      scale = __bfloat162float(__float2bfloat16(sig));
    }
  }
  __syncthreads();
  return __float2bfloat16(scale);
}

// The final-active contributions of one row in ascending expert id, summed
// in FP32 into one accumulator per element of the lane group.
template <int VEC>
__device__ __forceinline__ void moe_routed_acc(
    const bf16* __restrict__ contributions, const int* __restrict__ order,
    int live, int a0, long long hidden, long long d0, float* acc) {
#pragma unroll
  for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;
  for (int k = 0; k < live; ++k) {
    const bf16* __restrict__ src =
        contributions + (long long)(a0 + order[k]) * hidden + d0;
    if constexpr (VEC == 8) {
      Bf16x8 c;
      c.v = *reinterpret_cast<const float4*>(src);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += __bfloat162float(c.h[j]);
    } else {
      acc[0] += __bfloat162float(src[0]);
    }
  }
}

extern "C" __global__ void moe_combine_routed(
    const bf16* __restrict__ contributions, const int* __restrict__ ids,
    const unsigned* __restrict__ done, const unsigned* __restrict__ final_active,
    bf16* __restrict__ routed_out, int rows, int top_k, long long hidden) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  bf16* __restrict__ rr = routed_out + (long long)row * hidden;
  for (long long d = tid; d < hidden; d += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < live; ++k)
      acc += __bfloat162float(
          contributions[(long long)(a0 + order[k]) * hidden + d]);
    rr[d] = __float2bfloat16(acc);
  }
}

extern "C" __global__ void moe_combine_routed_vector(
    const bf16* __restrict__ contributions, const int* __restrict__ ids,
    const unsigned* __restrict__ done, const unsigned* __restrict__ final_active,
    bf16* __restrict__ routed_out, int rows, int top_k, long long hidden) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32 || hidden % 8 != 0) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  bf16* __restrict__ rr = routed_out + (long long)row * hidden;
  for (long long d0 = (long long)tid * 8; d0 < hidden;
       d0 += (long long)blockDim.x * 8) {
    float acc[8];
    moe_routed_acc<8>(contributions, order, live, a0, hidden, d0, acc);
    Bf16x8 out;
#pragma unroll
    for (int j = 0; j < 8; ++j) out.h[j] = __float2bfloat16(acc[j]);
    *reinterpret_cast<float4*>(rr + d0) = out.v;
  }
}

extern "C" __global__ void moe_join_routed(
    const bf16* __restrict__ routed_in, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ scale, bf16* __restrict__ y, int rows,
    long long hidden) {
  if (blockDim.x != 256) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const bf16* __restrict__ rr = routed_in + (long long)row * hidden;
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  if (scale != nullptr) {
    const bf16 s = scale[row];
    for (long long d = tid; d < hidden; d += blockDim.x) {
      const bf16 gated = __hmul(s, sh[d]);
      yr[d] = __hadd(rr[d], gated);
    }
  } else {
    for (long long d = tid; d < hidden; d += blockDim.x) {
      yr[d] = __hadd(rr[d], sh[d]);
    }
  }
}

// Fused combine group: the routed sum, the shared-expert sigmoid gate and the
// residual add in one launch.
//   residual[row, d] = residual[row, d] + BF16(routed[row, d] + gated[d]).
// grid = (rows, 1, 1), block = (256, 1, 1). seg == nullptr selects the
// ungated shared expert.
template <int VEC>
__device__ __forceinline__ void moe_combine_residual_prefill_body(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active, bf16* __restrict__ residual,
    int rows, int top_k, long long hidden) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  const bool gated = seg != nullptr;
  bf16 scale_row = __float2bfloat16(0.0f);
  if (gated)
    scale_row = moe_row_scale(x + (long long)row * x_stride, seg, hidden);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ res = residual + (long long)row * hidden;
  for (long long d0 = (long long)tid * 8; d0 < hidden;
       d0 += (long long)blockDim.x * 8) {
    float acc[8];
    moe_routed_acc<8>(contributions, order, live, a0, hidden, d0, acc);
    Bf16x8 s;
    s.v = *reinterpret_cast<const float4*>(sh + d0);
    Bf16x8 r;
    r.v = *reinterpret_cast<float4*>(res + d0);
    if (gated) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        r.h[j] = __hadd(r.h[j],
                        __hadd(__float2bfloat16(acc[j]),
                               __hmul(scale_row, s.h[j])));
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        r.h[j] = __hadd(r.h[j], __hadd(__float2bfloat16(acc[j]), s.h[j]));
    }
    *reinterpret_cast<float4*>(res + d0) = r.v;
  }
}

// Decode form: y_out selects the FFN delta; otherwise the residual is updated
// in place. grid = (rows * hidden_blocks, 1, 1), block = (256, 1, 1).
template <int VEC>
__device__ __forceinline__ void moe_combine_residual_decode_body(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active, bf16* __restrict__ residual,
    bf16* __restrict__ y_out, int rows, int top_k, long long hidden,
    int hidden_blocks) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32 || hidden_blocks < 1 ||
      gridDim.x != rows * hidden_blocks)
    __trap();
  if constexpr (VEC == 8) {
    if (hidden_blocks != 1) __trap();
  }
  const int row = blockIdx.x / hidden_blocks;
  if (row < 0 || row >= rows) __trap();
  const int hidden_block = blockIdx.x % hidden_blocks;
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  const bool gated = seg != nullptr;
  bf16 scale_row = __float2bfloat16(0.0f);
  if (gated)
    scale_row = moe_row_scale(x + (long long)row * x_stride, seg, hidden);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ res = nullptr;
  if (y_out == nullptr) res = residual + (long long)row * hidden;
  if constexpr (VEC == 8) {
    for (long long d0 = (long long)tid * 8; d0 < hidden;
         d0 += (long long)blockDim.x * 8) {
      float acc[8];
      moe_routed_acc<8>(contributions, order, live, a0, hidden, d0, acc);
      Bf16x8 s;
      s.v = *reinterpret_cast<const float4*>(sh + d0);
      Bf16x8 routed;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        routed.h[j] = __float2bfloat16(acc[j]);
      if (y_out != nullptr) {
        Bf16x8 out;
        if (gated) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            out.h[j] = __hadd(routed.h[j], __hmul(scale_row, s.h[j]));
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) out.h[j] = __hadd(routed.h[j], s.h[j]);
        }
        *reinterpret_cast<float4*>(y_out + (long long)row * hidden + d0) =
            out.v;
      } else {
        Bf16x8 r;
        r.v = *reinterpret_cast<float4*>(res + d0);
        if (gated) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            r.h[j] = __hadd(
                r.h[j],
                __hadd(routed.h[j], __hmul(scale_row, s.h[j])));
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            r.h[j] = __hadd(r.h[j], __hadd(routed.h[j], s.h[j]));
        }
        *reinterpret_cast<float4*>(res + d0) = r.v;
      }
    }
  } else {
    for (long long d = tid; d < hidden; d += blockDim.x) {
      if ((d / blockDim.x) % hidden_blocks != hidden_block) continue;
      float acc[1];
      moe_routed_acc<1>(contributions, order, live, a0, hidden, d, acc);
      const bf16 routed = __float2bfloat16(acc[0]);
      const bf16 ff = gated ? __hadd(routed, __hmul(scale_row, sh[d]))
                            : __hadd(routed, sh[d]);
      if (y_out != nullptr)
        y_out[(long long)row * hidden + d] = ff;
      else
        res[d] = __hadd(res[d], ff);
    }
  }
}

// The prefill peer join: the routed sum comes from the peer GPU; the shared
// expert and residual add are local.
template <int VEC>
__device__ __forceinline__ void moe_join_routed_residual_prefill_body(
    const bf16* __restrict__ routed_in, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    bf16* __restrict__ residual, int rows, long long hidden) {
  if (blockDim.x != 256) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const bool gated = seg != nullptr;
  bf16 scale_row = __float2bfloat16(0.0f);
  if (gated)
    scale_row = moe_row_scale(x + (long long)row * x_stride, seg, hidden);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  const bf16* __restrict__ rr = routed_in + (long long)row * hidden;
  bf16* __restrict__ res = residual + (long long)row * hidden;
  for (long long d0 = (long long)tid * 8; d0 < hidden;
       d0 += (long long)blockDim.x * 8) {
    Bf16x8 s;
    s.v = *reinterpret_cast<const float4*>(sh + d0);
    Bf16x8 r;
    r.v = *reinterpret_cast<const float4*>(rr + d0);
    Bf16x8 o;
    o.v = *reinterpret_cast<const float4*>(res + d0);
    if (gated) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o.h[j] = __hadd(o.h[j], __hadd(r.h[j], __hmul(scale_row, s.h[j])));
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o.h[j] = __hadd(o.h[j], __hadd(r.h[j], s.h[j]));
    }
    *reinterpret_cast<float4*>(res + d0) = o.v;
  }
}

template <int VEC>
__device__ __forceinline__ void moe_join_routed_residual_decode_body(
    const bf16* __restrict__ routed_in, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    bf16* __restrict__ residual, bf16* __restrict__ y_out, int rows,
    long long hidden, int hidden_blocks) {
  if (blockDim.x != 256 || hidden_blocks < 1 ||
      gridDim.x != rows * hidden_blocks)
    __trap();
  if constexpr (VEC == 8) {
    if (hidden_blocks != 1) __trap();
  }
  const int row = blockIdx.x / hidden_blocks;
  if (row < 0 || row >= rows) __trap();
  const int hidden_block = blockIdx.x % hidden_blocks;
  const int tid = threadIdx.x;
  const bool gated = seg != nullptr;
  bf16 scale_row = __float2bfloat16(0.0f);
  if (gated)
    scale_row = moe_row_scale(x + (long long)row * x_stride, seg, hidden);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  const bf16* __restrict__ rr = routed_in + (long long)row * hidden;
  bf16* __restrict__ res = nullptr;
  if (y_out == nullptr) res = residual + (long long)row * hidden;
  if constexpr (VEC == 8) {
    for (long long d0 = (long long)tid * 8; d0 < hidden;
         d0 += (long long)blockDim.x * 8) {
      Bf16x8 s;
      s.v = *reinterpret_cast<const float4*>(sh + d0);
      Bf16x8 r;
      r.v = *reinterpret_cast<const float4*>(rr + d0);
      if (y_out != nullptr) {
        Bf16x8 out;
        if (gated) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            out.h[j] = __hadd(r.h[j], __hmul(scale_row, s.h[j]));
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) out.h[j] = __hadd(r.h[j], s.h[j]);
        }
        *reinterpret_cast<float4*>(y_out + (long long)row * hidden + d0) =
            out.v;
      } else {
        Bf16x8 o;
        o.v = *reinterpret_cast<const float4*>(res + d0);
        if (gated) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            o.h[j] = __hadd(
                o.h[j],
                __hadd(r.h[j], __hmul(scale_row, s.h[j])));
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            o.h[j] = __hadd(o.h[j], __hadd(r.h[j], s.h[j]));
        }
        *reinterpret_cast<float4*>(res + d0) = o.v;
      }
    }
  } else {
    for (long long d = tid; d < hidden; d += blockDim.x) {
      if ((d / blockDim.x) % hidden_blocks != hidden_block) continue;
      const bf16 ff = gated ? __hadd(rr[d], __hmul(scale_row, sh[d]))
                            : __hadd(rr[d], sh[d]);
      if (y_out != nullptr)
        y_out[(long long)row * hidden + d] = ff;
      else
        res[d] = __hadd(res[d], ff);
    }
  }
}

extern "C" __global__ void moe_combine_residual(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active, bf16* __restrict__ residual,
    int rows, int top_k, long long hidden) {
  moe_combine_residual_prefill_body<8>(
      contributions, shared_out, seg, x, x_stride, ids, done, final_active,
      residual, rows, top_k, hidden);
}

extern "C" __global__ void moe_combine_residual_scalar(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active, bf16* __restrict__ residual,
    bf16* __restrict__ y_out, int rows, int top_k, long long hidden,
    int hidden_blocks) {
  moe_combine_residual_decode_body<1>(
      contributions, shared_out, seg, x, x_stride, ids, done, final_active,
      residual, y_out, rows, top_k, hidden, hidden_blocks);
}

extern "C" __global__ void moe_join_routed_residual(
    const bf16* __restrict__ routed_in, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    bf16* __restrict__ residual, int rows, long long hidden) {
  moe_join_routed_residual_prefill_body<8>(
      routed_in, shared_out, seg, x, x_stride, residual, rows, hidden);
}

extern "C" __global__ void moe_join_routed_residual_scalar(
    const bf16* __restrict__ routed_in, const bf16* __restrict__ shared_out,
    const bf16* __restrict__ seg, const bf16* __restrict__ x, long long x_stride,
    bf16* __restrict__ residual, bf16* __restrict__ y_out, int rows,
    long long hidden, int hidden_blocks) {
  moe_join_routed_residual_decode_body<1>(
      routed_in, shared_out, seg, x, x_stride, residual, y_out, rows, hidden,
      hidden_blocks);
}

// Combine with the accumulator precision as a parameter: fp32_accum 1
// accumulates the per-choice BF16 contributions and one weight-1 shared row
// in FP32 and rounds the output to BF16 once (model.py MoE.forward); 0
// rounds the routed sum to BF16 before the shared join (the family
// gated/ungated boundary). No activation bookkeeping: the caller wrote
// every (row, choice) slot before this launch.
extern "C" __global__ void moe_combine_accum(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared,
    bf16* __restrict__ y, int rows, int top_k, long long hidden,
    int fp32_accum) {
  if (blockDim.x != 256 || gridDim.y != 1 || gridDim.z != 1 || top_k < 1 ||
      top_k > 32 || hidden % 8 != 0 || (unsigned)fp32_accum > 1u)
    __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const bf16* __restrict__ cr =
      contributions + (long long)row * top_k * hidden;
  const bf16* __restrict__ sh = shared + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  for (long long d0 = (long long)tid * 8; d0 < hidden;
       d0 += (long long)blockDim.x * 8) {
    Bf16x8 s;
    s.v = *reinterpret_cast<const float4*>(sh + d0);
    // The definition's order (model.py MoE.forward): the routed rows accumulate
    // into a zero FP32 accumulator and the shared row is added last.
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float routed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < top_k; ++k) {
      Bf16x8 c;
      c.v = *reinterpret_cast<const float4*>(cr + (long long)k * hidden + d0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j] += __bfloat162float(c.h[j]);
        routed[j] += __bfloat162float(c.h[j]);
      }
    }
    if (fp32_accum == 1) {
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += __bfloat162float(s.h[j]);
    }
    Bf16x8 out;
    if (fp32_accum == 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        // The family's BF16 boundary: the routed sum rounds to BF16 before
        // the shared join, which itself rounds once.
        const bf16 routed_bf16 = __float2bfloat16(routed[j]);
        acc[j] = __bfloat162float(__float2bfloat16(
            __bfloat162float(routed_bf16) + __bfloat162float(s.h[j])));
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) out.h[j] = __float2bfloat16(acc[j]);
    *reinterpret_cast<float4*>(yr + d0) = out.v;
  }
}
extern "C" __global__ void moe_combine_bf16(
    const bf16* __restrict__ contributions, const bf16* __restrict__ shared_out,
    const int* __restrict__ ids, const unsigned* __restrict__ done,
    const unsigned* __restrict__ final_active,
    bf16* __restrict__ y, int rows, int top_k, long long hidden) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int order[32];
  const int live =
      moe_order_routed_row(ids, done, final_active, a0, top_k, order);
  const bf16* __restrict__ sh = shared_out + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  for (long long d = tid; d < hidden; d += blockDim.x) {
    bf16 acc = __float2bfloat16(0.0f);
    for (int k = 0; k < live; ++k)
      acc = __hadd(acc, contributions[(long long)(a0 + order[k]) * hidden + d]);
    yr[d] = __hadd(acc, sh[d]);
  }
}
