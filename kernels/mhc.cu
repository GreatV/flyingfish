// mHC hyper-connection mixing op: per-row mixer coefficients, the sinkhorn
// combination matrix, the stream collapse and the residual update.
// Numerics per the DeepSeek-V4.1-Flash and GLM-5.3 model definitions
// (publisher inference/model.py Block; modeling_glm5_next.py) and mHC:
// Manifold-Constrained Hyper-Connections (arXiv 2512.24880); the
// split/sinkhorn semantics are pinned by the black-box publisher fixtures.
//   mhc_mixes / mhc_mixes_bf16(stream, fn, base, scale, mult, hidden, iters,
//                              eps, norm_eps, pre, post, comb, rows):
//     stream [rows, mult, hidden] bf16 row-major, fn [(2+mult)*mult,
//     mult*hidden] f32 (mhc_mixes) or bf16 (mhc_mixes_bf16) row-major,
//     base [(2+mult)*mult] f32, scale [3] f32. The bf16 fn elements are
//     upcast exactly to FP32 registers before the dot (bf16->f32 is exact;
//     identical to fn.float() of the publisher). pre [rows, mult] f32,
//     post [rows, mult] f32, comb [rows, mult, mult] f32 with the
//     combination matrix rows indexed [j=out, k=in].
//     One block per row: the row is staged in dynamic shared memory, the
//     unweighted RMS factor over all mult*hidden elements is one block
//     reduction, and every mixer row is one FP32 dot product with one block
//     reduction. The scale is applied after the matmul.
//   mhc_collapse(stream, pre, mult, hidden, out, rows):
//     out [rows, hidden] bf16 = sum_j pre[j]*stream[j,h] in FP32.
//   mhc_update(sublayer_out, residual, post, comb, mult, hidden, rounding,
//              orientation, out, rows):
//     sublayer_out [rows, hidden] bf16, residual [rows, mult, hidden] bf16,
//     out [rows, mult, hidden] bf16 with
//     out[j,h] = post[j]*sublayer_out[h] + sum_k coeff(j,k)*residual[k,h].
//     The comb coefficients stay row-major [j=out, k=in]; orientation 0
//     applies coeff(j,k) = comb[j,k] (DeepSeek-V4.1), orientation 1 applies
//     coeff(j,k) = comb[k,j] (GLM-5.3, the transposed application).
//     rounding 0 uses separately rounded FP32 products and additions with one
//     BF16 output round. rounding 1 uses the GLM BF16 post product, BF16
//     transposed-combination matmul and BF16 add.
// mhc_mixes: one block per row, 256 threads. Dynamic shared memory holds the
// staged stream row, the mixer outputs and the combination matrix scratch.
// __trap() guards: grid beyond rows, mult below 2, hidden or iters below 1,
// non-positive eps or norm_eps. mhc_update additionally traps on a rounding
// or orientation mode above 1.
#include "gpu_compat.cuh"
#include <math.h>
#include <stdint.h>

#if defined(__HIP__)
#pragma clang fp contract(off)
#endif

#define MHC_THREADS 256

// Exact FP32 register upcast of one fn weight element.
__device__ __forceinline__ float mhc_fn_to_float(float v) { return v; }
__device__ __forceinline__ float mhc_fn_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Block-wide FP32 sum; every thread receives the result.
__device__ __forceinline__ float mhc_block_sum(float value, float* s_red) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarp = (int)blockDim.x >> 5;
  float v = value;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FF_WARP_MASK, v, off);
  if (lane == 0) s_red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float w = lane < nwarp ? s_red[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) w += __shfl_down_sync(FF_WARP_MASK, w, off);
    if (lane == 0) s_red[0] = w;
  }
  __syncthreads();
  const float total = s_red[0];
  __syncthreads();
  return total;
}

template <typename FnT>
__device__ __forceinline__ void mhc_mixes_body(
    char* mhc_smem, float* s_red, const __nv_bfloat16* __restrict__ stream,
    const FnT* __restrict__ fn, const float* __restrict__ base,
    const float* __restrict__ scale, int mult, int hidden, int iters, float eps,
    float norm_eps, float* __restrict__ pre, float* __restrict__ post,
    float* __restrict__ comb, int rows) {
  if ((int)blockIdx.x >= rows || mult < 2 || hidden <= 0 || iters < 1 ||
      !(eps > 0.0f) || !(norm_eps > 0.0f) || blockDim.x != MHC_THREADS)
    __trap();

  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int row = (int)blockIdx.x;
  const int n = mult * hidden;
  const int nmix = (2 + mult) * mult;

  __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(mhc_smem);
  float* s_mix = reinterpret_cast<float*>(mhc_smem + (size_t)n * 2);
  float* s_comb = s_mix + nmix;

  const __nv_bfloat16* x_row = stream + (size_t)row * n;
  for (int i = tid; i < n; i += nthr) s_x[i] = x_row[i];
  __syncthreads();

  float sumsq = 0.0f;
  for (int i = tid; i < n; i += nthr) {
    const float x = __bfloat162float(s_x[i]);
    sumsq += x * x;
  }
  const float rsqrt = 1.0f / sqrtf(mhc_block_sum(sumsq, s_red) / (float)n + norm_eps);

  for (int c = 0; c < nmix; ++c) {
    const FnT* fn_row = fn + (size_t)c * n;
    float acc = 0.0f;
    for (int d = tid; d < n; d += nthr)
      acc = fmaf(mhc_fn_to_float(fn_row[d]), __bfloat162float(s_x[d]), acc);
    const float dot = mhc_block_sum(acc, s_red);
    if (tid == 0) s_mix[c] = dot * rsqrt;
  }

  if (tid == 0) {
    for (int j = 0; j < mult; ++j)
      pre[(size_t)row * mult + j] =
          1.0f / (1.0f + expf(-(s_mix[j] * scale[0] + base[j]))) + eps;
    for (int j = 0; j < mult; ++j)
      post[(size_t)row * mult + j] =
          2.0f * (1.0f / (1.0f + expf(-(s_mix[mult + j] * scale[1] + base[mult + j]))));
    for (int j = 0; j < mult; ++j)
      for (int k = 0; k < mult; ++k)
        s_comb[j * mult + k] = s_mix[2 * mult + j * mult + k] * scale[2] +
                               base[2 * mult + j * mult + k];

    for (int j = 0; j < mult; ++j) {
      float mx = -INFINITY;
      for (int k = 0; k < mult; ++k) mx = fmaxf(mx, s_comb[j * mult + k]);
      float rowsum = 0.0f;
      for (int k = 0; k < mult; ++k) {
        const float e = expf(s_comb[j * mult + k] - mx);
        s_comb[j * mult + k] = e;
        rowsum += e;
      }
      for (int k = 0; k < mult; ++k)
        s_comb[j * mult + k] = s_comb[j * mult + k] / rowsum + eps;
    }
    for (int k = 0; k < mult; ++k) {
      float colsum = 0.0f;
      for (int j = 0; j < mult; ++j) colsum += s_comb[j * mult + k];
      for (int j = 0; j < mult; ++j)
        s_comb[j * mult + k] = s_comb[j * mult + k] / (colsum + eps);
    }
    for (int it = 1; it < iters; ++it) {
      for (int j = 0; j < mult; ++j) {
        float rowsum = 0.0f;
        for (int k = 0; k < mult; ++k) rowsum += s_comb[j * mult + k];
        for (int k = 0; k < mult; ++k)
          s_comb[j * mult + k] = s_comb[j * mult + k] / (rowsum + eps);
      }
      for (int k = 0; k < mult; ++k) {
        float colsum = 0.0f;
        for (int j = 0; j < mult; ++j) colsum += s_comb[j * mult + k];
        for (int j = 0; j < mult; ++j)
          s_comb[j * mult + k] = s_comb[j * mult + k] / (colsum + eps);
      }
    }
    for (int j = 0; j < mult; ++j)
      for (int k = 0; k < mult; ++k)
        comb[(size_t)row * mult * mult + j * mult + k] = s_comb[j * mult + k];
  }
}

extern "C" __global__ __launch_bounds__(MHC_THREADS) void mhc_mixes(
    const __nv_bfloat16* __restrict__ stream, const float* __restrict__ fn,
    const float* __restrict__ base, const float* __restrict__ scale, int mult,
    int hidden, int iters, float eps, float norm_eps,
    float* __restrict__ pre, float* __restrict__ post,
    float* __restrict__ comb, int rows) {
  extern __shared__ __align__(16) char mhc_smem[];
  __shared__ float s_red[8];
  mhc_mixes_body<float>(mhc_smem, s_red, stream, fn, base, scale, mult, hidden,
                        iters, eps, norm_eps, pre, post, comb, rows);
}

extern "C" __global__ __launch_bounds__(MHC_THREADS) void mhc_mixes_bf16(
    const __nv_bfloat16* __restrict__ stream,
    const __nv_bfloat16* __restrict__ fn, const float* __restrict__ base,
    const float* __restrict__ scale, int mult, int hidden, int iters, float eps,
    float norm_eps, float* __restrict__ pre, float* __restrict__ post,
    float* __restrict__ comb, int rows) {
  extern __shared__ __align__(16) char mhc_smem[];
  __shared__ float s_red[8];
  mhc_mixes_body<__nv_bfloat16>(mhc_smem, s_red, stream, fn, base, scale, mult,
                                hidden, iters, eps, norm_eps, pre, post, comb,
                                rows);
}

extern "C" __global__ __launch_bounds__(MHC_THREADS) void mhc_collapse(
    const __nv_bfloat16* __restrict__ stream, const float* __restrict__ pre,
    int mult, int hidden, __nv_bfloat16* __restrict__ out, int rows) {
  if ((int)blockIdx.x >= rows || mult < 2 || hidden <= 0 || blockDim.x != MHC_THREADS)
    __trap();

  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int row = (int)blockIdx.x;
  const __nv_bfloat16* x_row = stream + (size_t)row * mult * hidden;
  __nv_bfloat16* out_row = out + (size_t)row * hidden;
  for (int h = tid; h < hidden; h += nthr) {
    float acc = 0.0f;
    for (int j = 0; j < mult; ++j)
      acc = fmaf(pre[(size_t)row * mult + j],
                 __bfloat162float(x_row[(size_t)j * hidden + h]), acc);
    out_row[h] = __float2bfloat16_rn(acc);
  }
}

extern "C" __global__ __launch_bounds__(MHC_THREADS) void mhc_update(
    const __nv_bfloat16* __restrict__ sublayer_out,
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ post,
    const float* __restrict__ comb, int mult, int hidden, int rounding,
    int orientation, __nv_bfloat16* __restrict__ out, int rows) {
  if ((int)blockIdx.x >= rows || mult < 2 || hidden <= 0 ||
      (unsigned)rounding > 1u || (unsigned)orientation > 1u ||
      blockDim.x != MHC_THREADS)
    __trap();

  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int row = (int)blockIdx.x;
  const __nv_bfloat16* sub_row = sublayer_out + (size_t)row * hidden;
  const __nv_bfloat16* res_row = residual + (size_t)row * mult * hidden;
  __nv_bfloat16* out_row = out + (size_t)row * mult * hidden;
  if (rounding == 1) {
    // 50e35cc0:1335-1337,1346-1348: BF16 post, BF16 product, BF16 comb^T@
    // residual matmul with FP32 accumulation and one BF16 round, BF16 add.
    for (int h = tid; h < hidden; h += nthr) {
      const float sub = __bfloat162float(sub_row[h]);
      for (int j = 0; j < mult; ++j) {
        const float p = __bfloat162float(
            __float2bfloat16_rn(post[(size_t)row * mult + j]));
        const __nv_bfloat16 prod = __float2bfloat16_rn(p * sub);
        float acc = 0.0f;
        for (int k = 0; k < mult; ++k) {
          const int coeff = orientation == 0 ? j * mult + k : k * mult + j;
          const float c = __bfloat162float(__float2bfloat16_rn(
              comb[(size_t)row * mult * mult + coeff]));
          acc = fmaf(c, __bfloat162float(res_row[(size_t)k * hidden + h]),
                     acc);
        }
        const __nv_bfloat16 mm = __float2bfloat16_rn(acc);
        out_row[(size_t)j * hidden + h] =
            __float2bfloat16_rn(__bfloat162float(prod) + __bfloat162float(mm));
      }
    }
    return;
  }
  for (int h = tid; h < hidden; h += nthr) {
    const float sub = __bfloat162float(sub_row[h]);
    for (int j = 0; j < mult; ++j) {
      const float p = post[(size_t)row * mult + j];
      // model.py hc_post and modeling_glm5_next.py:1346-1348 form `post*x`
      // and each `comb*residual` before the additions.
      float acc = __fmul_rn(p, sub);
      for (int k = 0; k < mult; ++k) {
        const int coeff = orientation == 0 ? j * mult + k : k * mult + j;
        const float c = comb[(size_t)row * mult * mult + coeff];
        acc = __fadd_rn(
            acc,
            __fmul_rn(c, __bfloat162float(res_row[(size_t)k * hidden + h])));
      }
      out_row[(size_t)j * hidden + h] = __float2bfloat16_rn(acc);
    }
  }
}
