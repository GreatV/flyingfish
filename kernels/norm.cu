// norm.cu — RMSNorm, fused residual-add + RMSNorm, and residual add for
// MiniCPM5-2B (hidden = 2048), BF16 in / BF16 out with FP32 internal math.
// Drop-in replacement for the repo norm.cu entry points using the v8 vectorized
// implementations extracted from src/fused.cu: 16-byte vector I/O (8 bf16 per
// load/store), warp-shuffle block reduction, register-cached single pass
// (re-reads the row from L2 only when it does not fit in registers).
//
// Semantics
//   rms_norm:    y[r][d] = x[r][d] * rsqrt(mean_d(x[r][.]^2) + eps) * w[d]
//   add_rmsnorm: res_out[r][d] = bf16(res[r][d] + x[r][d])            (add in fp32,
//                single round at store, matching transformer-block residual math)
//                y[r][d] = bf16(res_out[r][d]) * rsqrt(mean_d(res_out[r][.]^2) + eps) * w[d]
//                (the norm consumes the rounded residual exactly as stored)
//   residual:    x[i] = bf16(x[i] + y[i])   (__hadd2 per pair; same round-to-nearest
//                result as the repo's scalar __hadd loop)
//   rms_norm/add_rmsnorm compute x*rstd*w in fp32 and round once at the store;
//   the repo rms_norm instead rounds x*scale to bf16 before a bf16 multiply by w.
//   All reductions accumulate in fp32.
//
// Layout / contract
//   * All tensors contiguous BF16 device buffers, element stride 1. x, y, res,
//     res_out are row-major [rows][dim]; w is one shared row of length dim.
//   * The 8-wide vector path is taken when dim % 8 == 0 (rms_norm, add_rmsnorm)
//     or n % 8 == 0 (residual) and requires 16-byte base alignment
//     (cudaMalloc guarantees 256 B); any other extent takes the scalar path.
//   * rms_norm / add_rmsnorm: the sum of squares is a per-block reduction, so
//     exactly one block per row: gridDim.x == rows. blockDim.x must be a
//     multiple of 32 and <= 1024 (warp shuffles assume full warps).
//   * add_rmsnorm: res_out may equal res (in-place residual stream, the decode
//     case); res/res_out are deliberately not __restrict__ for that reason.
//     x, w, y must not alias any output.
//   * residual: grid-stride over ceil(n/8) chunks (vector) or n elements
//     (scalar); any grid >= 1 block is correct. The repo launches
//     ceil(n/256) x 256, which this kernel handles unchanged.
//   * No global state, no host interaction, no dynamic allocation; 128 B static
//     __shared__ scratch per block. CUDA-Graph capturable.
//
// Measured (RTX 4090, sm_89, hidden = 2048, rows = 1, block = 256, hot /
// launch-bound batched launches; per-kernel ~2-3 us regime)
//   rms_norm     v8 1.84 us  vs repo rms_norm 3.7 us             (~2.0x)
//   add_rmsnorm  v8 1.93 us  vs repo residual + rms_norm 5.1 us  (~2.6x)
//   accuracy vs fp32 reference: max_rel err <= 0.0039
//
// Exported symbols (extern "C" __global__) and launch configs
//   void rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* w,
//                 __nv_bfloat16* y, int rows, int dim, float eps)
//       launch: <<<rows, 256>>>                     (repo-identical signature)
//   void add_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* res,
//                    const __nv_bfloat16* w, __nv_bfloat16* y,
//                    __nv_bfloat16* res_out, int rows, int dim, float eps)
//       launch: <<<rows, 256>>>                     (res_out == res allowed)
//   void residual(__nv_bfloat16* x, const __nv_bfloat16* y, int n)
//       launch: <<<ceil(n/2048), 256>>>             (grid-stride; any grid ok,
//               repo's <<<ceil(n/256), 256>>> included)

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

// Max elements cached per thread in registers. If the row does not fit, the
// norm kernels fall back to a strided loop that re-reads the row from L2.
constexpr int kMaxPerThread = 8;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// blockDim.x must be a multiple of 32 (host enforces) so every lane is active.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  const int tid = threadIdx.x;
  const int nw = blockDim.x >> 5;
  v = warp_reduce_sum(v);
  if ((tid & 31) == 0) smem[tid >> 5] = v;
  __syncthreads();
  if (tid < 32) {
    float t = (tid < nw) ? smem[tid] : 0.0f;
    t = warp_reduce_sum(t);
    if (tid == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

// Load VEC contiguous bf16 -> float[VEC]. Requires n % VEC == 0 and
// VEC*2-byte pointer alignment (guaranteed by the entry-point dispatch).
template <int VEC>
__device__ __forceinline__ void ld_vec(const __nv_bfloat16* __restrict__ p,
                                       float* __restrict__ f) {
  if constexpr (VEC % 8 == 0) {
    union U { float4 v; __nv_bfloat162 h[4]; } u;
    u.v = *reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 t = __bfloat1622float2(u.h[k]);
      f[2 * k] = t.x;
      f[2 * k + 1] = t.y;
    }
  } else if constexpr (VEC % 4 == 0) {
    union U { float2 v; __nv_bfloat162 h[2]; } u;
    u.v = *reinterpret_cast<const float2*>(p);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      float2 t = __bfloat1622float2(u.h[k]);
      f[2 * k] = t.x;
      f[2 * k + 1] = t.y;
    }
  } else if constexpr (VEC % 2 == 0) {
    __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(p);
    float2 t = __bfloat1622float2(h);
    f[0] = t.x;
    f[1] = t.y;
  } else {
#pragma unroll
    for (int k = 0; k < VEC; ++k) f[k] = __bfloat162float(p[k]);
  }
}

template <int VEC>
__device__ __forceinline__ void st_vec(__nv_bfloat16* __restrict__ p,
                                       const float* __restrict__ f) {
  if constexpr (VEC % 8 == 0) {
    union U { float4 v; __nv_bfloat162 h[4]; } u;
#pragma unroll
    for (int k = 0; k < 4; ++k) u.h[k] = __floats2bfloat162_rn(f[2 * k], f[2 * k + 1]);
    *reinterpret_cast<float4*>(p) = u.v;
  } else if constexpr (VEC % 4 == 0) {
    union U { float2 v; __nv_bfloat162 h[2]; } u;
#pragma unroll
    for (int k = 0; k < 2; ++k) u.h[k] = __floats2bfloat162_rn(f[2 * k], f[2 * k + 1]);
    *reinterpret_cast<float2*>(p) = u.v;
  } else if constexpr (VEC % 2 == 0) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(f[0], f[1]);
  } else {
#pragma unroll
    for (int k = 0; k < VEC; ++k) p[k] = __float2bfloat16(f[k]);
  }
}

// One block normalizes one row of n elements (nthr = blockDim.x, gid = threadIdx.x).
template <int VEC>
__device__ __forceinline__ void rmsnorm_row(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ w,
                                            __nv_bfloat16* __restrict__ y, int n,
                                            float eps) {
  __shared__ float smem[32];
  const int nthr = blockDim.x;
  const int gid = threadIdx.x;
  const int nv = n / VEC;                              // caller guarantees n % VEC == 0
  const bool cached = (nv <= nthr * (kMaxPerThread / VEC));   // block-uniform

  float r[kMaxPerThread];
  float ssq = 0.0f;
  if (cached) {
#pragma unroll
    for (int i = 0; i < kMaxPerThread / VEC; ++i) {
      const int c = gid + i * nthr;
      if (c < nv) {
        ld_vec<VEC>(x + (size_t)c * VEC, r + i * VEC);
#pragma unroll
        for (int j = 0; j < VEC; ++j) ssq += r[i * VEC + j] * r[i * VEC + j];
      }
    }
  } else {
    for (int c = gid; c < nv; c += nthr) {
      ld_vec<VEC>(x + (size_t)c * VEC, r);
#pragma unroll
      for (int j = 0; j < VEC; ++j) ssq += r[j] * r[j];
    }
  }
  const float rstd = rsqrtf(block_reduce_sum(ssq, smem) / (float)n + eps);

  if (cached) {
#pragma unroll
    for (int i = 0; i < kMaxPerThread / VEC; ++i) {
      const int c = gid + i * nthr;
      if (c < nv) {
        const size_t idx = (size_t)c * VEC;
        float ww[VEC], out[VEC];
        ld_vec<VEC>(w + idx, ww);
#pragma unroll
        for (int j = 0; j < VEC; ++j) out[j] = r[i * VEC + j] * rstd * ww[j];
        st_vec<VEC>(y + idx, out);
      }
    }
  } else {
    for (int c = gid; c < nv; c += nthr) {
      const size_t idx = (size_t)c * VEC;
      float ww[VEC], out[VEC];
      ld_vec<VEC>(x + idx, r);
      ld_vec<VEC>(w + idx, ww);
#pragma unroll
      for (int j = 0; j < VEC; ++j) out[j] = r[j] * rstd * ww[j];
      st_vec<VEC>(y + idx, out);
    }
  }
}

// Fused residual add + RMSNorm over one row per block. res/res_out are not
// __restrict__: res_out == res (in-place residual stream) is supported.
template <int VEC>
__device__ __forceinline__ void add_rmsnorm_row(const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* res,
                                                const __nv_bfloat16* __restrict__ w,
                                                __nv_bfloat16* __restrict__ y,
                                                __nv_bfloat16* res_out, int n,
                                                float eps) {
  __shared__ float smem[32];
  const int nthr = blockDim.x;
  const int gid = threadIdx.x;
  const int nv = n / VEC;
  const bool cached = (nv <= nthr * (kMaxPerThread / VEC));

  float rr[kMaxPerThread];
  float ssq = 0.0f;
  if (cached) {
#pragma unroll
    for (int i = 0; i < kMaxPerThread / VEC; ++i) {
      const int c = gid + i * nthr;
      if (c < nv) {
        const size_t idx = (size_t)c * VEC;
        float a[VEC], b[VEC];
        ld_vec<VEC>(x + idx, a);
        ld_vec<VEC>(res + idx, b);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          const __nv_bfloat16 s = __float2bfloat16(b[j] + a[j]);  // store, then normalize what was stored
          res_out[idx + j] = s;
          b[j] = __bfloat162float(s);
          rr[i * VEC + j] = b[j];
          ssq += b[j] * b[j];
        }
      }
    }
  } else {
    for (int c = gid; c < nv; c += nthr) {
      const size_t idx = (size_t)c * VEC;
      float a[VEC], b[VEC];
      ld_vec<VEC>(x + idx, a);
      ld_vec<VEC>(res + idx, b);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const __nv_bfloat16 s = __float2bfloat16(b[j] + a[j]);
        res_out[idx + j] = s;
        b[j] = __bfloat162float(s);
        ssq += b[j] * b[j];
      }
    }
  }
  const float rstd = rsqrtf(block_reduce_sum(ssq, smem) / (float)n + eps);

  if (cached) {
#pragma unroll
    for (int i = 0; i < kMaxPerThread / VEC; ++i) {
      const int c = gid + i * nthr;
      if (c < nv) {
        const size_t idx = (size_t)c * VEC;
        float ww[VEC], out[VEC];
        ld_vec<VEC>(w + idx, ww);
#pragma unroll
        for (int j = 0; j < VEC; ++j) out[j] = rr[i * VEC + j] * rstd * ww[j];
        st_vec<VEC>(y + idx, out);
      }
    }
  } else {
    for (int c = gid; c < nv; c += nthr) {
      const size_t idx = (size_t)c * VEC;
      float b[VEC], ww[VEC], out[VEC];
      ld_vec<VEC>(res_out + idx, b);
      ld_vec<VEC>(w + idx, ww);
#pragma unroll
      for (int j = 0; j < VEC; ++j) out[j] = b[j] * rstd * ww[j];
      st_vec<VEC>(y + idx, out);
    }
  }
}

}  // namespace

extern "C" {

// y[r][:] = rmsnorm(x[r][:], w) — one block per row, grid == rows.
__global__ __launch_bounds__(1024) void rms_norm(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* w,
                                                 __nv_bfloat16* y, int rows, int dim,
                                                 float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const size_t off = (size_t)row * dim;
  if ((dim & 7) == 0) rmsnorm_row<8>(x + off, w, y + off, dim, eps);
  else                rmsnorm_row<1>(x + off, w, y + off, dim, eps);
}

// res_out[r][:] = bf16(res[r][:] + x[r][:]); y[r][:] = rmsnorm(res_out[r][:], w).
// One block per row, grid == rows. res_out may alias res (in place).
__global__ __launch_bounds__(1024) void add_rmsnorm(const __nv_bfloat16* x,
                                                    const __nv_bfloat16* res,
                                                    const __nv_bfloat16* w,
                                                    __nv_bfloat16* y,
                                                    __nv_bfloat16* res_out, int rows,
                                                    int dim, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const size_t off = (size_t)row * dim;
  if ((dim & 7) == 0) add_rmsnorm_row<8>(x + off, res + off, w, y + off, res_out + off, dim, eps);
  else                add_rmsnorm_row<1>(x + off, res + off, w, y + off, res_out + off, dim, eps);
}

// x[i] += y[i] in bf16, flat grid-stride.
__global__ __launch_bounds__(1024) void residual(__nv_bfloat16* x,
                                                 const __nv_bfloat16* y, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthr = blockDim.x * gridDim.x;
  if ((n & 7) == 0) {
    const int nv = n >> 3;
    for (int c = gid; c < nv; c += nthr) {
      union U { float4 v; __nv_bfloat162 h[4]; } a, b;
      a.v = *reinterpret_cast<const float4*>(x + (size_t)c * 8);
      b.v = *reinterpret_cast<const float4*>(y + (size_t)c * 8);
#pragma unroll
      for (int k = 0; k < 4; ++k) a.h[k] = __hadd2(a.h[k], b.h[k]);
      *reinterpret_cast<float4*>(x + (size_t)c * 8) = a.v;
    }
  } else {
    for (int i = gid; i < n; i += nthr) x[i] = __hadd(x[i], y[i]);
  }
}

}  // extern "C"
