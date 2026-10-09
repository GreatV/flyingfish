// add_rmsnorm_capture.cu -- residual add + RMSNorm with the DSpark capture
// copy folded into the same launch: add_rmsnorm's x/res/y/res_out writes are
// bit-identical, and the capture destination receives the same bytes the 2D
// device-to-device copy would have moved. Only the capture layers (5 per
// round) use this entry point; the rest keep add_rmsnorm, decode keeps both.
//
// Math as norm.cu add_rmsnorm: res_out = bf16(res + x) (one rounding), then
// y = bf16(res_out * rsqrt(mean(res_out^2) + eps) * w) in fp32 with one round
// at the store; the same rounded res_out values are mirrored into
// capture[r * dst_pitch + dst_col + d].
//
// Layout / contract:
//   * x, res, y, res_out bf16 [rows][hidden] (res_out may alias res);
//     capture bf16 with row stride dst_pitch and slot offset dst_col.
//   * grid = (rows,1,1), block = (256,1,1), smem 128 B static; blocks past
//     rows return before any write.
//   * FD_GUARD (__trap__): hidden != 2048, rows < 1, eps negative or not
//     finite, any null buffer, blockDim.x not a multiple of 32 or outside
//     [32,1024], dst_col + hidden > dst_pitch, dst_pitch/dst_col not a
//     multiple of 8 elements (the capture store is a 16 B vector store).
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace add_norm_capture {

using bf16 = __nv_bfloat16;

constexpr int kHidden = 2048;      // hidden_size
constexpr int kVec = 8;            // bf16 elements per 16 B access
constexpr int kMaxPerThread = 8;   // elements cached per thread in registers

__device__ __forceinline__ void shape(int rows, int hidden, float eps, unsigned nthr) {
    if (hidden != kHidden) __trap();
    if (rows < 1) __trap();
    if (!(eps >= 0.f) || !isfinite(eps)) __trap();
    if (nthr < 32u || nthr > 1024u || (nthr & 31u)) __trap();
}

__device__ __forceinline__ void rect(int hidden, int dst_pitch, int dst_col) {
    if (dst_pitch <= 0 || dst_col < 0) __trap();
    if (dst_col + hidden > dst_pitch) __trap();
    if ((dst_pitch & (kVec - 1)) || (dst_col & (kVec - 1))) __trap();
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

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

template <int VEC>
__device__ __forceinline__ void ld_vec(const bf16* __restrict__ p, float* __restrict__ f) {
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
__device__ __forceinline__ void st_vec(bf16* __restrict__ p, const float* __restrict__ f) {
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

// Pack 8 already-rounded bf16 values into one 16 B store (bit packing only,
// no second rounding).
template <int VEC>
__device__ __forceinline__ void st_vec_bf16(bf16* __restrict__ p, const bf16* __restrict__ v) {
    static_assert(VEC % 8 == 0, "vector store needs a multiple of 8 elements");
    union U { float4 v; __nv_bfloat162 h[VEC / 2]; } u;
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) u.h[k] = __nv_bfloat162(v[2 * k], v[2 * k + 1]);
    *reinterpret_cast<float4*>(p) = u.v;
}

// One block: res_out[r] = bf16(res[r] + x[r]), y[r] = rmsnorm(res_out[r], w),
// and the same rounded res_out[r] mirrored into capture[r][dst_col + ...].
template <int VEC>
__device__ __forceinline__ void add_rmsnorm_capture_row(const bf16* __restrict__ x,
                                                        const bf16* res,
                                                        const bf16* __restrict__ w,
                                                        bf16* __restrict__ y,
                                                        bf16* res_out,
                                                        bf16* __restrict__ capture,
                                                        int n, float eps, int dst_pitch,
                                                        int dst_col) {
    __shared__ float smem[32];
    const int nthr = blockDim.x;
    const int gid = threadIdx.x;
    const int nv = n / VEC;
    const bool cached = (nv <= nthr * (kMaxPerThread / VEC));   // block-uniform
    bf16* cap = capture + (size_t)blockIdx.x * dst_pitch + dst_col;

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
                bf16 s[VEC];
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    s[j] = __float2bfloat16(b[j] + a[j]);
                    res_out[idx + j] = s[j];
                }
                st_vec_bf16<VEC>(cap + idx, s);
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    rr[i * VEC + j] = __bfloat162float(s[j]);
                    ssq += rr[i * VEC + j] * rr[i * VEC + j];
                }
            }
        }
    } else {
        for (int c = gid; c < nv; c += nthr) {
            const size_t idx = (size_t)c * VEC;
            float a[VEC], b[VEC];
            ld_vec<VEC>(x + idx, a);
            ld_vec<VEC>(res + idx, b);
            bf16 s[VEC];
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                s[j] = __float2bfloat16(b[j] + a[j]);
                res_out[idx + j] = s[j];
            }
            st_vec_bf16<VEC>(cap + idx, s);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                b[j] = __bfloat162float(s[j]);
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

}  // namespace add_norm_capture

extern "C" {

// add_rmsnorm plus the capture mirror. One block per row, grid == rows.
__global__ __launch_bounds__(1024) void add_rmsnorm_capture(const __nv_bfloat16* __restrict__ x,
                                                            const __nv_bfloat16* res,
                                                            const __nv_bfloat16* __restrict__ w,
                                                            __nv_bfloat16* __restrict__ y,
                                                            __nv_bfloat16* res_out,
                                                            __nv_bfloat16* __restrict__ capture,
                                                            int rows, int hidden, float eps,
                                                            int dst_pitch, int dst_col) {
    using namespace add_norm_capture;
    shape(rows, hidden, eps, blockDim.x);
    rect(hidden, dst_pitch, dst_col);
    if (!x || !res || !w || !y || !res_out || !capture) __trap();
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t off = (size_t)row * hidden;
    // hidden is pinned to 2048 by the guard, so the 8-wide register-cached
    // path is the one the unfused add_rmsnorm takes for this shape.
    add_rmsnorm_capture_row<8>(x + off, res + off, w, y + off, res_out + off, capture, hidden,
                               eps, dst_pitch, dst_col);
}

}  // extern "C"
