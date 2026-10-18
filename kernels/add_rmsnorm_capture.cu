// BF16 residual capture with standard or zero-centered RMSNorm.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "q38_contract.cuh"

namespace add_norm_capture {

using bf16 = __nv_bfloat16;

constexpr int kHidden = 2048;      // hidden_size
constexpr int kVec = 8;            // bf16 elements per 16 B access
constexpr int kMaxPerThread = 8;   // elements cached per thread in registers

__device__ __forceinline__ void shape(int rows, int hidden, float eps, unsigned nthr) {
    if (hidden != kHidden && hidden != 5120) __trap();
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

template <int VEC>
__device__ __forceinline__ void st_vec_bf16(bf16* __restrict__ p, const bf16* __restrict__ v) {
    static_assert(VEC % 8 == 0, "vector store needs a multiple of 8 elements");
    union U { float4 v; __nv_bfloat162 h[VEC / 2]; } u;
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) u.h[k] = __nv_bfloat162(v[2 * k], v[2 * k + 1]);
    *reinterpret_cast<float4*>(p) = u.v;
}

template <int VEC>
__device__ __forceinline__ void add_rmsnorm_capture_row(const bf16* __restrict__ x,
                                                        const bf16* res,
                                                        const bf16* __restrict__ w,
                                                        bf16* __restrict__ y,
                                                        bf16* res_out,
                                                        bf16* __restrict__ capture,
                                                        int n, float eps, int dst_pitch,
                                                        int dst_col, int row) {
    __shared__ float smem[32];
    const int nthr = blockDim.x;
    const int gid = threadIdx.x;
    const int nv = n / VEC;
    const bool cached = (nv <= nthr * (kMaxPerThread / VEC));   // block-uniform
    bf16* cap = capture + (size_t)row * dst_pitch + dst_col;

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

__device__ __forceinline__ float zero_scale(float a, float inv, bf16 w) {
    return __fmul_rn(__fmul_rn(a, inv), __fadd_rn(1.0f, __bfloat162float(w)));
}

__device__ __forceinline__ float zero_inv(float sum, int hidden, float eps) {
    return rsqrtf(__fadd_rn(sum / float(hidden), eps));
}

__device__ __forceinline__ void zero_row(
    const bf16* x, const bf16* res, const bf16* w, bf16* y,
    bf16* res_out, bf16* cap, int hidden, float eps) {
    __shared__ float sums[8];
    float values[20];
    float sum = 0.0f;
    const int count = hidden / 256;
    for (int c = 0; c < count; ++c) {
        int d = c * 256 + threadIdx.x;
        bf16 value = __hadd(res[d], x[d]);
        res_out[d] = value;
        cap[d] = value;
        values[c] = __bfloat162float(value);
        sum = __fadd_rn(sum, __fmul_rn(values[c], values[c]));
    }
    for (int delta = 16; delta; delta /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = sum;
    __syncthreads();
    sum = threadIdx.x < 8 ? sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int delta = 16; delta; delta /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
        if (threadIdx.x == 0) sums[0] = zero_inv(sum, hidden, eps);
    }
    __syncthreads();
    const float inv = sums[0];
    for (int c = 0; c < count; ++c) {
        int d = c * 256 + threadIdx.x;
        y[d] = __float2bfloat16_rn(zero_scale(values[c], inv, w[d]));
    }
}

union Vec4 {
    uint2 u;
    __nv_bfloat162 h[2];
    bf16 e[4];
};

__device__ __forceinline__ void zero_four_rows(
    const bf16* x, const bf16* res, const bf16* w, bf16* y,
    bf16* res_out, bf16* capture, int rows, int valid, int base,
    int pitch, int column, float eps) {
    const int begin = base + int(blockIdx.x) * 4;
    if (begin >= valid) return;
    const int slot = threadIdx.x >> 6;
    const int tid = threadIdx.x & 63;
    const int hw = tid >> 5;
    const int lane = tid & 31;
    const int offset = int(blockIdx.x) * 4 + slot;
    const int row = base + offset;
    const bool active = offset < rows && row < valid;
    uint2 values[20];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active) {
        const bf16* xr = x + size_t(row) * 5120;
        const bf16* rr = res + size_t(row) * 5120;
        bf16* dst = res_out + size_t(row) * 5120;
        bf16* cap = capture + size_t(row) * pitch + column;
        #pragma unroll
        for (int c = 0; c < 20; ++c) {
            int d4 = c * 64 + tid;
            Vec4 a, b;
            a.u = reinterpret_cast<const uint2*>(xr)[d4];
            b.u = reinterpret_cast<const uint2*>(rr)[d4];
            a.h[0] = __hadd2(a.h[0], b.h[0]);
            a.h[1] = __hadd2(a.h[1], b.h[1]);
            reinterpret_cast<uint2*>(dst)[d4] = a.u;
            reinterpret_cast<uint2*>(cap)[d4] = a.u;
            values[c] = a.u;
            float2 f0 = __bfloat1622float2(a.h[0]);
            float2 f1 = __bfloat1622float2(a.h[1]);
            acc[0] = __fadd_rn(acc[0], __fmul_rn(f0.x, f0.x));
            acc[1] = __fadd_rn(acc[1], __fmul_rn(f0.y, f0.y));
            acc[2] = __fadd_rn(acc[2], __fmul_rn(f1.x, f1.x));
            acc[3] = __fadd_rn(acc[3], __fmul_rn(f1.y, f1.y));
        }
    }
    #pragma unroll
    for (int delta = 4; delta >= 1; delta >>= 1) {
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            acc[k] = __fadd_rn(acc[k], __shfl_down_sync(0xffffffffu, acc[k], delta));
    }
    float ww = __fadd_rn(__fadd_rn(acc[0], acc[2]), __fadd_rn(acc[1], acc[3]));
    __shared__ float sums[4][4];
    if (hw == 1 && (lane & 7) == 0) sums[slot][lane >> 3] = ww;
    __syncthreads();
    if (hw == 0) {
        if ((lane & 7) == 0) ww = __fadd_rn(ww, sums[slot][lane >> 3]);
        ww = __fadd_rn(ww, __shfl_down_sync(0xffffffffu, ww, 16));
        ww = __fadd_rn(ww, __shfl_down_sync(0xffffffffu, ww, 8));
        if (lane == 0) sums[slot][0] = zero_inv(ww, 5120, eps);
    }
    __syncthreads();
    if (active) {
        const float inv = sums[slot][0];
        bf16* yr = y + size_t(row) * 5120;
        #pragma unroll
        for (int c = 0; c < 20; ++c) {
            const int d4 = c * 64 + tid;
            Vec4 a, weight, result;
            a.u = values[c];
            weight.u = reinterpret_cast<const uint2*>(w)[d4];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                result.e[k] = __float2bfloat16_rn(zero_scale(__bfloat162float(a.e[k]), inv, weight.e[k]));
            reinterpret_cast<uint2*>(yr)[d4] = result.u;
        }
    }
}

}

extern "C" __global__ __launch_bounds__(256) void add_rmsnorm_capture(
    const __nv_bfloat16* x, const __nv_bfloat16* res, const __nv_bfloat16* w,
    __nv_bfloat16* y, __nv_bfloat16* res_out, __nv_bfloat16* capture,
    int rows, int hidden, float eps, int pitch, int column, int mode,
    const Q38SequenceMeta* meta, int base) {
    using namespace add_norm_capture;
    shape(rows, hidden, eps, blockDim.x);
    rect(hidden, pitch, column);
    if (!x || !res || !w || !y || !res_out || !capture || mode < 0 || mode > 1 ||
        blockDim.x != 256 || base < 0) __trap();
    if (meta) q38_meta_guard(meta);
    const int valid = meta ? meta->valid_rows : rows;
    const bool four = mode == 1 && hidden == 5120 && rows > 16;
    if (int(gridDim.x) != (four ? (rows + 3) / 4 : rows)) __trap();
    if (four) {
        zero_four_rows(x, res, w, y, res_out, capture, rows, valid, base, pitch, column, eps);
        return;
    }
    const int row = base + int(blockIdx.x);
    if (row >= valid) return;
    const size_t offset = size_t(row) * hidden;
    if (mode == 1) {
        zero_row(x + offset, res + offset, w, y + offset, res_out + offset,
                 capture + size_t(row) * pitch + column, hidden, eps);
    } else {
        add_rmsnorm_capture_row<8>(x + offset, res + offset, w, y + offset,
                                   res_out + offset, capture, hidden, eps, pitch, column, row);
    }
}
