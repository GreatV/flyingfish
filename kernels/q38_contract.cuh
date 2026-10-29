#pragma once
#include "gpu_compat.cuh"
#if !defined(__HIP__)
#include <cuda_fp8.h>
#endif
#include <math.h>
#include <stdint.h>
struct Q38SequenceFields {
    int32_t prefix, valid_rows, key_count, capacity, causal, reserved[3];
};
struct alignas(16) Q38SequenceMeta : Q38SequenceFields {};
static_assert(sizeof(Q38SequenceMeta)==32,"Q38 ABI sequence size");
__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}
__device__ __forceinline__ float bround(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sigmoid(float x) {
    if (x>=0.0f) return 1.0f/(1.0f+expf(-x));
    float e=expf(x); return e/(1.0f+e);
}
__device__ __forceinline__ float silu(float x) {return x*sigmoid(x);}

__device__ __forceinline__ void q38_meta_guard(const Q38SequenceMeta* s, bool causal_required=true) {
    if (!s || (reinterpret_cast<uintptr_t>(s)&15)) { __trap(); return; }
    if (s->prefix<0 || s->valid_rows<=0 || s->capacity<=0 ||
        (s->causal!=1 && (causal_required || s->causal!=0)) ||
        int64_t(s->prefix)+s->valid_rows!=s->key_count || s->key_count>s->capacity ||
        s->reserved[0] || s->reserved[1] || s->reserved[2]) __trap();
}

// E8M0FN scale: the value is 2^(code - 127); 0xff is NaN.
__device__ __forceinline__ float e8m0_decode(uint8_t code) {
    if (code == 0xffu) return __uint_as_float(0x7fc00000u);
    return ldexpf(1.0f, int(code) - 127);
}

__device__ __forceinline__ float e4m3_decode(uint8_t code) {
    const uint32_t sign = uint32_t(code & 0x80) << 24;
    const uint32_t mag = code & 0x7f;
    if (mag == 0x7f) return __uint_as_float(sign | 0x7fc00000u);
    const uint32_t e = mag >> 3, m = mag & 7;
    if (e == 0) return __uint_as_float(__float_as_uint(float(m) * (1.0f/512.0f)) | sign);
    return __uint_as_float(sign | ((e + 120) << 23) | (m << 20));
}

#if defined(__HIP__)
// BF16-rounded channel weights below 2^-64 are scaled by 2^64 inside WMMA tiles.
__device__ __forceinline__ __nv_bfloat16 fp8_channel_tile_weight(
    uint8_t code, __nv_bfloat16 channel_scale) {
    const float scale = __bfloat162float(channel_scale);
    const float boost = scale < 0x1p-64f ? 0x1p64f : 1.0f;
    return __float2bfloat16(__bfloat162float(__float2bfloat16_rn(e4m3_decode(code) * scale)) * boost);
}
__device__ __forceinline__ float fp8_channel_tile_unscale(__nv_bfloat16 channel_scale) {
    return __bfloat162float(channel_scale) < 0x1p-64f ? 0x1p-64f : 1.0f;
}
#endif

#if defined(__HIP__)
__device__ __forceinline__ uint8_t e4m3_encode_finite(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t sign = (bits >> 24) & 128;
    const float magnitude = fabsf(value);
    if (magnitude < 0.015625f)
        return uint8_t(sign | uint32_t(nearbyintf(magnitude * 512.0f)));
    const uint32_t absolute = bits & 0x7fffffffu;
    uint32_t rounded = absolute >> 20;
    const uint32_t remainder = absolute & 0xfffffu;
    rounded += remainder > 0x80000u || (remainder == 0x80000u && (rounded & 1));
    return uint8_t(sign | min(126u, rounded - 960u));
}
#else
__device__ __forceinline__ uint8_t e4m3_encode_finite(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}
#endif

// ceil(log2(v)) of a positive normal float, from its bit pattern.
__device__ __forceinline__ int act_quant_log2_ceil(float v) {
    const unsigned bits = __float_as_uint(v);
    const int exponent = int((bits >> 23) & 0xFFu) - 127;
    return exponent + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
}

// 2^exponent.
__device__ __forceinline__ float act_quant_pow2(int exponent) {
    return __uint_as_float((unsigned)(exponent + 127) << 23);
}

// One 32-element activation-quant group, each lane owning one element: the amax
// is floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the code is
// E4M3 round-to-nearest-even of value/scale clamped to +-448 (model definition,
// publisher kernel.py act_quant, block 32, scale_fmt "ue8m0"). Shared by the
// W4A8/W8A8 expert paths and the DeepSeek-V4.1 latent ring and attention-output
// quantization.
__device__ __forceinline__ void act_quant32(float value, unsigned char* code,
                                            unsigned char* scale_code) {
    float amax = fabsf(value);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(FF_WARP_MASK, amax, off));
    amax = fmaxf(amax, 1e-4f);
    const int exponent = act_quant_log2_ceil(amax * (1.0f / 448.0f));
    if (exponent < -127 || exponent > 127) {
        printf("act_quant32 trap value=%f amax=%f exponent=%d block=%u thread=%u\n",
               value, amax, exponent, blockIdx.x, threadIdx.x);
        __trap();
    }
    const float scale = act_quant_pow2(exponent);
    *code = e4m3_encode_finite(fminf(448.0f, fmaxf(-448.0f, value / scale)));
    *scale_code = (unsigned char)(exponent + 127);
}

// E2M1 nibble magnitude and sign; both signed zeros are preserved.
__device__ __forceinline__ float e2m1_decode(uint8_t code) {
    const int magnitude = code & 7;
    const float value = magnitude < 4 ? float(magnitude) * 0.5f
        : magnitude == 4 ? 2.0f : magnitude == 5 ? 3.0f
        : magnitude == 6 ? 4.0f : 6.0f;
    return __uint_as_float(__float_as_uint(value) | (uint32_t(code & 8) << 28));
}

__device__ __forceinline__ uint8_t e2m1_encode(float value) {
    const float magnitude = fabsf(value);
    const uint8_t code = magnitude <= 0.25f ? 0 : magnitude < 0.75f ? 1
        : magnitude <= 1.25f ? 2 : magnitude < 1.75f ? 3
        : magnitude <= 2.5f ? 4 : magnitude < 3.5f ? 5
        : magnitude <= 5.0f ? 6 : 7;
    return code | uint8_t((__float_as_uint(value) >> 28) & 8);
}

// All 32 lanes participate; a 16-wide specialization reduces each half-warp separately.
template<int Block, bool PowerOfTwoScale>
__device__ __forceinline__ void fp4_act_quant(
    float value, uint8_t* code, uint8_t* scale_code) {
    static_assert((Block == 16 && !PowerOfTwoScale) || (Block == 32 && PowerOfTwoScale));
    if (!isfinite(value)) __trap();
    float amax = fabsf(value);
#pragma unroll
    for (int offset = Block / 2; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(FF_WARP_MASK, amax, offset));
    const float unrounded = __fdiv_rn(amax, 6.0f);
    float scale;
    if constexpr (PowerOfTwoScale) {
        const int exponent = act_quant_log2_ceil(fmaxf(unrounded, 0x1p-126f));
        if (exponent > 127) __trap();
        *scale_code = uint8_t(exponent + 127);
        scale = act_quant_pow2(exponent);
    } else {
        uint8_t rounded = e4m3_encode_finite(unrounded);
        *scale_code = rounded == 0 ? 1 : rounded;
        scale = e4m3_decode(*scale_code);
    }
    *code = e2m1_encode(__fdiv_rn(value, scale));
}

#if !defined(__HIP__)
#include <cuda_fp16.h>
// Two E4M3 codes per call. E4M3 is a subset of FP16, so the hardware form is
// exact for all 256 codes; the fix-up forces the canonical NaN for both NaN codes.
__device__ __forceinline__ float2 e4m3_decode_pair(uint16_t packed) {
#if __CUDA_ARCH__ >= 890
    uint32_t halfs;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halfs) : "h"(packed));
    float2 values = make_float2(__half2float(__ushort_as_half(uint16_t(halfs))),
                               __half2float(__ushort_as_half(uint16_t(halfs >> 16))));
    const uint32_t low = packed & 255u, high = packed >> 8;
    if ((low & 127u) == 127u)
        values.x = __uint_as_float(((low & 128u) << 24) | 0x7fc00000u);
    if ((high & 127u) == 127u)
        values.y = __uint_as_float(((high & 128u) << 24) | 0x7fc00000u);
    return values;
#else
    return make_float2(e4m3_decode(uint8_t(packed)), e4m3_decode(uint8_t(packed >> 8)));
#endif
}

__device__ __forceinline__ float fp8_block_scale(float value) { return value; }
__device__ __forceinline__ float fp8_block_scale(uint8_t value) { return e8m0_decode(value); }

template <typename Scale, int BLOCK>
__device__ __forceinline__ void fp8_block_decode(
        const uint8_t* w, const Scale* sw, const uint8_t* x, const Scale* sx,
        __nv_bfloat16* y, int m, int n, int k, uint64_t tile) {
    const uint64_t tiles = (uint64_t(n) + 7) / 8;
    const uint64_t row = tile / tiles;
    const uint64_t col = (tile % tiles) * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= uint64_t(m) || col >= uint64_t(n)) return;
    const uint64_t groups = (uint64_t(k) + BLOCK - 1) / BLOCK;
    float total = 0.0f;
    for (uint64_t group = 0; group < groups; ++group) {
        float sum = 0.0f;
        #pragma unroll
        for (int offset = lane; offset < BLOCK; offset += 32) {
            const uint64_t inner = group * BLOCK + offset;
            if (inner < uint64_t(k)) {
                const uint16_t pair = uint16_t(w[col * k + inner]) |
                                      (uint16_t(x[row * k + inner]) << 8);
                const float2 values = e4m3_decode_pair(pair);
                sum = fmaf(values.x, values.y, sum);
            }
        }
        #pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, shift);
        if (lane == 0) {
            const float scale = fp8_block_scale(sw[(col / BLOCK) * groups + group]) *
                                fp8_block_scale(sx[row * groups + group]);
            total = fmaf(sum, scale, total);
        }
    }
    if (lane == 0) y[row * n + col] = __float2bfloat16_rn(total);
}


#endif
