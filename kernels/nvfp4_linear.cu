// Row-major packed NVFP4 E2M1 codes (U8 [N, K/2]), E4M3 group-16 scales ([N, K/16]), F32 global scale.
// M1..16 rounds effective weights in registers; larger M materializes BF16 weights.
#include "gpu_compat.cuh"
#if !defined(__HIP__)
#include <cuda_fp8.h>
#endif
#include <stdint.h>
#include "q38_contract.cuh"

#if !defined(__HIP__)
// ModelOpt NVFP4 lookup table maps both 0x0 and 0x8 to +0.0f.
__device__ __forceinline__ float nvfp4_e2m1_decode(uint8_t code) {
    return (code & 7u) == 0u ? 0.0f : e2m1_decode(code);
}

// launch: block 256; grid ceil(N*(K/2)/256), capped to a positive device-valid grid.
extern "C" __global__ void nvfp4_dequant_bf16(
        const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ scales,
        const float* __restrict__ ws2,
        __nv_bfloat16* __restrict__ out,
        int n,
        int k) {
    if (n <= 0 || k <= 0 || (k & 15) != 0 || blockDim.x != 256 || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(ws2) || !aligned16(out)) {
        __trap();
        return;
    }
    const float global_scale = __ldg(ws2);
    if (!isfinite(global_scale)) {
        __trap();
        return;
    }
    const uint64_t pairs = uint64_t(n) * uint64_t(k / 2);
    for (uint64_t p = uint64_t(blockIdx.x) * 256 + threadIdx.x;
         p < pairs;
         p += uint64_t(gridDim.x) * 256) {
        const float s = e4m3_decode(__ldg(scales + (p >> 3))) * global_scale;
        const uint8_t b = __ldg(w + p);
        const uint32_t lo =
            __bfloat16_as_ushort(__float2bfloat16_rn(nvfp4_e2m1_decode(b & 15u) * s));
        const uint32_t hi =
            __bfloat16_as_ushort(__float2bfloat16_rn(nvfp4_e2m1_decode(b >> 4) * s));
        *reinterpret_cast<uint32_t*>(out + 2 * p) = lo | (hi << 16);
    }
}

// launch: grid ceil(N/4), block 128. K % 1024 == 0 takes the vectorized path.
extern "C" __global__ void nvfp4_gemv_bf16(
        const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ scales,
        const float* __restrict__ ws2,
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        int n,
        int k) {
    if (n <= 0 || k <= 0 || (k & 15) != 0 || blockDim.x != 128 || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        uint64_t(gridDim.x) != (uint64_t(n) + 3) / 4 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(ws2) || !aligned16(x) || !aligned16(y)) {
        __trap();
        return;
    }
    const float global_scale = __ldg(ws2);
    if (!isfinite(global_scale)) {
        __trap();
        return;
    }
    const uint64_t row = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= uint64_t(n)) return;
    float acc = 0.0f;
    if ((k & 1023) == 0) {
        float accum[4] = {};
        const uint8_t* const row_codes = w + row * uint64_t(k / 2);
        const uint8_t* const row_scales = scales + row * uint64_t(k / 16);
        for (int base = 0; base < k; base += 1024) {
            const int col = base + int(lane) * 32;
            const uint4 packed = __ldg(reinterpret_cast<const uint4*>(row_codes + col / 2));
            const uint16_t raw_scales =
                __ldg(reinterpret_cast<const uint16_t*>(row_scales + col / 16));
            const float2 bs = e4m3_decode_pair(raw_scales);
            const float s0 = bs.x * global_scale;
            const float s1 = bs.y * global_scale;
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                const float s = word < 2 ? s0 : s1;
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + col + 8 * word));
                const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const uint8_t b = uint8_t(words[word] >> (8 * byte));
                    const float w_lo =
                        __bfloat162float(__float2bfloat16_rn(nvfp4_e2m1_decode(b & 15u) * s));
                    const float w_hi =
                        __bfloat162float(__float2bfloat16_rn(nvfp4_e2m1_decode(b >> 4) * s));
                    const float x_lo =
                        __bfloat162float(__ushort_as_bfloat16(uint16_t(xw[byte])));
                    const float x_hi =
                        __bfloat162float(__ushort_as_bfloat16(uint16_t(xw[byte] >> 16)));
                    accum[byte] = fmaf(w_hi, x_hi, fmaf(w_lo, x_lo, accum[byte]));
                }
            }
        }
        acc = (accum[0] + accum[1]) + (accum[2] + accum[3]);
    } else {
        const uint8_t* const row_codes = w + row * uint64_t(k / 2);
        const uint8_t* const row_scales = scales + row * uint64_t(k / 16);
        for (uint64_t pair = lane; pair < uint64_t(k / 2); pair += 32) {
            const float s = e4m3_decode(__ldg(row_scales + (pair >> 3))) * global_scale;
            const uint8_t b = __ldg(row_codes + pair);
            const float w_lo =
                __bfloat162float(__float2bfloat16_rn(nvfp4_e2m1_decode(b & 15u) * s));
            const float w_hi =
                __bfloat162float(__float2bfloat16_rn(nvfp4_e2m1_decode(b >> 4) * s));
            const float x_lo = __bfloat162float(x[2 * pair]);
            const float x_hi = __bfloat162float(x[2 * pair + 1]);
            acc = fmaf(w_hi, x_hi, fmaf(w_lo, x_lo, acc));
        }
    }
    #pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
        acc += __shfl_down_sync(FF_WARP_MASK, acc, shift);
    if (lane == 0) y[row] = __float2bfloat16_rn(acc);
}

template <int M>
__device__ __forceinline__ void nvfp4_skinny(
        const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ scales,
        const float* __restrict__ ws2,
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        int n,
        int k) {
    if (n <= 0 || k <= 0 || (k & 15) != 0 || blockDim.x != 128 || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        uint64_t(gridDim.x) != (uint64_t(n) + 3) / 4 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(ws2) || !aligned16(x) || !aligned16(y)) {
        __trap();
        return;
    }
    const float global_scale = __ldg(ws2);
    if (!isfinite(global_scale)) {
        __trap();
        return;
    }
    const uint64_t row = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= uint64_t(n)) return;
    float sums[M][4] = {};
    if ((k & 1023) == 0) {
        const uint8_t* const row_codes = w + row * uint64_t(k / 2);
        const uint8_t* const row_scales = scales + row * uint64_t(k / 16);
        for (int base = 0; base < k; base += 1024) {
            const int col = base + int(lane) * 32;
            const uint4 packed = __ldg(reinterpret_cast<const uint4*>(row_codes + col / 2));
            const uint16_t raw_scales =
                __ldg(reinterpret_cast<const uint16_t*>(row_scales + col / 16));
            const float2 bs = e4m3_decode_pair(raw_scales);
            const float s0 = bs.x * global_scale;
            const float s1 = bs.y * global_scale;
            uint32_t words = packed.x;
            #pragma unroll 1
            for (int word = 0; word < 4; ++word) {
                if (word == 1) words = packed.y;
                else if (word == 2) words = packed.z;
                else if (word == 3) words = packed.w;
                const float s = word < 2 ? s0 : s1;
                uint32_t ew[4];
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const uint8_t b = uint8_t(words >> (8 * byte));
                    const uint32_t lo =
                        __bfloat16_as_ushort(__float2bfloat16_rn(nvfp4_e2m1_decode(b & 15u) * s));
                    const uint32_t hi =
                        __bfloat16_as_ushort(__float2bfloat16_rn(nvfp4_e2m1_decode(b >> 4) * s));
                    ew[byte] = lo | (hi << 16);
                }
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const __nv_bfloat16* const row_x = x + uint64_t(m) * uint64_t(k) + col + 8 * word;
                    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(row_x));
                    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        const float w_lo =
                            __bfloat162float(__ushort_as_bfloat16(uint16_t(ew[byte])));
                        const float w_hi =
                            __bfloat162float(__ushort_as_bfloat16(uint16_t(ew[byte] >> 16)));
                        const float x_lo =
                            __bfloat162float(__ushort_as_bfloat16(uint16_t(xw[byte])));
                        const float x_hi =
                            __bfloat162float(__ushort_as_bfloat16(uint16_t(xw[byte] >> 16)));
                        sums[m][byte] = fmaf(w_hi, x_hi, fmaf(w_lo, x_lo, sums[m][byte]));
                    }
                }
            }
        }
    } else {
        const uint8_t* const row_codes = w + row * uint64_t(k / 2);
        const uint8_t* const row_scales = scales + row * uint64_t(k / 16);
        for (uint64_t pair = lane; pair < uint64_t(k / 2); pair += 32) {
            const float s = e4m3_decode(__ldg(row_scales + (pair >> 3))) * global_scale;
            const uint8_t b = __ldg(row_codes + pair);
            const float w_lo =
                __bfloat162float(__float2bfloat16_rn(nvfp4_e2m1_decode(b & 15u) * s));
            const float w_hi =
                __bfloat162float(__float2bfloat16_rn(nvfp4_e2m1_decode(b >> 4) * s));
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const __nv_bfloat16* const row_x = x + uint64_t(m) * uint64_t(k) + 2 * pair;
                const float x_lo = __bfloat162float(row_x[0]);
                const float x_hi = __bfloat162float(row_x[1]);
                sums[m][0] = fmaf(w_hi, x_hi, fmaf(w_lo, x_lo, sums[m][0]));
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float value = (sums[m][0] + sums[m][1]) + (sums[m][2] + sums[m][3]);
        #pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1)
            value += __shfl_down_sync(FF_WARP_MASK, value, shift);
        if (lane == 0) y[uint64_t(m) * uint64_t(n) + row] = __float2bfloat16_rn(value);
    }
}

#define NVFP4_SKINNY_ENTRY(M) \
extern "C" __global__ __launch_bounds__(128, 4) void nvfp4_skinny_bf16_m##M( \
    const uint8_t* w, const uint8_t* scales, const float* ws2, \
    const __nv_bfloat16* x, __nv_bfloat16* y, int n, int k) { \
    nvfp4_skinny<M>(w, scales, ws2, x, y, n, k); \
}
NVFP4_SKINNY_ENTRY(2)
NVFP4_SKINNY_ENTRY(3)
NVFP4_SKINNY_ENTRY(4)
NVFP4_SKINNY_ENTRY(5)
NVFP4_SKINNY_ENTRY(6)
NVFP4_SKINNY_ENTRY(7)
NVFP4_SKINNY_ENTRY(8)
NVFP4_SKINNY_ENTRY(9)
NVFP4_SKINNY_ENTRY(10)
NVFP4_SKINNY_ENTRY(11)
NVFP4_SKINNY_ENTRY(12)
NVFP4_SKINNY_ENTRY(13)
NVFP4_SKINNY_ENTRY(14)
NVFP4_SKINNY_ENTRY(15)
NVFP4_SKINNY_ENTRY(16)
#undef NVFP4_SKINNY_ENTRY
#endif
