// Row-major packed NVFP4 E2M1 codes (U8 [N, K/2]), E4M3 group-16 scales ([N, K/16]), F32 global scale.
// M1..16 uses SIMT warp reduction; M>=17 uses m16n8k16 BF16 Tensor Core MMA with register dequantization.
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

__device__ __forceinline__ uint32_t nvfp4_unpack_bf16x2(uint8_t b, float s) {
    const uint32_t lo =
        __bfloat16_as_ushort(__float2bfloat16_rn(nvfp4_e2m1_decode(b & 15u) * s));
    const uint32_t hi =
        __bfloat16_as_ushort(__float2bfloat16_rn(nvfp4_e2m1_decode(b >> 4) * s));
    return lo | (hi << 16);
}

template <int BM, int WARP_M, int WARP_N>
__device__ __forceinline__ void nvfp4_mma_bf16_impl(
        const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ scales,
        const float* __restrict__ ws2,
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        int m,
        int n,
        int k) {
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int N_SUB = BN / (16 * WARP_N);
    const uint64_t tiles_m = (uint64_t(m) + uint64_t(BM - 1)) / uint64_t(BM);
    const uint64_t tiles_n = (uint64_t(n) + 127u) / 128u;
    if (m <= 0 || n <= 0 || k <= 0 || (k & 15) != 0 ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 ||
        uint64_t(gridDim.x) != tiles_m * tiles_n ||
        !aligned16(w) || !aligned16(scales) || !aligned16(ws2) || !aligned16(x) || !aligned16(y)) {
        __trap();
        return;
    }
    const float global_scale = __ldg(ws2);
    if (!isfinite(global_scale)) {
        __trap();
        return;
    }

    constexpr uint32_t GROUP_M = 8;
    const uint32_t pid = blockIdx.x;
    const uint32_t tm = uint32_t(tiles_m);
    const uint32_t tn = uint32_t(tiles_n);
    const uint32_t group_size = GROUP_M * tn;
    const uint32_t group_id = pid / group_size;
    const uint32_t first_m = group_id * GROUP_M;
    const uint32_t group_m = (tm - first_m < GROUP_M) ? (tm - first_m) : GROUP_M;
    const uint32_t in_group = pid % group_size;
    const int m0 = int(first_m + (in_group % group_m)) * BM;
    const int n0 = int(in_group / group_m) * BN;
    const int m_valid = (m - m0 < BM) ? (m - m0) : BM;
    const int n_valid = (n - n0 < BN) ? (n - n0) : BN;

    struct StageStorage {
        __nv_bfloat16 x[2][BM][72];
        uint8_t w_codes[2][BN][32];
        uint32_t w_scales[2][BN];
    };
    union SharedStorage {
        StageStorage pipe;
        uint16_t epilogue[BM][136];
    };
    __shared__ __align__(16) SharedStorage smem;

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp / WARP_N;
    const int warp_n = warp % WARP_N;
    const int group = lane >> 2;
    const int tig = lane & 3;

    float accum[N_SUB][8][4] = {};

    auto consume_stage = [&](int stage, int k_steps) {
        uint32_t s_pack[N_SUB][2];
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            const int r0 = warp_n * (16 * N_SUB) + ns * 16 + group;
            s_pack[ns][0] = smem.pipe.w_scales[stage][r0];
            s_pack[ns][1] = smem.pipe.w_scales[stage][r0 + 8];
        }
        #pragma unroll
        for (int step_k = 0; step_k < 4; ++step_k) {
            if (step_k >= k_steps) break;
            const int kk = step_k * 16;
            const int h = kk >> 5;
            const int sub8 = (kk >> 1) & 8;
            uint32_t a[N_SUB][4];
            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
                const int r0 = warp_n * (16 * N_SUB) + ns * 16 + group;
                const int r1 = r0 + 8;
                const int swz0 = (h ^ ((r0 >> 2) & 1)) * 16 + sub8;
                const int swz1 = (h ^ ((r1 >> 2) & 1)) * 16 + sub8;
                const uint2 c0 =
                    *reinterpret_cast<const uint2*>(&smem.pipe.w_codes[stage][r0][swz0]);
                const uint2 c1 =
                    *reinterpret_cast<const uint2*>(&smem.pipe.w_codes[stage][r1][swz1]);
                const uint8_t sc0 = uint8_t(s_pack[ns][0] >> (8 * step_k));
                const uint8_t sc1 = uint8_t(s_pack[ns][1] >> (8 * step_k));
                const float s0 = (n0 + r0 < n) ? (e4m3_decode(sc0) * global_scale) : 0.0f;
                const float s1 = (n0 + r1 < n) ? (e4m3_decode(sc1) * global_scale) : 0.0f;
                a[ns][0] = nvfp4_unpack_bf16x2(uint8_t(c0.x >> (8 * tig)), s0);
                a[ns][1] = nvfp4_unpack_bf16x2(uint8_t(c1.x >> (8 * tig)), s1);
                a[ns][2] = nvfp4_unpack_bf16x2(uint8_t(c0.y >> (8 * tig)), s0);
                a[ns][3] = nvfp4_unpack_bf16x2(uint8_t(c1.y >> (8 * tig)), s1);
            }
            #pragma unroll
            for (int ms = 0; ms < 8; ++ms) {
                unsigned b0, b1;
                const int x_row = warp_m * 64 + ms * 8 + (lane & 7);
                const int x_col = kk + ((lane >> 3) & 1) * 8;
                const unsigned addr =
                    (unsigned)__cvta_generic_to_shared(&smem.pipe.x[stage][x_row][x_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b0), "=r"(b1) : "r"(addr));
                #pragma unroll
                for (int ns = 0; ns < N_SUB; ++ns) {
                    float* d = accum[ns][ms];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                        : "r"(a[ns][0]), "r"(a[ns][1]), "r"(a[ns][2]), "r"(a[ns][3]),
                          "r"(b0), "r"(b1));
                }
            }
        }
    };

    if ((k & 63) == 0) {
        const int x_row_base = tid >> 3;
        const int x_col = (tid & 7) * 8;
        const int w_row = tid >> 1;
        const int w_half = tid & 1;
        const int w_swz_col = (w_half ^ ((w_row >> 2) & 1)) * 16;
        const bool w_live = (n0 + w_row) < n;
        const int w_bytes = w_live ? 16 : 0;
        const uint8_t* w_base = w_live
            ? (w + uint64_t(n0 + w_row) * uint64_t(k >> 1) + uint64_t(w_half * 16))
            : w;
        auto stage_tile = [&](int stage, int k0) {
            #pragma unroll
            for (int i = 0; i < BM / 32; ++i) {
                const int xr = x_row_base + i * 32;
                const bool x_live = xr < m_valid;
                const __nv_bfloat16* x_src = x_live
                    ? (x + uint64_t(m0 + xr) * uint64_t(k) + uint64_t(k0 + x_col))
                    : x;
                const int x_bytes = x_live ? 16 : 0;
                const unsigned x_dst =
                    (unsigned)__cvta_generic_to_shared(&smem.pipe.x[stage][xr][x_col]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                    :: "r"(x_dst), "l"(x_src), "r"(x_bytes) : "memory");
            }
            const uint8_t* w_src = w_live ? (w_base + uint64_t(k0 >> 1)) : w;
            const unsigned w_dst =
                (unsigned)__cvta_generic_to_shared(&smem.pipe.w_codes[stage][w_row][w_swz_col]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                :: "r"(w_dst), "l"(w_src), "r"(w_bytes) : "memory");
            if (tid < BN) {
                smem.pipe.w_scales[stage][tid] = (n0 + tid < n)
                    ? __ldg(reinterpret_cast<const uint32_t*>(
                          scales + uint64_t(n0 + tid) * uint64_t(k >> 4) + uint64_t(k0 >> 4)))
                    : 0u;
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        };
        stage_tile(0, 0);
        for (int k0 = 0, stage = 0; k0 < k; k0 += BK, stage ^= 1) {
            if (k0 + BK < k) {
                stage_tile(stage ^ 1, k0 + BK);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
            consume_stage(stage, 4);
            __syncthreads();
        }
    } else {
        for (int k0 = 0; k0 < k; k0 += BK) {
            for (int idx = tid; idx < BM * BK; idx += 256) {
                const int xr = idx >> 6;
                const int xc = idx & 63;
                smem.pipe.x[0][xr][xc] = (xr < m_valid && k0 + xc < k)
                    ? x[uint64_t(m0 + xr) * uint64_t(k) + uint64_t(k0 + xc)]
                    : __ushort_as_bfloat16(0);
            }
            for (int idx = tid; idx < BN * 32; idx += 256) {
                const int wr = idx >> 5;
                const int wc = idx & 31;
                const int h = wc >> 4;
                const int swz = (h ^ ((wr >> 2) & 1)) * 16 + (wc & 15);
                smem.pipe.w_codes[0][wr][swz] = (n0 + wr < n && (k0 >> 1) + wc < (k >> 1))
                    ? __ldg(w + uint64_t(n0 + wr) * uint64_t(k >> 1) + uint64_t((k0 >> 1) + wc))
                    : 0u;
            }
            if (tid < BN) {
                uint32_t sp = 0u;
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    if (n0 + tid < n && (k0 >> 4) + g < (k >> 4)) {
                        sp |= uint32_t(__ldg(
                                  scales + uint64_t(n0 + tid) * uint64_t(k >> 4) +
                                  uint64_t((k0 >> 4) + g)))
                              << (8 * g);
                    }
                }
                smem.pipe.w_scales[0][tid] = sp;
            }
            __syncthreads();
            const int k_rem = k - k0;
            consume_stage(0, k_rem < BK ? (k_rem >> 4) : 4);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        const int base_n = warp_n * (16 * N_SUB) + ns * 16 + group;
        #pragma unroll
        for (int ms = 0; ms < 8; ++ms) {
            const int base_m = warp_m * 64 + ms * 8 + tig * 2;
            smem.epilogue[base_m + 0][base_n + 0] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[ns][ms][0]));
            smem.epilogue[base_m + 1][base_n + 0] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[ns][ms][1]));
            smem.epilogue[base_m + 0][base_n + 8] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[ns][ms][2]));
            smem.epilogue[base_m + 1][base_n + 8] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[ns][ms][3]));
        }
    }
    __syncthreads();
    if (n_valid == BN && (n & 7) == 0) {
        const int col8 = (tid & 15) * 8;
        for (int r = (tid >> 4); r < m_valid; r += 16) {
            const uint4 v = *reinterpret_cast<const uint4*>(&smem.epilogue[r][col8]);
            *reinterpret_cast<uint4*>(y + uint64_t(m0 + r) * uint64_t(n) + uint64_t(n0 + col8)) = v;
        }
    } else {
        const int col = tid & 127;
        for (int r = (tid >> 7); r < m_valid; r += 2) {
            if (col < n_valid) {
                y[uint64_t(m0 + r) * uint64_t(n) + uint64_t(n0 + col)] =
                    __ushort_as_bfloat16(smem.epilogue[r][col]);
            }
        }
    }
}

// launch: block 256, grid ceil(M/64)*ceil(N/128); 3 CTAs/SM (24 warps/SM).
extern "C" __global__ __launch_bounds__(256, 3) void nvfp4_mma_bf16_m64(
        const uint8_t* w,
        const uint8_t* scales,
        const float* ws2,
        const __nv_bfloat16* x,
        __nv_bfloat16* y,
        int m,
        int n,
        int k) {
    nvfp4_mma_bf16_impl<64, 1, 8>(w, scales, ws2, x, y, m, n, k);
}

// launch: block 256, grid ceil(M/128)*ceil(N/128); 2 CTAs/SM (16 warps/SM).
extern "C" __global__ __launch_bounds__(256, 2) void nvfp4_mma_bf16_m128(
        const uint8_t* w,
        const uint8_t* scales,
        const float* ws2,
        const __nv_bfloat16* x,
        __nv_bfloat16* y,
        int m,
        int n,
        int k) {
    nvfp4_mma_bf16_impl<128, 2, 4>(w, scales, ws2, x, y, m, n, k);
}
#endif
