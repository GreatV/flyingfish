// Row-major E4M3 codes; BF16 block-128 or per-output-channel scales.
// M1..16 entry points form the rounded weight in registers, accumulate FP32 and
// write BF16 with no matrix scratch; larger M materializes effective BF16 weights.
// Format: __nv_fp8_e4m3 (CUDA Math API).
#include "gpu_compat.cuh"
#if !defined(__HIP__)
#include <cuda_fp8.h>
#endif
#include <stdint.h>
#include "q38_contract.cuh"

template <bool PER_CHANNEL>
__device__ __forceinline__ __nv_bfloat16 weight_at(
        const uint8_t* w, const __nv_bfloat16* scales, uint64_t n,
        uint64_t k, uint64_t in, uint64_t scale_stride) {
    const float s = PER_CHANNEL
        #if defined(__HIP__)
        ? __bfloat162float(scales[n])
        #else
        ? __bfloat162float(__ldg(scales + n))
        #endif
        : __bfloat162float(scales[(n/128)*scale_stride + k/128]);
    return __float2bfloat16_rn(e4m3_decode(w[n*in+k]) * s);
}
// launch: block256; grid ceil(N*K/256), capped to a positive device-valid grid.
template <bool PER_CHANNEL>
__device__ __forceinline__ void fp8_dequant(
        const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        __nv_bfloat16* __restrict__ out, int n, int k) {
    if (n <= 0 || k <= 0 || blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || !aligned16(w) || !aligned16(scales) || !aligned16(out)) {
        __trap(); return;
    }
    const uint64_t count = uint64_t(n)*uint64_t(k);
    const uint64_t scale_stride = (uint64_t(k)+127)/128;
    for (uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;
         i<count; i+=uint64_t(gridDim.x)*256) {
        out[i] = weight_at<PER_CHANNEL>(w,scales,i/uint64_t(k),i%uint64_t(k),k,scale_stride);
    }
}
extern "C" __global__ void fp8_dequant_bf16(
        const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        __nv_bfloat16* __restrict__ out, int n, int k) {
    fp8_dequant<false>(w, scales, out, n, k);
}
extern "C" __global__ void fp8_dequant_bf16_per_channel(
        const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        __nv_bfloat16* __restrict__ out, int n, int k) {
    fp8_dequant<true>(w, scales, out, n, k);
}
#if !defined(__HIP__)
__device__ __forceinline__ float2 gemv_decode_pair(uint16_t packed) {
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

__device__ __forceinline__ void gemv_load_pair(const uint8_t* first,
                                              uint4& a, uint4& b) {
    asm volatile(
        "ld.global.v4.u32 {%0,%1,%2,%3}, [%8];\n\t"
        "ld.global.v4.u32 {%4,%5,%6,%7}, [%9];"
        : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w),
          "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
        : "l"(first), "l"(first + 512));
}

__device__ __forceinline__ void gemv_accumulate_tiles(
        uint4 first, uint4 next, const __nv_bfloat16* x,
        float scale0, float scale1, float& a0, float& a1, float& a2, float& a3) {
    const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(x));
    const uint4 x1 = __ldg(reinterpret_cast<const uint4*>(x + 8));
    const uint4 x2 = __ldg(reinterpret_cast<const uint4*>(x + 512));
    const uint4 x3 = __ldg(reinterpret_cast<const uint4*>(x + 520));
    const uint32_t code0[4] = {first.x, first.y, first.z, first.w};
    const uint32_t code1[4] = {next.x, next.y, next.z, next.w};
    const uint32_t inputs0[8] = {x0.x,x0.y,x0.z,x0.w,x1.x,x1.y,x1.z,x1.w};
    const uint32_t inputs1[8] = {x2.x,x2.y,x2.z,x2.w,x3.x,x3.y,x3.z,x3.w};
    float accum[4] = {a0,a1,a2,a3};
    #pragma unroll
    for (int j=0; j<4; ++j) {
        #pragma unroll
        for (int byte=0; byte<4; ++byte) {
            // pair = (byte of code0[j], same byte of code1[j])
            const uint16_t pair = uint16_t(__byte_perm(code0[j], code1[j], byte | ((byte+4)<<4)));
            const float2 values = gemv_decode_pair(pair);
            const float w0 = __bfloat162float(__float2bfloat16_rn(values.x * scale0));
            const float w1 = __bfloat162float(__float2bfloat16_rn(values.y * scale1));
            const uint32_t in0 = inputs0[2*j+byte/2] >> ((byte&1)*16);
            const uint32_t in1 = inputs1[2*j+byte/2] >> ((byte&1)*16);
            accum[byte] = fmaf(w0, __bfloat162float(__ushort_as_bfloat16(uint16_t(in0))), accum[byte]);
            accum[(byte+2)&3] = fmaf(w1, __bfloat162float(__ushort_as_bfloat16(uint16_t(in1))), accum[(byte+2)&3]);
        }
    }
    a0=accum[0]; a1=accum[1]; a2=accum[2]; a3=accum[3];
}

// launch: grid ceil(N/4), block 128. K % 1024 == 0 takes the vectorized path.
#define FP8_GEMV_ENTRY(NAME, PER_CHANNEL) \
extern "C" __global__ void NAME( \
        const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales, \
        const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, \
        int n, int k) { \
    if (n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || \
        gridDim.y != 1 || gridDim.z != 1 || uint64_t(gridDim.x) != (uint64_t(n)+3)/4 || \
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) { \
        __trap(); return; \
    } \
    const uint64_t row = uint64_t(blockIdx.x)*4 + threadIdx.x/32; \
    const uint32_t lane = threadIdx.x%32; \
    if (row >= uint64_t(n)) return; \
    const uint64_t scale_stride = (uint64_t(k)+127)/128; \
    float acc=0.0f; \
    if ((k & 1023) == 0) { \
        float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f; \
        const uint8_t* const row_codes = w + row * uint64_t(k); \
        const __nv_bfloat16* const row_scales = PER_CHANNEL \
            ? scales + row \
            : scales + (row / 128) * scale_stride; \
        for (int base=0; base<k; base+=1024) { \
            const int col = base + int(lane) * 16; \
            uint4 first, next; \
            gemv_load_pair(row_codes + col, first, next); \
            float scale0=0.0f, scale1=0.0f; \
            if constexpr (PER_CHANNEL) { \
                scale0 = __bfloat162float(__ldg(row_scales)); \
                scale1 = scale0; \
            } else { \
                if ((lane & 7u) == 0) { \
                    scale0 = __bfloat162float(__ldg(row_scales + col / 128)); \
                    scale1 = __bfloat162float(__ldg(row_scales + (col + 512) / 128)); \
                } \
                scale0 = __shfl_sync(0xffffffffu, scale0, 0, 8); \
                scale1 = __shfl_sync(0xffffffffu, scale1, 0, 8); \
            } \
            gemv_accumulate_tiles(first, next, x + col, scale0, scale1, a0, a1, a2, a3); \
        } \
        acc = (a0+a1)+(a2+a3); \
    } else { \
        for (uint64_t col=lane; col<uint64_t(k); col+=32) { \
            const float weight = __bfloat162float(weight_at<PER_CHANNEL>(w,scales,row,col,k,scale_stride)); \
            acc = fmaf(weight,__bfloat162float(x[col]),acc); \
        } \
    } \
    for (int shift=16; shift>0; shift/=2) acc += __shfl_down_sync(0xffffffffu,acc,shift); \
    if (lane==0) y[row]=__float2bfloat16_rn(acc); \
}
FP8_GEMV_ENTRY(fp8_gemv_bf16, false)
FP8_GEMV_ENTRY(fp8_gemv_bf16_per_channel, true)
#undef FP8_GEMV_ENTRY

template<int M, bool PER_CHANNEL>
__device__ __forceinline__ void fp8_skinny(
        const uint8_t* w, const __nv_bfloat16* scales,
        const __nv_bfloat16* x, __nv_bfloat16* y, int n, int k) {
    if (n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || uint64_t(gridDim.x) != (uint64_t(n)+3)/4 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) {
        __trap(); return;
    }
    const uint64_t row = uint64_t(blockIdx.x)*4 + threadIdx.x/32;
    const uint32_t lane = threadIdx.x%32;
    if (row >= uint64_t(n)) return;
    const uint64_t stride = (uint64_t(k)+127)/128;
    float sums[M][4] = {};
    if ((k & 1023) == 0) {
        const uint8_t* const codes = w + row*uint64_t(k);
        const __nv_bfloat16* const row_scales = PER_CHANNEL
            ? scales + row : scales + (row/128)*stride;
        for (int base=0; base<k; base+=1024) {
            const int col = base + int(lane)*16;
            uint4 first, next;
            gemv_load_pair(codes+col, first, next);
            float s0=0.0f,s1=0.0f;
            if constexpr (PER_CHANNEL) {
                s0=__bfloat162float(__ldg(row_scales));
                s1=s0;
            } else {
            if ((lane&7u)==0) {
                s0=__bfloat162float(__ldg(row_scales+col/128));
                s1=__bfloat162float(__ldg(row_scales+(col+512)/128));
            }
            s0=__shfl_sync(0xffffffffu,s0,0,8);
            s1=__shfl_sync(0xffffffffu,s1,0,8);
            }
            const uint32_t c0[4]={first.x,first.y,first.z,first.w};
            const uint32_t c1[4]={next.x,next.y,next.z,next.w};
            #pragma unroll
            for (int word=0; word<4; ++word) {
                uint32_t effective[4];
                #pragma unroll
                for (int byte=0; byte<4; ++byte) {
                    const uint16_t pair=uint16_t(__byte_perm(c0[word],c1[word],byte|((byte+4)<<4)));
                    const float2 values=gemv_decode_pair(pair);
                    const uint32_t a=__bfloat16_as_ushort(__float2bfloat16_rn(values.x*s0));
                    const uint32_t b=__bfloat16_as_ushort(__float2bfloat16_rn(values.y*s1));
                    effective[byte]=a|(b<<16);
                }
                #pragma unroll
                for (int m=0; m<M; ++m) {
                    const __nv_bfloat16* const row_x=x+uint64_t(m)*uint64_t(k)+col+4*word;
                    const uint2 x0=__ldg(reinterpret_cast<const uint2*>(row_x));
                    const uint2 x1=__ldg(reinterpret_cast<const uint2*>(row_x+512));
                    const uint32_t a[2]={x0.x,x0.y}, b[2]={x1.x,x1.y};
                    #pragma unroll
                    for (int byte=0; byte<4; ++byte) {
                        const float w0=__bfloat162float(__ushort_as_bfloat16(uint16_t(effective[byte])));
                        const float w1=__bfloat162float(__ushort_as_bfloat16(uint16_t(effective[byte]>>16)));
                        const float v0=__bfloat162float(__ushort_as_bfloat16(uint16_t(a[byte/2]>>((byte&1)*16))));
                        const float v1=__bfloat162float(__ushort_as_bfloat16(uint16_t(b[byte/2]>>((byte&1)*16))));
                        sums[m][byte]=fmaf(w0,v0,sums[m][byte]);
                        sums[m][(byte+2)&3]=fmaf(w1,v1,sums[m][(byte+2)&3]);
                    }
                }
            }
        }
    } else {
        for (uint64_t col=lane; col<uint64_t(k); col+=32) {
            const float weight=__bfloat162float(weight_at<PER_CHANNEL>(w,scales,row,col,k,stride));
            #pragma unroll
            for (int m=0; m<M; ++m)
                sums[m][0]=fmaf(weight,__bfloat162float(x[uint64_t(m)*uint64_t(k)+col]),sums[m][0]);
        }
    }
    #pragma unroll
    for (int m=0; m<M; ++m) {
        float value=(sums[m][0]+sums[m][1])+(sums[m][2]+sums[m][3]);
        for (int shift=16; shift>0; shift/=2)
            value+=__shfl_down_sync(0xffffffffu,value,shift);
        if (lane==0) y[uint64_t(m)*uint64_t(n)+row]=__float2bfloat16_rn(value);
    }
}

#define FP8_SKINNY_ENTRY(M) \
extern "C" __global__ __launch_bounds__(128, 5) void fp8_skinny_bf16_m##M( \
    const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x, \
    __nv_bfloat16* y, int n, int k) { fp8_skinny<M, false>(w,scales,x,y,n,k); } \
extern "C" __global__ __launch_bounds__(128, 5) void fp8_skinny_bf16_m##M##_per_channel( \
    const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x, \
    __nv_bfloat16* y, int n, int k) { fp8_skinny<M, true>(w,scales,x,y,n,k); }
FP8_SKINNY_ENTRY(2)
FP8_SKINNY_ENTRY(3)
FP8_SKINNY_ENTRY(4)
FP8_SKINNY_ENTRY(5)
FP8_SKINNY_ENTRY(6)
FP8_SKINNY_ENTRY(7)
FP8_SKINNY_ENTRY(8)
FP8_SKINNY_ENTRY(9)
FP8_SKINNY_ENTRY(10)
FP8_SKINNY_ENTRY(11)
FP8_SKINNY_ENTRY(12)
FP8_SKINNY_ENTRY(13)
FP8_SKINNY_ENTRY(14)
FP8_SKINNY_ENTRY(15)
FP8_SKINNY_ENTRY(16)
#undef FP8_SKINNY_ENTRY
#else
template<int M, bool PER_CHANNEL>
__device__ void fp8_small_hip(
        const uint8_t* w, const __nv_bfloat16* scales,
        const __nv_bfloat16* x, __nv_bfloat16* y, int n, int k) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= n) return;
    float sums[M] = {};
    for (int col = lane; col < k; col += 32) {
        const float weight = __bfloat162float(weight_at<PER_CHANNEL>(
            w, scales, row, col, k, (uint64_t(k) + 127) / 128));
#pragma unroll
        for (int m = 0; m < M; ++m)
            sums[m] = fmaf(weight, __bfloat162float(x[(size_t)m * k + col]), sums[m]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float value = sums[m];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down(value, offset);
        if (lane == 0) y[(size_t)m * n + row] = __float2bfloat16(value);
    }
}
#define HIP_FP8_SMALL(NAME, M, CHANNEL) \
extern "C" __global__ void NAME( \
    const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x, \
    __nv_bfloat16* y, int n, int k) { \
    if (n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || \
        gridDim.x != (uint64_t(n)+3)/4 || gridDim.y != 1 || gridDim.z != 1 || \
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) __trap(); \
    fp8_small_hip<M, CHANNEL>(w,scales,x,y,n,k); \
}
HIP_FP8_SMALL(fp8_gemv_bf16, 1, false)
HIP_FP8_SMALL(fp8_gemv_bf16_per_channel, 1, true)
#define HIP_FP8_ROW(M) \
HIP_FP8_SMALL(fp8_skinny_bf16_m##M, M, false) \
HIP_FP8_SMALL(fp8_skinny_bf16_m##M##_per_channel, M, true)
HIP_FP8_ROW(2)
HIP_FP8_ROW(3)
HIP_FP8_ROW(4)
HIP_FP8_ROW(5)
HIP_FP8_ROW(6)
HIP_FP8_ROW(7)
HIP_FP8_ROW(8)
HIP_FP8_ROW(9)
HIP_FP8_ROW(10)
HIP_FP8_ROW(11)
HIP_FP8_ROW(12)
HIP_FP8_ROW(13)
HIP_FP8_ROW(14)
HIP_FP8_ROW(15)
HIP_FP8_ROW(16)
#undef HIP_FP8_ROW
#undef HIP_FP8_SMALL

extern "C" __global__ void fp8_channel_bf16_simt(
        const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x,
        __nv_bfloat16* y, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (uint64_t(n)+3)/4 || gridDim.y != uint64_t(m) || gridDim.z != 1 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) __trap();
    fp8_small_hip<1,true>(w,scales,x+(size_t)blockIdx.y*k,y+(size_t)blockIdx.y*n,n,k);
}

using ChannelBf16 = short __attribute__((ext_vector_type(16)));
using ChannelAcc = float __attribute__((ext_vector_type(8)));

extern "C" __global__ void fp8_channel_bf16_wmma(
        const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x,
        __nv_bfloat16* y, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (uint64_t(n)+63)/64 || gridDim.y != (uint64_t(m)+15)/16 || gridDim.z != 1 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) __trap();
    __shared__ __nv_bfloat16 weight[64][32], input[16][32];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int n0 = blockIdx.x * 64, m0 = blockIdx.y * 16;
    ChannelAcc acc = {};
    for (int k0 = 0; k0 < k; k0 += 32) {
        for (int cell = tid; cell < 64 * 32; cell += 128) {
            const int row = n0 + cell / 32, col = k0 + cell % 32;
            const float boost = row < n && __bfloat162float(scales[row]) < 0x1p-64f
                ? 0x1p64f : 1.0f;
            weight[cell / 32][cell % 32] = row < n && col < k
                ? __float2bfloat16(__bfloat162float(weight_at<true>(w,scales,row,col,k,1)) * boost)
                : __float2bfloat16(0.0f);
        }
        for (int cell = tid; cell < 16 * 32; cell += 128) {
            const int row = m0 + cell / 32, col = k0 + cell % 32;
            input[cell / 32][cell % 32] = row < m && col < k
                ? x[(size_t)row*k+col] : __float2bfloat16(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            ChannelBf16 a,b;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                a[j] = reinterpret_cast<const unsigned short*>(weight[wave*16+lane%16])[part*16+j];
                b[j] = reinterpret_cast<const unsigned short*>(input[lane%16])[part*16+j];
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,acc);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = m0 + lane % 16, col = n0 + wave * 16 + 2*j + lane/16;
        if (row < m && col < n) {
            const float unscale = __bfloat162float(scales[col]) < 0x1p-64f ? 0x1p-64f : 1.0f;
            y[(size_t)row*n+col] = __float2bfloat16(acc[j] * unscale);
        }
    }
}

__device__ __forceinline__ uint8_t channel_e4m3_finite(float value) {
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
#endif

// scale and normalized quotient round to BF16 before E4M3 conversion.
extern "C" __global__ void e4m3_fake_quant_bf16(
        __nv_bfloat16* x,
        int rows, int cols, const Q38SequenceMeta* meta,
        uint32_t row_base, uint32_t row_capacity) {
    if (rows <= 0 || cols <= 0 || blockDim.x != 256 || blockDim.y != 1 ||
        blockDim.z != 1 || !row_capacity ||
        uint64_t(row_base) + row_capacity > uint64_t(rows) ||
        gridDim.x != row_capacity || gridDim.y != 1 ||
        gridDim.z != 1 || !aligned16(x)) { __trap(); return; }
    const uint64_t row = uint64_t(row_base) + blockIdx.x;
    if (meta) {
        q38_meta_guard(meta);
        if (row >= uint64_t(meta->valid_rows)) return;
    } else if (row_base || row_capacity != uint32_t(rows)) { __trap(); return; }
    __shared__ float maxima[8];
    __shared__ float scale;
    const uint64_t base = row * uint64_t(cols);
    const int lane = threadIdx.x & 31;
    float amax = 0.0f;
    for (uint64_t col = threadIdx.x; col < uint64_t(cols); col += 256) {
        const float value = __bfloat162float(x[base + col]);
        if (!isfinite(value)) { __trap(); return; }
        amax = fmaxf(amax, fabsf(value));
    }
    for (int shift = 16; shift > 0; shift /= 2)
        amax = fmaxf(amax, __shfl_down_sync(FF_WARP_MASK, amax, shift));
    if (lane == 0) maxima[threadIdx.x / 32] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        amax = 0.0f;
        for (int warp = 0; warp < 8; ++warp) amax = fmaxf(amax, maxima[warp]);
        scale = __bfloat162float(__float2bfloat16_rn(amax / 448.0f));
        if (scale == 0.0f) scale = 0.0078125f;
    }
    __syncthreads();
    for (uint64_t col = threadIdx.x; col < uint64_t(cols); col += 256) {
        const float value = __bfloat162float(x[base + col]);
        const float normalized = __bfloat162float(__float2bfloat16_rn(value / scale));
        #if defined(__HIP__)
        const uint8_t code = channel_e4m3_finite(fminf(448.0f, fmaxf(-448.0f, normalized)));
        #else
        const uint8_t code = __nv_cvt_float_to_fp8(
            fminf(448.0f, fmaxf(-448.0f, normalized)), __NV_SATFINITE, __NV_E4M3);
        #endif
        x[base + col] = __float2bfloat16_rn(e4m3_decode(code) * scale);
    }
}
