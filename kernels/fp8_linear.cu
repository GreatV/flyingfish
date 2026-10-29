// Row-major E4M3 codes; BF16 block-128 or per-output-channel scales.
// CUDA M1..16 rounds weights in registers; larger M materializes BF16 weights.
// HIP projections decode BF16-rounded weights in SIMT registers or WMMA tiles.
// Format: __nv_fp8_e4m3 (CUDA Math API).
#include "gpu_compat.cuh"
#if !defined(__HIP__)
#include <cuda_fp8.h>
#endif
#include <stdint.h>
#include "q38_contract.cuh"

__device__ __forceinline__ float fp8_block_scale(float value) { return value; }
__device__ __forceinline__ float fp8_block_scale(uint8_t value) { return e8m0_decode(value); }

template <typename Scale, int BLOCK>
__device__ __forceinline__ void fp8_block_decode(
        const uint8_t* w, const Scale* sw, const uint8_t* x, const Scale* sx,
        __nv_bfloat16* y, int m, int n, int k) {
    const uint64_t tiles = (uint64_t(n) + 7) / 8;
    const uint64_t row = blockIdx.x / tiles;
    const uint64_t col = (blockIdx.x % tiles) * 8 + threadIdx.x / 32;
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

template <int COUNT>
__device__ __forceinline__ uint32_t fp8_block_codes(
        const uint8_t* data, uint64_t row, uint64_t col, int rows, int k) {
    if (row >= uint64_t(rows)) return 0;
    const uint8_t* ptr = data + row * k + col;
    if (col + COUNT <= uint64_t(k) && (reinterpret_cast<uintptr_t>(ptr) & (COUNT - 1)) == 0) {
        if constexpr (COUNT == 4) return *reinterpret_cast<const uint32_t*>(ptr);
        else return *reinterpret_cast<const uint16_t*>(ptr);
    }
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < COUNT; ++i)
        if (col + i < uint64_t(k)) packed |= uint32_t(data[row * k + col + i]) << (8 * i);
    return packed;
}

__device__ __forceinline__ uint32_t fp8_block_bf16(uint32_t packed) {
    const float2 value = e4m3_decode_pair(uint16_t(packed));
    return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(value.x))) |
           (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(value.y))) << 16);
}

template <typename Scale, int BLOCK>
__device__ __forceinline__ void fp8_block_mma(
        const uint8_t* w, const Scale* sw, const uint8_t* x, const Scale* sx,
        __nv_bfloat16* y, int m, int n, int k) {
    const uint64_t tiles = (uint64_t(n) + 31) / 32;
    const uint64_t row_base = (blockIdx.x / tiles) * 16;
    const uint64_t col_base = (blockIdx.x % tiles) * 32 + (threadIdx.x / 32) * 8;
    const int lane = threadIdx.x & 31, group = lane >> 2, thread = lane & 3;
    const uint64_t groups = (uint64_t(k) + BLOCK - 1) / BLOCK;
    float result[4] = {};
    for (uint64_t scale_col = 0; scale_col < groups; ++scale_col) {
        float c[4] = {};
        // sm_89 CUDA 13.0 native FP8 MMA exceeds FP64 bounds in 14/1024 outputs at [M,N,K]=[16,64,32].
#if __CUDA_ARCH__ >= 900
        #pragma unroll
        for (int offset = 0; offset < BLOCK; offset += 32) {
            const uint64_t inner = scale_col * BLOCK + offset + 4 * thread;
            const uint32_t a0 = fp8_block_codes<4>(x, row_base + group, inner, m, k);
            const uint32_t a1 = fp8_block_codes<4>(x, row_base + group + 8, inner, m, k);
            const uint32_t a2 = fp8_block_codes<4>(x, row_base + group, inner + 16, m, k);
            const uint32_t a3 = fp8_block_codes<4>(x, row_base + group + 8, inner + 16, m, k);
            const uint32_t b0 = fp8_block_codes<4>(w, col_base + group, inner, n, k);
            const uint32_t b1 = fp8_block_codes<4>(w, col_base + group, inner + 16, n, k);
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
#else
        #pragma unroll
        for (int offset = 0; offset < BLOCK; offset += 16) {
            const uint64_t inner = scale_col * BLOCK + offset + 2 * thread;
            const uint32_t a0 = fp8_block_bf16(fp8_block_codes<2>(x, row_base + group, inner, m, k));
            const uint32_t a1 = fp8_block_bf16(fp8_block_codes<2>(x, row_base + group + 8, inner, m, k));
            const uint32_t a2 = fp8_block_bf16(fp8_block_codes<2>(x, row_base + group, inner + 8, m, k));
            const uint32_t a3 = fp8_block_bf16(fp8_block_codes<2>(x, row_base + group + 8, inner + 8, m, k));
            const uint32_t b0 = fp8_block_bf16(fp8_block_codes<2>(w, col_base + group, inner, n, k));
            const uint32_t b1 = fp8_block_bf16(fp8_block_codes<2>(w, col_base + group, inner + 8, n, k));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
#endif
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint64_t row = row_base + group + (i / 2) * 8;
            const uint64_t col = col_base + 2 * thread + (i & 1);
            if (row < uint64_t(m) && col < uint64_t(n)) {
                const float scale = fp8_block_scale(sw[(col / BLOCK) * groups + scale_col]) *
                                    fp8_block_scale(sx[row * groups + scale_col]);
                result[i] = fmaf(c[i], scale, result[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t row = row_base + group + (i / 2) * 8;
        const uint64_t col = col_base + 2 * thread + (i & 1);
        if (row < uint64_t(m) && col < uint64_t(n))
            y[row * n + col] = __float2bfloat16_rn(result[i]);
    }
}

#define FP8_BLOCK_ENTRY(NAME, SCALE, BLOCK, TILED) \
extern "C" __global__ void NAME( \
        const uint8_t* w, const SCALE* sw, const uint8_t* x, const SCALE* sx, \
        __nv_bfloat16* y, int m, int n, int k) { \
    if (m <= 0 || n <= 0 || k <= 0 || blockDim.x != (TILED ? 128 : 256) || \
        blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) { \
        __trap(); return; \
    } \
    if constexpr (TILED) fp8_block_mma<SCALE, BLOCK>(w, sw, x, sx, y, m, n, k); \
    else fp8_block_decode<SCALE, BLOCK>(w, sw, x, sx, y, m, n, k); \
}
FP8_BLOCK_ENTRY(fp8_block_f32_128_decode, float, 128, false)
FP8_BLOCK_ENTRY(fp8_block_f32_128_mma, float, 128, true)
FP8_BLOCK_ENTRY(fp8_block_e8m0_32_decode, uint8_t, 32, false)
FP8_BLOCK_ENTRY(fp8_block_e8m0_32_mma, uint8_t, 32, true)
#undef FP8_BLOCK_ENTRY

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
template<bool PER_CHANNEL>
__device__ void fp8_simt_hip(
        const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x,
        __nv_bfloat16* y, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (uint64_t(n)+3)/4 || gridDim.y != uint64_t(m) || gridDim.z != 1 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) __trap();
    fp8_small_hip<1,PER_CHANNEL>(w,scales,x+(size_t)blockIdx.y*k,y+(size_t)blockIdx.y*n,n,k);
}

using ChannelBf16 = short __attribute__((ext_vector_type(16)));
using ChannelAcc = float __attribute__((ext_vector_type(8)));

template<bool PER_CHANNEL>
__device__ void fp8_wmma_hip(
        const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x,
        __nv_bfloat16* y, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (uint64_t(n)+63)/64 || gridDim.y != (uint64_t(m)+15)/16 || gridDim.z != 1 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) __trap();
    __shared__ __nv_bfloat16 weight[64][32], input[16][32];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int n0 = blockIdx.x * 64, m0 = blockIdx.y * 16;
    ChannelAcc acc = {}, partial = {};
    for (int k0 = 0; k0 < k; k0 += 32) {
        for (int cell = tid; cell < 64 * 32; cell += 128) {
            const int row = n0 + cell / 32, col = k0 + cell % 32;
            const size_t scale = PER_CHANNEL ? row : (size_t)(row/128)*((k+127)/128)+col/128;
            weight[cell / 32][cell % 32] = row < n && col < k
                ? fp8_channel_tile_weight(w[(size_t)row*k+col], scales[scale])
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
            if constexpr (PER_CHANNEL)
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,acc);
            else
                partial = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,partial);
        }
        __syncthreads();
        if constexpr (!PER_CHANNEL) {
            if ((k0+32)%128 == 0 || k0+32 >= k) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = n0 + wave*16 + 2*j + lane/16;
                    const float unscale = col < n
                        ? fp8_channel_tile_unscale(scales[(size_t)(col/128)*((k+127)/128)+k0/128])
                        : 1.0f;
                    acc[j] += partial[j] * unscale;
                    partial[j] = 0.0f;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = m0 + lane % 16, col = n0 + wave * 16 + 2*j + lane/16;
        if (row < m && col < n) {
            const float unscale = PER_CHANNEL ? fp8_channel_tile_unscale(scales[col]) : 1.0f;
            y[(size_t)row*n+col] = __float2bfloat16(acc[j] * unscale);
        }
    }
}

#define HIP_FP8_PROJECTION(NAME, IMPL, PER_CHANNEL) \
extern "C" __global__ void NAME( \
        const uint8_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* x, \
        __nv_bfloat16* y, int m, int n, int k) { \
    IMPL<PER_CHANNEL>(w,scales,x,y,m,n,k); \
}
HIP_FP8_PROJECTION(fp8_channel_bf16_simt, fp8_simt_hip, true)
HIP_FP8_PROJECTION(fp8_channel_bf16_wmma, fp8_wmma_hip, true)
HIP_FP8_PROJECTION(fp8_block_bf16_simt, fp8_simt_hip, false)
HIP_FP8_PROJECTION(fp8_block_bf16_wmma, fp8_wmma_hip, false)
#undef HIP_FP8_PROJECTION

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
