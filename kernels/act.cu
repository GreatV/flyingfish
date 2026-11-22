#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 && !defined(__HIP__)
// sm75 16-bit load is cvt.f32.f16. sm75 16-bit store is cvt.rn.f16.f32.
#include "gpu_compat.cuh"
#include "q38_contract.cuh"
#include "ff_sm75_f16.cuh"
#define __bfloat162float(v) ff_f32_from_f16((v))
#define __bfloat1622float2(v) ff_f32x2((v))
#define __float2bfloat16(v) ff_f16_from_f32<__nv_bfloat16>((v))
#define __float2bfloat16_rn(v) ff_f16_from_f32<__nv_bfloat16>((v))
#define __floats2bfloat162_rn(a, b) ff_f16x2((a), (b))
#define __hadd(a, b) ff_hadd((a), (b))
#define __hadd2(a, b) ff_hadd2((a), (b))
#define __hsub(a, b) ff_hsub((a), (b))
#define __hmul(a, b) ff_hmul((a), (b))

// mode: 0 = SiLU, product rounded twice; 1 = SiLU, rounded once; 2 = GELU (tanh form), product rounded twice;
// 3 = SiLU of the upper half times the lower half, product rounded twice.
__device__ __forceinline__ float gelu_tanh(float x) {
    const float cube = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" __global__ void silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* y,
                                   int rows, int dim, int mode) {
    if ((unsigned)mode > 3u || rows <= 0 || dim <= 0 || (dim & 7) != 0 ||
        (long long)rows * dim > 2147483647LL || !gu || !y ||
        ((reinterpret_cast<uintptr_t>(gu) | reinterpret_cast<uintptr_t>(y)) &
         15u) != 0u) {
        __trap();
        return;
    }
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_dim = dim >> 3;
    if (vec >= rows * vec_dim) return;
    const int row = vec / vec_dim;
    const int vd = vec % vec_dim;
    const uint4 gv =
        reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vd];
    const uint4 uv =
        reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vec_dim + vd];
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
    uint4 yv;
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(&yv);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float g = __bfloat162float(gp[j]);
        const float u = __bfloat162float(up[j]);
        if (mode == 3) {
            yp[j] = __hmul(__float2bfloat16(u / (1.0f + expf(-u))), gp[j]);
        } else if (mode == 2) {
            yp[j] = __hmul(__float2bfloat16(gelu_tanh(g)), up[j]);
        } else if (mode == 1) {
            yp[j] = __float2bfloat16(g / (1.0f + expf(-g)) * u);
        } else {
            const __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
            yp[j] = __hmul(s, up[j]);
        }
    }
    reinterpret_cast<uint4*>(y)[vec] = yv;
}

// Gate and up are dense [rows,dim] regions; y is disjoint from both inputs.
extern "C" __global__ void silu_mul_split(const __nv_bfloat16* gate,
                                         const __nv_bfloat16* up,
                                         __nv_bfloat16* y, int count,
                                         int mode) {
    if ((unsigned)mode > 2u || count < 0 || (count & 7) != 0 ||
        !gate || !up || !y ||
        ((reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) |
          reinterpret_cast<uintptr_t>(y)) &
         15u) != 0u) {
        __trap();
        return;
    }
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec >= (count >> 3)) return;
    const uint4 gv = reinterpret_cast<const uint4*>(gate)[vec];
    const uint4 uv = reinterpret_cast<const uint4*>(up)[vec];
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* up_p = reinterpret_cast<const __nv_bfloat16*>(&uv);
    uint4 yv;
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(&yv);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float g = __bfloat162float(gp[j]);
        const float u = __bfloat162float(up_p[j]);
        if (mode == 2) {
            yp[j] = __hmul(__float2bfloat16_rn(gelu_tanh(g)), up_p[j]);
        } else if (mode == 1) {
            yp[j] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
        } else {
            const __nv_bfloat16 s = __float2bfloat16_rn(g / (1.0f + expf(-g)));
            yp[j] = __hmul(s, up_p[j]);
        }
    }
    reinterpret_cast<uint4*>(y)[vec] = yv;
}

extern "C" __global__ void silu_mul_split_f32(const float* gate,
                                              const float* up,
                                              float* y, int count,
                                              int mode) {
    if ((unsigned)mode > 2u || count < 0 || !gate || !up || !y) {
        __trap();
        return;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        const float u = up[i];
        if (mode == 2) {
            y[i] = __fmul_rn(gelu_tanh(g), u);
        } else {
            y[i] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u);
        }
    }
}

// Clamped SwiGLU over dense [rows, width] gate and up regions (ClampedTwice):
// gate clamps above at the limit, up on both sides; the SiLU rounds to BF16,
// then the product rounds once. Grid (vectors, row_capacity), 256 threads,
// eight elements per thread; rows beyond the sequence meta are skipped.
extern "C" __global__ void silu_mul_split_clamped(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* y,
    const Q38SequenceMeta* meta, unsigned row_base, unsigned row_capacity,
    unsigned width, float limit) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.z != 1 || !width || width % 8 || !row_capacity ||
        gridDim.y != row_capacity || !gate || !up || !y ||
        !(limit > 0.0f) || !isfinite(limit) ||
        (reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) |
         reinterpret_cast<uintptr_t>(y)) % 16) {__trap(); return;}
    q38_meta_guard(meta);
    const unsigned vectors = width / 8;
    if (gridDim.x != vectors / 256 + (vectors % 256 != 0)) {__trap(); return;}
    const uint64_t row = uint64_t(row_base) + blockIdx.y;
    if (row >= uint64_t(meta->valid_rows)) return;
    const unsigned vector = blockIdx.x * 256 + threadIdx.x;
    if (vector >= vectors) return;
    const __nv_bfloat16* grow = gate + row * width;
    const __nv_bfloat16* urow = up + row * width;
    const uint4 a = reinterpret_cast<const uint4*>(grow)[vector];
    const uint4 b = reinterpret_cast<const uint4*>(urow)[vector];
    const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&b);
    uint4 result;
    __nv_bfloat16* rv = reinterpret_cast<__nv_bfloat16*>(&result);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float g = fminf(__bfloat162float(av[i]), limit);
        const float u = fminf(fmaxf(__bfloat162float(bv[i]), -limit), limit);
        const float s =
            __bfloat162float(__float2bfloat16_rn(silu(g)));
        rv[i] = __float2bfloat16_rn(s * u);
    }
    reinterpret_cast<uint4*>(y + row * width)[vector] = result;
}

// x = bf16(bf16(tanh(bf16(x / cap))) * cap), in place.
extern "C" __global__ void softcap(__nv_bfloat16* x, long long count, float cap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float z = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i]) / cap));
    const float t = __bfloat162float(__float2bfloat16_rn(tanhf(z)));
    x[i] = __float2bfloat16_rn(t * cap);
}

// Canvas head form: y = cap * tanh(x / cap) in FP32 from BF16 logits.
extern "C" __global__ void softcap_cast_f32(const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y, long long count, float cap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = __bfloat162float(x[i]);
    y[i] = cap * tanhf(v / cap);
}

// y = bf16(bf16(a + b) * scale); two ordered BF16 roundings, elementwise.
extern "C" __global__ void residual_scale(const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, float scale, __nv_bfloat16* __restrict__ y,
    long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const __nv_bfloat16 sum =
        __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    y[i] = __float2bfloat16_rn(__bfloat162float(sum) * scale);
}


// Gated FF in F32: pairs [rows, 2*dim] split half/half,
// y = first * silu(second).
extern "C" __global__ void silu_mul_f32(const float* __restrict__ gu,
    float* __restrict__ y, long long rows, int dim) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * (long long)dim) return;
    const long long row = i / dim;
    const int d = (int)(i - row * dim);
    const float value = gu[row * 2 * dim + d];
    const float gate = gu[row * 2 * dim + dim + d];
    y[i] = value * (gate / (1.0f + expf(-gate)));
}

// In-place silu in F32.
extern "C" __global__ void silu_f32(float* __restrict__ x, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

extern "C" __global__ void silu_bf16(__nv_bfloat16* __restrict__ x, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16_rn(v / (1.0f + expf(-v)));
}

// Mode-3 product without the store check: fp16(silu(upper)) * lower.
__device__ __forceinline__ float ff_f16_roundtrip(float v) {
    unsigned short u;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(u) : "f"(v));
    float f;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(u));
    return f;
}

__device__ __forceinline__ float ff_mode3_product(float lower, float upper) {
    const float silu = upper / (1.0f + expf(-upper));
    return ff_f16_roundtrip(silu) * lower;
}

// One atomicMax per block of the uint bits of fabsf of the mode-3 product.
// A NaN's bits win that max. The scaled store then faults.
// Does not store and does not set ff_fp16_store_fault.
extern "C" __global__ void silu_mul_maxabs(const __nv_bfloat16* gu, float* max_abs,
                                          int rows, int dim) {
    if (rows <= 0 || dim <= 0 || (dim & 7) != 0 ||
        (long long)rows * dim > 2147483647LL || !gu || !max_abs ||
        (reinterpret_cast<uintptr_t>(gu) & 15u) != 0u ||
        (blockDim.x & 31u) != 0u || blockDim.x > 1024u) {
        __trap();
        return;
    }
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_dim = dim >> 3;
    const int nvec = rows * vec_dim;
    unsigned bits = 0;
    if (vec < nvec) {
        const int row = vec / vec_dim;
        const int vd = vec % vec_dim;
        const uint4 gv = reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vd];
        const uint4 uv = reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vec_dim + vd];
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned b = __float_as_uint(fabsf(
                ff_mode3_product(__bfloat162float(gp[j]), __bfloat162float(up[j]))));
            if (b > bits) bits = b;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned other = __shfl_xor_sync(FF_WARP_MASK, bits, o);
        if (other > bits) bits = other;
    }
    __shared__ unsigned warp_bits[32];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) warp_bits[warp] = bits;
    __syncthreads();
    if (warp == 0) {
        const unsigned nwarps = blockDim.x >> 5;
        unsigned block = lane < nwarps ? warp_bits[lane] : 0u;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const unsigned other = __shfl_xor_sync(FF_WARP_MASK, block, o);
            if (other > block) block = other;
        }
        if (lane == 0) atomicMax(reinterpret_cast<unsigned int*>(max_abs), block);
    }
}

// alpha = 2^k. k is the smallest integer with max_abs / 2^k <= 32768, by halving, capped at 127.
extern "C" __global__ void silu_pow2_alpha(const float* max_abs, float* alpha) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || !max_abs || !alpha) return;
    float m = *max_abs;
    int k = 0;
    while (m > 32768.0f && k < 127) {
        m *= 0.5f;
        ++k;
    }
    *alpha = __uint_as_float(static_cast<unsigned>(127 + k) << 23);
}

// Stores product / 2^k. The silu round and the scaled store use the FP16 check.
extern "C" __global__ void silu_mul_scaled(const __nv_bfloat16* gu, __nv_bfloat16* y,
                                          const float* alpha, int rows, int dim) {
    if (rows <= 0 || dim <= 0 || (dim & 7) != 0 ||
        (long long)rows * dim > 2147483647LL || !gu || !y || !alpha ||
        ((reinterpret_cast<uintptr_t>(gu) | reinterpret_cast<uintptr_t>(y)) & 15u) != 0u) {
        __trap();
        return;
    }
    const float scale = *alpha;
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_dim = dim >> 3;
    if (vec >= rows * vec_dim) return;
    const int row = vec / vec_dim;
    const int vd = vec % vec_dim;
    const uint4 gv = reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vd];
    const uint4 uv = reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vec_dim + vd];
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
    uint4 yv;
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(&yv);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float u = __bfloat162float(up[j]);
        const float g = __bfloat162float(gp[j]);
        const __nv_bfloat16 silu = __float2bfloat16(u / (1.0f + expf(-u)));
        const float prod = __bfloat162float(silu) * g;
        yp[j] = __float2bfloat16(prod / scale);
    }
    reinterpret_cast<uint4*>(y)[vec] = yv;
}
#else /* ff-sm80 */
#include "gpu_compat.cuh"
#include "q38_contract.cuh"

// mode: 0 = SiLU, product rounded twice; 1 = SiLU, rounded once; 2 = GELU (tanh form), product rounded twice;
// 3 = SiLU of the upper half times the lower half, product rounded twice.
__device__ __forceinline__ float gelu_tanh(float x) {
    const float cube = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" __global__ void silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* y,
                                   int rows, int dim, int mode) {
    if ((unsigned)mode > 3u || rows <= 0 || dim <= 0 || (dim & 7) != 0 ||
        (long long)rows * dim > 2147483647LL || !gu || !y ||
        ((reinterpret_cast<uintptr_t>(gu) | reinterpret_cast<uintptr_t>(y)) &
         15u) != 0u) {
        __trap();
        return;
    }
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_dim = dim >> 3;
    if (vec >= rows * vec_dim) return;
    const int row = vec / vec_dim;
    const int vd = vec % vec_dim;
    const uint4 gv =
        reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vd];
    const uint4 uv =
        reinterpret_cast<const uint4*>(gu)[row * 2 * vec_dim + vec_dim + vd];
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
    uint4 yv;
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(&yv);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float g = __bfloat162float(gp[j]);
        const float u = __bfloat162float(up[j]);
        if (mode == 3) {
            yp[j] = __hmul(__float2bfloat16(u / (1.0f + expf(-u))), gp[j]);
        } else if (mode == 2) {
            yp[j] = __hmul(__float2bfloat16(gelu_tanh(g)), up[j]);
        } else if (mode == 1) {
            yp[j] = __float2bfloat16(g / (1.0f + expf(-g)) * u);
        } else {
            const __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
            yp[j] = __hmul(s, up[j]);
        }
    }
    reinterpret_cast<uint4*>(y)[vec] = yv;
}

// Gate and up are dense [rows,dim] regions; y is disjoint from both inputs.
extern "C" __global__ void silu_mul_split(const __nv_bfloat16* gate,
                                         const __nv_bfloat16* up,
                                         __nv_bfloat16* y, int count,
                                         int mode) {
    if ((unsigned)mode > 2u || count < 0 || (count & 7) != 0 ||
        !gate || !up || !y ||
        ((reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) |
          reinterpret_cast<uintptr_t>(y)) &
         15u) != 0u) {
        __trap();
        return;
    }
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec >= (count >> 3)) return;
    const uint4 gv = reinterpret_cast<const uint4*>(gate)[vec];
    const uint4 uv = reinterpret_cast<const uint4*>(up)[vec];
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* up_p = reinterpret_cast<const __nv_bfloat16*>(&uv);
    uint4 yv;
    __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(&yv);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float g = __bfloat162float(gp[j]);
        const float u = __bfloat162float(up_p[j]);
        if (mode == 2) {
            yp[j] = __hmul(__float2bfloat16_rn(gelu_tanh(g)), up_p[j]);
        } else if (mode == 1) {
            yp[j] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
        } else {
            const __nv_bfloat16 s = __float2bfloat16_rn(g / (1.0f + expf(-g)));
            yp[j] = __hmul(s, up_p[j]);
        }
    }
    reinterpret_cast<uint4*>(y)[vec] = yv;
}

extern "C" __global__ void silu_mul_split_f32(const float* gate,
                                              const float* up,
                                              float* y, int count,
                                              int mode) {
    if ((unsigned)mode > 2u || count < 0 || !gate || !up || !y) {
        __trap();
        return;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        const float u = up[i];
        if (mode == 2) {
            y[i] = __fmul_rn(gelu_tanh(g), u);
        } else {
            y[i] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u);
        }
    }
}

// Clamped SwiGLU over dense [rows, width] gate and up regions (ClampedTwice):
// gate clamps above at the limit, up on both sides; the SiLU rounds to BF16,
// then the product rounds once. Grid (vectors, row_capacity), 256 threads,
// eight elements per thread; rows beyond the sequence meta are skipped.
extern "C" __global__ void silu_mul_split_clamped(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* y,
    const Q38SequenceMeta* meta, unsigned row_base, unsigned row_capacity,
    unsigned width, float limit) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.z != 1 || !width || width % 8 || !row_capacity ||
        gridDim.y != row_capacity || !gate || !up || !y ||
        !(limit > 0.0f) || !isfinite(limit) ||
        (reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) |
         reinterpret_cast<uintptr_t>(y)) % 16) {__trap(); return;}
    q38_meta_guard(meta);
    const unsigned vectors = width / 8;
    if (gridDim.x != vectors / 256 + (vectors % 256 != 0)) {__trap(); return;}
    const uint64_t row = uint64_t(row_base) + blockIdx.y;
    if (row >= uint64_t(meta->valid_rows)) return;
    const unsigned vector = blockIdx.x * 256 + threadIdx.x;
    if (vector >= vectors) return;
    const __nv_bfloat16* grow = gate + row * width;
    const __nv_bfloat16* urow = up + row * width;
    const uint4 a = reinterpret_cast<const uint4*>(grow)[vector];
    const uint4 b = reinterpret_cast<const uint4*>(urow)[vector];
    const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&b);
    uint4 result;
    __nv_bfloat16* rv = reinterpret_cast<__nv_bfloat16*>(&result);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float g = fminf(__bfloat162float(av[i]), limit);
        const float u = fminf(fmaxf(__bfloat162float(bv[i]), -limit), limit);
        const float s =
            __bfloat162float(__float2bfloat16_rn(silu(g)));
        rv[i] = __float2bfloat16_rn(s * u);
    }
    reinterpret_cast<uint4*>(y + row * width)[vector] = result;
}

// x = bf16(bf16(tanh(bf16(x / cap))) * cap), in place.
extern "C" __global__ void softcap(__nv_bfloat16* x, long long count, float cap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float z = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i]) / cap));
    const float t = __bfloat162float(__float2bfloat16_rn(tanhf(z)));
    x[i] = __float2bfloat16_rn(t * cap);
}

// Canvas head form: y = cap * tanh(x / cap) in FP32 from BF16 logits.
extern "C" __global__ void softcap_cast_f32(const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y, long long count, float cap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = __bfloat162float(x[i]);
    y[i] = cap * tanhf(v / cap);
}

// y = bf16(bf16(a + b) * scale); two ordered BF16 roundings, elementwise.
extern "C" __global__ void residual_scale(const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, float scale, __nv_bfloat16* __restrict__ y,
    long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const __nv_bfloat16 sum =
        __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    y[i] = __float2bfloat16_rn(__bfloat162float(sum) * scale);
}


// Gated FF in F32: pairs [rows, 2*dim] split half/half,
// y = first * silu(second).
extern "C" __global__ void silu_mul_f32(const float* __restrict__ gu,
    float* __restrict__ y, long long rows, int dim) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * (long long)dim) return;
    const long long row = i / dim;
    const int d = (int)(i - row * dim);
    const float value = gu[row * 2 * dim + d];
    const float gate = gu[row * 2 * dim + dim + d];
    y[i] = value * (gate / (1.0f + expf(-gate)));
}

// In-place silu in F32.
extern "C" __global__ void silu_f32(float* __restrict__ x, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

extern "C" __global__ void silu_bf16(__nv_bfloat16* __restrict__ x, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16_rn(v / (1.0f + expf(-v)));
}
#endif /* ff-sm80 */
