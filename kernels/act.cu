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
