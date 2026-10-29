#include "gpu_compat.cuh"
#include "q38_contract.cuh"

extern "C" __global__ void silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* y,
                                   int rows, int dim, int rounding_once) {
    if ((unsigned)rounding_once > 1u) __trap();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    int row = i / dim;
    int d = i % dim;
    float g = __bfloat162float(gu[row * 2 * dim + d]);
    const float u = __bfloat162float(gu[row * 2 * dim + dim + d]);
    if (rounding_once) {
      y[i] = __float2bfloat16(g / (1.0f + expf(-g)) * u);
    } else {
      const __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
      y[i] = __hmul(s, gu[row * 2 * dim + dim + d]);
    }
}

// Gate and up are dense [rows,dim] regions; y is disjoint from both inputs.
extern "C" __global__ void silu_mul_split(const __nv_bfloat16* gate,
                                         const __nv_bfloat16* up,
                                         __nv_bfloat16* y, int count,
                                         int rounding_once) {
    if ((unsigned)rounding_once > 1u) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    if (rounding_once) {
      y[i] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
    } else {
      const __nv_bfloat16 s = __float2bfloat16_rn(g / (1.0f + expf(-g)));
      y[i] = __hmul(s, up[i]);
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
