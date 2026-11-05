// Row-indexed modulation from a parameter table. Row r of [rows, dim] BF16 activations reads table row
// index[r]; table row p starts at table + p * stride, so one table holds several parameter vectors per row.
//   row_modulate:   y[r][d] = bf16(bf16(x[r][d] * bf16(1 + scale[p][d])) + shift[p][d])
//   row_gated_add:  x[r][d] = bf16(x[r][d] + bf16(gate[p][d] * y[r][d]))
// Grid (rows, ceil(dim / 256)), 256 threads. x, y dense [rows, dim]; y may equal x.
#include "gpu_compat.cuh"

namespace {

__device__ __forceinline__ float mod_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float mod_to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float mod_to_f32(float v) { return v; }

template <typename T>
__device__ __forceinline__ T mod_from_f32(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 mod_from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ __half mod_from_f32<__half>(float v) {
    return __float2half_rn(v);
}
template <>
__device__ __forceinline__ float mod_from_f32<float>(float v) {
    return v;
}

// NULLABLE: `index` may be null, row r then uses entry r / (rows / entries). The BF16 symbols keep the dense-index form.
template <typename T, bool NULLABLE, typename P = T>
__device__ __forceinline__ void row_modulate_impl(
    const T* x, const P* shift, const P* scale, const unsigned* index,
    T* y, int rows, int dim, unsigned long long stride, unsigned entries) {
    const int row = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || d >= dim) return;
    unsigned p;
    if constexpr (NULLABLE) {
        if (entries == 0 || (!index && (rows % (int)entries) != 0)) __trap();
        p = index ? index[row] : (unsigned)(row / (rows / (int)entries));
    } else {
        p = index[row];
    }
    if (p >= entries) __trap();
    const size_t at = (size_t)row * dim + d;
    const size_t param = (size_t)p * stride + d;
    const T one_plus = mod_from_f32<T>(1.0f + mod_to_f32(scale[param]));
    const T scaled = mod_from_f32<T>(mod_to_f32(x[at]) * mod_to_f32(one_plus));
    y[at] = mod_from_f32<T>(mod_to_f32(scaled) + mod_to_f32(mod_from_f32<T>(mod_to_f32(shift[param]))));
}

template <typename T, bool NULLABLE, typename P = T>
__device__ __forceinline__ void row_gated_add_impl(
    T* x, const T* y, const P* gate, const unsigned* index,
    int rows, int dim, unsigned long long stride, unsigned entries) {
    const int row = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || d >= dim) return;
    unsigned p;
    if constexpr (NULLABLE) {
        if (entries == 0 || (!index && (rows % (int)entries) != 0)) __trap();
        p = index ? index[row] : (unsigned)(row / (rows / (int)entries));
    } else {
        p = index[row];
    }
    if (p >= entries) __trap();
    const size_t at = (size_t)row * dim + d;
    const T gated = mod_from_f32<T>(mod_to_f32(mod_from_f32<T>(mod_to_f32(gate[(size_t)p * stride + d]))) * mod_to_f32(y[at]));
    x[at] = mod_from_f32<T>(mod_to_f32(x[at]) + mod_to_f32(gated));
}

}  // namespace

extern "C" __global__ void row_modulate(const __nv_bfloat16* x, const __nv_bfloat16* shift,
                                        const __nv_bfloat16* scale, const unsigned* index,
                                        __nv_bfloat16* y, int rows, int dim,
                                        unsigned long long stride, unsigned entries) {
    row_modulate_impl<__nv_bfloat16, false>(x, shift, scale, index, y, rows, dim, stride, entries);
}

extern "C" __global__ void row_gated_add(__nv_bfloat16* x, const __nv_bfloat16* y,
                                         const __nv_bfloat16* gate, const unsigned* index,
                                         int rows, int dim, unsigned long long stride,
                                         unsigned entries) {
    row_gated_add_impl<__nv_bfloat16, false>(x, y, gate, index, rows, dim, stride, entries);
}

extern "C" __global__ void row_modulate_f16(const __half* x, const __half* shift,
                                            const __half* scale, const unsigned* index,
                                            __half* y, int rows, int dim,
                                            unsigned long long stride, unsigned entries) {
    row_modulate_impl<__half, true>(x, shift, scale, index, y, rows, dim, stride, entries);
}

extern "C" __global__ void row_gated_add_f16(__half* x, const __half* y,
                                             const __half* gate, const unsigned* index,
                                             int rows, int dim, unsigned long long stride,
                                             unsigned entries) {
    row_gated_add_impl<__half, true>(x, y, gate, index, rows, dim, stride, entries);
}

extern "C" __global__ void row_modulate_f32(const float* x, const float* shift,
                                            const float* scale, const unsigned* index,
                                            float* y, int rows, int dim,
                                            unsigned long long stride, unsigned entries) {
    row_modulate_impl<float, true>(x, shift, scale, index, y, rows, dim, stride, entries);
}

extern "C" __global__ void add_row_bf16_f32(
    const __nv_bfloat16* x, const float* row, __nv_bfloat16* y,
    unsigned long long count, unsigned width) {
    if (!x || !row || !y || !width || count % width) { __trap(); return; }
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
        y[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(x[i]), row[i % width]));
    }
}

extern "C" __global__ void row_modulate_f16_f32(const __half* x, const float* shift,
    const float* scale, const unsigned* index, __half* y, int rows, int dim,
    unsigned long long stride, unsigned entries) {
    row_modulate_impl<__half, true, float>(x, shift, scale, index, y, rows, dim, stride, entries);
}
extern "C" __global__ void row_gated_add_f16_f32(__half* x, const __half* y,
    const float* gate, const unsigned* index, int rows, int dim,
    unsigned long long stride, unsigned entries) {
    row_gated_add_impl<__half, true, float>(x, y, gate, index, rows, dim, stride, entries);
}
extern "C" __global__ void add_row_f16_f32(const __half* x, const float* row, float* y,
    unsigned long long count, unsigned width) {
    if (!x || !row || !y || !width || count % width) { __trap(); return; }
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = __fadd_rn(__half2float(x[i]), row[i % width]);
}
