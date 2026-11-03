// Row-indexed modulation from a parameter table. Row r of [rows, dim] BF16 activations reads table row
// index[r]; table row p starts at table + p * stride, so one table holds several parameter vectors per row.
//   row_modulate:   y[r][d] = bf16(bf16(x[r][d] * bf16(1 + scale[p][d])) + shift[p][d])
//   row_gated_add:  x[r][d] = bf16(x[r][d] + bf16(gate[p][d] * y[r][d]))
// Grid (rows, ceil(dim / 256)), 256 threads. x, y dense [rows, dim]; y may equal x.
#include "gpu_compat.cuh"

extern "C" __global__ void row_modulate(const __nv_bfloat16* x, const __nv_bfloat16* shift,
                                        const __nv_bfloat16* scale, const unsigned* index,
                                        __nv_bfloat16* y, int rows, int dim,
                                        unsigned long long stride, unsigned entries) {
    const int row = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || d >= dim) return;
    const unsigned p = index[row];
    if (p >= entries) __trap();
    const size_t at = (size_t)row * dim + d;
    const size_t param = (size_t)p * stride + d;
    const __nv_bfloat16 one_plus = __float2bfloat16_rn(1.0f + __bfloat162float(scale[param]));
    const __nv_bfloat16 scaled = __float2bfloat16_rn(__bfloat162float(x[at]) * __bfloat162float(one_plus));
    y[at] = __float2bfloat16_rn(__bfloat162float(scaled) + __bfloat162float(shift[param]));
}

extern "C" __global__ void row_gated_add(__nv_bfloat16* x, const __nv_bfloat16* y,
                                         const __nv_bfloat16* gate, const unsigned* index,
                                         int rows, int dim, unsigned long long stride,
                                         unsigned entries) {
    const int row = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || d >= dim) return;
    const unsigned p = index[row];
    if (p >= entries) __trap();
    const size_t at = (size_t)row * dim + d;
    const __nv_bfloat16 gated =
        __float2bfloat16_rn(__bfloat162float(gate[(size_t)p * stride + d]) * __bfloat162float(y[at]));
    x[at] = __float2bfloat16_rn(__bfloat162float(x[at]) + __bfloat162float(gated));
}
