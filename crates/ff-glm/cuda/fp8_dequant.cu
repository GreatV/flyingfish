#include <cuda_bf16.h>
#include <stdint.h>

// E4M3FN has finite exponent-15 values except mantissa 7 (NaN).
// Decode through F32 bits so SM80+ needs no native FP8 cast instruction.
__device__ __forceinline__ float e4m3_to_f32(uint8_t bits) {
    const uint32_t sign = uint32_t(bits & 128) << 24;
    const uint32_t exponent = (bits >> 3) & 15;
    const uint32_t mantissa = bits & 7;
    if (exponent == 0) {
        const float magnitude = float(mantissa) * 0x1p-9f;
        return __uint_as_float(__float_as_uint(magnitude) | sign);
    }
    if ((bits & 127) == 127) return __uint_as_float(sign | 0x7fc00000u);
    return __uint_as_float(sign | ((exponent + 120) << 23) | (mantissa << 20));
}

template<typename Output>
__device__ __forceinline__ void dequant(const uint8_t* weight, const float* scale,
                                       Output* output, uint32_t rows, uint32_t cols) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= uint64_t(rows) * cols) return;
    const uint32_t scale_cols = (cols + 127) / 128;
    const uint32_t block = ((i / cols) / 128) * scale_cols + (i % cols) / 128;
    const float value = __fmul_rn(e4m3_to_f32(weight[i]), scale[block]);
    output[i] = value;
}

extern "C" __global__ void glm_fp8_dequant_f32(
    const uint8_t* weight, const float* scale, float* output, uint32_t rows, uint32_t cols) {
    dequant(weight, scale, output, rows, cols);
}

extern "C" __global__ void glm_fp8_dequant_bf16(
    const uint8_t* weight, const float* scale, __nv_bfloat16* output,
    uint32_t rows, uint32_t cols) {
    dequant(weight, scale, output, rows, cols);
}
