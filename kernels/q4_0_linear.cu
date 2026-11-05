// GGUF Q4_0: 18 bytes per 32 weights. F16 scale d at byte 0; low nibble = j, high nibble = j+16.
// Blocks run along the official row-major [out, in] flat order; no transpose.
#include "gpu_compat.cuh"
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float q4_0_scale(const uint8_t* block) {
    return __half2float(__ushort_as_half(uint16_t(block[0]) | (uint16_t(block[1]) << 8)));
}

__device__ __forceinline__ void q4_0_block_bf16(const uint8_t* block, __nv_bfloat16* out) {
    const float d = q4_0_scale(block);
    const uint8_t* qs = block + 2;
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        out[j] = __float2bfloat16_rn((float)((qs[j] & 0x0f) - 8) * d);
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        out[16 + j] = __float2bfloat16_rn((float)((qs[j] >> 4) - 8) * d);
}

__device__ __forceinline__ void q4_0_block_f32(const uint8_t* block, float* out) {
    const float d = q4_0_scale(block);
    const uint8_t* qs = block + 2;
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        out[j] = (float)((qs[j] & 0x0f) - 8) * d;
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        out[16 + j] = (float)((qs[j] >> 4) - 8) * d;
}

extern "C" __global__ void q4_0_dequant_bf16(
        const uint8_t* __restrict__ blocks, __nv_bfloat16* __restrict__ out,
        long long blocks_count) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= blocks_count) return;
    q4_0_block_bf16(blocks + index * 18, out + index * 32);
}

extern "C" __global__ void q4_0_dequant_f32(
        const uint8_t* __restrict__ blocks, float* __restrict__ out,
        long long blocks_count) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= blocks_count) return;
    q4_0_block_f32(blocks + index * 18, out + index * 32);
}

extern "C" __global__ void q4_0_dequant_swap_halves_bf16(
        const uint8_t* __restrict__ blocks, __nv_bfloat16* __restrict__ out,
        long long rows, long long cols) {
    if (rows <= 0 || (rows & 1) != 0 || cols <= 0 || (cols & 31) != 0) { __trap(); return; }
    const long long per_row = cols >> 5;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * per_row) return;
    const long long row = index / per_row;
    const long long column = index - row * per_row;
    const long long half = rows >> 1;
    const long long dest = row < half ? row + half : row - half;
    q4_0_block_bf16(blocks + index * 18, out + dest * cols + column * 32);
}

extern "C" __global__ void f16_to_bf16(
        const uint16_t* __restrict__ src, __nv_bfloat16* __restrict__ dst, long long n) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    dst[index] = __float2bfloat16_rn(__half2float(__ushort_as_half(src[index])));
}
