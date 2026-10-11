// Q38 ABI v1: row-major E4M3 codes, BF16 [ceil(N/128),ceil(K/128)] scales.
// Prefill materializes effective BF16 weights. Decode forms the same rounded
// weight in registers, accumulates FP32, and writes BF16; no matrix scratch.
// Format source: NVIDIA CUDA 13.0 Math API, __nv_fp8_e4m3.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float e4m3_decode(uint8_t code) {
    const uint32_t sign = uint32_t(code & 0x80) << 24;
    const uint32_t mag = code & 0x7f;
    if (mag == 0x7f) return __uint_as_float(sign | 0x7fc00000u);
    const uint32_t e = mag >> 3, m = mag & 7;
    if (e == 0) return __uint_as_float(__float_as_uint(float(m) * (1.0f/512.0f)) | sign);
    return __uint_as_float(sign | ((e + 120) << 23) | (m << 20));
}
__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}
__device__ __forceinline__ __nv_bfloat16 weight_at(
        const uint8_t* w, const __nv_bfloat16* scales, uint64_t n,
        uint64_t k, uint64_t in, uint64_t scale_stride) {
    const float s = __bfloat162float(scales[(n/128)*scale_stride + k/128]);
    return __float2bfloat16_rn(e4m3_decode(w[n*in+k]) * s);
}
// launch: block256; grid ceil(N*K/256), capped to a positive device-valid grid.
extern "C" __global__ void fp8_dequant_bf16(
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
        out[i] = weight_at(w,scales,i/uint64_t(k),i%uint64_t(k),k,scale_stride);
    }
}
// launch: grid ceil(N/4), block128. One warp per output, lanes partition K.
extern "C" __global__ void fp8_gemv_bf16(
        const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y,
        int n, int k) {
    if (n <= 0 || k <= 0 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || uint64_t(gridDim.x) != (uint64_t(n)+3)/4 ||
        !aligned16(w) || !aligned16(scales) || !aligned16(x) || !aligned16(y)) {
        __trap(); return;
    }
    const uint64_t row = uint64_t(blockIdx.x)*4 + threadIdx.x/32;
    const uint32_t lane = threadIdx.x%32;
    if (row >= uint64_t(n)) return;
    const uint64_t scale_stride = (uint64_t(k)+127)/128;
    float acc=0.0f;
    for (uint64_t col=lane; col<uint64_t(k); col+=32) {
        const float weight = __bfloat162float(weight_at(w,scales,row,col,k,scale_stride));
        acc = fmaf(weight,__bfloat162float(x[col]),acc);
    }
    for (int shift=16; shift>0; shift/=2) acc += __shfl_down_sync(0xffffffffu,acc,shift);
    if (lane==0) y[row]=__float2bfloat16_rn(acc);
}
