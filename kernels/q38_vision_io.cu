#include "gpu_compat.cuh"
#include <stdint.h>
#include <math.h>
#include "q38_contract.cuh"

__device__ __forceinline__ bool flat_launch() {
    return blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
           gridDim.y == 1 && gridDim.z == 1;
}

extern "C" __global__ void q38_vision_cast_patches(
    const float* input, __nv_bfloat16* output, uint64_t count) {
    if (!flat_launch() || !input || !output || !count || count % 1536) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * 256) {
        float x = input[i];
        __nv_bfloat16 y = __float2bfloat16_rn(x);
        if (!isfinite(x) || !isfinite(__bfloat162float(y))) { __trap(); return; }
        output[i] = y;
    }
}

// FP32 interpolation casts before position addition; optional bias rounds in BF16 first.
extern "C" __global__ void q38_vision_position_add(
    const __nv_bfloat16* table, const int32_t* indices, const float* weights,
    const __nv_bfloat16* patches, const __nv_bfloat16* bias,
    __nv_bfloat16* trace_patches, __nv_bfloat16* hidden, float* interpolated,
    uint32_t rows) {
#if defined(__HIP__)
#pragma clang fp contract(off)
#endif
    if (!flat_launch() || !table || !indices || !weights || !patches ||
        !hidden || !interpolated || !rows || (trace_patches && !bias)) { __trap(); return; }
    uint64_t count = uint64_t(rows) * 1152;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * 256) {
        uint32_t r = i / 1152, d = i % 1152;
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t k = 0; k < 4; ++k) {
            uint64_t off = uint64_t(k) * rows + r;
            int32_t index = indices[off];
            float w = weights[off];
            if (index < 0 || index >= 2304 || !isfinite(w)) { __trap(); return; }
#if defined(__HIP__)
            const float product = __bfloat162float(table[uint64_t(index) * 1152 + d]) * w;
            sum = sum + product;
#else
            sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(table[uint64_t(index) * 1152 + d]), w));
#endif
        }
        if (!isfinite(sum)) { __trap(); return; }
        interpolated[i] = sum;
        const __nv_bfloat16 patch = bias ? __hadd(patches[i], bias[d]) : patches[i];
        if (trace_patches) trace_patches[i] = patch;
        hidden[i] = __hadd(patch, __float2bfloat16_rn(sum));
    }
}

template <uint32_t WIDTH>
__device__ __forceinline__ void vision_splice(
    const __nv_bfloat16* features, __nv_bfloat16* output,
    uint32_t feature_rows, uint32_t output_rows,
    uint32_t source_first, uint32_t destination_first, uint32_t rows) {
    if (!flat_launch() || !features || !output || !rows ||
        uint64_t(source_first) + rows > feature_rows ||
        uint64_t(destination_first) + rows > output_rows) { __trap(); return; }
    uint64_t count = uint64_t(rows) * WIDTH;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * 256) {
        output[uint64_t(destination_first) * WIDTH + i] =
            features[uint64_t(source_first) * WIDTH + i];
    }
}

extern "C" __global__ void q38_vision_splice(
    const __nv_bfloat16* features, __nv_bfloat16* output,
    uint32_t feature_rows, uint32_t output_rows,
    uint32_t source_first, uint32_t destination_first, uint32_t rows) {
    vision_splice<5120>(features, output, feature_rows, output_rows,
                        source_first, destination_first, rows);
}

extern "C" __global__ void vision_splice_2048(
    const __nv_bfloat16* features, __nv_bfloat16* output,
    uint32_t feature_rows, uint32_t output_rows,
    uint32_t source_first, uint32_t destination_first, uint32_t rows) {
    vision_splice<2048>(features, output, feature_rows, output_rows,
                        source_first, destination_first, rows);
}

// One image range from the shared PatchPacking; QKV is [rows,3,16,72].
extern "C" __global__ void q38_vision_pack(
    const __nv_bfloat16* qkv, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    uint32_t rows, uint32_t padded, uint32_t first, uint32_t key_first, uint32_t n) {
    if (!flat_launch() || !qkv || !q || !k || !v || !n ||
        uint64_t(first)+n>rows || uint64_t(key_first)+n>padded) { __trap(); return; }
    for (uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x; i<uint64_t(n)*1152;
         i+=uint64_t(gridDim.x)*256) {
        uint32_t r=i/1152, h=(i%1152)/72, d=i%72;
        uint64_t src=uint64_t(first+r)*3456+h*72+d;
        uint64_t dst=(uint64_t(h)*rows+first+r)*72+d;
        q[dst]=qkv[src]; k[dst]=qkv[src+1152];
        v[(uint64_t(h)*padded+key_first+r)*72+d]=qkv[src+2304];
    }
}

extern "C" __global__ void q38_vision_residual(
    __nv_bfloat16* hidden, const __nv_bfloat16* delta, uint64_t count) {
    if (!flat_launch() || !hidden || !delta || !count) { __trap(); return; }
    for (uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x; i<count;
         i+=uint64_t(gridDim.x)*256) hidden[i]=__hadd(hidden[i],delta[i]);
}


// Both buffers have stable addresses. Only SequenceMeta advances in decode graphs.
extern "C" __global__ void q38_text_positions(
    const int32_t* table, int32_t* active, const Q38SequenceMeta* meta, uint32_t max_rows) {
    q38_meta_guard(meta);
    if (!flat_launch() || !table || !active || uint32_t(meta->valid_rows)>max_rows) {
        __trap(); return;
    }
    uint64_t count=uint64_t(meta->valid_rows)*3;
    for (uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x; i<count;
         i+=uint64_t(gridDim.x)*256) active[i]=table[uint64_t(meta->prefix)*3+i];
}
