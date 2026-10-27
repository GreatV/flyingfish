#pragma once
#include "gpu_compat.cuh"
#include <math.h>
#include <stdint.h>
struct Q38SequenceFields {
    int32_t prefix, valid_rows, key_count, capacity, causal, reserved[3];
};
struct alignas(16) Q38SequenceMeta : Q38SequenceFields {};
static_assert(sizeof(Q38SequenceMeta)==32,"Q38 ABI sequence size");
__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}
__device__ __forceinline__ float bround(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sigmoid(float x) {
    if (x>=0.0f) return 1.0f/(1.0f+expf(-x));
    float e=expf(x); return e/(1.0f+e);
}
__device__ __forceinline__ float silu(float x) {return x*sigmoid(x);}

__device__ __forceinline__ void q38_meta_guard(const Q38SequenceMeta* s, bool causal_required=true) {
    if (!s || (reinterpret_cast<uintptr_t>(s)&15)) { __trap(); return; }
    if (s->prefix<0 || s->valid_rows<=0 || s->capacity<=0 ||
        (s->causal!=1 && (causal_required || s->causal!=0)) ||
        int64_t(s->prefix)+s->valid_rows!=s->key_count || s->key_count>s->capacity ||
        s->reserved[0] || s->reserved[1] || s->reserved[2]) __trap();
}

__device__ __forceinline__ float e4m3_decode(uint8_t code) {
    const uint32_t sign = uint32_t(code & 0x80) << 24;
    const uint32_t mag = code & 0x7f;
    if (mag == 0x7f) return __uint_as_float(sign | 0x7fc00000u);
    const uint32_t e = mag >> 3, m = mag & 7;
    if (e == 0) return __uint_as_float(__float_as_uint(float(m) * (1.0f/512.0f)) | sign);
    return __uint_as_float(sign | ((e + 120) << 23) | (m << 20));
}
