#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// sm75 16-bit load is cvt.f32.f16; sm75 16-bit store is cvt.rn.f16.f32; a non-finite value or a magnitude above 65504 sets ff_fp16_store_fault.

extern "C" {
__device__ unsigned ff_fp16_store_fault;
}

template <typename T>
__device__ __forceinline__ unsigned short ff_u16(T v) {
    static_assert(sizeof(T) == 2, "ff_u16");
    unsigned short u;
    __builtin_memcpy(&u, &v, 2);
    return u;
}

template <typename T>
__device__ __forceinline__ T ff_from_u16(unsigned short u) {
    static_assert(sizeof(T) == 2, "ff_from_u16");
    T v;
    __builtin_memcpy(&v, &u, 2);
    return v;
}

__device__ __forceinline__ void ff_fp16_store_check(float v) {
    if (!isfinite(v) || v > 65504.0f || v < -65504.0f) atomicExch(&ff_fp16_store_fault, 1u);
}

template <typename T>
__device__ __forceinline__ float ff_f32_from_f16(T v) {
    const unsigned short u = ff_u16(v);
    float f;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(u));
    return f;
}

template <typename T>
__device__ __forceinline__ T ff_f16_from_f32(float v) {
    ff_fp16_store_check(v);
    unsigned short u;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(u) : "f"(v));
    return ff_from_u16<T>(u);
}

__device__ __forceinline__ float2 ff_f32x2(__nv_bfloat162 v) {
    float2 r;
    r.x = ff_f32_from_f16(v.x);
    r.y = ff_f32_from_f16(v.y);
    return r;
}

__device__ __forceinline__ __nv_bfloat162 ff_f16x2(float a, float b) {
    __nv_bfloat162 r;
    r.x = ff_f16_from_f32<__nv_bfloat16>(a);
    r.y = ff_f16_from_f32<__nv_bfloat16>(b);
    return r;
}

__device__ __forceinline__ __nv_bfloat16 ff_hadd(__nv_bfloat16 a, __nv_bfloat16 b) {
    return ff_f16_from_f32<__nv_bfloat16>(ff_f32_from_f16(a) + ff_f32_from_f16(b));
}

__device__ __forceinline__ __nv_bfloat16 ff_hsub(__nv_bfloat16 a, __nv_bfloat16 b) {
    return ff_f16_from_f32<__nv_bfloat16>(ff_f32_from_f16(a) - ff_f32_from_f16(b));
}

__device__ __forceinline__ __nv_bfloat16 ff_hmul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return ff_f16_from_f32<__nv_bfloat16>(ff_f32_from_f16(a) * ff_f32_from_f16(b));
}

__device__ __forceinline__ __nv_bfloat162 ff_hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    const float2 af = ff_f32x2(a);
    const float2 bf = ff_f32x2(b);
    return ff_f16x2(af.x + bf.x, af.y + bf.y);
}
