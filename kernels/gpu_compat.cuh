#pragma once

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#if !defined(__HIP_PLATFORM_AMD__)
#error "HIP compiler is not targeting AMD"
#endif

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

#define FF_WARP_MASK 0xffffffffULL
#define __trap() __builtin_trap()
#define FF_TRAP() __builtin_trap()

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(float low, float high) {
    return __halves2bfloat162(__float2bfloat16(low), __float2bfloat16(high));
}
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define FF_WARP_MASK 0xffffffffu
#define FF_TRAP() asm volatile("trap;")
#endif
