// BF16 convolution outputs widen before FP32 Q/K L2 normalization.
// Q/K expand to value heads; Q is scaled by 1/sqrt(DK).
#include "../gpu_compat.cuh"
#include <stdint.h>
#include "../q38_contract.cuh"

#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define GDN_PREP_HEADS GDN_HEADS
#ifndef GDN_KHEADS
#define GDN_KHEADS 16
#endif
#define GDN_PREP_KHEADS GDN_KHEADS
#define GDN_PREP_DK 128
#define GDN_PREP_DV 128

// The row bound is read from the device meta.
// Exact BF16 -> FP32 widen of beta [M, GDN_PREP_HEADS].
extern "C" __global__ void gdn_beta_widen(
        const __nv_bfloat16* __restrict__ beta_bf16,   // [M, GDN_PREP_HEADS]
        float* __restrict__ beta_fp32,                 // [M, GDN_PREP_HEADS]
        const Q38SequenceFields* __restrict__ meta) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;   // value head
    if (h >= GDN_PREP_HEADS) return;
    const int m = blockIdx.y;                              // LOCAL row
    if (m >= meta->valid_rows) return;
    beta_fp32[(size_t)m * GDN_PREP_HEADS + h] =
        __bfloat162float(beta_bf16[(size_t)m * GDN_PREP_HEADS + h]);
}

extern "C" __global__ void gdn_prep_l2_fp32(
        const __nv_bfloat16* __restrict__ q_bf16,
        const __nv_bfloat16* __restrict__ k_bf16,
        const __nv_bfloat16* __restrict__ v_bf16,
        float* __restrict__ q_fp32,
        float* __restrict__ k_fp32,
        float* __restrict__ v_fp32,
        uint32_t row_stride,
        const Q38SequenceFields* __restrict__ meta,
        float l2_eps) {
    if (blockDim.x != GDN_PREP_DK || blockDim.y != 1 || blockDim.z != 1
        || gridDim.x != 1 || gridDim.y != GDN_PREP_KHEADS) {
        __trap();
        return;
    }
    const int row = blockIdx.z;
    if (row >= meta->valid_rows) return;
    const int head = blockIdx.y;
    const int d = threadIdx.x;
    const int lane = d & 31;
    const int warp = d >> 5;
    const size_t source = (size_t)row * row_stride + head * GDN_PREP_DK + d;
    const float q = __bfloat162float(q_bf16[source]);
    const float k = __bfloat162float(k_bf16[source]);
    float qs = q * q;
    float ks = k * k;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        qs += __shfl_down_sync(FF_WARP_MASK, qs, delta);
        ks += __shfl_down_sync(FF_WARP_MASK, ks, delta);
    }
    __shared__ float sums[2][GDN_PREP_DK / 32];
    if (lane == 0) {
        sums[0][warp] = qs;
        sums[1][warp] = ks;
    }
    __syncthreads();
    if (warp == 0) {
        qs = lane < GDN_PREP_DK / 32 ? sums[0][lane] : 0.0f;
        ks = lane < GDN_PREP_DK / 32 ? sums[1][lane] : 0.0f;
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            qs += __shfl_down_sync(FF_WARP_MASK, qs, delta);
            ks += __shfl_down_sync(FF_WARP_MASK, ks, delta);
        }
        if (lane == 0) {
            sums[0][0] = rsqrtf(qs + l2_eps);
            sums[1][0] = rsqrtf(ks + l2_eps);
        }
    }
    __syncthreads();
    const float qn = q * sums[0][0];
    const float kn = k * sums[1][0];
    const float scaled_q = qn * (1.0f / sqrtf((float)GDN_PREP_DK));
    #pragma unroll
    for (int child = 0; child < GDN_PREP_HEADS / GDN_PREP_KHEADS; ++child) {
        const int value_head = head * (GDN_PREP_HEADS / GDN_PREP_KHEADS) + child;
        const size_t out = ((size_t)row * GDN_PREP_HEADS + value_head) * GDN_PREP_DV + d;
        q_fp32[out] = scaled_q;
        k_fp32[out] = kn;
        if (v_fp32 != nullptr) {
            v_fp32[out] = __bfloat162float(
                v_bf16[(size_t)row * row_stride + value_head * GDN_PREP_DV + d]);
        }
    }
}
