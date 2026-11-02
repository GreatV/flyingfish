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

template<bool ScaleQuery>
__device__ void gdn_prep_l2(
        const __nv_bfloat16* __restrict__ q_bf16,
        const __nv_bfloat16* __restrict__ k_bf16,
        const __nv_bfloat16* __restrict__ v_bf16,
        float* __restrict__ q_fp32,
        float* __restrict__ k_fp32,
        float* __restrict__ v_fp32,
        uint32_t row_stride,
        int rows,
        float l2_eps) {
    if (blockDim.x != GDN_PREP_DK || blockDim.y != 1 || blockDim.z != 1
        || gridDim.x != 1 || gridDim.y != GDN_PREP_KHEADS) {
        __trap();
        return;
    }
    const int row = blockIdx.z;
    if (row >= rows) return;
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
    const float scaled_q = ScaleQuery ? qn * (1.0f / sqrtf((float)GDN_PREP_DK)) : qn;
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

extern "C" __global__ void gdn_prep_l2_fp32(
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
        float* q_out, float* k_out, float* v_out, uint32_t stride,
        const Q38SequenceFields* meta, float eps) {
    gdn_prep_l2<true>(q, k, v, q_out, k_out, v_out, stride, meta->valid_rows, eps);
}

extern "C" __global__ void gdn_block_l2_gates(
        const __nv_bfloat16* mixed, const __nv_bfloat16* a, const __nv_bfloat16* b,
        const __nv_bfloat16* a_log, const __nv_bfloat16* dt,
        float* q, float* k, float* v, float* g, float* beta, int rows, int preview) {
    constexpr int stride = (2 * GDN_KHEADS + GDN_HEADS) * 128;
    if (rows < 1 || rows > 32 || gridDim.z != rows || (preview != 0 && preview != 1) ||
        !aligned16(mixed) || !aligned16(a) || !aligned16(a_log) || !aligned16(dt) ||
        !aligned16(q) || !aligned16(k) || !aligned16(v) || !aligned16(g) ||
        (!preview && (!aligned16(b) || !aligned16(beta)))) { __trap(); return; }
    if (preview)
        gdn_prep_l2<false>(mixed, mixed + GDN_KHEADS * 128, mixed + 2 * GDN_KHEADS * 128,
                          q, k, v, stride, rows, 1e-6f);
    else
        gdn_prep_l2<true>(mixed, mixed + GDN_KHEADS * 128, mixed + 2 * GDN_KHEADS * 128,
                         q, k, v, stride, rows, 1e-6f);
    if (threadIdx.x < GDN_HEADS / GDN_KHEADS) {
        const int head = blockIdx.y * (GDN_HEADS / GDN_KHEADS) + threadIdx.x;
        const int index = blockIdx.z * GDN_HEADS + head;
        const float x = __bfloat162float(a[index]) + __bfloat162float(dt[head]);
        const float softplus = x > 20.0f ? x : log1pf(expf(x));
        g[index] = -expf(__bfloat162float(a_log[head])) * softplus;
        if (!preview)
            beta[index] = __bfloat162float(__float2bfloat16_rn(sigmoid(__bfloat162float(b[index]))));
    }
}
