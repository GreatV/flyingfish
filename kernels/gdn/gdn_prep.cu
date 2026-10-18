// GDN preprocessing: widen BF16 Q/K/V to FP32, expand Q and K across the value heads, and
// scale Q by 1/sqrt(DK). One pass over [M, HV, DV]:
//   q_fp32[m][h][t] = f32(q_bf16[m][h / GQA][t]) * (1/sqrt(DK))
//   k_fp32[m][h][t] = f32(k_bf16[m][h / GQA][t])
//   v_fp32[m][h][t] = f32(v_bf16[m][h][t])
// Value head h reads key head h / (HV / 16), with HV = 32 or 48.
// Q is scaled by multiplying with the FP32 scalar 1/sqrt(DK), the model definition's form.
// One thread per (row, value head, element).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define GDN_PREP_HEADS GDN_HEADS
#define GDN_PREP_KHEADS 16
#define GDN_PREP_DK 128
#define GDN_PREP_DV 128

// The row bound is read from the device meta.
struct GdnPrepMeta { int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28; };

extern "C" __global__ void gdn_prep_fp32(
        const __nv_bfloat16* __restrict__ q_bf16,   // [M, 16, 128]  KEY heads
        const __nv_bfloat16* __restrict__ k_bf16,   // [M, 16, 128]  KEY heads
        const __nv_bfloat16* __restrict__ v_bf16,   // [M, GDN_PREP_HEADS, 128]  VALUE heads already
        float* __restrict__ q_fp32,                 // [M, GDN_PREP_HEADS, 128]  expanded to VALUE heads
        float* __restrict__ k_fp32,                 // [M, GDN_PREP_HEADS, 128]  expanded to VALUE heads
        float* __restrict__ v_fp32,                 // [M, GDN_PREP_HEADS, 128]
        // V is read in place from the V columns of act; its row stride is v_row_stride.
        uint32_t v_row_stride,
        const GdnPrepMeta* __restrict__ meta) {
    const int valid_rows = meta->valid_rows;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;   // element within the head
    if (t >= GDN_PREP_DV) return;
    const int h = blockIdx.y;                               // VALUE head
    const int m = blockIdx.z;                               // LOCAL row
    if (m >= valid_rows) return;

    const size_t kv = (size_t)m * GDN_PREP_HEADS * GDN_PREP_DV + (size_t)h * GDN_PREP_DV + t;
    const int kh = h / (GDN_PREP_HEADS / GDN_PREP_KHEADS); // GQA: value head -> key head
    const size_t kk = (size_t)m * GDN_PREP_KHEADS * GDN_PREP_DK
                    + (size_t)kh * GDN_PREP_DK + t;

    q_fp32[kv] = __bfloat162float(q_bf16[kk]) * (1.0f / sqrtf((float)GDN_PREP_DK));
    k_fp32[kv] = __bfloat162float(k_bf16[kk]);
    const size_t vv = (size_t)m * v_row_stride + (size_t)h * GDN_PREP_DV + t;
    v_fp32[kv] = __bfloat162float(v_bf16[vv]);
}
// Exact BF16 -> FP32 widen of beta [M, GDN_PREP_HEADS].
extern "C" __global__ void gdn_beta_widen(
        const __nv_bfloat16* __restrict__ beta_bf16,   // [M, GDN_PREP_HEADS]
        float* __restrict__ beta_fp32,                 // [M, GDN_PREP_HEADS]
        const GdnPrepMeta* __restrict__ meta) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;   // value head
    if (h >= GDN_PREP_HEADS) return;
    const int m = blockIdx.y;                              // LOCAL row
    if (m >= meta->valid_rows) return;
    beta_fp32[(size_t)m * GDN_PREP_HEADS + h] =
        __bfloat162float(beta_bf16[(size_t)m * GDN_PREP_HEADS + h]);
}

#if GDN_HEADS == 32
extern "C" __global__ void gdn_prep_l2_fp32(
        const __nv_bfloat16* __restrict__ q_bf16,
        const __nv_bfloat16* __restrict__ k_bf16,
        const __nv_bfloat16* __restrict__ v_bf16,
        float* __restrict__ q_fp32,
        float* __restrict__ k_fp32,
        float* __restrict__ v_fp32,
        uint32_t row_stride,
        const GdnPrepMeta* __restrict__ meta) {
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
        qs += __shfl_down_sync(0xffffffff, qs, delta);
        ks += __shfl_down_sync(0xffffffff, ks, delta);
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
            qs += __shfl_down_sync(0xffffffff, qs, delta);
            ks += __shfl_down_sync(0xffffffff, ks, delta);
        }
        if (lane == 0) {
            sums[0][0] = rsqrtf(qs + 1e-6f);
            sums[1][0] = rsqrtf(ks + 1e-6f);
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
        v_fp32[out] = __bfloat162float(
            v_bf16[(size_t)row * row_stride + value_head * GDN_PREP_DV + d]);
    }
}
#endif
