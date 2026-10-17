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
