// Q38 ABI section 6, GDN preprocessing tail: widen the BF16 chain to the FP32 the core
// consumes, expand Q and K across the value heads, and scale Q by 1/sqrt(DK).
//
// This is the step between the common ops and the core. ABI section 5 fixes the rounding on
// the BF16 side (silu, l2_heads, gdn_gates all end in BF16 or FP32 exactly as declared); the
// core then runs in FP32. So the widen must happen HERE, once, after the last BF16 boundary
// -- not folded into the BF16 ops, which would move a declared cast.
//
// Three things happen in one pass over [M, HV, DV]:
//   q_fp32[m][h][t] = f32(q_bf16[m][h][t]) / sqrt(DK)
//   k_fp32[m][h][t] = f32(k_bf16[m][h / GQA][t])      <-- GQA expand, value head h
//   v_fp32[m][h][t] = f32(v_bf16[m][h][t])
//
// ABI section 6: value head h corresponds to key head floor(h / 3). HK = HV / GQA = 16.
// The core indexes k as [M, HEADS, DK] with HEADS == 48, so the expansion has to happen
// before the core is launched; handing it [M, 16, DK] would silently mix up the strides.
//
// Q is scaled by MULTIPLICATION by 1/sqrt(DK). The model does
// `scale = 1 / (query.shape[-1] ** 0.5); query = query * scale`
// (modeling_qwen3_5.py:275-276 chunked, 340-341 recurrent), so dividing by sqrt(DK) instead
// is wrong: the two differ on about 41% of elements by 1 ulp in FP32.
//
// One thread per (row, value head, element); the three outputs are written by the same
// thread that reads the same coordinate, so there is no shared state and no barrier.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define GDN_PREP_HEADS 48
#define GDN_PREP_KHEADS 16
#define GDN_PREP_DK 128
#define GDN_PREP_DV 128

// Same device metadata the other three kernels take. The row bound is read from
// here, NOT from a host scalar: `run` never learns valid_rows, and a captured
// Graph must not bake it in.
struct GdnPrepMeta { int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28; };

extern "C" __global__ void gdn_prep_fp32(
        const __nv_bfloat16* __restrict__ q_bf16,   // [M, 16, 128]  KEY heads
        const __nv_bfloat16* __restrict__ k_bf16,   // [M, 16, 128]  KEY heads
        const __nv_bfloat16* __restrict__ v_bf16,   // [M, 48, 128]  VALUE heads already
        float* __restrict__ q_fp32,                 // [M, 48, 128]  expanded to VALUE heads
        float* __restrict__ k_fp32,                 // [M, 48, 128]  expanded to VALUE heads
        float* __restrict__ v_fp32,                 // [M, 48, 128]
        // V is not normed, so it is read IN PLACE from the V columns of `act`, whose row
        // stride is conv_dim (10240), not 48*DV (6144). Q and K come from their own
        // contiguous workspace regions, so they need no stride. Passing V's base without
        // this would read the wrong row for every row after the first.
        uint32_t v_row_stride,
        const GdnPrepMeta* __restrict__ meta) {
    const int valid_rows = meta->valid_rows;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;   // element within the head
    if (t >= GDN_PREP_DV) return;
    const int h = blockIdx.y;                               // VALUE head
    const int m = blockIdx.z;                               // LOCAL row
    if (m >= valid_rows) return;

    const size_t kv = (size_t)m * GDN_PREP_HEADS * GDN_PREP_DV + (size_t)h * GDN_PREP_DV + t;
    // BOTH Q and K expand. The projection's Q and K slices are each 16 key heads
    // (2048 columns), but the core indexes both as [M, 48, 128] with VALUE heads, so
    // value head h reads key head h/3. Expanding only K is the easy half-mistake: it
    // compiles, and every value head then reads the wrong key head for h not divisible by 3.
    const int kh = h / (GDN_PREP_HEADS / GDN_PREP_KHEADS); // GQA: value head -> key head
    const size_t kk = (size_t)m * GDN_PREP_KHEADS * GDN_PREP_DK
                    + (size_t)kh * GDN_PREP_DK + t;

    // The 1/sqrt(DK) scale is applied HERE, after the widen and after the expansion, as a
    // MULTIPLY by the FP32 scalar -- modeling_qwen3_5.py:275-276 (chunked) and 340-341
    // (recurrent) both do `scale = 1 / (query.shape[-1] ** 0.5); query = query * scale`.
    // A division by sqrt(DK) is NOT equivalent: they differ by up to 1 ulp in FP32.
    q_fp32[kv] = __bfloat162float(q_bf16[kk]) * (1.0f / sqrtf((float)GDN_PREP_DK));
    k_fp32[kv] = __bfloat162float(k_bf16[kk]);
    const size_t vv = (size_t)m * v_row_stride + (size_t)h * GDN_PREP_DV + t;
    v_fp32[kv] = __bfloat162float(v_bf16[vv]);
}
// gdn_gates emits `beta = BF16(sigmoid(b.float))` -- ABI section 5 fixes that cast -- but
// BOTH cores declare `const float* beta`. So beta needs an EXACT, value-preserving widen.
// It is not a rounding step: the value is already BF16-rounded by gdn_gates, and this only
// retypes it. Handing the core beta's BF16 buffer instead is a silent 2-byte/4-byte stride
// bug that compiles cleanly and reads garbage from the second element onward.
//
// Separate kernel rather than a fourth leg of gdn_prep_fp32: beta is [M, 48], not
// [M, 48, 128], so it shares no index space with the head tiles, and keeping it separate
// makes the cast boundary obvious in the module list.
extern "C" __global__ void gdn_beta_widen(
        const __nv_bfloat16* __restrict__ beta_bf16,   // [M, 48]
        float* __restrict__ beta_fp32,                 // [M, 48]
        const GdnPrepMeta* __restrict__ meta) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;   // value head
    if (h >= GDN_PREP_HEADS) return;
    const int m = blockIdx.y;                              // LOCAL row
    if (m >= meta->valid_rows) return;
    beta_fp32[(size_t)m * GDN_PREP_HEADS + h] =
        __bfloat162float(beta_bf16[(size_t)m * GDN_PREP_HEADS + h]);
}
