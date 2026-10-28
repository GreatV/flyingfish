// GDN recurrent core, per v-head and per row:
//     S     = S * exp(g)                       <- decay first
//     kv    = k^T S                            <- read from the decayed S
//     delta = (v - kv) * beta
//     S     = S + k (x) delta                  <- rank-one, thread-local
//     o     = q^T S                            <- read from the updated S
// In matrix form from the model definition:
//     S_t = exp(g_t)(I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
//     o_t = q_t^T S_t
//
// S[:, j] is independent across the value axis; no cross-split reduction.
// All state math is FP32. The caller applies FP32 L2 normalization and scales q
// by 1/sqrt(HEAD_K). SINGLE requires valid_rows == 1; otherwise valid_rows in
// [1, bound], bound <= 16 (h32) or 5 (h48).
// launch: grid (VSPLIT, NUM_V_HEADS), block GDN_THREADS.
#pragma once
#include "../gpu_compat.cuh"
#include <stdint.h>
#include "../q38_contract.cuh"

#define GDN_THREADS 512
#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define GDN_DK      128
#define GDN_DV      128

template <int VSPLIT, bool SINGLE, bool SNAPSHOT = false>
__device__
void gdn_recurrent_t(
        const float* __restrict__ q,      // FP32 [bound, HEADS, DK]
        const float* __restrict__ k,
        const float* __restrict__ v,
        const float* __restrict__ beta,   // FP32 [bound, HEADS]
        const float* __restrict__ g,
        float* __restrict__ S,            // FP32 [HEADS, DK, DV]
        __nv_bfloat16* __restrict__ out,  // BF16 [bound, HEADS, DV]
        const Q38SequenceFields* __restrict__ meta, int bound, float* snapshots) {
    constexpr int VW = GDN_DV / VSPLIT;
    const int head = blockIdx.y;
    const int j0 = blockIdx.x * VW;
    const int tid = threadIdx.x;

    if (VSPLIT < 1 || VSPLIT > 4 || blockDim.x != GDN_THREADS ||
        blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1 ||
        gridDim.y != GDN_HEADS || gridDim.x != VSPLIT ||
        !aligned16(q) || !aligned16(k) || !aligned16(v) ||
        !aligned16(S) || !aligned16(out)) {
        __trap(); return;
    }
    if (SNAPSHOT && !aligned16(snapshots)) { __trap(); return; }
    const int prefix = meta->prefix;
    const int valid_rows = meta->valid_rows;
    if (SINGLE) {
        if (valid_rows != 1 || prefix < 0 || meta->causal != 1 ||
            meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
            __trap(); return;
        }
    } else {
        constexpr int row_limit = GDN_HEADS == 32 ? 16 : 5;
        if (bound < 1 || bound > row_limit || valid_rows < 1 || valid_rows > bound || prefix < 0 || meta->causal != 1 ||
            meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
            __trap(); return;
        }
    }

    // 512 threads tile the [DK][VW] state as 16 row groups x 32 column groups, RT x TC each.
    // Loops indexing Sr are fully unrolled.
    constexpr int RT = GDN_DK / 16, TC = VW / 32, TG = GDN_DK / RT;
    const int ti = tid / 32, tj = tid % 32;
    const int i0 = ti * RT, jb = tj * TC;

    extern __shared__ float smem[];
    float* const sRed = smem;                    // [TG][VW]: the two cross-thread reductions

    float Sr[RT][TC];
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            Sr[a][b] = S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b];

    const int rows = SINGLE ? 1 : valid_rows;
    for (int row = 0; row < rows; ++row) {
    const float gv = g[row * GDN_HEADS + head];
    const float bv = beta[row * GDN_HEADS + head];
    const size_t kv0 = ((size_t)row * GDN_HEADS + head) * GDN_DK;    // local row; prefix never indexes input

    float dl[TC];
    // 1. Decay before kv.
    const float alpha = __expf(gv);
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b) Sr[a][b] *= alpha;
    __syncthreads();

    // 2. kv = k^T S from the decayed S; one staged partial per row group.
    #pragma unroll
    for (int b = 0; b < TC; ++b) {
        float p = 0.0f;
        #pragma unroll
        for (int a = 0; a < RT; ++a) p = fmaf(k[kv0 + i0 + a], Sr[a][b], p);
        sRed[ti * VW + jb + b] = p;
    }
    __syncthreads();
    #pragma unroll
    for (int b = 0; b < TC; ++b) {
        float kv = 0.0f;
        for (int u = 0; u < TG; ++u) kv += sRed[u * VW + jb + b];
        dl[b] = (v[((size_t)row * GDN_HEADS + head) * GDN_DV + j0 + jb + b] - kv) * bv;
    }

    // 3. Rank-one update, thread-local.
    #pragma unroll
    for (int a = 0; a < RT; ++a) {
        const float ka = k[kv0 + i0 + a];
        #pragma unroll
        for (int b = 0; b < TC; ++b) Sr[a][b] = fmaf(ka, dl[b], Sr[a][b]);
    }
    __syncthreads();

    // The last verify row is not snapshotted; snapshot_row reads rows < live only.
    // float2 stores: TC and every (j0 + jb) are even.
    if (SNAPSHOT && row + 1 < valid_rows) {
        static_assert(TC % 2 == 0, "float2 snapshot stores need even TC");
        #pragma unroll
        for (int a = 0; a < RT; ++a) {
            float2* dst = reinterpret_cast<float2*>(
                snapshots + ((size_t)row * GDN_HEADS + head) * GDN_DK * GDN_DV +
                (i0 + a) * GDN_DV + j0 + jb);
            #pragma unroll
            for (int b2 = 0; b2 < TC / 2; ++b2)
                dst[b2] = make_float2(Sr[a][2 * b2], Sr[a][2 * b2 + 1]);
        }
    }
    // 4. o = q^T S from the updated S. SINGLE stores from every row group.
    #pragma unroll
    for (int b = 0; b < TC; ++b) {
        float p = 0.0f;
        #pragma unroll
        for (int a = 0; a < RT; ++a) p = fmaf(q[kv0 + i0 + a], Sr[a][b], p);
        sRed[ti * VW + jb + b] = p;
    }
    __syncthreads();
    #pragma unroll
    for (int b = 0; b < TC; ++b) {
        float o = 0.0f;
        for (int u = 0; u < TG; ++u) o += sRed[u * VW + jb + b];
        if (SINGLE || ti == 0) out[((size_t)row * GDN_HEADS + head) * GDN_DV + j0 + jb + b] = __float2bfloat16_rn(o);
    }

        if (!SINGLE) __syncthreads();
    }

    // 5. Write the state back.
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b] = Sr[a][b];
}
