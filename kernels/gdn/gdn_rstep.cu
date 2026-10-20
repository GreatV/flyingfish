// Ordered recurrent rows: h32 1..16, h48 1..5. Each (head,value split) CTA holds FP32
// state in registers. The live row count is read from the device meta; bound is the
// prepared allocation. No metadata mutation and no cross-CTA state dependency.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define GDN_THREADS 512
#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define GDN_DK      128
#define GDN_DV      128

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <int VSPLIT, bool SNAPSHOT = false>
__device__
void gdn_rstep_core_t(
        const float* __restrict__ q,      // FP32 [bound, HEADS, DK]
        const float* __restrict__ k,
        const float* __restrict__ v,
        const float* __restrict__ beta,   // FP32 [bound, HEADS]
        const float* __restrict__ g,
        float* __restrict__ S,            // FP32 [HEADS, DK, DV]
        __nv_bfloat16* __restrict__ out,  // BF16 [bound, HEADS, DV]
        const GdnMeta* __restrict__ meta, int bound, float* snapshots = nullptr) {
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
    constexpr int row_limit = GDN_HEADS == 32 ? 16 : 5;
    if (bound < 1 || bound > row_limit || valid_rows < 1 || valid_rows > bound || prefix < 0 || meta->causal != 1 ||
        meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap(); return;
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

    for (int r = 0; r < valid_rows; ++r) {
#ifdef GDN_RSTEP_WRONG_ROW
        const int row = 0; // Negative control: reuse the first row.
#else
        const int row = r;
#endif
#ifdef GDN_RSTEP_STALE_STATE
        // Negative control: restart every row from the transaction's old state.
        #pragma unroll
        for (int a = 0; a < RT; ++a)
            #pragma unroll
            for (int b = 0; b < TC; ++b)
                Sr[a][b] = S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b];
#endif
    const float gv = g[row * GDN_HEADS + head];
    const float bv = beta[row * GDN_HEADS + head];
    const size_t kv0 = ((size_t)row * GDN_HEADS + head) * GDN_DK;    // local row; prefix never indexes input

    float dl[TC];
    // 1. Decay before kv.
// Negative-control build; never defined by build.rs.
#ifdef GDN_RSTEP_WRONG_DECAY
    // Decay applied after kv instead of before.
    #pragma unroll
    for (int b = 0; b < TC; ++b) {
        float p = 0.0f;
        #pragma unroll
        for (int a = 0; a < RT; ++a) p = fmaf(k[kv0 + i0 + a], Sr[a][b], p);
        sRed[ti * VW + jb + b] = p;
    }
    __syncthreads();
    const float alpha = __expf(gv);
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b) Sr[a][b] *= alpha;
    __syncthreads();
    #pragma unroll
    for (int b = 0; b < TC; ++b) {
        float kv = 0.0f;
        for (int u = 0; u < TG; ++u) kv += sRed[u * VW + jb + b];
        dl[b] = (v[((size_t)row * GDN_HEADS + head) * GDN_DV + j0 + jb + b] - kv) * bv;
    }
    __syncthreads();
#else
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
#endif

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
    if (SNAPSHOT && r + 1 < valid_rows) {
        static_assert(TC % 2 == 0, "float2 snapshot stores need even TC");
        #pragma unroll
        for (int a = 0; a < RT; ++a) {
            float2* dst = reinterpret_cast<float2*>(
                snapshots + ((size_t)r * GDN_HEADS + head) * GDN_DK * GDN_DV +
                (i0 + a) * GDN_DV + j0 + jb);
            #pragma unroll
            for (int b2 = 0; b2 < TC / 2; ++b2)
                dst[b2] = make_float2(Sr[a][2 * b2], Sr[a][2 * b2 + 1]);
        }
    }
    // 4. o = q^T S from the updated S.
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
        if (ti == 0) out[((size_t)row * GDN_HEADS + head) * GDN_DV + j0 + jb + b] = __float2bfloat16_rn(o);
    }

        __syncthreads();
    }

    // 5. Write the state back.
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b] = Sr[a][b];
}

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_rstep_core_v1(const float* q, const float* k, const float* v,
                       const float* beta, const float* g, float* S,
                       __nv_bfloat16* out, const GdnMeta* meta, int bound) {
    gdn_rstep_core_t<2>(q, k, v, beta, g, S, out, meta, bound);
}

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_rstep_core_snapshot(const float* q, const float* k, const float* v,
                            const float* beta, const float* g, float* S,
                            __nv_bfloat16* out, const GdnMeta* meta, int bound,
                            float* snapshots) {
    gdn_rstep_core_t<2, true>(q, k, v, beta, g, S, out, meta, bound, snapshots);
}
