// Q38 ABI v1 section 6, GDN recurrent core. M = 1, decode, value_split = 2 (P1 plan).
//
// One token, per v-head:
//     S     = S * exp(g)                       <- decay FIRST, this is the contract
//     kv    = k^T S                            <- read from the DECAYED S
//     delta = (v - kv) * beta
//     S     = S + k (x) delta                  <- rank-one, thread-local
//     o     = q^T S                            <- read from the UPDATED S
// In matrix form from the model definition:
//     S_t = exp(g_t)(I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
//     o_t = q_t^T S_t
// The order is load-bearing: applying the decay after computing kv is a DIFFERENT recurrence
// and differs by core 7.05e-02 / state 9.58e-03 (see gdn_kernel_design.md 6b) -- enough that
// a generic state tolerance would NOT catch it, so it is a dedicated negative control in G5.
//
// S[:, j] is independent across the value axis, so the value split is free here: no
// cross-split reduction anywhere in this kernel.
//
// All state math is FP32 (ABI section 6). q arrives pre-scaled by 1/sqrt(HEAD_K); the caller
// applied the shared l2_heads and widened. There is no neutral-padding question at M=1, but
// valid_rows is still checked: the recurrent path is only for valid_rows == 1, and anything
// else belongs to gdn_chunked (ABI section 6).
//
// launch: grid (DV/VSPLIT, NUM_V_HEADS). The only parallelism is across heads and the value
// split; the state pass is inherently serial in time.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define GDN_THREADS 512
#define GDN_HEADS   48
#define GDN_DK      128
#define GDN_DV      128

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <int VSPLIT>
__device__
void gdn_recurrent_core_t(
        const float* __restrict__ q,      // FP32 [1, HEADS, DK] (or [M,HEADS,DK] with M==1)
        const float* __restrict__ k,
        const float* __restrict__ v,
        const float* __restrict__ beta,   // FP32 [1, HEADS]
        const float* __restrict__ g,
        float* __restrict__ S,            // FP32 [HEADS, DK, DV]
        __nv_bfloat16* __restrict__ out,  // BF16  [1, HEADS, DV]
        const GdnMeta* __restrict__ meta) {
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
    const int prefix = meta->prefix;
    const int valid_rows = meta->valid_rows;
    if (valid_rows != 1 || prefix < 0 || meta->causal != 1 ||
        meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap(); return;   // M>1 is the chunked path
    }

    // 512 threads tile the [DK][VW] state as 16 row groups x 32 column groups, RT x TC each,
    // i.e. 16 floats per thread. Every loop indexing Sr MUST be unrolled or the tile lands
    // in local memory -- see gdn_chunked.cu for the same trap and its cost.
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

    const float gv = g[head];
    const float bv = beta[head];
    const size_t kv0 = (size_t)head * GDN_DK;    // row 0 of this head's [DK] slice

    float dl[TC];   // declared before the control-build fork: both branches produce it
    // ---- 1. decay FIRST
// Negative-control build; never defined by build.rs.
#ifdef GDN_WRONG_DECAY
    // NEGATIVE CONTROL BUILD (never shipped): decay applied AFTER kv instead of before.
    // The whole point of the G5 order control is that this is a real kernel build, so it is
    // the same source with the order swapped, not a perturbation of the outputs.
    float kvp[TC];
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
        dl[b] = (v[(size_t)head * GDN_DV + j0 + jb + b] - kv) * bv;
    }
    __syncthreads();
#else
    const float alpha = __expf(gv);
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b) Sr[a][b] *= alpha;
    __syncthreads();

    // ---- 2. kv = k^T S, from the DECAYED S. Stage one partial per row group.
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
        dl[b] = (v[(size_t)head * GDN_DV + j0 + jb + b] - kv) * bv;
    }
#endif

    // ---- 3. rank-one update, entirely thread-local
    #pragma unroll
    for (int a = 0; a < RT; ++a) {
        const float ka = k[kv0 + i0 + a];
        #pragma unroll
        for (int b = 0; b < TC; ++b) Sr[a][b] = fmaf(ka, dl[b], Sr[a][b]);
    }
    __syncthreads();

    // ---- 4. o = q^T S, from the UPDATED S
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
        out[(size_t)head * GDN_DV + j0 + jb + b] = __float2bfloat16_rn(o);
    }

    // ---- 5. write the state back
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b] = Sr[a][b];
}

// Decode entry point. value_split is a property of the prepared plan (P1: 2), fixed in
// setup, not chosen per call.
extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_recurrent_core_v2(const float* q, const float* k, const float* v,
                           const float* beta, const float* g, float* S,
                           __nv_bfloat16* out, const GdnMeta* meta) {
    gdn_recurrent_core_t<2>(q, k, v, beta, g, S, out, meta);
}
