// GDN chunked core: C = GDN_CHUNK, value_split VSPLIT, GDN_THREADS threads/block.
// WY / UT-transform form. Per chunk, per v-head, G_j = within-chunk cumsum of g
// (g <= 0, so G_j <= 0):
//     L[j,i] = exp(G_j - G_i) for j>=i, 0 for j<i
//     A[j,i] = -beta_j (k_j . k_i) L[j,i]                for j>i, 0 otherwise
//     T      = (I - A)^-1                               unit lower triangular
//     w      = T (beta .* V)
//     v_new  = w - (T (beta .* K .* e^G)) @ S
//     o      = (q .* e^G) @ S + [tril(q k^T .* L)] @ v_new
//     S      = S * e^{G_{C-1}} + (k .* e^{G_{C-1}-G})^T @ v_new
// kdec = T(beta.*K.*e^G) is not materialised: kbS = kb @ S ([C][DK] x [DK][VW]) is
// formed once and contracted with T.
// Every exp() argument is non-positive; exp(G_j - G_i) is formed from the difference.
// State math is FP32. q arrives pre-scaled by 1/sqrt(HEAD_K).
// Rows at or beyond valid_rows carry q=k=v=beta=0, g=0; S is bit-unchanged by them.
// launch: grid (VSPLIT, NUM_V_HEADS); each block owns S[:, j0:j0+VW] in registers and
//         walks every chunk sequentially.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef GDN_CHUNK
#define GDN_CHUNK 32
#endif
#define GDN_SUB   16          // UT sub-block
#define GDN_THREADS 512
#define GDN_HEADS 48
#define GDN_DK    128
#define GDN_DV    128

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <int VSPLIT, bool SNAPSHOT = false>
__device__
void gdn_chunked_core_t(
        const float* __restrict__ q,      // FP32 [M, HEADS, DK], pre-scaled by 1/sqrt(DK)
        const float* __restrict__ k,      // FP32 [M, HEADS, DK]
        const float* __restrict__ v,      // FP32 [M, HEADS, DV]
        const float* __restrict__ beta,   // FP32 [M, HEADS]
        const float* __restrict__ g,      // FP32 [M, HEADS]
        float* __restrict__ S,            // FP32 [HEADS, DK, DV]
        __nv_bfloat16* __restrict__ out,  // BF16 [M, HEADS, DV]
        const GdnMeta* __restrict__ meta,
        int valid_rows_in, float* snapshots
// Negative-control build; never defined by build.rs.
#ifdef GDN_DUMP_PARTS
        , float* __restrict__ dump_inter, float* __restrict__ dump_attn
#endif
        ) {
    constexpr int VW = GDN_DV / VSPLIT;
    const int head = blockIdx.y;
    const int j0 = blockIdx.x * VW;
    const int tid = threadIdx.x;

    if (VSPLIT < 1 || VSPLIT > 4 || blockDim.x != GDN_THREADS || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.z != 1 || gridDim.y != GDN_HEADS || gridDim.x != VSPLIT ||
        !aligned16(q) || !aligned16(k) || !aligned16(v) ||
        !aligned16(S) || !aligned16(out)) {
        __trap(); return;
    }
    const int prefix = meta->prefix;
    const int valid_rows = valid_rows_in > 0 ? valid_rows_in : meta->valid_rows;
    if (valid_rows <= 0 || prefix < 0 || meta->causal != 1 ||
        meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap(); return;
    }
    if (SNAPSHOT && (!aligned16(snapshots) || valid_rows > 5)) { __trap(); return; }

    // Dynamic shared memory; the size exceeds 49,152 B and needs the cudaFuncSetAttribute opt-in.
    extern __shared__ float smem[];
    float* const sKb   = smem;                       // [C][DK] raw k
    float* const sVb   = sKb   + GDN_CHUNK * GDN_DK;  // [C][VW] beta.V, then kbS
    float* const sKbS  = sVb   + GDN_CHUNK * VW;      // [C][VW] w, then v_new
    float* const sT    = sKbS  + GDN_CHUNK * VW;      // [C][C]  L -> A -> T
    float* const sQr   = sT    + GDN_CHUNK * GDN_CHUNK;// [DK]   one q row at a time
    float* const sBeta = sQr   + GDN_DK;               // [C]
    float* const sG    = sBeta + GDN_CHUNK;            // [C]
    // S lives in registers; sums over all DK rows stage through sRed [TG][VW] and sQk [C].
    float* const sRed  = sG    + GDN_CHUNK;            // [TG][VW]
    float* const sQk   = sRed  + 16 * VW;              // [C]

    // S[:, j0:j0+VW] stays in registers: 512 threads tile [DK][VW] as 16 row groups x 32
    // column groups, RT x TC each. Loops indexing Sr are fully unrolled.
    constexpr int RT = 8, TC = GDN_DV / 32, TG = GDN_DK / RT;
    const int ti = tid / 32, tj = tid % 32;
    const int i0 = ti * RT, jb = tj * TC;
    float Sr[RT][TC];
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            Sr[a][b] = S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b];

    const int nchunk = (valid_rows + GDN_CHUNK - 1) / GDN_CHUNK;

    for (int c = 0; c < nchunk; ++c) {
        const int row0 = c * GDN_CHUNK;

        for (int r = tid; r < GDN_CHUNK; r += GDN_THREADS) {
            const int gr = row0 + r;
            const bool live = gr < valid_rows;
            sBeta[r] = live ? beta[(size_t)gr * GDN_HEADS + head] : 0.0f;
            sG[r]   = live ? g[(size_t)gr * GDN_HEADS + head] : 0.0f;
        }
        __syncthreads();
        // Single-thread in-place prefix sum over sG.
        if (tid == 0) {
            for (int r = 1; r < GDN_CHUNK; ++r) sG[r] += sG[r - 1];
        }
        __syncthreads();

        // sKb holds unscaled k; beta*e^G is applied in kbS and in the state update.
        for (int idx = tid; idx < GDN_CHUNK * GDN_DK; idx += GDN_THREADS) {
            const int r = idx / GDN_DK, t = idx - r * GDN_DK;
            const int gr = row0 + r;
            sKb[r * GDN_DK + t] = (gr < valid_rows)
                ? k[((size_t)gr * GDN_HEADS + head) * GDN_DK + t] : 0.0f;
        }
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, t = idx - r * VW;
            const int gr = row0 + r;
            sVb[r * VW + t] = (gr < valid_rows)
                ? v[((size_t)gr * GDN_HEADS + head) * GDN_DV + j0 + t] * sBeta[r] : 0.0f;
        }
        __syncthreads();

        // A[j,i] = -beta_j (k_j.k_i) exp(G_j - G_i), j>i, into sT.
        for (int sj = 0; sj < GDN_CHUNK / GDN_SUB; ++sj)
            for (int si = 0; si < GDN_CHUNK / GDN_SUB; ++si)
                for (int idx = tid; idx < GDN_SUB * GDN_SUB; idx += GDN_THREADS) {
                    const int jl = idx / GDN_SUB, il = idx - jl * GDN_SUB;
                    const int j = sj * GDN_SUB + jl, i = si * GDN_SUB + il;
                    if (j <= i || sBeta[j] == 0.0f) { sT[j * GDN_CHUNK + i] = 0.0f; continue; }
                    float dot = 0.0f;
                    for (int t = 0; t < GDN_DK; ++t) dot = fmaf(sKb[j * GDN_DK + t], sKb[i * GDN_DK + t], dot);
                    sT[j * GDN_CHUNK + i] = -sBeta[j] * dot * __expf(sG[j] - sG[i]);
                }
        __syncthreads();

        // T = (I - A)^-1 by forward substitution; thread r computes row r, one barrier per row.
        for (int r = 1; r < GDN_CHUNK; ++r) {
            if (tid == r) {
                // T[r][j] = M[r][j] + sum_{m=j+1}^{r-1} M[r][m] * T[m][j]
                // Forward j order: sT[r][m] for m > j still holds M[r][m]; rows m < r are final.
                for (int j = 0; j < r; ++j) {
                    float acc = sT[r * GDN_CHUNK + j];
                    for (int m = j + 1; m < r; ++m)
                        acc = fmaf(sT[r * GDN_CHUNK + m], sT[m * GDN_CHUNK + j], acc);
                    sT[r * GDN_CHUNK + j] = acc;
                }
            }
            __syncthreads();
        }
        for (int idx = tid; idx < GDN_CHUNK; idx += GDN_THREADS) sT[idx * GDN_CHUNK + idx] += 1.0f;
        __syncthreads();

        // w = T (beta .* V) into sKbS
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int i = 0; i <= r; ++i) acc = fmaf(sT[r * GDN_CHUNK + i], sVb[i * VW + j], acc);
            sKbS[r * VW + j] = acc;
        }
        __syncthreads();

        // kbS = kb @ S (kb = beta.K.e^G) into sVb, one row at a time: per-row-group partials
        // stage through sRed [TG][VW] and reduce between barriers.
        for (int r = 0; r < GDN_CHUNK; ++r) {
            const float sc = sBeta[r] * __expf(sG[r]);
            #pragma unroll
            for (int b = 0; b < TC; ++b) {
                float acc = 0.0f;
                #pragma unroll
                for (int a = 0; a < RT; ++a)
                    acc = fmaf(sKb[r * GDN_DK + i0 + a] * sc, Sr[a][b], acc);
                sRed[ti * VW + jb + b] = acc;
            }
            __syncthreads();
            for (int j = tid; j < VW; j += GDN_THREADS) {
                float acc = 0.0f;
                for (int u = 0; u < GDN_DK / RT; ++u) acc += sRed[u * VW + j];
                sVb[r * VW + j] = acc;       // sVb now holds kbS row r
            }
            __syncthreads();
        }
        __syncthreads();

// Negative-control build; never defined by build.rs.
#ifdef GDN_RAW_V
        // v_new from raw V instead of w = T(beta.*V).
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int t = 0; t <= r; ++t) acc = fmaf(sT[r * GDN_CHUNK + t], sVb[t * VW + j], acc);
            sKbS[r * VW + j] = sVb[r * VW + j] - acc;   // raw v, not w
        }
        __syncthreads();
#else
        // v_new = w - T @ kbS, in place in sKbS
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int t = 0; t <= r; ++t) acc = fmaf(sT[r * GDN_CHUNK + t], sVb[t * VW + j], acc);
            sKbS[r * VW + j] -= acc;
        }
        __syncthreads();
#endif

        // o = (q.e^G) @ S + [tril(q k^T .* L)] @ v_new
        for (int r = 0; r < GDN_CHUNK; ++r) {
            const int gr = row0 + r;
            // sQr holds raw q; inter is scaled by e^{G_r} once and the attention term uses e^{G_r-G_i}.
            if (gr < valid_rows) {
                for (int t = tid; t < GDN_DK; t += GDN_THREADS)
                    sQr[t] = q[((size_t)gr * GDN_HEADS + head) * GDN_DK + t];
            } else {
                for (int t = tid; t < GDN_DK; t += GDN_THREADS) sQr[t] = 0.0f;
            }
            __syncthreads();
            // inter partials: each thread contributes its RT rows for its own columns
            #pragma unroll
            for (int b = 0; b < TC; ++b) {
                float p = 0.0f;
                #pragma unroll
                for (int a = 0; a < RT; ++a) p = fmaf(sQr[i0 + a], Sr[a][b], p);
                sRed[ti * VW + jb + b] = p;
            }
            // q.k row products, shared across columns
            for (int i = tid; i <= r; i += GDN_THREADS) {
                float qk = 0.0f;
                for (int t = 0; t < GDN_DK; ++t) qk = fmaf(sQr[t], sKb[i * GDN_DK + t], qk);
                sQk[i] = qk;
            }
            __syncthreads();
// Negative-control build; never defined by build.rs.
#ifdef GDN_DUMP_PARTS
            // Writes the inter and attn terms before rounding.
            for (int j = tid; j < VW; j += GDN_THREADS) {
                const float eG = __expf(sG[r]);
                float i0 = 0.0f;
                for (int u = 0; u < GDN_DK / RT; ++u) i0 += sRed[u * VW + j];
                float a0 = 0.0f;
                for (int i = 0; i <= r; ++i)
                    a0 = fmaf(sQk[i] * __expf(sG[r] - sG[i]), sKbS[i * VW + j], a0);
                if (gr < valid_rows) {
                    const size_t oi = ((size_t)gr * GDN_HEADS + head) * GDN_DV + j0 + j;
                    dump_inter[oi] = i0 * eG;
                    dump_attn[oi] = a0;
                }
            }
#endif
            for (int j = tid; j < VW; j += GDN_THREADS) {
                const float eG = __expf(sG[r]);
                float inter = 0.0f;
                for (int u = 0; u < GDN_DK / RT; ++u) inter += sRed[u * VW + j];
                inter *= eG;
                float attn = 0.0f;
                for (int i = 0; i <= r; ++i)
                    attn = fmaf(sQk[i] * __expf(sG[r] - sG[i]), sKbS[i * VW + j], attn);
                if (gr < valid_rows)
                    out[((size_t)gr * GDN_HEADS + head) * GDN_DV + j0 + j] =
                        __float2bfloat16_rn(inter + attn);
            }
            __syncthreads();
        }

        // The last row is not snapshotted; snapshot_row reads rows < live only.
        if (SNAPSHOT) {
            for (int r = 0; r + 1 < valid_rows; ++r) {
                #pragma unroll
                for (int a = 0; a < RT; ++a)
                    #pragma unroll
                    for (int b = 0; b < TC; ++b) {
                        float acc = Sr[a][b] * __expf(sG[r]);
                        for (int t = 0; t <= r; ++t)
                            acc = fmaf(sKb[t * GDN_DK + i0 + a] * __expf(sG[r] - sG[t]),
                                       sKbS[t * VW + jb + b], acc);
                        snapshots[((size_t)r * GDN_HEADS + head) * GDN_DK * GDN_DV +
                                  (i0 + a) * GDN_DV + j0 + jb + b] = acc;
                    }
            }
        }

        // S = S * e^{G_{C-1}} + (k .* e^{G_{C-1}-G})^T @ v_new
        {
            const float decay = __expf(sG[GDN_CHUNK - 1]);
            const float invc = 1.0f / fmaxf(decay, 1e-38f);
            #pragma unroll
            for (int a = 0; a < RT; ++a)
                #pragma unroll
                for (int b = 0; b < TC; ++b) {
                    float acc = Sr[a][b] * decay;
                    const int i = i0 + a;
                    #pragma unroll 4
                    for (int r = 0; r < GDN_CHUNK; ++r) {
                        // want k_r[i] * e^{Glast - G_r}; sKb holds raw k
                        const float f = __expf(sG[GDN_CHUNK - 1] - sG[r]);
                        acc = fmaf(sKb[r * GDN_DK + i] * f, sKbS[r * VW + jb + b], acc);
                    }
                    Sr[a][b] = acc;
                }
        }
        // End-of-chunk barrier: the next iteration overwrites sBeta/sG/sKb.
        __syncthreads();
    }

    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b] = Sr[a][b];
}

// Negative-control build; never defined by build.rs.
#ifdef GDN_DUMP_PARTS
#define GDN_DUMP_ARGS , nullptr, nullptr
#else
#define GDN_DUMP_ARGS
#endif

// VSPLIT is fixed at setup by the prepared plan.
extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_chunked_core_v1(const float* q, const float* k, const float* v,
                        const float* beta, const float* g, float* S,
                        __nv_bfloat16* out, const GdnMeta* meta, int valid_rows) {
    gdn_chunked_core_t<1>(q, k, v, beta, g, S, out, meta, valid_rows, nullptr GDN_DUMP_ARGS);
}

// Negative-control build; never defined by build.rs.
#ifdef GDN_DUMP_PARTS
// Additionally writes the output phase's inter and attn terms before rounding.
extern "C" __global__ void gdn_chunked_core_v1_dump(const float* q, const float* k, const float* v,
    const float* beta, const float* g, float* S, __nv_bfloat16* out,
    const GdnMeta* meta, int valid_rows, float* dump_inter, float* dump_attn) {
    gdn_chunked_core_t<1>(q, k, v, beta, g, S, out, meta, valid_rows, nullptr, dump_inter, dump_attn);
}
#endif

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_chunked_core_snapshot(const float* q, const float* k, const float* v,
                              const float* beta, const float* g, float* S,
                              __nv_bfloat16* out, const GdnMeta* meta, int valid_rows,
                              float* snapshots) {
    gdn_chunked_core_t<1, true>(q, k, v, beta, g, S, out, meta, valid_rows, snapshots GDN_DUMP_ARGS);
}
