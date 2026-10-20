// GDN chunkwise prefill, C = 64. Per head and chunk: sG = cumsum(g),
// sA = -tril(beta.K.K^T.exp(G_j-G_i),-1), sT = blocked inverse of (I-A) in 16x16 FP32
// leaves with 16->32->64 TC merges, W = bf16(T)@bf16(beta.K.e^G),
// qk = bf16(q)@bf16(K)^T.exp(G_j-G_i) tril inclusive; then per value strip
// U = bf16(T)@bf16(beta.V), v_new = U - bf16(W)@bf16(S), QS = bf16(q.e^G)@bf16(S),
// AD = bf16(qk)@bf16(v_new), out = BF16_RNE(QS+AD) and
// S = S.exp(G_last) + bf16(k.e^{G_last-G})^T@bf16(v_new).
// One BF16 RNE copy per GEMM operand for 48 value heads.
// Stage-H Q/K use BF16 high/residual terms for 32 value heads.
// FP32 accumulators and FP32 T leaves; __expf only
// on differences of non-positive log-cumulatives. BM = C = 64, BN = VW = 128/s, BK = 16,
// 256 threads = 8 warps, S[:,j0:j0+VW] in registers, one CTA per SM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "../q38_contract.cuh"

#include "gdn_cw_p1_layout.h"

#define P1_C       64     // chunk rows
#define P1_SUB     16     // UT inverse diagonal block
#define P1_THREADS 256
#define P1_WARPS   (P1_THREADS / 32)
#define P1_DK      128
#define P1_DV      128
#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define P1_HEADS GDN_HEADS
#define P1_MT      (P1_C / 16)          // row tiles of 16
#define P1_NT_C    (P1_C / 8)           // column tiles of 8 in the [C][C] products
#define P1_VL      72                    // sV row stride for the CHUNK axis, bf16 (+8 off 64)
#define P1_VLDK    136                   // sV row stride for the d_k axis, bf16 (+8 off 128)
#define P1_A_LD    68                    // sA row stride, floats (+4 off 64)
#define P1_T_LD    72                    // sT row stride, bf16 (+8 off 64)

// Shared arena, in order: sA | sT | sV | sVn | sG | sBeta | sSc; sV is [output column] x
// [reduction] (P1_VL, p1_b_sbf) and the region map is in gdn_cw_p1_layout.h.
#define P1_SMEM_BYTES(VW)                                                        \
    (P1_C * P1_A_LD * 4 + P1_C * P1_T_LD * 2 + P1_C * P1_VL * 2                 \
     + P1_C * (VW) * 4 + 3 * P1_C * 4)

#include "gdn_cw_frag.cuh"

// A from FP32 rows, one scale per row (beta, and the q liveness mask): register 1 is the
// +8 row at the low columns and register 2 the base row at the +8 columns.
template<bool Residual = false>
__device__ __forceinline__ AFrag p1_a_rows(const float* lo, const float* hi, int col0,
                                           int t2, float s_lo, float s_hi) {
    AFrag f;
    f.r[0] = p1_pack<Residual>(lo[col0 + t2] * s_lo, lo[col0 + t2 + 1] * s_lo);
    f.r[1] = p1_pack<Residual>(hi[col0 + t2] * s_hi, hi[col0 + t2 + 1] * s_hi);
    f.r[2] = p1_pack<Residual>(lo[col0 + t2 + 8] * s_lo, lo[col0 + t2 + 9] * s_lo);
    f.r[3] = p1_pack<Residual>(hi[col0 + t2 + 8] * s_hi, hi[col0 + t2 + 9] * s_hi);
    return f;
}

// B from a row-major FP32 buffer: fragment row n, columns 2tig + {0,1} and +8.
__device__ __forceinline__ BFrag p1_b_rows(const float* p, int ld, int n, int col0, int t2,
                                           float s0, float s1, float s8, float s9) {
    const float* r = p + (size_t)n * ld;
    BFrag f;
    f.r[0] = p1_pack(r[col0 + t2] * s0, r[col0 + t2 + 1] * s1);
    f.r[1] = p1_pack(r[col0 + t2 + 8] * s8, r[col0 + t2 + 9] * s9);
    return f;
}

template <int VW>
__device__ __forceinline__ void gdn_chunkwise_p1_t(
        const float* __restrict__ q,      // FP32 [M, HEADS, DK], pre-scaled by 1/sqrt(DK)
        const float* __restrict__ k,      // FP32 [M, HEADS, DK]
        const float* __restrict__ v,      // FP32 [M, HEADS, DV]
        const float* __restrict__ beta,   // FP32 [M, HEADS]
        const float* __restrict__ g,      // FP32 [M, HEADS]
        float* __restrict__ S,            // FP32 [HEADS, DK, DV]
        __nv_bfloat16* __restrict__ out,  // BF16 [M, HEADS, DV]
        const Q38SequenceFields* __restrict__ meta,
        int valid_rows_in) {
    static_assert(VW == 8 || VW == 16 || VW == 32,
                  "the value strip must be DV/4, DV/8 or DV/16 (protocol §A s = 4/8/16)");
    static_assert(P1_SMEM_BYTES(VW) == (VW == 32 ? GDN_CW_P1_S4_BYTES
                                       : VW == 16 ? GDN_CW_P1_S8_BYTES
                                                  : GDN_CW_P1_S16_BYTES),
                  "kernel arena and gdn_cw_p1_layout.h disagree on the shared-memory size");
    constexpr int NT = VW / 8;      // 8-column value tiles
    constexpr int VLD = P1_VL;      // sV row stride in bf16; sV is staged [value][reduction]

    const int head = blockIdx.y;
    const int j0 = blockIdx.x * VW;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = CW_GID(lane);
    const int t2 = CW_T2(lane);

    const bool aligned = (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k)
                          | reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(beta)
                          | reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(S)
                          | reinterpret_cast<uintptr_t>(out)) % 16 == 0;
    if (blockDim.x != P1_THREADS || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || gridDim.y != P1_HEADS || gridDim.x * VW != P1_DV || !aligned) {
        __trap();
        return;
    }
    const int prefix = meta->prefix;
    const int valid_rows = valid_rows_in > 0 ? valid_rows_in : meta->valid_rows;
    if (valid_rows <= 0 || prefix < 0 || meta->causal != 1
        || meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap();
        return;
    }

    extern __shared__ __align__(16) char p1_smem[];
    float* const sA    = reinterpret_cast<float*>(p1_smem);                    // [C][A_LD]
    __nv_bfloat16* const sT = reinterpret_cast<__nv_bfloat16*>(sA + P1_C * P1_A_LD);  // [C][T_LD]
    __nv_bfloat16* const sV = sT + P1_C * P1_T_LD;                             // [C][VLD]
    float* const sVn  = reinterpret_cast<float*>(sV + P1_C * VLD);             // [C][VW]
    // sVn doubles as the 16->32 merge's second-product BF16 staging: the fp32
    // intermediate takes its first 1024 B, the transposed T[a,a] the second half.
    __nv_bfloat16* const sVnB = reinterpret_cast<__nv_bfloat16*>(sVn);
    float* const sG   = sVn + P1_C * VW;
    float* const sBeta = sG + P1_C;
    float* const sSc  = sBeta + P1_C;

    const size_t rstride = (size_t)P1_HEADS * P1_DK;
    const float* const kh = k + (size_t)head * P1_DK;
    const float* const qh = q + (size_t)head * P1_DK;
    const float* const vh = v + (size_t)head * P1_DV + j0;

    // S[:, j0:j0+VW] stays in registers for the whole call; every index into Sr is
    // compile-time.
    float Sr[2][NT][2];
    {
        // Warp w owns d_k rows 16w..16w+15.
        float* srow = S + (size_t)head * P1_DK * P1_DV
                    + (size_t)warp * 16 * P1_DV + j0 + gid * P1_DV + t2;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                Sr[0][nt][c] = srow[8 * nt + c];
                Sr[1][nt][c] = srow[8 * nt + c + 8 * P1_DV];
            }
    }

    const int nchunk = (valid_rows + P1_C - 1) / P1_C;
    for (int c = 0; c < nchunk; ++c) {
        const int row0 = c * P1_C;
        // ---- gates and the in-chunk cumsum; the in-place scan runs on one thread.
        for (int r = tid; r < P1_C; r += P1_THREADS) {
            const int gr = row0 + r;
            sBeta[r] = gr < valid_rows ? beta[(size_t)gr * P1_HEADS + head] : 0.0f;
            sG[r]   = gr < valid_rows ? g[(size_t)gr * P1_HEADS + head] : 0.0f;
        }
        __syncthreads();
        if (tid == 0)
            for (int r = 1; r < P1_C; ++r) sG[r] += sG[r - 1];
        __syncthreads();
        for (int r = tid; r < P1_C; r += P1_THREADS)
            sSc[r] = sBeta[r] * __expf(sG[r]);

        // ---- A = -tril( (bf16(beta.K) @ bf16(K)^T) . exp(G_j - G_i), -1 ) -> sA
        __syncthreads();
        for (int tile = warp; tile < P1_MT * P1_NT_C; tile += P1_WARPS) {
            const int mt = tile / P1_NT_C, nt = tile - mt * P1_NT_C;
            const int cl = mt * 16 + gid, ch = cl + 8, bn = nt * 8 + gid;
            const float* al = kh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
            const float* ah = kh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
            const float* bp = kh + (size_t)((row0 + bn) < valid_rows ? row0 + bn : row0) * rstride;
            const float sl = sBeta[cl], sh = sBeta[ch];     // zero on a dead row
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P1_DK / 16; ++ks) {
                const AFrag a = p1_a_rows(al, ah, ks * 16, t2, sl, sh);
                const BFrag b = p1_b_rows(bp, P1_DK, 0, ks * 16, t2, 1.0f, 1.0f, 1.0f, 1.0f);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();
        // tril is applied again after the exp: exp of a masked entry is 1.0.
        for (int idx = tid; idx < P1_C * P1_C; idx += P1_THREADS) {
            const int j = idx / P1_C, i = idx - j * P1_C;
            sA[j * P1_A_LD + i] = j > i ? -sA[j * P1_A_LD + i] * __expf(sG[j] - sG[i]) : 0.0f;
        }
        __syncthreads();

        // ---- T = (I - A)^-1, blocked: 16x16 FP32 leaves by forward substitution, then
        // 16->32 and 32->64 merges. One row per iteration with a barrier between rows, one
        // thread per diagonal block; each sA row has one writer.
        for (int rr = 1; rr < P1_SUB; ++rr) {
            if (tid < P1_C / P1_SUB) {
                const int blk = tid;
                // Columns are within the diagonal block's column window c0.
                const int c0 = blk * P1_SUB;
                float* row = sA + (size_t)(blk * P1_SUB + rr) * P1_A_LD + c0;
                for (int j = 0; j < rr; ++j) {
                    float acc = row[j];
                    #pragma unroll
                    for (int m = j + 1; m < P1_SUB; ++m) {
                        if (m < rr)
                            acc = fmaf(row[m],
                                       sA[(size_t)(blk * P1_SUB + m) * P1_A_LD + c0 + j],
                                       acc);
                    }
                    row[j] = acc;
                }
            }
            __syncthreads();
        }
        for (int r = tid; r < P1_C; r += P1_THREADS) sA[r * P1_A_LD + r] += 1.0f;
        __syncthreads();   // the encode reads all of sA, including the incremented diagonal
        for (int idx = tid; idx < P1_C * P1_C; idx += P1_THREADS) {
            const int r = idx / P1_C, cc = idx - r * P1_C;
            sT[r * P1_T_LD + cc] = __float2bfloat16_rn(sA[r * P1_A_LD + cc]);
        }
        __syncthreads();

        // ---- merges: L = I - A gives L21 = -A21, so each level is
        // T[b,a] = T[b,b] @ A[b,a] @ T[a,a] with no leading sign.
        for (int p = 0; p < P1_C / (2 * P1_SUB); ++p) {
            const int a0 = p * 2 * P1_SUB, b0 = a0 + P1_SUB;
            __syncthreads();
            // Merge operands are staged as [output column][reduction]: A[b,a] as [a][b] and
            // T[a,a] as [a'][a]. The off-diagonal block of sA still holds the original A.
            for (int idx = tid; idx < P1_SUB * P1_SUB; idx += P1_THREADS) {
                const int bl = idx / P1_SUB, al = idx - bl * P1_SUB;
                sV[al * 32 + bl] = __float2bfloat16_rn(
                    sA[(size_t)(b0 + bl) * P1_A_LD + (a0 + al)]);
            }
            __syncthreads();
            if (warp < 2) {                       // 2 column tiles of a 16-wide block
                const int nt = warp;              // tile a in [nt*8, nt*8+8)
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const AFrag a = p1_a_sbf(sT, P1_T_LD, b0, b0, gid, t2);
                const BFrag b = p1_b_sbf(sV, 32, gid + nt * 8, 0, t2);
                CW_MMA(d, a, b);
                p1_store_f32(sVn, P1_SUB, 0, nt * 8, gid, t2, d);
            }
            __syncthreads();
            for (int idx = tid; idx < P1_SUB * P1_SUB; idx += P1_THREADS) {
                const int al = idx / P1_SUB, bl = idx - al * P1_SUB;
                // memory[n=a][k=a'] holds T11[a'][a]: destination [al][bl], source
                // [b=a'][a=al] of sT.
                sVnB[512 + al * 32 + bl] = __float2bfloat16_rn(
                    sT[(size_t)(a0 + bl) * P1_T_LD + (a0 + al)]);
            }
            __syncthreads();
            if (warp < 2) {
                const int nt = warp;              // tile a in [nt*8, nt*8+8)
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const AFrag a = p1_a_sf32(sVn, P1_SUB, 0, 0, gid, t2);
                const BFrag b = p1_b_sbf(sVnB + 512, 32, gid + nt * 8, 0, t2);
                CW_MMA(d, a, b);
                p1_store_bf16(sT, P1_T_LD, b0, a0 + nt * 8, gid, t2, d);
            }
            __syncthreads();
        }
        {   // 32 -> 64. The (x@y) intermediate goes to sA[0:32][32:64], strictly ABOVE
            // A's lower triangle and so disjoint from A[32:64][0:32].
            const int mt = warp / 4, nt = warp - (warp / 4) * 4;
            // A[32:64][0:32] transposed into sV as [a][b].
            for (int idx = tid; idx < 32 * 32; idx += P1_THREADS) {
                const int bl = idx / 32, al = idx - bl * 32;
                sV[al * 64 + bl] = __float2bfloat16_rn(sA[(size_t)(32 + bl) * P1_A_LD + al]);
            }
            __syncthreads();
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            // Both products reduce over a 32-wide axis: two 16-wide k-steps.
            #pragma unroll
            for (int ks = 0; ks < P1_C / (2 * P1_SUB); ++ks) {
                const AFrag a = p1_a_sbf(sT, P1_T_LD, 32 + mt * 16, 32 + ks * 16, gid, t2);
                // The staged tile is indexed by the local b.
                const BFrag b = p1_b_sbf(sV, 64, nt * 8 + gid, ks * 16, t2);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, 32 + nt * 8, gid, t2, d);
            __syncthreads();
            for (int idx = tid; idx < 32 * 32; idx += P1_THREADS) {
                const int bl = idx / 32, al = idx - bl * 32;
                sV[al * 64 + 32 + bl] = __float2bfloat16_rn(
                    sT[(size_t)(bl) * P1_T_LD + al]);
            }
            __syncthreads();
            float e[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P1_C / (2 * P1_SUB); ++ks) {
                const AFrag a = p1_a_sf32(sA, P1_A_LD, mt * 16, 32 + ks * 16, gid, t2);
                const BFrag b = p1_b_sbf(sV, 64, nt * 8 + gid, 32 + ks * 16, t2);
                CW_MMA(e, a, b);
            }
            p1_store_bf16(sT, P1_T_LD, 32 + mt * 16, nt * 8, gid, t2, e);
            __syncthreads();
        }

        // ---- U = T @ (beta . V) -> sVn.
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            const int gr = row0 + r;
            sV[cc * VLD + r] = __float2bfloat16_rn(
                gr < valid_rows ? vh[(size_t)gr * rstride + cc] * sBeta[r] : 0.0f);
        }
        __syncthreads();
        for (int tile = warp; tile < P1_MT * NT; tile += P1_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P1_C / 16; ++ks) {
                const AFrag a = p1_a_sbf(sT, P1_T_LD, mt * 16, ks * 16, gid, t2);
                const BFrag b = p1_b_sbf(sV, VLD, nt * 8 + gid, ks * 16, t2);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sVn, VW, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();

        // ---- WS = W @ S: W is rounded to bf16 and multiplies a bf16 copy of S. Each warp
        // stages its own 16-row d_k slice; sV holds the whole d_k axis as [value][d_k].
        const int kblk = warp;           // the dk window this warp's S rows belong to
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                sV[(8 * nt + t2 + cc) * P1_VLDK + kblk * 16 + gid] =
                    __float2bfloat16_rn(Sr[0][nt][cc]);
                sV[(8 * nt + t2 + cc) * P1_VLDK + kblk * 16 + gid + 8] =
                    __float2bfloat16_rn(Sr[1][nt][cc]);
            }
        __syncthreads();
        // The owning warp of a chunk-row tile contracts the whole d_k axis over ks = 0..7;
        // every output element has one writer.
        if (warp < P1_MT) {
            const int mt = warp;
            float d[NT][4] = {{0.0f}};
            #pragma unroll 2
            for (int ks = 0; ks < P1_DK / 16; ++ks) {
                float w0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float w1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll 1
                for (int cs = 0; cs < P1_C / 16; ++cs) {
                    const AFrag a = p1_a_sbf(sT, P1_T_LD, mt * 16, cs * 16, gid, t2);
                    // W's B fragment reads the input TRANSPOSED (fragment row = d_k
                    // index, columns = chunk rows), over the k window [cs*16, cs*16+16).
                    const int c0 = cs * 16 + t2;
                    const int dk0 = ks * 16 + gid;
                    const BFrag b0{ {p1_pack(p1_kbg(kh, rstride, row0, valid_rows, c0, dk0, sSc),
                                             p1_kbg(kh, rstride, row0, valid_rows, c0 + 1, dk0, sSc)),
                                    p1_pack(p1_kbg(kh, rstride, row0, valid_rows, c0 + 8, dk0, sSc),
                                            p1_kbg(kh, rstride, row0, valid_rows, c0 + 9, dk0, sSc))} };
                    const BFrag b1{ {p1_pack(p1_kbg(kh, rstride, row0, valid_rows, c0, dk0 + 8, sSc),
                                             p1_kbg(kh, rstride, row0, valid_rows, c0 + 1, dk0 + 8, sSc)),
                                    p1_pack(p1_kbg(kh, rstride, row0, valid_rows, c0 + 8, dk0 + 8, sSc),
                                            p1_kbg(kh, rstride, row0, valid_rows, c0 + 9, dk0 + 8, sSc))} };
                    CW_MMA(w0, a, b0);
                    CW_MMA(w1, a, b1);
                }
                // Same register order as p1_a_rows; w0 is the first 8-wide d_k group of
                // the window, w1 the second.
                const AFrag wa{ {p1_pack(w0[0], w0[1]), p1_pack(w0[2], w0[3]),
                                 p1_pack(w1[0], w1[1]), p1_pack(w1[2], w1[3])} };
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const BFrag b = p1_b_sbf(sV, P1_VLDK, nt * 8 + gid, ks * 16, t2);
                    CW_MMA(d[nt], wa, b);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                p1_store_f32(sA, P1_A_LD, mt * 16, nt * 8, gid, t2, d[nt]);
        }

        // ---- v_new = U - WS, in place in sVn
        __syncthreads();
        __syncthreads();
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            sVn[r * VW + cc] -= sA[r * P1_A_LD + cc];
        }

        // ---- qk = (bf16(q) @ bf16(K)^T) . exp(G_j - G_i), tril with the diagonal, into sA.
        __syncthreads();
        for (int tile = warp; tile < P1_MT * P1_NT_C; tile += P1_WARPS) {
            const int mt = tile / P1_NT_C, nt = tile - mt * P1_NT_C;
            const int cl = mt * 16 + gid, ch = cl + 8, bn = nt * 8 + gid;
            const float* al = qh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
            const float* ah = qh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
            const float* bp = kh + (size_t)((row0 + bn) < valid_rows ? row0 + bn : row0) * rstride;
            const float sl = (row0 + cl) < valid_rows ? 1.0f : 0.0f;
            const float sh = (row0 + ch) < valid_rows ? 1.0f : 0.0f;
            const float sb = (row0 + bn) < valid_rows ? 1.0f : 0.0f;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P1_DK / 16; ++ks) {
                const AFrag a = p1_a_rows(al, ah, ks * 16, t2, sl, sh);
                const BFrag b = p1_b_rows(bp, P1_DK, 0, ks * 16, t2, sb, sb, sb, sb);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();
        for (int idx = tid; idx < P1_C * P1_C; idx += P1_THREADS) {
            const int j = idx / P1_C, i = idx - j * P1_C;
            sA[j * P1_A_LD + i] = j >= i ? sA[j * P1_A_LD + i] * __expf(sG[j] - sG[i]) : 0.0f;
        }
        __syncthreads();
        for (int idx = tid; idx < P1_C * P1_C; idx += P1_THREADS) {
            const int r = idx / P1_C, cc = idx - r * P1_C;
            sT[r * P1_T_LD + cc] = __float2bfloat16_rn(sA[r * P1_A_LD + cc]);
        }

        // ---- v_new as bf16, for AD and for the state update
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            sV[cc * VLD + r] = __float2bfloat16_rn(sVn[r * VW + cc]);
        }
        __syncthreads();

        // ---- AD = qk @ v_new, accumulating into sA (the output tile)
        for (int tile = warp; tile < P1_MT * NT; tile += P1_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P1_C / 16; ++ks) {
                const AFrag a = p1_a_sbf(sT, P1_T_LD, mt * 16, ks * 16, gid, t2);
                const BFrag b = p1_b_sbf(sV, VLD, nt * 8 + gid, ks * 16, t2);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();

        // ---- QS = (q . e^G) @ S onto AD's tile; S is re-staged into sV.
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                sV[(8 * nt + t2 + cc) * P1_VLDK + kblk * 16 + gid] =
                    __float2bfloat16_rn(Sr[0][nt][cc]);
                sV[(8 * nt + t2 + cc) * P1_VLDK + kblk * 16 + gid + 8] =
                    __float2bfloat16_rn(Sr[1][nt][cc]);
            }
        __syncthreads();
        // The owning warp contracts the full d_k axis onto AD's tile.
        for (int tile = warp; tile < P1_MT * NT; tile += P1_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            const int cl = mt * 16 + gid, ch = cl + 8;
            const float* al = qh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
            const float* ah = qh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
            const float sl = (row0 + cl) < valid_rows ? __expf(sG[cl]) : 0.0f;
            const float sh = (row0 + ch) < valid_rows ? __expf(sG[ch]) : 0.0f;
            float* o = sA + (size_t)(mt * 16 + gid) * P1_A_LD + nt * 8 + t2;
            float d[4] = {o[0], o[1], o[8 * P1_A_LD], o[8 * P1_A_LD + 1]};
            #pragma unroll
            for (int ks = 0; ks < P1_DK / 16; ++ks) {
                const AFrag a = p1_a_rows(al, ah, ks * 16, t2, sl, sh);
                const BFrag b = p1_b_sbf(sV, P1_VLDK, nt * 8 + gid, ks * 16, t2);
                CW_MMA(d, a, b);
#if GDN_HEADS == 32
                const AFrag residual = p1_a_rows<true>(al, ah, ks * 16, t2, sl, sh);
                CW_MMA(d, residual, b);
#endif
            }
            o[0] = d[0];
            o[1] = d[1];
            o[8 * P1_A_LD] = d[2];
            o[8 * P1_A_LD + 1] = d[3];
        }
        __syncthreads();

        // ---- out = BF16_RNE(QS + AD)
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            const int gr = row0 + r;
            if (gr < valid_rows)
                out[((size_t)gr * P1_HEADS + head) * P1_DV + j0 + cc] =
                    __float2bfloat16_rn(sA[r * P1_A_LD + cc]);
        }

        // ---- S = S . exp(G_last) + (k . e^{G_last-G})^T @ v_new; the A operand is the
        // TRANSPOSE (fragment rows are d_k indices), so the scale is per column here.
        __syncthreads();
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            sV[cc * VLD + r] = __float2bfloat16_rn(sVn[r * VW + cc]);
        }
        __syncthreads();
        {
            const float decay = __expf(sG[P1_C - 1]);
            const float gl = sG[P1_C - 1];
            // Every o[nt] contracts over ascending ks, then all Sr rows update in one
            // sweep. NT is 1/2/4 for s = 16/8/4.
            float o[NT][4] = {{0.0f}};
            #pragma unroll
            for (int ks = 0; ks < P1_C / 16; ++ks) {
                // ks*16 is the chunk-row offset of the 16-wide k-step.
                const AFrag a = p1_a_kdec(kh, rstride, row0, valid_rows, sG, gl,
                                          warp * 16 + gid, ks * 16 + t2);
#if GDN_HEADS == 32
                const AFrag residual = p1_a_kdec<true>(kh, rstride, row0, valid_rows, sG, gl,
                                                      warp * 16 + gid, ks * 16 + t2);
#endif
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const BFrag b = p1_b_sbf(sV, VLD, nt * 8 + gid, ks * 16, t2);
                    CW_MMA(o[nt], a, b);
#if GDN_HEADS == 32
                    CW_MMA(o[nt], residual, b);
#endif
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    Sr[0][nt][cc] = fmaf(decay, Sr[0][nt][cc], o[nt][cc]);
                    Sr[1][nt][cc] = fmaf(decay, Sr[1][nt][cc], o[nt][cc + 2]);
                }
            }
        }
        __syncthreads();   // the next chunk rewrites sG/sBeta/sSc while slow threads read them
    }

    {
        // gid is a d_k ROW here, so it scales by P1_DV; t2 is the value column.
        float* srow = S + (size_t)head * P1_DK * P1_DV
                    + (size_t)(warp * 16 + gid) * P1_DV + j0 + t2;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                srow[8 * nt + c] = Sr[0][nt][c];
                srow[8 * nt + c + 8 * P1_DV] = Sr[1][nt][c];
            }
    }
}

// Value strip splits s = 4/8/16 (VW = 32/16/8).
extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_core_s4(const float* q, const float* k, const float* v,
                              const float* beta, const float* g, float* S,
                              __nv_bfloat16* out, const Q38SequenceFields* meta, int valid_rows) {
    gdn_chunkwise_p1_t<32>(q, k, v, beta, g, S, out, meta, valid_rows);
}

extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_core_s8(const float* q, const float* k, const float* v,
                              const float* beta, const float* g, float* S,
                              __nv_bfloat16* out, const Q38SequenceFields* meta, int valid_rows) {
    gdn_chunkwise_p1_t<16>(q, k, v, beta, g, S, out, meta, valid_rows);
}

extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_core_s16(const float* q, const float* k, const float* v,
                               const float* beta, const float* g, float* S,
                               __nv_bfloat16* out, const Q38SequenceFields* meta, int valid_rows) {
    gdn_chunkwise_p1_t<8>(q, k, v, beta, g, S, out, meta, valid_rows);
}
