// GDN chunkwise prefill as two kernels, C = 64, with FP32 accumulators.
// Stage H uses BF16 high/residual Q/K terms for 32 value heads.
//
//   Q/K sources are expanded FP32 [M,HV,128], with Q already scaled.
//   beta and g are FP32.
//
//   stage A  grid (nchunk, P2_HEADS): gates and the cumsum, A = -tril(...), the blocked
//             (I-A)^-1 with both merge levels, U = T@(beta.V) for all DV columns,
//             W = T@(beta.K.e^G) materialized bf16, qk, and the FP32[64] vectors
//             Gc, Be, Sc.
//   stage H  grid (4, P2_HEADS), 32-wide value strips: the chunk loop in order with
//             S[:, j0:j0+32] in registers; WS = W@S, v_new = U - WS, out, and the
//             state update. S is written back at the end.
//
// The workspace slot of one (chunk, head) is GDN_CW_P2_SLOT_BYTES at
// ws + (chunk * P2_HEADS + head) * GDN_CW_P2_SLOT_BYTES, in this byte order:
//   P2_OFF_W   0      W  bf16[64][128]  row stride 128   A writes, H reads
//   P2_OFF_U   16384  U  fp32[64][128]  row stride 128   A writes, H reads
//   P2_OFF_QK  49152  qk bf16[64][72]   row stride 72    A writes, H reads
//   P2_OFF_GC  58368  Gc fp32[64]                        A writes, H reads
//   P2_OFF_BE  58624  Be fp32[64]                        A writes
//   P2_OFF_SC  58880  Sc fp32[64]                        A writes
// The qk row stride is P1_T_LD so p1_a_sbf reads it in place.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "../q38_contract.cuh"

#include "gdn_cw_p1_layout.h"
#include "gdn_cw_p2_layout.h"

#define P2_C       64     // chunk rows
#define P2_SUB     16     // UT inverse diagonal block
#define P2_THREADS 256
#define P2_WARPS   (P2_THREADS / 32)
#define P2_DK      128
#define P2_DV      128
#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define P2_HEADS GDN_HEADS
#define P2_MT      (P2_C / 16)          // row tiles of 16
#define P2_NT_C    (P2_C / 8)           // column tiles of 8 in the [C][C] products
#define P2_VL      72                    // sV row stride for the CHUNK axis, bf16 (+8 off 64)
#define P2_VLDK    136                   // sV row stride for the d_k axis, bf16 (+8 off 128)
#define P2_A_LD    68                    // sA row stride, floats (+4 off 64)
#define P2_T_LD    72                    // sT row stride, bf16 (+8 off 64)

// One arena for both stages, in order: sA | sT | sV | sVn | sG | sBeta | sSc.
#ifdef GDN_COMPACT_QK
#define P2_QK_ROW (16 * P2_DK)
#define P2_SOURCE_HEAD(h) ((h) / 3)
using AQuery = __nv_bfloat16;
using AKey = __nv_bfloat16;
#else
#define P2_QK_ROW (P2_HEADS * P2_DK)
#define P2_SOURCE_HEAD(h) (h)
using AQuery = float;
using AKey = float;
#endif
#define P2_QUERY_SCALE 1.0f

#define P2_SMEM_BYTES GDN_CW_P2_SMEM_BYTES

static_assert(P2_SMEM_BYTES == P2_C * P2_A_LD * 4 + P2_C * P2_T_LD * 2 + P2_C * P2_VL * 2
                  + P2_C * 32 * 4 + 3 * P2_C * 4,
              "kernel arena and gdn_cw_p2_layout.h disagree on the shared-memory size");
static_assert(GDN_CW_P2_SLOT_BYTES == 16384 + 32768 + 9216 + 3 * 256,
              "workspace slot size disagrees with the field offsets below");

#define P2_SLOT_BYTES GDN_CW_P2_SLOT_BYTES
#define P2_OFF_W      0
#define P2_OFF_U      16384
#define P2_OFF_QK     49152
#define P2_OFF_GC     58368
#define P2_OFF_BE     58624
#define P2_OFF_SC     58880

// Stage H value strip width and stage A U column tile width.
#define P2_VW        32
#define P2_VT        8

#include "gdn_cw_frag.cuh"

// A from Q/K source rows, one scale per row (beta, and the q liveness mask): register 1 is the
// +8 row at the low columns and register 2 the base row at the +8 columns. `qscale` is
// 1.0f for the prepared Q/K sources, applied before the row scale.
template<bool Residual = false, typename T>
__device__ __forceinline__ AFrag p1_a_rows(const T* lo, const T* hi,
                                           int col0, int t2, float qscale,
                                           float s_lo, float s_hi) {
    AFrag f;
    f.r[0] = p1_pack<Residual>((float)lo[col0 + t2] * qscale * s_lo,
                     (float)lo[col0 + t2 + 1] * qscale * s_lo);
    f.r[1] = p1_pack<Residual>((float)hi[col0 + t2] * qscale * s_hi,
                     (float)hi[col0 + t2 + 1] * qscale * s_hi);
    f.r[2] = p1_pack<Residual>((float)lo[col0 + t2 + 8] * qscale * s_lo,
                     (float)lo[col0 + t2 + 9] * qscale * s_lo);
    f.r[3] = p1_pack<Residual>((float)hi[col0 + t2 + 8] * qscale * s_hi,
                     (float)hi[col0 + t2 + 9] * qscale * s_hi);
    return f;
}

// B from a row-major Q/K source: fragment row n, columns 2tig + {0,1} and +8.
template<typename T>
__device__ __forceinline__ BFrag p1_b_rows(const T* p, int ld, int n, int col0,
                                           int t2, float s0, float s1, float s8, float s9) {
    const T* r = p + (size_t)n * ld;
    BFrag f;
    f.r[0] = p1_pack((float)r[col0 + t2] * s0, (float)r[col0 + t2 + 1] * s1);
    f.r[1] = p1_pack((float)r[col0 + t2 + 8] * s8,
                     (float)r[col0 + t2 + 9] * s9);
    return f;
}

// The arena is sA | sT | sV | sVn | sG | sBeta | sSc; sV is [output column] x
// [reduction] (P2_VL, p1_b_sbf).
struct P2Arena {
    float* sA;
    __nv_bfloat16* sT;
    __nv_bfloat16* sV;
    float* sVn;
    float* sG;
    float* sBeta;
    float* sSc;
};

__device__ __forceinline__ P2Arena p2_arena(char* smem) {
    P2Arena a;
    a.sA = reinterpret_cast<float*>(smem);
    a.sT = reinterpret_cast<__nv_bfloat16*>(a.sA + P2_C * P2_A_LD);
    a.sV = a.sT + P2_C * P2_T_LD;
    a.sVn = reinterpret_cast<float*>(a.sV + P2_C * P2_VL);
    // sVn doubles as the 16->32 merge's second-product BF16 staging: the fp32
    // intermediate takes its first 1024 B, the transposed T[a,a] the second half.
    a.sG = a.sVn + P2_C * P2_VW;
    a.sBeta = a.sG + P2_C;
    a.sSc = a.sBeta + P2_C;
    return a;
}

// The slot of one (chunk, head); slots are contiguous and in that order.
__device__ __forceinline__ char* p2_slot(char* ws, int chunk, int head) {
    return ws + ((size_t)chunk * P2_HEADS + (size_t)head) * P2_SLOT_BYTES;
}

// valid_rows_in == 0 selects meta->valid_rows; key_count == prefix + valid_rows bounds
// the rows a kernel may read.
__device__ __forceinline__ int p2_valid_rows(const Q38SequenceFields* meta, int valid_rows_in) {
    const int prefix = meta->prefix;
    const int valid_rows = valid_rows_in > 0 ? valid_rows_in : meta->valid_rows;
    if (valid_rows <= 0 || prefix < 0 || meta->causal != 1
        || meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap();
    }
    return valid_rows;
}


// Stage A: one CTA per (chunk, head); no S dependency. Grid x is ceil(max_rows / P2_C);
// CTAs past the valid chunks exit before the first barrier.
extern "C" __global__ __launch_bounds__(P2_THREADS, 1)
void gdn_cw_p2_stage_a(const AQuery* __restrict__ q,
                       const float* __restrict__ k,
                       const __nv_bfloat16* __restrict__ v,  // BF16 [M, HEADS, DV] VALUE heads
                       const float* __restrict__ beta,// FP32 [M, HEADS]
                       const float* __restrict__ g,   // FP32 [M, HEADS]
                       char* __restrict__ ws,
                       const Q38SequenceFields* __restrict__ meta, int valid_rows_in,
                       uint32_t v_row_stride
#ifdef GDN_COMPACT_QK
                       , const __nv_bfloat16* __restrict__ k_bf16
#endif
                       ) {
    const int head = blockIdx.y;
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = CW_GID(lane);
    const int t2 = CW_T2(lane);

    const bool aligned = (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k)
                          | reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(beta)
                          | reinterpret_cast<uintptr_t>(g)
                          | reinterpret_cast<uintptr_t>(ws)) % 256 == 0;
    if (v_row_stride <= 0) __trap();
    if (blockDim.x != P2_THREADS || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || gridDim.y != P2_HEADS || !aligned) {
        __trap();
        return;
    }
    const int valid_rows = p2_valid_rows(meta, valid_rows_in);
    const int nchunk = (valid_rows + P2_C - 1) / P2_C;
    if (chunk >= nchunk) return;
    const int row0 = chunk * P2_C;
#ifdef GDN_COMPACT_QK
    if (!k_bf16 || reinterpret_cast<uintptr_t>(k_bf16) % 256 != 0) { __trap(); return; }
#endif

    extern __shared__ __align__(16) char p2_smem[];
    const P2Arena ar = p2_arena(p2_smem);
    float* const sA = ar.sA;
    __nv_bfloat16* const sT = ar.sT;
    __nv_bfloat16* const sV = ar.sV;
    float* const sVn = ar.sVn;
    __nv_bfloat16* const sVnB = reinterpret_cast<__nv_bfloat16*>(sVn);
    float* const sG = ar.sG;
    float* const sBeta = ar.sBeta;
    float* const sSc = ar.sSc;

    // Q/K source rows follow P2_QK_ROW; V rows are v_row_stride elements apart.
    constexpr size_t rstride = P2_QK_ROW;
    const float* const kh = k + (size_t)P2_SOURCE_HEAD(head) * P2_DK;
    const AQuery* const qh = q + (size_t)P2_SOURCE_HEAD(head) * P2_DK;
#ifdef GDN_COMPACT_QK
    const AKey* const kb = k_bf16 + (size_t)P2_SOURCE_HEAD(head) * P2_DK;
#else
    const AKey* const kb = kh;
#endif
    const __nv_bfloat16* const vh = v + (size_t)head * P2_DV;
    const float qscale = P2_QUERY_SCALE;

    char* const slot = p2_slot(ws, chunk, head);

    // ---- gates and the in-chunk cumsum; the in-place scan runs on one thread.
    for (int r = tid; r < P2_C; r += P2_THREADS) {
        const int gr = row0 + r;
        sBeta[r] = gr < valid_rows ? beta[(size_t)gr * P2_HEADS + head] : 0.0f;
        sG[r]   = gr < valid_rows ? g[(size_t)gr * P2_HEADS + head] : 0.0f;
    }
    __syncthreads();
    if (tid == 0)
        for (int r = 1; r < P2_C; ++r) sG[r] += sG[r - 1];
    __syncthreads();
    for (int r = tid; r < P2_C; r += P2_THREADS)
        sSc[r] = sBeta[r] * __expf(sG[r]);

    // ---- A = -tril( (bf16(beta.K) @ bf16(K)^T) . exp(G_j - G_i), -1 ) -> sA
    __syncthreads();
    for (int tile = warp; tile < P2_MT * P2_NT_C; tile += P2_WARPS) {
        const int mt = tile / P2_NT_C, nt = tile - mt * P2_NT_C;
        const int cl = mt * 16 + gid, ch = cl + 8, bn = nt * 8 + gid;
        const float* al = kh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
        const float* ah = kh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
        const AKey* bp = kb + (size_t)((row0 + bn) < valid_rows ? row0 + bn : row0) * rstride;
        const float sl = sBeta[cl], sh = sBeta[ch];     // zero on a dead row
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ks = 0; ks < P2_DK / 16; ++ks) {
            const AFrag a = p1_a_rows(al, ah, ks * 16, t2, 1.0f, sl, sh);
            const BFrag b = p1_b_rows(bp, P2_DK, 0, ks * 16, t2, 1.0f, 1.0f, 1.0f, 1.0f);
            CW_MMA(d, a, b);
        }
        p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d);
    }
    __syncthreads();
    // tril is applied again after the exp: exp of a masked entry is 1.0.
    for (int idx = tid; idx < P2_C * P2_C; idx += P2_THREADS) {
        const int j = idx / P2_C, i = idx - j * P2_C;
        sA[j * P2_A_LD + i] = j > i ? -sA[j * P2_A_LD + i] * __expf(sG[j] - sG[i]) : 0.0f;
    }
    __syncthreads();

    // ---- T = (I - A)^-1, blocked: 16x16 FP32 leaves by forward substitution, then
    // 16->32 and 32->64 merges. One warp per diagonal block, lane j = row j; row r reads
    // the final rows m < r through __shfl_sync(lw, t[j], m) in an ascending-m fma chain.
    // All lanes in lw execute every shuffle; lanes 16..31 do not participate.
    {
        const unsigned lw = (1u << P2_SUB) - 1u;   // the 16 leaf lanes
        if (warp < P2_C / P2_SUB && lane < P2_SUB) {
            // Columns are within the diagonal block's column window c0.
            const int c0 = warp * P2_SUB;
            float t[P2_SUB];
            #pragma unroll
            for (int j = 0; j < P2_SUB; ++j)
                t[j] = sA[(size_t)(warp * P2_SUB + lane) * P2_A_LD + c0 + j];
            __syncwarp(lw);
            for (int step = 1; step < P2_SUB; ++step) {
                float acc[P2_SUB];
                if (lane == step)
                    #pragma unroll
                    for (int j = 0; j < step; ++j) acc[j] = t[j];
                #pragma unroll
                for (int j = 0; j < P2_SUB; ++j) {
                    if (j < step)
                        #pragma unroll
                        for (int m = j + 1; m < P2_SUB; ++m) {
                            if (m < step) {
                                const float tmj = __shfl_sync(lw, t[j], m);
                                if (lane == step) acc[j] = fmaf(t[m], tmj, acc[j]);
                            }
                        }
                }
                if (lane == step)
                    #pragma unroll
                    for (int j = 0; j < step; ++j) t[j] = acc[j];
                __syncwarp(lw);
            }
            #pragma unroll
            for (int j = 0; j < P2_SUB; ++j)
                sA[(size_t)(warp * P2_SUB + lane) * P2_A_LD + c0 + j] = t[j];
        }
    }
    __syncthreads();
    for (int r = tid; r < P2_C; r += P2_THREADS) sA[r * P2_A_LD + r] += 1.0f;
    __syncthreads();   // the encode reads all of sA, including the incremented diagonal
    for (int idx = tid; idx < P2_C * P2_C; idx += P2_THREADS) {
        const int r = idx / P2_C, cc = idx - r * P2_C;
        sT[r * P2_T_LD + cc] = __float2bfloat16_rn(sA[r * P2_A_LD + cc]);
    }
    __syncthreads();

    // ---- merges: L = I - A gives L21 = -A21, so each level is
    // T[b,a] = T[b,b] @ A[b,a] @ T[a,a] with no leading sign.
    for (int p = 0; p < P2_C / (2 * P2_SUB); ++p) {
        const int a0 = p * 2 * P2_SUB, b0 = a0 + P2_SUB;
        __syncthreads();
        // Merge operands are staged as [output column][reduction]: A[b,a] as [a][b] and
        // T[a,a] as [a'][a]. The off-diagonal block of sA still holds the original A.
        for (int idx = tid; idx < P2_SUB * P2_SUB; idx += P2_THREADS) {
            const int bl = idx / P2_SUB, al = idx - bl * P2_SUB;
            sV[al * 32 + bl] = __float2bfloat16_rn(
                sA[(size_t)(b0 + bl) * P2_A_LD + (a0 + al)]);
        }
        __syncthreads();
        if (warp < 2) {                       // 2 column tiles of a 16-wide block
            const int nt = warp;              // tile a in [nt*8, nt*8+8)
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const AFrag a = p1_a_sbf(sT, P2_T_LD, b0, b0, gid, t2);
            const BFrag b = p1_b_sbf(sV, 32, gid + nt * 8, 0, t2);
            CW_MMA(d, a, b);
            p1_store_f32(sVn, P2_SUB, 0, nt * 8, gid, t2, d);
        }
        __syncthreads();
        for (int idx = tid; idx < P2_SUB * P2_SUB; idx += P2_THREADS) {
            const int al = idx / P2_SUB, bl = idx - al * P2_SUB;
            // memory[n=a][k=a'] holds T11[a'][a]: destination [al][bl], source
            // [b=a'][a=al] of sT.
            sVnB[512 + al * 32 + bl] = __float2bfloat16_rn(
                sT[(size_t)(a0 + bl) * P2_T_LD + (a0 + al)]);
        }
        __syncthreads();
        if (warp < 2) {
            const int nt = warp;              // tile a in [nt*8, nt*8+8)
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const AFrag a = p1_a_sf32(sVn, P2_SUB, 0, 0, gid, t2);
            const BFrag b = p1_b_sbf(sVnB + 512, 32, gid + nt * 8, 0, t2);
            CW_MMA(d, a, b);
            p1_store_bf16(sT, P2_T_LD, b0, a0 + nt * 8, gid, t2, d);
        }
        __syncthreads();
    }
    {   // 32 -> 64. The (x@y) intermediate goes to sA[0:32][32:64], strictly ABOVE
        // A's lower triangle and so disjoint from A[32:64][0:32].
        const int mt = warp / 4, nt = warp - (warp / 4) * 4;
        // A[32:64][0:32] transposed into sV as [a][b].
        for (int idx = tid; idx < 32 * 32; idx += P2_THREADS) {
            const int bl = idx / 32, al = idx - bl * 32;
            sV[al * 64 + bl] = __float2bfloat16_rn(sA[(size_t)(32 + bl) * P2_A_LD + al]);
        }
        __syncthreads();
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        // Both products reduce over a 32-wide axis: two 16-wide k-steps.
        #pragma unroll
        for (int ks = 0; ks < P2_C / (2 * P2_SUB); ++ks) {
            const AFrag a = p1_a_sbf(sT, P2_T_LD, 32 + mt * 16, 32 + ks * 16, gid, t2);
            // The staged tile is indexed by the local b.
            const BFrag b = p1_b_sbf(sV, 64, nt * 8 + gid, ks * 16, t2);
            CW_MMA(d, a, b);
        }
        p1_store_f32(sA, P2_A_LD, mt * 16, 32 + nt * 8, gid, t2, d);
        __syncthreads();
        for (int idx = tid; idx < 32 * 32; idx += P2_THREADS) {
            const int bl = idx / 32, al = idx - bl * 32;
            sV[al * 64 + 32 + bl] = __float2bfloat16_rn(
                sT[(size_t)(bl) * P2_T_LD + al]);
        }
        __syncthreads();
        float e[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ks = 0; ks < P2_C / (2 * P2_SUB); ++ks) {
            const AFrag a = p1_a_sf32(sA, P2_A_LD, mt * 16, 32 + ks * 16, gid, t2);
            const BFrag b = p1_b_sbf(sV, 64, nt * 8 + gid, 32 + ks * 16, t2);
            CW_MMA(e, a, b);
        }
        p1_store_bf16(sT, P2_T_LD, 32 + mt * 16, nt * 8, gid, t2, e);
        __syncthreads();
    }

    // ---- U = T @ (beta . V), all DV columns in 16 tiles of P2_VT; each tile's fp32
    // accumulator goes through sVn to the ws U slot. Tiles split columns, not k.
    float* const wU = reinterpret_cast<float*>(slot + P2_OFF_U);
    for (int cb = 0; cb < P2_DV / P2_VT; ++cb) {
        for (int idx = tid; idx < P2_C * P2_VT; idx += P2_THREADS) {
            const int r = idx / P2_VT, cc = idx - r * P2_VT;
            const int gr = row0 + r;
            sV[cc * P2_VL + r] = __float2bfloat16_rn(
                gr < valid_rows
                    ? __bfloat162float(vh[(size_t)gr * (size_t)v_row_stride
                                            + cb * P2_VT + cc])
                          * sBeta[r]
                    : 0.0f);
        }
        __syncthreads();
        for (int tile = warp; tile < P2_MT; tile += P2_WARPS) {
            const int mt = tile;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P2_C / 16; ++ks) {
                const AFrag a = p1_a_sbf(sT, P2_T_LD, mt * 16, ks * 16, gid, t2);
                const BFrag b = p1_b_sbf(sV, P2_VL, gid, ks * 16, t2);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sVn, P2_VT, mt * 16, 0, gid, t2, d);
        }
        __syncthreads();
        for (int idx = tid; idx < P2_C * P2_VT; idx += P2_THREADS) {
            const int r = idx / P2_VT, cc = idx - r * P2_VT;
            wU[r * P2_DV + cb * P2_VT + cc] = sVn[r * P2_VT + cc];
        }
        __syncthreads();
    }

    // ---- W = T @ (beta . K . e^G), materialized bf16 for stage H's A fragment. Warp mt
    // owns all eight 16-wide d_k windows of its rows.
    __nv_bfloat16* const wW = reinterpret_cast<__nv_bfloat16*>(slot + P2_OFF_W);
    if (warp < P2_MT) {
        const int mt = warp;
        #pragma unroll 2
        for (int ks = 0; ks < P2_DK / 16; ++ks) {
            float w0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float w1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll 1
            for (int cs = 0; cs < P2_C / 16; ++cs) {
                const AFrag a = p1_a_sbf(sT, P2_T_LD, mt * 16, cs * 16, gid, t2);
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
            // w0 holds d_k columns [ks*16, +8) and w1 [ks*16+8, +8) of the window.
            const float g0[4] = {w0[0], w0[1], w0[2], w0[3]};
            const float g1[4] = {w1[0], w1[1], w1[2], w1[3]};
            p1_store_bf16(wW, P2_DK, mt * 16, ks * 16, gid, t2, g0);
            p1_store_bf16(wW, P2_DK, mt * 16, ks * 16 + 8, gid, t2, g1);
        }
    }
    __syncthreads();

    // ---- qk = (bf16(q) @ bf16(K)^T) . exp(G_j - G_i), tril with the diagonal,
    // into sA, then to the ws slot.
    __syncthreads();
    for (int tile = warp; tile < P2_MT * P2_NT_C; tile += P2_WARPS) {
        const int mt = tile / P2_NT_C, nt = tile - mt * P2_NT_C;
        const int cl = mt * 16 + gid, ch = cl + 8, bn = nt * 8 + gid;
        const AQuery* al = qh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
        const AQuery* ah = qh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
        const AKey* bp = kb + (size_t)((row0 + bn) < valid_rows ? row0 + bn : row0) * rstride;
        const float sl = (row0 + cl) < valid_rows ? 1.0f : 0.0f;
        const float sh = (row0 + ch) < valid_rows ? 1.0f : 0.0f;
        const float sb = (row0 + bn) < valid_rows ? 1.0f : 0.0f;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ks = 0; ks < P2_DK / 16; ++ks) {
            const AFrag a = p1_a_rows(al, ah, ks * 16, t2, qscale, sl, sh);
            const BFrag b = p1_b_rows(bp, P2_DK, 0, ks * 16, t2, sb, sb, sb, sb);
            CW_MMA(d, a, b);
        }
        p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d);
    }
    __syncthreads();
    for (int idx = tid; idx < P2_C * P2_C; idx += P2_THREADS) {
        const int j = idx / P2_C, i = idx - j * P2_C;
        sA[j * P2_A_LD + i] = j >= i ? sA[j * P2_A_LD + i] * __expf(sG[j] - sG[i]) : 0.0f;
    }
    __syncthreads();
    for (int idx = tid; idx < P2_C * P2_C; idx += P2_THREADS) {
        const int r = idx / P2_C, cc = idx - r * P2_C;
        reinterpret_cast<__nv_bfloat16*>(slot + P2_OFF_QK)[r * P2_T_LD + cc] =
            __float2bfloat16_rn(sA[r * P2_A_LD + cc]);
    }

    // ---- the FP32[64] vectors stage H reads.
    float* const wGc = reinterpret_cast<float*>(slot + P2_OFF_GC);
    float* const wBe = reinterpret_cast<float*>(slot + P2_OFF_BE);
    float* const wSc = reinterpret_cast<float*>(slot + P2_OFF_SC);
    for (int r = tid; r < P2_C; r += P2_THREADS) {
        wGc[r] = sG[r];
        wBe[r] = sBeta[r];
        wSc[r] = sSc[r];
    }
}


// Stage H: one CTA per (32-wide value strip, head), the chunk loop in order,
// S[:, j0:j0+32] in registers. v, beta and g are not read.
extern "C" __global__ __launch_bounds__(P2_THREADS, 1)
void gdn_cw_p2_stage_h(const float* __restrict__ q,
                       const float* __restrict__ k,
                       const __nv_bfloat16* __restrict__ v,
                       const float* __restrict__ beta,
                       const float* __restrict__ g,
                       float* __restrict__ S,            // FP32 [HEADS, DK, DV]
                       __nv_bfloat16* __restrict__ out,  // BF16 [rows, HEADS, DV]
                       char* __restrict__ ws,
                       const Q38SequenceFields* __restrict__ meta,
                       int valid_rows_in) {
    const int head = blockIdx.y;
    const int j0 = blockIdx.x * P2_VW;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = CW_GID(lane);
    const int t2 = CW_T2(lane);
    constexpr int NT = P2_VW / 8;   // 8-column value tiles of the 32-wide strip

    const bool aligned = (reinterpret_cast<uintptr_t>(k)
                          | reinterpret_cast<uintptr_t>(S)
                          | reinterpret_cast<uintptr_t>(out)
                          | reinterpret_cast<uintptr_t>(ws)) % 256 == 0;
    if (blockDim.x != P2_THREADS || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || gridDim.y != P2_HEADS || gridDim.x * P2_VW != P2_DV || !aligned) {
        __trap();
        return;
    }
    const int valid_rows = p2_valid_rows(meta, valid_rows_in);

    extern __shared__ __align__(16) char p2_smem[];
    const P2Arena ar = p2_arena(p2_smem);
    float* const sA = ar.sA;
    __nv_bfloat16* const sV = ar.sV;
    float* const sVn = ar.sVn;
    float* const sG = ar.sG;
    float* const sSc = ar.sSc;

    constexpr size_t rstride = P2_QK_ROW;
    const float* const kh = k + (size_t)P2_SOURCE_HEAD(head) * P2_DK;
    const float* const qh = q + (size_t)P2_SOURCE_HEAD(head) * P2_DK;
    const float qscale = P2_QUERY_SCALE;

    // S[:, j0:j0+VW] stays in registers for the whole call; every index into Sr is
    // compile-time.
    float Sr[2][NT][2];
    {
        // Warp w owns d_k rows 16w..16w+15.
        float* srow = S + (size_t)head * P2_DK * P2_DV
                    + (size_t)warp * 16 * P2_DV + j0 + gid * P2_DV + t2;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                Sr[0][nt][c] = srow[8 * nt + c];
                Sr[1][nt][c] = srow[8 * nt + c + 8 * P2_DV];
            }
    }

    const int nchunk = (valid_rows + P2_C - 1) / P2_C;
    for (int c = 0; c < nchunk; ++c) {
        const int row0 = c * P2_C;
        char* const slot = p2_slot(ws, c, head);
        const __nv_bfloat16* const wW =
            reinterpret_cast<const __nv_bfloat16*>(slot + P2_OFF_W);
        const float* const wU = reinterpret_cast<const float*>(slot + P2_OFF_U);
        const __nv_bfloat16* const wqk =
            reinterpret_cast<const __nv_bfloat16*>(slot + P2_OFF_QK);
        const float* const wGc = reinterpret_cast<const float*>(slot + P2_OFF_GC);
        const float* const wSc = reinterpret_cast<const float*>(slot + P2_OFF_SC);
        for (int r = tid; r < P2_C; r += P2_THREADS) {
            sG[r] = wGc[r];
            sSc[r] = wSc[r];
        }

        // ---- WS = W @ S: bf16 W times a bf16 copy of S. Each warp stages its own 16-row
        // d_k slice; sV holds the whole d_k axis as [value][d_k].
        const int kblk = warp;           // the dk window this warp's S rows belong to
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                sV[(8 * nt + t2 + cc) * P2_VLDK + kblk * 16 + gid] =
                    __float2bfloat16_rn(Sr[0][nt][cc]);
                sV[(8 * nt + t2 + cc) * P2_VLDK + kblk * 16 + gid + 8] =
                    __float2bfloat16_rn(Sr[1][nt][cc]);
            }
        __syncthreads();
        // The owning warp of a chunk-row tile contracts the whole d_k axis over ks = 0..7;
        // every output element has one writer.
        if (warp < P2_MT) {
            const int mt = warp;
            float d[NT][4] = {{0.0f}};
            #pragma unroll 2
            for (int ks = 0; ks < P2_DK / 16; ++ks) {
                const AFrag a = p1_a_sbf(wW, P2_DK, mt * 16, ks * 16, gid, t2);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const BFrag b = p1_b_sbf(sV, P2_VLDK, nt * 8 + gid, ks * 16, t2);
                    CW_MMA(d[nt], a, b);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d[nt]);
        }

        // ---- v_new = U - WS into sVn; U comes from the ws slot.
        __syncthreads();
        __syncthreads();
        for (int idx = tid; idx < P2_C * P2_VW; idx += P2_THREADS) {
            const int r = idx / P2_VW, cc = idx - r * P2_VW;
            sVn[r * P2_VW + cc] = wU[r * P2_DV + j0 + cc] - sA[r * P2_A_LD + cc];
        }

        // ---- v_new as bf16, staged for the out contraction and the state update.
        for (int idx = tid; idx < P2_C * P2_VW; idx += P2_THREADS) {
            const int r = idx / P2_VW, cc = idx - r * P2_VW;
            const __nv_bfloat16 e = __float2bfloat16_rn(sVn[r * P2_VW + cc]);
            sV[cc * P2_VL + r] = e;
        }
        __syncthreads();

        // ---- AD = qk @ v_new into sA. qk is read from the ws slot in place (stride
        // P2_T_LD); v_new is in sV as [value][chunk].
        for (int tile = warp; tile < P2_MT * NT; tile += P2_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P2_C / 16; ++ks) {
                const AFrag a = p1_a_sbf(wqk, P2_T_LD, mt * 16, ks * 16, gid, t2);
                const BFrag b = p1_b_sbf(sV, P2_VL, nt * 8 + gid, ks * 16, t2);
                CW_MMA(d, a, b);
            }
            p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();

        // ---- QS = (q . e^G) @ S_{t-1} onto AD's tile; the pre-update Sr is staged in
        // bf16 per warp d_k slice.
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                sV[(8 * nt + t2 + cc) * P2_VLDK + kblk * 16 + gid] =
                    __float2bfloat16_rn(Sr[0][nt][cc]);
                sV[(8 * nt + t2 + cc) * P2_VLDK + kblk * 16 + gid + 8] =
                    __float2bfloat16_rn(Sr[1][nt][cc]);
            }
        __syncthreads();
        // The owning warp contracts the full d_k axis onto AD's tile.
        for (int tile = warp; tile < P2_MT * NT; tile += P2_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            const int cl = mt * 16 + gid, ch = cl + 8;
            const float* al =
                qh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
            const float* ah =
                qh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
            const float sl = (row0 + cl) < valid_rows ? __expf(sG[cl]) : 0.0f;
            const float sh = (row0 + ch) < valid_rows ? __expf(sG[ch]) : 0.0f;
            float* o = sA + (size_t)(mt * 16 + gid) * P2_A_LD + nt * 8 + t2;
            float d[4] = {o[0], o[1], o[8 * P2_A_LD], o[8 * P2_A_LD + 1]};
            #pragma unroll
            for (int ks = 0; ks < P2_DK / 16; ++ks) {
                const AFrag a = p1_a_rows(al, ah, ks * 16, t2, qscale, sl, sh);
                const BFrag b = p1_b_sbf(sV, P2_VLDK, nt * 8 + gid, ks * 16, t2);
                CW_MMA(d, a, b);
#if GDN_HEADS == 32
                const AFrag residual = p1_a_rows<true>(al, ah, ks * 16, t2, qscale, sl, sh);
                CW_MMA(d, residual, b);
#endif
            }
            o[0] = d[0];
            o[1] = d[1];
            o[8 * P2_A_LD] = d[2];
            o[8 * P2_A_LD + 1] = d[3];
        }
        __syncthreads();

        // ---- out = BF16_RNE(QS + AD)
        for (int idx = tid; idx < P2_C * P2_VW; idx += P2_THREADS) {
            const int r = idx / P2_VW, cc = idx - r * P2_VW;
            const int gr = row0 + r;
            if (gr < valid_rows)
                out[((size_t)gr * P2_HEADS + head) * P2_DV + j0 + cc] =
                    __float2bfloat16_rn(sA[r * P2_A_LD + cc]);
        }
        __syncthreads();   // the QS staging overwrote sV; the decay restages it below

        // ---- S = S . exp(G_last) + (k . e^{G_last-G})^T @ v_new; the A operand is the
        // TRANSPOSE (fragment rows are d_k indices), so the scale is per column here.
        __syncthreads();
        for (int idx = tid; idx < P2_C * P2_VW; idx += P2_THREADS) {
            const int r = idx / P2_VW, cc = idx - r * P2_VW;
            sV[cc * P2_VL + r] = __float2bfloat16_rn(sVn[r * P2_VW + cc]);
        }
        __syncthreads();
        {
            const float decay = __expf(sG[P2_C - 1]);
            const float gl = sG[P2_C - 1];
            // Every o[nt] contracts over ascending ks, then all Sr rows update in one sweep.
            float o[NT][4] = {{0.0f}};
            #pragma unroll
            for (int ks = 0; ks < P2_C / 16; ++ks) {
                // ks*16 is the chunk-row offset of the 16-wide k-step.
                const AFrag a = p1_a_kdec(kh, rstride, row0, valid_rows, sG, gl,
                                          warp * 16 + gid, ks * 16 + t2);
#if GDN_HEADS == 32
                const AFrag residual = p1_a_kdec<true>(kh, rstride, row0, valid_rows, sG, gl,
                                                      warp * 16 + gid, ks * 16 + t2);
#endif
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const BFrag b = p1_b_sbf(sV, P2_VL, nt * 8 + gid, ks * 16, t2);
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
        __syncthreads();   // the next chunk rewrites sG/sSc while slow threads read them
    }

    {
        // gid is a d_k ROW here, so it scales by P2_DV; t2 is the value column.
        float* srow = S + (size_t)head * P2_DK * P2_DV
                    + (size_t)(warp * 16 + gid) * P2_DV + j0 + t2;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                srow[8 * nt + c] = Sr[0][nt][c];
                srow[8 * nt + c + 8 * P2_DV] = Sr[1][nt][c];
            }
    }
}

