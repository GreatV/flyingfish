// Q38 GDN chunkwise prefill, the P2 SPLIT of the precision protocol P1 core, C = 64,
// module gdn/gdn_cw_p2. gdn/gdn_chunkwise_p1 runs one kernel per step; this file runs
// two, with the same fp32 accumulators, the same bf16 RNE rounding points, the same
// mma fragment staging and the same loop and accumulation orders, so the two agree bit
// for bit. Only the kernel boundaries and the placement of the materialized products
// change.
//
//   P2 reads the BF16 sources directly: the widen (__bfloat162float at each load), the
//   GQA expansion (value head h reads key head h / P2_GQA, row stride P2_KROW = 16*128)
//   and the Q scale by 1/sqrt(DK) as a separate FIRST multiply happen at the load sites;
//   no gdn_prep_fp32 launch. beta stays FP32 (gdn_beta_widen) and g is FP32.
//
//   stage A  grid (nchunk, 48): P1 :263-274 gates and the cumsum, :276-301 A = -tril(...),
//             :303-410 the blocked (I-A)^-1 with both merge levels, then U = T@(beta.V) for
//             all DV columns (:412-431 tiled), W = T@(beta.K.e^G) materialized bf16 (:447-487
//             inner), qk (:497-527) and the FP32[64] vectors Gc, Be, Sc.
//   stage H  grid (4, 48), 32-wide value strips (P1's s4 slice): the chunk loop in order,
//             with S[:, j0:j0+32] in registers (:246-258), WS = W@S (:435-487), v_new =
//             U - WS (:492-495), the pre-update S snapshot, and the state update
//             (:596-630). S is written back at :634-645.
//
// The workspace slot of one (chunk, head) is GDN_CW_P2_SLOT_BYTES at
// ws + (chunk * 48 + head) * GDN_CW_P2_SLOT_BYTES, in this byte order:
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

#include "gdn_cw_p1_layout.h"
#include "gdn_cw_p2_layout.h"

#define P2_C       64     // chunk rows
#define P2_SUB     16     // UT inverse diagonal block (protocol section A)
#define P2_THREADS 256    // protocol section E: 256 threads for the state pass
#define P2_WARPS   (P2_THREADS / 32)
#define P2_DK      128
#define P2_DV      128
#define P2_HEADS   48
#define P2_MT      (P2_C / 16)          // row tiles of 16
#define P2_KHEADS  16                   // KEY heads in the projection's Q and K
#define P2_GQA     (P2_HEADS / P2_KHEADS)   // value head -> key head, h / GQA
#define P2_KROW    (P2_KHEADS * P2_DK)  // a KEY-head row of the bf16 sources
#define P2_NT_C    (P2_C / 8)           // column tiles of 8 in the [C][C] products
#define P2_VL      72                    // sV row stride for the CHUNK axis, bf16 (+8 off 64)
#define P2_VLDK    136                   // sV row stride for the d_k axis, bf16 (+8 off 128)
#define P2_A_LD    68                    // sA row stride, floats (+4 off 64)
#define P2_T_LD    72                    // sT row stride, bf16 (+8 off 64)

// One arena for both stages, in order: sA | sT | sV | sVn | sG | sBeta | sSc; it is
// P1's s4 arena (gdn_cw_p1.cu :24-26), the largest of the split set.
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

// The value strip stage H walks, and the U column tile stage A walks: 32-wide
// strips are P1's s4, the 8-wide tile is P1's s16 accumulator width.
#define P2_VW        32
#define P2_VT        8

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

struct AFrag { uint32_t r[4]; };
struct BFrag { uint32_t r[2]; };

// C and D alias: the accumulator is both the mma input and its output.
#define P2_MMA(d, a, b)                                                        \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"     \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])              \
                 : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]),         \
                   "r"(b.r[0]), "r"(b.r[1]))

// Fragment geometry (PTX ISA 9.7.16.5.8, mma.m16n8k16), groupID = laneid >> 2, tig = laneid & 3: A row groupID (a0,a1) / groupID+8 (a2,a3) at column 2*tig+(i&1), +8 for a2,a3; B column groupID, row 2*tig+(i&1) for b0 and +8 for b1; C/D row groupID (c0,c1) / groupID+8 (c2,c3) at column 2*tig+(i&1); register i holds element 2i in its LOW half.
#define P2_GID(l) ((l) >> 2)
#define P2_T2(l)  (((l) & 3) << 1)

__device__ __forceinline__ uint32_t p1_pack(float lo, float hi) {
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(lo))
         | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hi)) << 16);
}



// ---- fragment loads ----------------------------------------------------------------

// A from BF16 rows, one scale per ROW (beta, and the q liveness mask): register 1 is the
// +8 row at the LOW columns and register 2 the base row at the +8 columns, and the two
// are not interchangeable. `qscale` is 1/sqrt(DK) for Q and 1.0f for K, applied before
// the row scale. x * 1.0f is exact, so the same helper serves K unchanged.
__device__ __forceinline__ AFrag p1_a_rows(const __nv_bfloat16* lo, const __nv_bfloat16* hi,
                                           int col0, int t2, float qscale,
                                           float s_lo, float s_hi) {
    AFrag f;
    f.r[0] = p1_pack(__bfloat162float(lo[col0 + t2]) * qscale * s_lo,
                     __bfloat162float(lo[col0 + t2 + 1]) * qscale * s_lo);
    f.r[1] = p1_pack(__bfloat162float(hi[col0 + t2]) * qscale * s_hi,
                     __bfloat162float(hi[col0 + t2 + 1]) * qscale * s_hi);
    f.r[2] = p1_pack(__bfloat162float(lo[col0 + t2 + 8]) * qscale * s_lo,
                     __bfloat162float(lo[col0 + t2 + 9]) * qscale * s_lo);
    f.r[3] = p1_pack(__bfloat162float(hi[col0 + t2 + 8]) * qscale * s_hi,
                     __bfloat162float(hi[col0 + t2 + 9]) * qscale * s_hi);
    return f;
}

// One element of (k . e^{G_last-G}) at chunk row ci and d_k column dk; the state update's A operand is the TRANSPOSE, so a register pair's two elements sit in DIFFERENT chunk rows at the SAME d_k column.
__device__ __forceinline__ float p1_kdec(const __nv_bfloat16* kh, size_t ld, int row0,
                                          int valid, int ci, int dk, const float* sG, float gl) {
    const int gr = row0 + ci;
    if (gr >= valid) return 0.0f;
    return __bfloat162float(kh[(size_t)gr * ld + dk]) * __expf(gl - sG[ci]);
}

__device__ __forceinline__ AFrag p1_a_kdec(const __nv_bfloat16* kh, size_t ld, int row0,
                                           int valid, const float* sG, float gl, int dk, int t2) {
    const int c0 = t2;
    AFrag f;
    f.r[0] = p1_pack(p1_kdec(kh, ld, row0, valid, c0, dk, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 1, dk, sG, gl));
    f.r[1] = p1_pack(p1_kdec(kh, ld, row0, valid, c0, dk + 8, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 1, dk + 8, sG, gl));
    f.r[2] = p1_pack(p1_kdec(kh, ld, row0, valid, c0 + 8, dk, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 9, dk, sG, gl));
    f.r[3] = p1_pack(p1_kdec(kh, ld, row0, valid, c0 + 8, dk + 8, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 9, dk + 8, sG, gl));
    return f;
}

// B from a row-major BF16 buffer: fragment row n, columns 2tig + {0,1} and +8.
__device__ __forceinline__ BFrag p1_b_rows(const __nv_bfloat16* p, int ld, int n, int col0,
                                           int t2, float s0, float s1, float s8, float s9) {
    const __nv_bfloat16* r = p + (size_t)n * ld;
    BFrag f;
    f.r[0] = p1_pack(__bfloat162float(r[col0 + t2]) * s0, __bfloat162float(r[col0 + t2 + 1]) * s1);
    f.r[1] = p1_pack(__bfloat162float(r[col0 + t2 + 8]) * s8,
                     __bfloat162float(r[col0 + t2 + 9]) * s9);
    return f;
}

// A from a BF16 buffer (T class, encoded qk, materialized W): shared or global, the
// register order is p1_a_rows'. Same register order as p1_a_rows.
__device__ __forceinline__ AFrag p1_a_sbf(const __nv_bfloat16* base, int ld, int row0,
                                          int col0, int gid, int t2) {
    const uint32_t* lo = reinterpret_cast<const uint32_t*>(base + (size_t)(row0 + gid) * ld);
    const uint32_t* hi = reinterpret_cast<const uint32_t*>(base + (size_t)(row0 + gid + 8) * ld);
    AFrag f;
    f.r[0] = lo[(col0 + t2) >> 1];
    f.r[1] = hi[(col0 + t2) >> 1];
    f.r[2] = lo[(col0 + t2 + 8) >> 1];
    f.r[3] = hi[(col0 + t2 + 8) >> 1];
    return f;
}

// B from a BF16 row-major buffer (beta.V, the encoded v_new, the S copy).
__device__ __forceinline__ BFrag p1_b_sbf(const __nv_bfloat16* base, int ld, int n,
                                          int col0, int t2) {
    const uint32_t* r = reinterpret_cast<const uint32_t*>(base + (size_t)n * ld);
    BFrag f;
    f.r[0] = r[(col0 + t2) >> 1];
    f.r[1] = r[(col0 + t2 + 8) >> 1];
    return f;
}

// A from an FP32 shared buffer, rounded to BF16 (A[b,a] of the inverse merge).
__device__ __forceinline__ AFrag p1_a_sf32(const float* base, int ld, int row0, int col0,
                                           int gid, int t2) {
    const float* lo = base + (size_t)(row0 + gid) * ld;
    const float* hi = base + (size_t)(row0 + gid + 8) * ld;
    AFrag f;
    f.r[0] = p1_pack(lo[col0 + t2], lo[col0 + t2 + 1]);
    f.r[1] = p1_pack(hi[col0 + t2], hi[col0 + t2 + 1]);
    f.r[2] = p1_pack(lo[col0 + t2 + 8], lo[col0 + t2 + 9]);
    f.r[3] = p1_pack(hi[col0 + t2 + 8], hi[col0 + t2 + 9]);
    return f;
}

// B from an FP32 shared buffer, rounded to BF16 (A[b,a] as the merge's non-T side).
__device__ __forceinline__ BFrag p1_b_sf32(const float* base, int ld, int n, int col0, int t2) {
    const float* r = base + (size_t)n * ld;
    BFrag f;
    f.r[0] = p1_pack(r[col0 + t2], r[col0 + t2 + 1]);
    f.r[1] = p1_pack(r[col0 + t2 + 8], r[col0 + t2 + 9]);
    return f;
}

// One element of (beta . K . e^G) at chunk row ci, d_k column dk; the only GEMM whose
// B fragment reads the input TRANSPOSED (fragment row = d_k index, column = chunk row).
__device__ __forceinline__ float p1_kbg(const __nv_bfloat16* kh, size_t ld, int row0, int valid,
                                        int ci, int dk, const float* sSc) {
    const int gr = row0 + ci;
    const __nv_bfloat16* p = kh + (size_t)(gr < valid ? gr : row0) * ld + dk;
    return __bfloat162float(p[0]) * sSc[ci];
}

// D fragment -> FP32 shared, at (row0 + gid, col0 + 2*tig) and the +8 row.
__device__ __forceinline__ void p1_store_f32(float* base, int ld, int row0, int col0,
                                             int gid, int t2, const float* d) {
    float* a = base + (size_t)(row0 + gid) * ld + col0 + t2;
    a[0] = d[0];
    a[1] = d[1];
    float* b = a + 8 * ld;
    b[0] = d[2];
    b[1] = d[3];
}

// D fragment -> BF16 shared.
__device__ __forceinline__ void p1_store_bf16(__nv_bfloat16* base, int ld, int row0, int col0,
                                              int gid, int t2, const float* d) {
    __nv_bfloat16* a = base + (size_t)(row0 + gid) * ld + col0 + t2;
    a[0] = __float2bfloat16_rn(d[0]);
    a[1] = __float2bfloat16_rn(d[1]);
    __nv_bfloat16* b = a + 8 * ld;
    b[0] = __float2bfloat16_rn(d[2]);
    b[1] = __float2bfloat16_rn(d[3]);
}


// ---- shared arena and the workspace slot --------------------------------------------

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

// P1's meta guard (gdn_chunkwise_p1.cu :218-224): valid_rows resolves the host sentinel
// 0, and key_count == prefix + valid_rows bounds the rows a kernel may read.
__device__ __forceinline__ int p2_valid_rows(const GdnMeta* meta, int valid_rows_in) {
    const int prefix = meta->prefix;
    const int valid_rows = valid_rows_in > 0 ? valid_rows_in : meta->valid_rows;
    if (valid_rows <= 0 || prefix < 0 || meta->causal != 1
        || meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap();
    }
    return valid_rows;
}


// ---- stage A ------------------------------------------------------------------------

// One CTA per (chunk, head). No S dependency: everything stage H reads is
// written here. The host grid is ceil(max_rows / P2_C) wide, so a tail step leaves spare
// CTAs, and they exit before the first barrier.
extern "C" __global__ __launch_bounds__(P2_THREADS, 1)
void gdn_cw_p2_stage_a(const __nv_bfloat16* __restrict__ q,  // BF16 [M, 16, DK] KEY heads
                       const __nv_bfloat16* __restrict__ k,  // BF16 [M, 16, DK] KEY heads
                       const __nv_bfloat16* __restrict__ v,  // BF16 [M, HEADS, DV] VALUE heads
                       const float* __restrict__ beta,// FP32 [M, HEADS]
                       const float* __restrict__ g,   // FP32 [M, HEADS]
                       char* __restrict__ ws,
                       const GdnMeta* __restrict__ meta, int valid_rows_in,
                       uint32_t v_row_stride) {
    const int head = blockIdx.y;
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = P2_GID(lane);
    const int t2 = P2_T2(lane);

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

    // Value head h reads key head h / GQA at row stride P2_KROW; V is a conv_out column
    // window whose rows are v_row_stride elements apart, not HEADS*DV.
    constexpr size_t rstride = P2_KROW;
    const __nv_bfloat16* const kh = k + (size_t)(head / P2_GQA) * P2_DK;
    const __nv_bfloat16* const qh = q + (size_t)(head / P2_GQA) * P2_DK;
    const __nv_bfloat16* const vh = v + (size_t)head * P2_DV;
    const float qscale = 1.0f / sqrtf((float)P2_DK);

    char* const slot = p2_slot(ws, chunk, head);

    // ---- gates and the in-chunk cumsum: ONE thread per scan then a barrier, since interleaved in-place scans apply G without bound (e^G overflow and NaNs).
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
        const __nv_bfloat16* al = kh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
        const __nv_bfloat16* ah = kh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
        const __nv_bfloat16* bp = kh + (size_t)((row0 + bn) < valid_rows ? row0 + bn : row0) * rstride;
        const float sl = sBeta[cl], sh = sBeta[ch];     // zero on a dead row
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ks = 0; ks < P2_DK / 16; ++ks) {
            const AFrag a = p1_a_rows(al, ah, ks * 16, t2, 1.0f, sl, sh);
            const BFrag b = p1_b_rows(bp, P2_DK, 0, ks * 16, t2, 1.0f, 1.0f, 1.0f, 1.0f);
            P2_MMA(d, a, b);
        }
        p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d);
    }
    __syncthreads();
    // tril BEFORE the exp and AGAIN after it: exp of a masked entry is 1.0, so
    // without the second mask every token reads the future.
    for (int idx = tid; idx < P2_C * P2_C; idx += P2_THREADS) {
        const int j = idx / P2_C, i = idx - j * P2_C;
        sA[j * P2_A_LD + i] = j > i ? -sA[j * P2_A_LD + i] * __expf(sG[j] - sG[i]) : 0.0f;
    }
    __syncthreads();

    // ---- T = (I - A)^-1, blocked: 16x16 FP32 leaves by forward substitution then 16->32 and 32->64 merges. ONE WARP per diagonal block, lane j = row j: row r still reads the
    // already-final entries of rows m < r, but through __shfl_sync(lw, t[j], m) — lane
    // m's current t[j], final by the row-synchronous step order — so each element keeps
    // the thread form's seed and ascending-m fma chain and the leaves are bit-identical.
    // All lanes named in lw execute every shuffle (the non-owning lanes discard); the
    // guard keeps lanes 16..31 of a leaf warp from touching the next block's rows.
    {
        const unsigned lw = (1u << P2_SUB) - 1u;   // the 16 leaf lanes
        if (warp < P2_C / P2_SUB && lane < P2_SUB) {
            // The substituted row and the rows it reads both live in the diagonal block's COLUMN window; without c0 the pass substitutes the off-diagonal block at columns 0..15 and destroys the A[b,a] the merges consume.
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
    __syncthreads();   // the encode reads ALL of sA, including diagonals other
                       // threads just incremented
    // The fp32 T is dead from here, so sA is free for the qk accumulator.
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
        // Both merge operands are staged TRANSPOSED, because a B fragment reads [output column][reduction]: A[b,a] as [a][b] and T[a,a] as [a'][a]; the block here is still the ORIGINAL A, the leaf substitution writing only the diagonal 16x16 blocks.
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
            P2_MMA(d, a, b);
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
            P2_MMA(d, a, b);
            p1_store_bf16(sT, P2_T_LD, b0, a0 + nt * 8, gid, t2, d);
        }
        __syncthreads();
    }
    {   // 32 -> 64. The (x@y) intermediate goes to sA[0:32][32:64], strictly ABOVE
        // A's lower triangle and so disjoint from A[32:64][0:32].
        const int mt = warp / 4, nt = warp - (warp / 4) * 4;
        // A[32:64][0:32] transposed into sV as [a][b], for the same reason as above.
        for (int idx = tid; idx < 32 * 32; idx += P2_THREADS) {
            const int bl = idx / 32, al = idx - bl * 32;
            sV[al * 64 + bl] = __float2bfloat16_rn(sA[(size_t)(32 + bl) * P2_A_LD + al]);
        }
        __syncthreads();
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        // BOTH products reduce over a 32-wide axis, i.e. TWO 16-wide k-steps; a
        // bound of P2_SUB/16 = 1 contracts half of it and is wrong on the mt=1 half.
        #pragma unroll
        for (int ks = 0; ks < P2_C / (2 * P2_SUB); ++ks) {
            const AFrag a = p1_a_sbf(sT, P2_T_LD, 32 + mt * 16, 32 + ks * 16, gid, t2);
            // col0 = ks*16: the staged tile is indexed by the LOCAL b, so without
            // ks*16 both steps read local b 0..15.
            const BFrag b = p1_b_sbf(sV, 64, nt * 8 + gid, ks * 16, t2);
            P2_MMA(d, a, b);
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
            P2_MMA(e, a, b);
        }
        p1_store_bf16(sT, P2_T_LD, 32 + mt * 16, nt * 8, gid, t2, e);
        __syncthreads();
    }

    // ---- U = T @ (beta . V), all DV columns in 16 tiles of P2_VT: sVn is one tile wide,
    // so each tile's fp32 accumulator goes to the ws U slot at once. Every element is one
    // mma over the same ascending ks, and the tile boundary splits columns, not k.
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
                P2_MMA(d, a, b);
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

    // ---- W = T @ (beta . K . e^G) MATERIALIZED: the same w0/w1 accumulators over cs and
    // the same p1_pack as P1 :476-477, written out so stage H can read them back as the A
    // fragment. Each warp mt owns all eight 16-wide d_k windows of its rows.
    __nv_bfloat16* const wW = reinterpret_cast<__nv_bfloat16*>(slot + P2_OFF_W);
    if (warp < P2_MT) {
        const int mt = warp;
        // ks unrolled by 2, cs not at all: cs accumulates serially into w0/w1, so
        // unrolling it buys no independent work.
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
                P2_MMA(w0, a, b0);
                P2_MMA(w1, a, b1);
            }
            // Same register order as p1_a_rows; w0 is the first 8-wide d_k group of
            // the window, w1 the second. p1_store_bf16 writes d[0],d[1] on the base row
            // and d[2],d[3] on its +8 row, so these two stores are r[0],r[1] from w0
            // at the window base and r[2],r[3] from w1 at the window's +8 d_k columns.
            const float g0[4] = {w0[0], w0[1], w0[2], w0[3]};
            const float g1[4] = {w1[0], w1[1], w1[2], w1[3]};
            p1_store_bf16(wW, P2_DK, mt * 16, ks * 16, gid, t2, g0);
            p1_store_bf16(wW, P2_DK, mt * 16, ks * 16 + 8, gid, t2, g1);
        }
    }
    __syncthreads();

    // ---- qk = (bf16(q) @ bf16(K)^T) . exp(G_j - G_i), tril WITH the diagonal,
    // into sA (free since the fp32 T is encoded), then to the ws slot.
    __syncthreads();
    for (int tile = warp; tile < P2_MT * P2_NT_C; tile += P2_WARPS) {
        const int mt = tile / P2_NT_C, nt = tile - mt * P2_NT_C;
        const int cl = mt * 16 + gid, ch = cl + 8, bn = nt * 8 + gid;
        const __nv_bfloat16* al = qh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
        const __nv_bfloat16* ah = qh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
        const __nv_bfloat16* bp = kh + (size_t)((row0 + bn) < valid_rows ? row0 + bn : row0) * rstride;
        const float sl = (row0 + cl) < valid_rows ? 1.0f : 0.0f;
        const float sh = (row0 + ch) < valid_rows ? 1.0f : 0.0f;
        const float sb = (row0 + bn) < valid_rows ? 1.0f : 0.0f;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ks = 0; ks < P2_DK / 16; ++ks) {
            const AFrag a = p1_a_rows(al, ah, ks * 16, t2, qscale, sl, sh);
            const BFrag b = p1_b_rows(bp, P2_DK, 0, ks * 16, t2, sb, sb, sb, sb);
            P2_MMA(d, a, b);
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


// ---- stage H ------------------------------------------------------------------------

// One CTA per (32-wide value block, head), the chunk loop in order, S[:, j0:j0+32] in
// registers. This stage reads q for the out contraction, k for the decay, the state S,
// the out buffer and the ws slot; v, beta and g are part of the common stage signature
// and are not read.
extern "C" __global__ __launch_bounds__(P2_THREADS, 1)
void gdn_cw_p2_stage_h(const __nv_bfloat16* __restrict__ q,
                       const __nv_bfloat16* __restrict__ k,
                       const __nv_bfloat16* __restrict__ v,
                       const float* __restrict__ beta,
                       const float* __restrict__ g,
                       float* __restrict__ S,            // FP32 [HEADS, DK, DV]
                       __nv_bfloat16* __restrict__ out,  // BF16 [rows, HEADS, DV]
                       char* __restrict__ ws,
                       const GdnMeta* __restrict__ meta,
                       int valid_rows_in) {
    const int head = blockIdx.y;
    const int j0 = blockIdx.x * P2_VW;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = P2_GID(lane);
    const int t2 = P2_T2(lane);
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

    constexpr size_t rstride = P2_KROW;
    const __nv_bfloat16* const kh = k + (size_t)(head / P2_GQA) * P2_DK;
    const __nv_bfloat16* const qh = q + (size_t)(head / P2_GQA) * P2_DK;
    const float qscale = 1.0f / sqrtf((float)P2_DK);

    // S[:, j0:j0+VW] stays in REGISTERS for the whole call; every index into Sr is
    // compile-time, which is what keeps the tile out of local memory.
    float Sr[2][NT][2];
    {
        // Warp w owns d_k rows 16w..16w+15, so the 8 warps cover all P2_DK = 128; the
        // warp*16 term is what keeps 112 state rows from being skipped entirely.
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

        // ---- WS = W @ S with W MATERIALIZED: W is a GEMM output, so it is rounded to BF16 (WU class) before it multiplies the BF16 S copy; each warp stages S into its OWN slice of the 128-row d_k axis, so sV holds the whole axis as [value][d_k] and no warp contracts W against another warp's rows.
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
        // The OWNING warp of a chunk-row tile contracts the WHOLE 128-row d_k axis itself over ks = 0..7, so every output element has exactly one writer; warps with no tile of their own wait at the barrier below, and W depends on (mt, ks) only.
        if (warp < P2_MT) {
            const int mt = warp;
            float d[NT][4] = {{0.0f}};
            #pragma unroll 2
            for (int ks = 0; ks < P2_DK / 16; ++ks) {
                // The A fragment is the materialized W, the same 16 bits P1 packed into
                // registers at :476-477, in the same register order.
                const AFrag a = p1_a_sbf(wW, P2_DK, mt * 16, ks * 16, gid, t2);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const BFrag b = p1_b_sbf(sV, P2_VLDK, nt * 8 + gid, ks * 16, t2);
                    P2_MMA(d[nt], a, b);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d[nt]);
        }

        // ---- v_new = U - WS into sVn; U comes from the ws slot in the same subtraction
        // order (U - WS) as P1 :494.
        __syncthreads();
        __syncthreads();
        for (int idx = tid; idx < P2_C * P2_VW; idx += P2_THREADS) {
            const int r = idx / P2_VW, cc = idx - r * P2_VW;
            sVn[r * P2_VW + cc] = wU[r * P2_DV + j0 + cc] - sA[r * P2_A_LD + cc];
        }

        // ---- v_new as the WU-class operand (bf16), staged for the out contraction and
        // the state update.
        for (int idx = tid; idx < P2_C * P2_VW; idx += P2_THREADS) {
            const int r = idx / P2_VW, cc = idx - r * P2_VW;
            const __nv_bfloat16 e = __float2bfloat16_rn(sVn[r * P2_VW + cc]);
            sV[cc * P2_VL + r] = e;
        }
        __syncthreads();

        // ---- AD = qk @ v_new, accumulating into sA (the output tile). qk is read from
        // the ws slot in place (it keeps P2_T_LD). v_new is staged in sV in
        // [value][chunk] layout — p1_b_sbf packs two consecutive CHUNK values per
        // register.
        for (int tile = warp; tile < P2_MT * NT; tile += P2_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P2_C / 16; ++ks) {
                const AFrag a = p1_a_sbf(wqk, P2_T_LD, mt * 16, ks * 16, gid, t2);
                const BFrag b = p1_b_sbf(sV, P2_VL, nt * 8 + gid, ks * 16, t2);
                P2_MMA(d, a, b);
            }
            p1_store_f32(sA, P2_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();

        // ---- QS = (q . e^G) @ S_{t-1} onto AD's tile; the PRE-update state is this
        // CTA's Sr registers, staged here in bf16 per warp dk-slice exactly as P1
        // :553-561 does.
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
        // The whole 128-row d_k axis of S is in sV, so the owning warp does the FULL
        // d_k contraction over the eight 16-wide windows onto AD's tile.
        for (int tile = warp; tile < P2_MT * NT; tile += P2_WARPS) {
            const int mt = tile / NT, nt = tile - (tile / NT) * NT;
            const int cl = mt * 16 + gid, ch = cl + 8;
            const __nv_bfloat16* al =
                qh + (size_t)((row0 + cl) < valid_rows ? row0 + cl : row0) * rstride;
            const __nv_bfloat16* ah =
                qh + (size_t)((row0 + ch) < valid_rows ? row0 + ch : row0) * rstride;
            const float sl = (row0 + cl) < valid_rows ? __expf(sG[cl]) : 0.0f;
            const float sh = (row0 + ch) < valid_rows ? __expf(sG[ch]) : 0.0f;
            float* o = sA + (size_t)(mt * 16 + gid) * P2_A_LD + nt * 8 + t2;
            float d[4] = {o[0], o[1], o[8 * P2_A_LD], o[8 * P2_A_LD + 1]};
            #pragma unroll
            for (int ks = 0; ks < P2_DK / 16; ++ks) {
                const AFrag a = p1_a_rows(al, ah, ks * 16, t2, qscale, sl, sh);
                const BFrag b = p1_b_sbf(sV, P2_VLDK, nt * 8 + gid, ks * 16, t2);
                P2_MMA(d, a, b);
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
            // ONE pass: every o[nt] contracts over ascending ks, then all Sr rows are
            // updated in one sweep. NT is 1/2/4 by split.
            float o[NT][4] = {{0.0f}};
            #pragma unroll
            for (int ks = 0; ks < P2_C / 16; ++ks) {
                // ks*16 is the chunk-row offset of the 16-wide k-step; without it every
                // step contracts chunk rows 0..15 again.
                const AFrag a = p1_a_kdec(kh, rstride, row0, valid_rows, sG, gl,
                                          warp * 16 + gid, ks * 16 + t2);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const BFrag b = p1_b_sbf(sV, P2_VL, nt * 8 + gid, ks * 16, t2);
                    P2_MMA(o[nt], a, b);
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

