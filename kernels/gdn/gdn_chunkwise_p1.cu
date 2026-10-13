// =====================================================================================
// Q38 GDN chunkwise prefill, phase-2 precision protocol candidate P1. C = 64.
//
// ADDITIVE. The shipped recursive core (kernels/gdn/gdn_chunked.cu,
// gdn_chunked_core_v1) and its dispatch are untouched; this is a separate entry
// point on a separate module (gdn/gdn_chunkwise_p1).
//
// Structure, function for function against the CPU numerics reference
// (gdn_phase2_model_w.py), which is what the GPU round will diff this against:
//
//   per chunk, ONCE per head          _chunk_shared
//     G_j   = within-chunk cumsum of g, FP32                     -> sG
//     A     = -tril( bf16(beta.K) @ bf16(K)^T . exp(G_j-G_i), -1) -> sA
//     T     = blocked inverse of (I - A): 16x16 FP32 leaves, 16->32->64 TC merges -> sT
//     W     = bf16(T) @ bf16(beta.K.e^G)   MATERIALIZED GEMM output, WU class
//     qk    = bf16(q) @ bf16(K)^T . exp(G_j-G_i)   tril INCLUSIVE, Q class -> sT
//   per value strip, blockIdx.x       _chunk_strip
//     U     = bf16(T) @ bf16(beta.V)                            WU class
//     v_new = U - bf16(W) @ bf16(S)                             WU class operand
//     QS    = bf16(q.e^G) @ bf16(S)
//     AD    = bf16(qk) @ bf16(v_new)
//     out   = BF16_RNE(QS + AD)
//                                         _state_update
//     S     = S . exp(G_last) + bf16(k.e^{G_last-G})^T @ bf16(v_new)
//
// NUMERICS. Protocol §A row P1: every GEMM operand is a SINGLE BF16 RNE copy, every
// accumulator is FP32 through mma.m16n8k16. The T leaves are FP32 forward
// substitution (not a TC op) and the merges are TC GEMMs whose T sides carry the T
// encoding and whose A[b,a] operand is the ordinary BF16 copy. Persistent S, the G
// cumsum and every elementwise step are FP32. exp is __expf with no global
// fast-math, and every argument is a difference of two non-positive log-cumulatives
// (G_j - G_i, G_last - G_i), so gamma is never formed: G reaches -87.7 for the
// fastest-decaying heads and exp(G)/exp(G) would be 0/0.
//
// GEOMETRY. BM = C = 64 chunk rows, BN = VW = 128/s value columns per CTA
// (blockIdx.x is the strip, unique column writer, §A's s = 4/8/16), BK = 16 (the mma
// k-step; the state operand streams in 16-row d_k blocks). 256 threads = 8 warps.
// S[:, j0:j0+VW] stays in registers for the whole call: 8/16/4 floats per thread at
// s=8/4/16, which is §E's 8/4/2 per-split register line.
//
// MMA COUNT. Per CTA per chunk at s=8 (VW=16): 1032 mma.m16n8k16, which is 8 warps
// x 132 less the 24 instructions the six warps that skip the 16->32 merge do not
// issue. Per head per chunk the eight strips run 8256, against protocol §D's 2856
// for the phase-separated form. The difference is exactly the fused form repeating
// the per-(head,chunk) local work (KK^T, merges, W, QK = 808) once per strip; §D's
// 2856 is unchanged as the contract.
//
// ARITHMETIC PER CTA PER CHUNK (mma.m16n8k16, protocol §D geometry):
//   KK^T [64x128]x[128x64]      256     merges 16->32->64      40
//   W = T @ (beta.K.e^G)        256     QK^T                   256
//   U = T @ (beta.V)             32     AD                     32
//   WS = W @ S                   64     QS                     64
//   state kdec^T @ v_new         64                          ----
//                                                1064 -> 1032 issued (see above)
//
// RESOURCES. ptxas -v, nvcc 13.2.86, sm_86 and sm_89 identical:
//   entry                      registers  spill stores/loads  dynamic smem
//   gdn_chunkwise_p1_core_s4         228            0               40,704 B
//   gdn_chunkwise_p1_core_s8         168            0               34,560 B
//   gdn_chunkwise_p1_core_s16        168            0               31,488 B
// The disassembly carries ZERO LDL/STL, so the state tile never leaves the register
// file; 256 x 168 = 43,008 registers of the 65,536 an SM has, so one CTA per SM.
// §E's 96/80/72 register line was a conservative initial bound and the 40-42 KiB
// local/inverse/output smem target is met by all three arenas.
// =====================================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "gdn_cw_p1_layout.h"

#define P1_C       64     // chunk rows
#define P1_SUB     16     // UT inverse diagonal block (protocol §A)
#define P1_THREADS 256    // protocol §E: 256 threads for the state pass
#define P1_WARPS   (P1_THREADS / 32)
#define P1_DK      128
#define P1_DV      128
#define P1_HEADS   48
#define P1_MT      (P1_C / 16)          // row tiles of 16
#define P1_NT_C    (P1_C / 8)           // column tiles of 8 in the [C][C] products
#define P1_VL      72                    // sV row stride for the CHUNK axis, bf16 (+8 off 64)
#define P1_VLDK    136                   // sV row stride for the d_k axis, bf16 (+8 off 128)
#define P1_A_LD    68                    // sA row stride, floats (+4 off 64)
#define P1_T_LD    72                    // sT row stride, bf16 (+8 off 64)

// Shared arena, in order: sA | sT | sV | sVn | sG | sBeta | sSc. sV is [output column]
// x [reduction] — see P1_VL and p1_b_sbf. The region map and its lifetimes are in
// gdn_cw_p1_layout.h; the aliases (sA also carries the WS accumulator and the AD+QS
// output; sT also carries the encoded qk; sV also carries the S staging) are noted at
// each use.
#define P1_SMEM_BYTES(VW)                                                        \
    (P1_C * P1_A_LD * 4 + P1_C * P1_T_LD * 2 + P1_C * P1_VL * 2                 \
     + P1_C * (VW) * 4 + 3 * P1_C * 4)

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

// -------------------------------------------------------------------------------------
// DEBUG DUMP PATH (additive; production entry points instantiate DUMP=false and compile
// every block below away). One CTA — head 0, value strip 0 — writes each pipeline stage
// to global memory so it can be diffed against gdn_phase2_model_w.py's same-stage arrays.
// Offsets are a single contiguous float array; P1_DBG_LAYOUT is the index it shares with
// the host runner, so the two cannot drift.
// -------------------------------------------------------------------------------------
#define P1_DBG_G      0                       // [64]        within-chunk cumsum G
#define P1_DBG_A      64                      // [64][64]    A, masked
#define P1_DBG_T      (64 + 64 * 64)          // [64][64]    T, T-class bf16 as f32
#define P1_DBG_W      (64 + 64 * 64 + 64 * 64)// [4][8][16][16] W per (row tile, dk block)
#define P1_DBG_U      (64 + 128 * 64 + 4 * 8 * 256)          // [64][VW]
#define P1_DBG_WS     (P1_DBG_U + 64 * 32)    // [64][VW]
#define P1_DBG_VNEW   (P1_DBG_WS + 64 * 32)   // [64][VW]
#define P1_DBG_QK     (P1_DBG_VNEW + 64 * 32) // [64][64]    qk, T-class bf16 as f32
#define P1_DBG_AD     (P1_DBG_QK + 64 * 64)   // [64][VW]
#define P1_DBG_OUT    (P1_DBG_AD + 64 * 32)   // [64][VW]    the BF16 output as f32
#define P1_DBG_SNEW   (P1_DBG_OUT + 64 * 32)  // [128][VW]   the state strip after the update
#define P1_DBG_BETA   (P1_DBG_SNEW + 128 * 32 * 4)  // [64]    the per-row beta scale
#define P1_DBG_RAWA   (P1_DBG_BETA + 64)      // [64][64]    A before the tril/exp mask
#define P1_DBG_SV     (P1_DBG_RAWA + 64 * 64) // [8][16][16] sV after the WS S staging
#define P1_DBG_SR     (P1_DBG_SV + 8 * 16 * 16)  // [8][16][16] Sr right after the load
#define P1_DBG_SR2    (P1_DBG_SR + 8 * 16 * 16)   // [8][16][16] Sr after the state update
// 32->64 merge chain, dumped one 32x32 block each: T11 | T22 | A21 | X (=T22@A21)
#define P1_DBG_MT11   (P1_DBG_SR2 + 8 * 16 * 16)
#define P1_DBG_MT22   (P1_DBG_MT11 + 32 * 32)
#define P1_DBG_MA21   (P1_DBG_MT22 + 32 * 32)
#define P1_DBG_MX     (P1_DBG_MA21 + 32 * 32)
#define P1_DBG_BYTES  (P1_DBG_MX + 32 * 32)

struct AFrag { uint32_t r[4]; };
struct BFrag { uint32_t r[2]; };

// C and D alias: the accumulator is both the mma input and its output.
#define P1_MMA(d, a, b)                                                        \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"     \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])              \
                 : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]),         \
                   "r"(b.r[0]), "r"(b.r[1]))

// Fragment geometry (PTX ISA 9.7.16.5.8, mma.m16n8k16 floating point):
//   groupID = laneid >> 2, threadID_in_group = laneid & 3
//   A: row groupID (a0,a1) / groupID+8 (a2,a3); col 2*threadID_in_group + (i&1)
//      for a0,a1 and +8 for a2,a3.  B: col groupID; row 2*tig + (i&1) for b0 and
//      +8 for b1.  C/D: row groupID (c0,c1) / groupID+8 (c2,c3); col 2*tig+(i&1).
// Register i holds element 2i in its LOW half.
#define P1_GID(l) ((l) >> 2)
#define P1_T2(l)  (((l) & 3) << 1)

__device__ __forceinline__ uint32_t p1_pack(float lo, float hi) {
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(lo))
         | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hi)) << 16);
}



// ---- fragment loads ----------------------------------------------------------------

// A from FP32 rows, one scale per ROW (beta, and the q liveness mask).
// PTX ISA 9.7.16.5.8, mma.m16n8k16 matrix A: elements 0,1 are row groupID, elements
// 2,3 are row groupID+8, elements 4,5 are row groupID, elements 6,7 are row groupID+8,
// and the column is 2*tig + (i&1), +8 for i >= 4. So register 1 is the +8 row at the
// LOW columns and register 2 the base row at the +8 columns — the two are not
// interchangeable, and swapping them computes a different product.
__device__ __forceinline__ AFrag p1_a_rows(const float* lo, const float* hi, int col0,
                                           int t2, float s_lo, float s_hi) {
    AFrag f;
    f.r[0] = p1_pack(lo[col0 + t2] * s_lo, lo[col0 + t2 + 1] * s_lo);
    f.r[1] = p1_pack(hi[col0 + t2] * s_hi, hi[col0 + t2 + 1] * s_hi);
    f.r[2] = p1_pack(lo[col0 + t2 + 8] * s_lo, lo[col0 + t2 + 9] * s_lo);
    f.r[3] = p1_pack(hi[col0 + t2 + 8] * s_hi, hi[col0 + t2 + 9] * s_hi);
    return f;
}

// One element of (k . e^{G_last-G}) at chunk row ci and d_k column dk. The state
// update's A operand is the TRANSPOSE: the fragment's rows are d_k indices and its
// columns are chunk rows, so a register pair's two elements are in DIFFERENT chunk
// rows at the SAME d_k column, not adjacent elements of one row.
__device__ __forceinline__ float p1_kdec(const float* kh, size_t ld, int row0, int valid,
                                          int ci, int dk, const float* sG, float gl) {
    const int gr = row0 + ci;
    if (gr >= valid) return 0.0f;
    return kh[(size_t)gr * ld + dk] * __expf(gl - sG[ci]);
}

__device__ __forceinline__ AFrag p1_a_kdec(const float* kh, size_t ld, int row0, int valid,
                                           const float* sG, float gl, int dk, int t2) {
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

// B from a row-major FP32 buffer: fragment row n, columns 2tig + {0,1} and +8.
__device__ __forceinline__ BFrag p1_b_rows(const float* p, int ld, int n, int col0, int t2,
                                           float s0, float s1, float s8, float s9) {
    const float* r = p + (size_t)n * ld;
    BFrag f;
    f.r[0] = p1_pack(r[col0 + t2] * s0, r[col0 + t2 + 1] * s1);
    f.r[1] = p1_pack(r[col0 + t2 + 8] * s8, r[col0 + t2 + 9] * s9);
    return f;
}

// A from a BF16 shared buffer (T class, encoded qk). Same register order as p1_a_rows.
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

// B from a BF16 shared buffer (beta.V, the encoded v_new, the S copy).
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

// One element of (beta . K . e^G) at chunk row ci, d_k column dk. This is the only
// GEMM whose B fragment reads the input TRANSPOSED: its fragment row is a d_k index
// and its fragment column is a chunk row.
__device__ __forceinline__ float p1_kbg(const float* kh, size_t ld, int row0, int valid,
                                        int ci, int dk, const float* sSc) {
    const int gr = row0 + ci;
    const float* p = kh + (size_t)(gr < valid ? gr : row0) * ld + dk;
    return p[0] * sSc[ci];
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

// ---- debug dump helpers (template parameter DUMP, so production compiles them away) ----

__device__ __forceinline__ float p1_lo(uint32_t r) {
    return __bfloat162float(__ushort_as_bfloat16((unsigned short)(r & 0xffffu)));
}
__device__ __forceinline__ float p1_hi(uint32_t r) {
    return __bfloat162float(__ushort_as_bfloat16((unsigned short)(r >> 16)));
}

template <bool DUMP>
__device__ __forceinline__ void p1_dump_sf32(float* dst, const float* src, int ld, int rows,
                                             int cols) {
    if (!DUMP) return;
    for (int r = threadIdx.x; r < rows; r += P1_THREADS)
        for (int c = 0; c < cols; ++c) dst[(size_t)r * cols + c] = src[(size_t)r * ld + c];
}

template <bool DUMP>
__device__ __forceinline__ void p1_dump_sbf16(float* dst, const __nv_bfloat16* src, int ld,
                                              int rows, int cols) {
    if (!DUMP) return;
    for (int r = threadIdx.x; r < rows; r += P1_THREADS)
        for (int c = 0; c < cols; ++c)
            dst[(size_t)r * cols + c] = __bfloat162float(src[(size_t)r * ld + c]);
}

// The W block lives in four packed registers, in the mma.m16n8k16 A order: r0 is row
// gid at the low columns, r1 is row gid+8 at the low columns, r2 is row gid at +8 and
// r3 is row gid+8 at +8. Each register's low/high halves are the pair 2*tig and 2*tig+1
// (r0, r1) or 2*tig+8 and +9 (r2, r3) of the 16-wide dk step.
template <bool DUMP>
__device__ __forceinline__ void p1_dump_w(float* dst, const AFrag& wa, int gid, int t2) {
    if (!DUMP) return;
    float* lo_row = dst + (size_t)gid * 16 + t2;
    float* hi_row = dst + (size_t)(gid + 8) * 16 + t2;
    lo_row[0] = p1_lo(wa.r[0]);
    lo_row[1] = p1_hi(wa.r[0]);
    lo_row[8] = p1_lo(wa.r[2]);
    lo_row[9] = p1_hi(wa.r[2]);
    hi_row[0] = p1_lo(wa.r[1]);
    hi_row[1] = p1_hi(wa.r[1]);
    hi_row[8] = p1_lo(wa.r[3]);
    hi_row[9] = p1_hi(wa.r[3]);
}

// ---- core ---------------------------------------------------------------------------

template <int VW, bool DUMP = false>
__device__ __forceinline__ void gdn_chunkwise_p1_t(
        const float* __restrict__ q,      // FP32 [M, HEADS, DK], pre-scaled by 1/sqrt(DK)
        const float* __restrict__ k,      // FP32 [M, HEADS, DK]
        const float* __restrict__ v,      // FP32 [M, HEADS, DV]
        const float* __restrict__ beta,   // FP32 [M, HEADS]
        const float* __restrict__ g,      // FP32 [M, HEADS]
        float* __restrict__ S,            // FP32 [HEADS, DK, DV]
        __nv_bfloat16* __restrict__ out,  // BF16 [M, HEADS, DV]
        const GdnMeta* __restrict__ meta,
        int valid_rows_in,
        float* __restrict__ dbg) {
    static_assert(VW == 8 || VW == 16 || VW == 32,
                  "the value strip must be DV/4, DV/8 or DV/16 (protocol §A s = 4/8/16)");
    static_assert(P1_SMEM_BYTES(VW) == (VW == 32 ? GDN_CW_P1_S4_BYTES
                                       : VW == 16 ? GDN_CW_P1_S8_BYTES
                                                  : GDN_CW_P1_S16_BYTES),
                  "kernel arena and gdn_cw_p1_layout.h disagree on the shared-memory size");
    constexpr int NT = VW / 8;      // 8-column value tiles
    constexpr int VLD = P1_VL;      // sV row stride in bf16. The OUTPUT column is the row and
                                   // the mma reduction axis is the column: p1_b_sbf packs two
                                   // consecutive reduction values into one 32-bit register, so
                                   // every sV staging below writes [value][reduction].

    const int head = blockIdx.y;
    const int j0 = blockIdx.x * VW;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = P1_GID(lane);
    const int t2 = P1_T2(lane);

    const bool aligned = (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k)
                          | reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(beta)
                          | reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(S)
                          | reinterpret_cast<uintptr_t>(out)) % 16 == 0;
    if (blockDim.x != P1_THREADS || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1
        || gridDim.y != P1_HEADS || gridDim.x * VW != P1_DV || !aligned) {
        __trap();
        return;
    }
    // DEBUG: which head/strip this CTA dumps (meta->r20/r24; 0 in production).
    const int dbg_head = meta->r20;
    const int dbg_strip = meta->r24;
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
    // sVn is reused as a BF16 staging area by the 16->32 merge's second product. The
    // 16->32 intermediate is P1_SUB x P1_SUB fp32 (its first 1024 B of this region); the
    // transposed T[a,a] staging goes in the SECOND half (offset 512 bf16 elements), so
    // the two never overlap.
    __nv_bfloat16* const sVnB = reinterpret_cast<__nv_bfloat16*>(sVn);
    float* const sG   = sVn + P1_C * VW;
    float* const sBeta = sG + P1_C;
    float* const sSc  = sBeta + P1_C;

    const size_t rstride = (size_t)P1_HEADS * P1_DK;
    const float* const kh = k + (size_t)head * P1_DK;
    const float* const qh = q + (size_t)head * P1_DK;
    const float* const vh = v + (size_t)head * P1_DV + j0;

    // S[:, j0:j0+VW] stays in REGISTERS for the whole call. Warp w owns the d_k rows
    // 16w..16w+15; lane l owns rows +gid and +gid+8 and, per 8-column tile, columns
    // 8*nt + 2*t + c. Every index into Sr is compile-time, which is what keeps the
    // tile out of local memory.
    float Sr[2][NT][2];
    {
        // Warp w owns d_k rows 16w..16w+15, so the 8 warps cover all P1_DK = 128 of
        // them. Without the warp*16 term every warp loads the same 16 rows and 112
        // rows of the state are never read, never updated and never written back.
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

    if (DUMP && dbg != nullptr && head == dbg_head && blockIdx.x == dbg_strip)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                dbg[P1_DBG_SR + warp * 256 + gid * 16 + 8 * nt + t2 + c] = Sr[0][nt][c];
                dbg[P1_DBG_SR + warp * 256 + (gid + 8) * 16 + 8 * nt + t2 + c] = Sr[1][nt][c];
            }
    const int nchunk = (valid_rows + P1_C - 1) / P1_C;
    // One CTA writes the dump: head 0, value strip 0. CTA-uniform, so no __syncthreads
    // inside the dump helpers can be reached by only part of the block.
    const bool dumping = DUMP && dbg != nullptr && head == dbg_head && blockIdx.x == dbg_strip;
    for (int c = 0; c < nchunk; ++c) {
        const int row0 = c * P1_C;
        // Every dump below is of chunk 0 only: the stage hunt runs on the smallest
        // fixture, which is a single chunk.
        const bool dmp = dumping && c == 0;

        // ---- gates and the in-chunk cumsum. ONE thread scans, then a barrier:
        // interleaved in-place scans on shared memory apply G repeatedly and
        // without bound, which is what produced e^G overflow and NaNs.
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
                P1_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();
        if (dmp) {
            p1_dump_sf32<DUMP>(dbg + P1_DBG_BETA, sBeta, 1, P1_C, 1);
            p1_dump_sf32<DUMP>(dbg + P1_DBG_RAWA, sA, P1_A_LD, P1_C, P1_C);
            __syncthreads();
        }
        // tril BEFORE the exp and AGAIN after it: exp of a masked entry is 1.0, and
        // without the second mask every token reads the future.
        for (int idx = tid; idx < P1_C * P1_C; idx += P1_THREADS) {
            const int j = idx / P1_C, i = idx - j * P1_C;
            sA[j * P1_A_LD + i] = j > i ? -sA[j * P1_A_LD + i] * __expf(sG[j] - sG[i]) : 0.0f;
        }
        if (dmp) {
            p1_dump_sf32<DUMP>(dbg + P1_DBG_G, sG, 1, P1_C, 1);
            __syncthreads();
            p1_dump_sf32<DUMP>(dbg + P1_DBG_A, sA, P1_A_LD, P1_C, P1_C);
        }
        __syncthreads();

        // ---- T = (I - A)^-1, blocked: 16x16 FP32 leaves by forward substitution,
        // then 16->32 and 32->64 merges. This REPLACES the shipped kernel's row-wise
        // form, whose history records one blocked UT form that did not compute the
        // inverse at all (off by 1.5-2.5 max-abs against a dense inverse).
        // ONE row per iteration, with the barrier between rows. Row r reads the
        // already-final entries sA[m][j] of rows m < r, so the rows must be separated
        // by a barrier; and each row must be substituted once — repeating the pass over
        // all rows re-applies it to values that are already T and destroys them.
        for (int rr = 1; rr < P1_SUB; ++rr) {
            if (tid < P1_C) {
                const int blk = tid >> 4;
                float* row = sA + (size_t)(blk * P1_SUB + rr) * P1_A_LD;
                for (int j = 0; j < rr; ++j) {
                    float acc = row[j];
                    #pragma unroll
                    for (int m = j + 1; m < P1_SUB; ++m) {
                        if (m < rr)
                            acc = fmaf(row[m], sA[(size_t)(blk * P1_SUB + m) * P1_A_LD + j],
                                       acc);
                    }
                    row[j] = acc;
                }
            }
            __syncthreads();
        }
        for (int r = tid; r < P1_C; r += P1_THREADS) sA[r * P1_A_LD + r] += 1.0f;
        __syncthreads();   // the encode below reads ALL of sA, including diagonals other
                           // threads just incremented; without this the diagonal is
                           // intermittently 0 and T is not the inverse at all
        // The leaves become the merge operands in the T encoding; the fp32 T is dead
        // from here, so sA is free for the WS accumulator and later for qk.
        for (int idx = tid; idx < P1_C * P1_C; idx += P1_THREADS) {
            const int r = idx / P1_C, cc = idx - r * P1_C;
            sT[r * P1_T_LD + cc] = __float2bfloat16_rn(sA[r * P1_A_LD + cc]);
        }
        __syncthreads();

        // ---- merges. L = [[L11,0],[L21,L22]] and L = I - A give L21 = -A21, so the
        // two signs cancel and each level is T[b,a] = T[b,b] @ A[b,a] @ T[a,a] with no
        // leading sign, on the ORIGINAL off-diagonal block of A.
        for (int p = 0; p < P1_C / (2 * P1_SUB); ++p) {
            const int a0 = p * 2 * P1_SUB, b0 = a0 + P1_SUB;
            __syncthreads();
            // A[b,a] TRANSPOSED into sV as [a][b]: p1_b_sbf packs two consecutive
            // reduction (b) indices into one 32-bit register, so b must be contiguous;
            // sA holds the block as [b][a] and cannot be read directly. A is the
            // ordinary BF16 copy in a merge (protocol §A), and the diagonal 16x16
            // blocks of sA are the only ones the leaf substitution wrote, so the
            // off-diagonal block here is still the ORIGINAL A.
            // Two transposed stagings, both because a B fragment reads memory indexed
            // [output column][reduction]: A[b,a] as [a][b] for the first product, and
            // T[a,a] as [a'][a] for the second. Reading T straight out of sT instead
            // computes X @ T^T.
            for (int idx = tid; idx < P1_SUB * P1_SUB; idx += P1_THREADS) {
                const int bl = idx / P1_SUB, al = idx - bl * P1_SUB;
                sV[al * 32 + bl] = __float2bfloat16_rn(
                    sA[(size_t)(b0 + bl) * P1_A_LD + (a0 + al)]);
            }
            __syncthreads();
            if (warp < 2) {                       // 2 column tiles of a 16-wide block
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const AFrag a = p1_a_sbf(sT, P1_T_LD, b0, b0, gid, t2);
                const BFrag b = p1_b_sbf(sV, 32, gid, 0, t2);
                P1_MMA(d, a, b);
                p1_store_f32(sVn, P1_SUB, 0, 0, gid, t2, d);
            }
            __syncthreads();
            for (int idx = tid; idx < P1_SUB * P1_SUB; idx += P1_THREADS) {
                const int al = idx / P1_SUB, bl = idx - al * P1_SUB;
                sVnB[512 + bl * 32 + al] = __float2bfloat16_rn(
                    sT[(size_t)(a0 + bl) * P1_T_LD + (a0 + al)]);
            }
            __syncthreads();
            if (warp < 2) {
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const AFrag a = p1_a_sf32(sVn, P1_SUB, 0, 0, gid, t2);
                const BFrag b = p1_b_sbf(sVnB + 512, 32, gid, 0, t2);
                P1_MMA(d, a, b);
                p1_store_bf16(sT, P1_T_LD, b0, a0, gid, t2, d);
            }
            __syncthreads();
        }
        {   // 32 -> 64. The (x@y) intermediate goes to sA[0:32][32:64], which is
            // strictly ABOVE A's lower triangle and so disjoint from A[32:64][0:32].
            const int mt = warp / 4, nt = warp - (warp / 4) * 4;
            // A[32:64][0:32] transposed into sV as [a][b], for the same reason as above.
            for (int idx = tid; idx < 32 * 32; idx += P1_THREADS) {
                const int bl = idx / 32, al = idx - bl * 32;
                sV[al * 64 + bl] = __float2bfloat16_rn(sA[(size_t)(32 + bl) * P1_A_LD + al]);
            }
            __syncthreads();
            if (DUMP && dbg != nullptr && head == dbg_head && blockIdx.x == dbg_strip)
                for (int idx = tid; idx < 32 * 32; idx += P1_THREADS) {
                    const int bl = idx / 32, al = idx - bl * 32;
                    dbg[P1_DBG_MT11 + idx] = __bfloat162float(sT[(size_t)al * P1_T_LD + bl]);
                    dbg[P1_DBG_MT22 + idx] =
                        __bfloat162float(sT[(size_t)(32 + al) * P1_T_LD + 32 + bl]);
                    dbg[P1_DBG_MA21 + idx] = __bfloat162float(sV[al * 64 + bl]);
                }
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            // The reduction axis of BOTH products here is 32 wide (b over [32,64) in the
            // first, a over [0,32) in the second), i.e. TWO 16-wide k-steps. A bound of
            // P1_SUB/16 = 1 contracts over half of it — invisible on the mt=0 half,
            // where T22's upper 16 columns are zero, and wrong on the mt=1 half.
            #pragma unroll
            for (int ks = 0; ks < P1_C / (2 * P1_SUB); ++ks) {
                const AFrag a = p1_a_sbf(sT, P1_T_LD, 32 + mt * 16, 32 + ks * 16, gid, t2);
                // col0 = ks*16: the reduction axis is b over [32,64), i.e. TWO 16-wide
                // steps, and the staged tile is indexed by the LOCAL b. Without ks*16
                // both steps read local b 0..15 and half the contraction is wrong.
                const BFrag b = p1_b_sbf(sV, 64, nt * 8 + gid, ks * 16, t2);
                P1_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, 32 + nt * 8, gid, t2, d);
            __syncthreads();
            if (DUMP && dbg != nullptr && head == dbg_head && blockIdx.x == dbg_strip)
                for (int idx = tid; idx < 32 * 32; idx += P1_THREADS)
                    dbg[P1_DBG_MX + idx] = sA[(size_t)(idx / 32) * P1_A_LD + 32 + (idx % 32)];
            // T[a,a] transposed into the tail of the same sV rows (offset 32), for the
            // second product's B operand: memory[n=a][k=a'] must hold T[a'][a].
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
                P1_MMA(e, a, b);
            }
            p1_store_bf16(sT, P1_T_LD, 32 + mt * 16, nt * 8, gid, t2, e);
            __syncthreads();
        }
        if (dmp) p1_dump_sbf16<DUMP>(dbg + P1_DBG_T, sT, P1_T_LD, P1_C, P1_C);

        // ---- U = T @ (beta . V) -> sVn. T is the T-class operand, beta.V the WU one.
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
                P1_MMA(d, a, b);
            }
            p1_store_f32(sVn, VW, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();
        if (dmp) p1_dump_sf32<DUMP>(dbg + P1_DBG_U, sVn, VW, P1_C, VW);

        // ---- WS = W @ S with W MATERIALIZED. W is a GEMM output, so it is rounded
        // to BF16 (WU class) before it multiplies the BF16 S copy: the fused form
        // produces one [16 x 16] d_k block of W, rounds it, and feeds it straight
        // into the next mma, which is element-for-element the model's W then WS.
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW;
            sA[r * P1_A_LD + (idx - r * VW)] = 0.0f;
        }
        // The S copy, staged ONCE per warp into its OWN slice of the 128-row d_k axis.
        // Warp w owns d_k rows 16w..16w+15 in registers (Sr) and the WS contraction is
        // partitioned over exactly those rows, so sV holds all eight warps' rows at once,
        // as [value][d_k]. All warps execute this staging, so they must not share the
        // destination: one shared slice makes each warp contract W against whichever
        // warp's rows landed last.
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
        for (int wm = 0; wm < P1_MT; ++wm) {
            float w0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, w1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < P1_C / 16; ++ks) {
                const AFrag a = p1_a_sbf(sT, P1_T_LD, wm * 16, ks * 16, gid, t2);
                #pragma unroll
                for (int nb = 0; nb < 2; ++nb) {
                    const int dk = kblk * 16 + nb * 8 + gid;
                    // The B fragment's k positions are the 16-wide window [ks*16,
                    // ks*16+16), so the chunk row is ks*16 + the in-window position.
                    const int c0 = ks * 16 + t2;
                    const BFrag b{ {p1_pack(p1_kbg(kh, rstride, row0, valid_rows, c0, dk, sSc),
                                            p1_kbg(kh, rstride, row0, valid_rows, c0 + 1, dk, sSc)),
                                    p1_pack(p1_kbg(kh, rstride, row0, valid_rows, c0 + 8, dk, sSc),
                                            p1_kbg(kh, rstride, row0, valid_rows, c0 + 9, dk, sSc))} };
                    float* wa16 = nb ? w1 : w0;
                    P1_MMA(wa16, a, b);
                }
            }
            AFrag wa;
            // Same register order as p1_a_rows: register 1 is row gid+8 at the low
            // columns, register 2 is row gid at the +8 columns. w0 is the first 8-wide
            // dk group of the block, w1 the second.
            wa.r[0] = p1_pack(w0[0], w0[1]);
            wa.r[1] = p1_pack(w0[2], w0[3]);
            wa.r[2] = p1_pack(w1[0], w1[1]);
            wa.r[3] = p1_pack(w1[2], w1[3]);
            if (dmp)
                p1_dump_w<DUMP>(dbg + P1_DBG_W + (size_t)(wm * 8 + kblk) * 256, wa, gid, t2);
            float* acc = sA + (size_t)(wm * 16 + gid) * P1_A_LD + t2;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const BFrag b = p1_b_sbf(sV, P1_VLDK, nt * 8 + gid, kblk * 16, t2);
                P1_MMA(o, wa, b);
                atomicAdd(acc + 8 * nt, o[0]);
                atomicAdd(acc + 8 * nt + 1, o[1]);
                atomicAdd(acc + 8 * P1_A_LD + 8 * nt, o[2]);
                atomicAdd(acc + 8 * P1_A_LD + 8 * nt + 1, o[3]);
            }
        }

        // ---- v_new = U - WS, in place in sVn
        __syncthreads();
        if (dmp) p1_dump_sf32<DUMP>(dbg + P1_DBG_WS, sA, P1_A_LD, P1_C, VW);
        __syncthreads();
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            sVn[r * VW + cc] -= sA[r * P1_A_LD + cc];
        }
        if (dmp) {
            __syncthreads();
            p1_dump_sf32<DUMP>(dbg + P1_DBG_VNEW, sVn, VW, P1_C, VW);
        }

        // ---- qk = (bf16(q) @ bf16(K)^T) . exp(G_j - G_i), tril WITH the diagonal,
        // into sA. sA held WS and is free now; sT held T and is dead since W.
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
                P1_MMA(d, a, b);
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
        if (dmp) {
            __syncthreads();
            p1_dump_sbf16<DUMP>(dbg + P1_DBG_QK, sT, P1_T_LD, P1_C, P1_C);
        }

        // ---- v_new as the WU-class operand (bf16), for AD and for the state update
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
                P1_MMA(d, a, b);
            }
            p1_store_f32(sA, P1_A_LD, mt * 16, nt * 8, gid, t2, d);
        }
        __syncthreads();
        if (dmp) p1_dump_sf32<DUMP>(dbg + P1_DBG_AD, sA, P1_A_LD, P1_C, VW);

        // ---- QS = (q . e^G) @ S, accumulated onto AD's tile. S is re-staged because
        // sV now holds the encoded v_new, which the state update still needs from the
        // FP32 sVn instead.
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
        // The whole 128-row d_k axis of S is in sV by now (every warp staged its own 16
        // rows above), so the owning warp of each tile does the FULL d_k contraction
        // itself over the eight 16-wide windows and adds the result onto AD's tile.
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
                P1_MMA(d, a, b);
            }
            o[0] = d[0];
            o[1] = d[1];
            o[8 * P1_A_LD] = d[2];
            o[8 * P1_A_LD + 1] = d[3];
        }
        __syncthreads();
        if (dmp) p1_dump_sf32<DUMP>(dbg + P1_DBG_OUT, sA, P1_A_LD, P1_C, VW);

        // ---- out = BF16_RNE(QS + AD)
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            const int gr = row0 + r;
            if (gr < valid_rows)
                out[((size_t)gr * P1_HEADS + head) * P1_DV + j0 + cc] =
                    __float2bfloat16_rn(sA[r * P1_A_LD + cc]);
        }

        // ---- S = S . exp(G_last) + (k . e^{G_last-G})^T @ v_new. The state update's
        // A operand is the TRANSPOSE, so its fragment rows are d_k indices and its
        // fragment columns are chunk rows: the scale is per column here.
        __syncthreads();
        for (int idx = tid; idx < P1_C * VW; idx += P1_THREADS) {
            const int r = idx / VW, cc = idx - r * VW;
            sV[cc * VLD + r] = __float2bfloat16_rn(sVn[r * VW + cc]);
        }
        __syncthreads();
        {
            const float decay = __expf(sG[P1_C - 1]);
            const float gl = sG[P1_C - 1];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float o[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
                #pragma unroll
                for (int ks = 0; ks < P1_C / 16; ++ks) {
                    // ks*16 is the chunk-row offset of the 16-wide k-step: without it
                    // every step contracts chunk rows 0..15 again, which is correct only
                    // while the chunk has <= 16 live rows.
                    const AFrag a = p1_a_kdec(kh, rstride, row0, valid_rows, sG, gl,
                                              warp * 16 + gid, ks * 16 + t2);
                    #pragma unroll
                    for (int nt2 = 0; nt2 < NT; ++nt2) {
                        const BFrag b = p1_b_sbf(sV, VLD, nt2 * 8 + gid, ks * 16, t2);
                        P1_MMA(o[nt2], a, b);
                    }
                }
                for (int cc = 0; cc < 2; ++cc) {
                    Sr[0][nt][cc] = fmaf(decay, Sr[0][nt][cc], o[nt][cc]);
                    Sr[1][nt][cc] = fmaf(decay, Sr[1][nt][cc], o[nt][cc + 2]);
                }
            }
        }
        if (DUMP && dbg != nullptr && head == dbg_head && blockIdx.x == dbg_strip)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    dbg[P1_DBG_SR2 + warp * 256 + gid * 16 + 8 * nt + t2 + cc] = Sr[0][nt][cc];
                    dbg[P1_DBG_SR2 + warp * 256 + (gid + 8) * 16 + 8 * nt + t2 + cc] =
                        Sr[1][nt][cc];
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
    if (DUMP && dbg != nullptr && head == dbg_head && blockIdx.x == dbg_strip) {
        // The store above is per thread; the dump reads the WHOLE 128-row strip, so it
        // must wait for every warp's stores to land.
        __syncthreads();
        p1_dump_sf32<DUMP>(dbg + P1_DBG_SNEW,
                           S + (size_t)head * P1_DK * P1_DV + j0, P1_DV, P1_DK, VW);
    }
}

// ABI-shaped wrappers. The value strip width is a compile-time property: the register
// state tile is indexed by it, so a runtime split would push the tile into local
// memory. Protocol §A fixes the split set at 4/8/16, one entry point each.
extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_core_s4(const float* q, const float* k, const float* v,
                              const float* beta, const float* g, float* S,
                              __nv_bfloat16* out, const GdnMeta* meta, int valid_rows) {
    gdn_chunkwise_p1_t<32>(q, k, v, beta, g, S, out, meta, valid_rows, nullptr);
}

extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_core_s8(const float* q, const float* k, const float* v,
                              const float* beta, const float* g, float* S,
                              __nv_bfloat16* out, const GdnMeta* meta, int valid_rows) {
    gdn_chunkwise_p1_t<16>(q, k, v, beta, g, S, out, meta, valid_rows, nullptr);
}

extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_core_s16(const float* q, const float* k, const float* v,
                               const float* beta, const float* g, float* S,
                               __nv_bfloat16* out, const GdnMeta* meta, int valid_rows) {
    gdn_chunkwise_p1_t<8>(q, k, v, beta, g, S, out, meta, valid_rows, nullptr);
}

// DEBUG entry points. Same computation, same grid, plus a P1_DBG_BYTES float buffer that
// head 0 / strip 0 fills with every pipeline stage. Nothing above is reachable from
// these, and nothing in them is reachable from the production entries above.
extern "C" __global__ __launch_bounds__(P1_THREADS, 1)
void gdn_chunkwise_p1_dbg_s8(const float* q, const float* k, const float* v,
                             const float* beta, const float* g, float* S,
                             __nv_bfloat16* out, const GdnMeta* meta, int valid_rows,
                             float* dbg) {
    gdn_chunkwise_p1_t<16, true>(q, k, v, beta, g, S, out, meta, valid_rows, dbg);
}