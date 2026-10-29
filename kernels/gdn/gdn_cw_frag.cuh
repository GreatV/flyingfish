// mma.m16n8k16 BF16 fragment loads, packs and D stores shared by the chunkwise GDN kernels.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

struct AFrag { uint32_t r[4]; };
struct BFrag { uint32_t r[2]; };

// C and D alias: the accumulator is both the mma input and its output.
#define CW_MMA(d, a, b)                                                        \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"     \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])              \
                 : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]),         \
                   "r"(b.r[0]), "r"(b.r[1]))

// Fragment geometry (PTX ISA 9.7.16.5.8, mma.m16n8k16), groupID = laneid >> 2, tig = laneid & 3: A row groupID (a0,a1) / groupID+8 (a2,a3) at column 2*tig+(i&1), +8 for a2,a3; B column groupID, row 2*tig+(i&1) for b0 and +8 for b1; C/D row groupID (c0,c1) / groupID+8 (c2,c3) at column 2*tig+(i&1); register i holds element 2i in its LOW half.
#define CW_GID(l) ((l) >> 2)
#define CW_T2(l)  (((l) & 3) << 1)

template<bool Residual = false>
__device__ __forceinline__ uint32_t p1_pack(float lo, float hi) {
    if constexpr (Residual) {
        lo = __fsub_rn(lo, __bfloat162float(__float2bfloat16_rn(lo)));
        hi = __fsub_rn(hi, __bfloat162float(__float2bfloat16_rn(hi)));
    }
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(lo))
         | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hi)) << 16);
}

// One element of (k . e^{G_last-G}) at chunk row ci and d_k column dk; the state update's A operand is the TRANSPOSE, so a register pair's two elements sit in DIFFERENT chunk rows at the SAME d_k column.
__device__ __forceinline__ float p1_kdec(const float* kh, size_t ld, int row0,
                                          int valid, int ci, int dk, const float* sG, float gl) {
    const int gr = row0 + ci;
    if (gr >= valid) return 0.0f;
    return kh[(size_t)gr * ld + dk] * __expf(gl - sG[ci]);
}

template<bool Residual = false>
__device__ __forceinline__ AFrag p1_a_kdec(const float* kh, size_t ld, int row0,
                                           int valid, const float* sG, float gl, int dk, int t2) {
    const int c0 = t2;
    AFrag f;
    f.r[0] = p1_pack<Residual>(p1_kdec(kh, ld, row0, valid, c0, dk, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 1, dk, sG, gl));
    f.r[1] = p1_pack<Residual>(p1_kdec(kh, ld, row0, valid, c0, dk + 8, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 1, dk + 8, sG, gl));
    f.r[2] = p1_pack<Residual>(p1_kdec(kh, ld, row0, valid, c0 + 8, dk, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 9, dk, sG, gl));
    f.r[3] = p1_pack<Residual>(p1_kdec(kh, ld, row0, valid, c0 + 8, dk + 8, sG, gl),
                     p1_kdec(kh, ld, row0, valid, c0 + 9, dk + 8, sG, gl));
    return f;
}

template<bool Residual = false>
__device__ __forceinline__ float p1_kdec_channel(const float* kh, size_t ld, int row0,
                                                int valid, int ci, int dk,
                                                const float* g, int head, int channel,
                                                int last) {
    const int gr = row0 + ci;
    if (gr >= valid) return 0.0f;
    const float gl = g[((size_t)last * P2_HEADS + head) * P2_DV + channel];
    const float gi = g[((size_t)gr * P2_HEADS + head) * P2_DV + channel];
    return kh[(size_t)gr * ld + dk] * __expf(gl - gi);
}

template<bool Residual = false>
__device__ __forceinline__ AFrag p1_a_kdec_channel(const float* kh, size_t ld, int row0,
                                                  int valid, const float* g, int head,
                                                  int channel, int last, int dk, int t2) {
    const int c0 = t2;
    AFrag f;
    f.r[0] = p1_pack<Residual>(p1_kdec_channel(kh, ld, row0, valid, c0, dk, g, head, channel, last),
                               p1_kdec_channel(kh, ld, row0, valid, c0 + 1, dk, g, head, channel, last));
    f.r[1] = p1_pack<Residual>(p1_kdec_channel(kh, ld, row0, valid, c0, dk + 8, g, head, channel, last),
                               p1_kdec_channel(kh, ld, row0, valid, c0 + 1, dk + 8, g, head, channel, last));
    f.r[2] = p1_pack<Residual>(p1_kdec_channel(kh, ld, row0, valid, c0 + 8, dk, g, head, channel, last),
                               p1_kdec_channel(kh, ld, row0, valid, c0 + 9, dk, g, head, channel, last));
    f.r[3] = p1_pack<Residual>(p1_kdec_channel(kh, ld, row0, valid, c0 + 8, dk + 8, g, head, channel, last),
                               p1_kdec_channel(kh, ld, row0, valid, c0 + 9, dk + 8, g, head, channel, last));
    return f;
}

// A from a BF16 buffer (T, encoded qk, materialized W), shared or global; same register
// order as p1_a_rows.
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
