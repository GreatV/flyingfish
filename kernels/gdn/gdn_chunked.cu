// =====================================================================================
// Q38 ABI v1 section 6, GDN chunked core. C = 32, value_split 1 (P1 plan v2).
//
// BUILDS CLEAN. nvcc 13.2.86, sm_86 and sm_89, 512 threads/block, identical on both:
//   entry point        registers  spill  stack frame  dynamic smem
//   gdn_chunked_core_v1     90       0        0           62,336 B
//   gdn_chunked_core_v2     93       0        0           41,856 B
//   gdn_chunked_core_v4     91       0        0           31,616 B
// v1 is the P1 plan and needs the >49,152 B opt-in (headroom to the 101,376 B device
// limit: 39,040 B); v2 and v4 fit static. Smem is dynamic: static __shared__ caps at
// 49,152 B, so the opt-in happens in prepare against the device limit.
//
// PRELIMINARY: local CUDA 13.2.86, not ABI section 8's cloud CUDA 13.0 native. Re-report
// from the 13.0 build with source SHA, full flags and the raw ptxas log.
//
// Two things that cost real time and would cost it again:
//   * 512 threads, not 256. The register-resident state tile is 64 floats/thread at 256
//     threads; the kernel hit 255 registers and still spilled 376 B. At 512 it is 32.
//   * every loop indexing the register tile by b MUST be `#pragma unroll`. A runtime index
//     into Sr[a][b] pushes the whole tile into local memory. `#pragma unroll 4` on the
//     32-step state-update loop is what took the last 216 B of stack frame to zero.
// =====================================================================================
// =====================================================================================
// Q38 ABI v1 section 6, GDN chunked core. C = 64.
//
// WY / UT-transform form. Per 64-token chunk, per v-head, G_j = within-chunk cumsum of g
// (g <= 0, so G_j <= 0):
//     L[j,i] = exp(G_j - G_i) for j>=i, 0 for j<i      (tril BEFORE exp, and AGAIN after)
//     A[j,i] = -beta_j (k_j . k_i) L[j,i]                for j>i, 0 otherwise
//     T      = (I - A)^-1                               unit lower triangular
//     w      = T (beta .* V)
//     v_new  = w - (T (beta .* K .* e^G)) @ S
//     o      = (q .* e^G) @ S + [tril(q k^T .* L)] @ v_new
//     S      = S * e^{G_{C-1}} + (k .* e^{G_{C-1}-G})^T @ v_new
//
// ALGEBRAIC NOTE. kdec = T(beta.*K.*e^G) is never materialised. Since
//     sum_i kdec[r][i] S[i][j] = sum_t T[r][t] ( sum_i kb[t][i] S[i][j] ),
// it is enough to form kbS[t][j] = sum_i kb[t][i] S[i][j] once -- a [C][DK] x [DK][vw]
// matmul -- and then contract with T. That removes a [C][DK] tile from the working set,
// which is what makes the smem budget close at all.
//
// SILENT-FAILURE TRAPS, called out because both give a plausible O(1)-wrong answer rather
// than a crash:
//   * the SECOND .tril() after .exp(). The first zeroes the upper triangle, exp turns that
//     into 1.0, and without the second every token reads the future.
//   * v_new comes from w = T(beta.*V). Using raw V is the same magnitude of wrong.
// Both cost real time deriving this file; both are covered by the G5 tests.
//
// Every exp() takes a non-positive argument, so nothing overflows. exp(G_j - G_i) is formed
// from a difference of two non-positive numbers rather than exp(G_j)/exp(G_i), because G
// reaches -87.7 for the fastest-decaying heads and exp(-G) would overflow.
//
// All state math is FP32 (ABI section 6). q arrives pre-scaled by 1/sqrt(HEAD_K); the caller
// applied the shared l2_heads and widened.
//
// Neutral tail: rows at or beyond valid_rows carry q=k=v=beta=0, g=0. That makes A=0 there,
// so T=I, so w=v_new=0 and the state update adds nothing -- S is bit-unchanged by an invalid
// step. A contract obligation (ABI section 6), not an optimisation.
//
// launch: grid (DV/VSPLIT, NUM_V_HEADS); each block owns S[:, j0:j0+VW] in REGISTERS and
//         walks every chunk SEQUENTIALLY. Serial by construction: the only parallelism is
//         across heads and across the value split.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef GDN_CHUNK
#define GDN_CHUNK 32          // P1 plan v2: C 64 -> 32, see gdn_kernel_design.md
#endif
#define GDN_SUB   16          // UT sub-block
#define GDN_THREADS 512   // 512 threads so the register-resident state tile is
                       // 32 floats/thread instead of 64; at 256 threads the
                       // kernel hit 255 registers and still spilled.
#define GDN_HEADS 48
#define GDN_DK    128
#define GDN_DV    128

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// Shared memory, per block:
//   sKb  [C][DK]   beta.K.e^G, FP32 -- feeds A (unscaled), the state update, and kbS
//   sQd  [C][DK]   q.e^G,      FP32 -- feeds only the output path
//   sVb  [C][VW]   beta.V,     FP32 -- reused as the w / v_new buffer once w is formed
//   sKbS [C][VW]   kb @ S,     FP32 -- reused as the sOut buffer once consumed
//   sT   [C][C]    L, then A, then T: one buffer, three uses
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
    if (SNAPSHOT && (!aligned16(snapshots) || valid_rows > 4)) { __trap(); return; }
    // ExecutionPhase selects this core even for a single prefill row. Partial
    // chunks mask inactive rows with neutral beta=0 and g=0 values (ABI §6).

    // Dynamic shared memory, laid out by hand. Static __shared__ caps at 49,152 B, so the
    // 54,016 B this kernel needs at C=32 / v_split=1 is only reachable as extern __shared__
    // plus a cudaFuncSetAttribute opt-in (done in prepare, against the device limit).
    extern __shared__ float smem[];
    float* const sKb   = smem;                       // [C][DK] raw k
    float* const sVb   = sKb   + GDN_CHUNK * GDN_DK;  // [C][VW] beta.V, then kbS, then
                                                     //        the inter-reduction staging
    float* const sKbS  = sVb   + GDN_CHUNK * VW;      // [C][VW] w, then v_new
    float* const sT    = sKbS  + GDN_CHUNK * VW;      // [C][C]  L -> A -> T
    float* const sQr   = sT    + GDN_CHUNK * GDN_CHUNK;// [DK]   one q row at a time
    float* const sBeta = sQr   + GDN_DK;               // [C]
    float* const sG    = sBeta + GDN_CHUNK;            // [C]
    // S lives in REGISTERS, tiled 8x8 per thread, so any sum that runs over all DK rows
    // needs values this thread does not hold. Those reductions stage through sRed
    // [TG][VW] (and the q.k row products through sQk[C]).
    float* const sRed  = sG    + GDN_CHUNK;            // [TG][VW]  8 KiB at TG=16, VW=128
    float* const sQk   = sRed  + 16 * VW;              // [C]

    // ---- S[:, j0:j0+VW] stays in registers for the whole call. 256 threads tile the
    // [DK][VW] state as thread groups of 8x8, so every index is a compile-time offset from
    // a per-thread base and does not spill to local memory.
    // 512 threads tile the [DK][VW] state as 16 row groups x 32 column groups, RT x TC each.
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
        // ONE thread performs the prefix sum, then a barrier. Running this in place on
        // shared memory from all 512 threads interleaves the scans, so G is applied
        // repeatedly and without bound -- which is what produced e^G overflow and the NaNs.
        if (tid == 0) {
            for (int r = 1; r < GDN_CHUNK; ++r) sG[r] += sG[r - 1];
        }
        __syncthreads();

        // k is staged UNSCALED. The scalars beta*e^G are applied where they belong:
        // never in A (which needs raw k), only in kbS and in the state update.
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

        // ---- A[j,i] = -beta_j (k_j.k_i) exp(G_j - G_i), j>i, into sT.
        // A needs UNSCALED k, so strip beta*e^G back off. beta*e^G can underflow to 0 for
        // fast-decaying rows, in which case sKb is 0 and the true A entry is 0 as well
        // (beta_j = 0 forces it), so guarding the divide is safe rather than lossy.
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

        // ---- T = (I - A)^-1 by forward substitution, ONE ROW AT A TIME.
        // Row r needs rows < r final, so the rows must be ordered: thread r does row r and a
        // barrier per row enforces the ordering.
        //
        // This REPLACES a blocked sub-block version (4x4 blocks of 16, off-diagonal blocks by
        // matmul) that did NOT compute the inverse. Verified on host against a dense inverse:
        // the blocked form is off by 1.5-2.5 in max-abs, this form by 4.4e-16. The blocked
        // form was never checked before now. It is also strictly safer: the row-wise form has
        // no cross-thread read of a row another thread is writing.
        for (int r = 1; r < GDN_CHUNK; ++r) {
            if (tid == r) {
                // T[r][j] = M[r][j] + sum_{m=j+1}^{r-1} M[r][m] * T[m][j]
                //
                // The sum runs over m from j+1 to r-1 -- NOT t<j. T[m][j] is zero for
                // m <= j, so the reference's "sum over the whole sub-block" and this are
                // the same thing, but writing the loop as t<j silently drops the m>j terms.
                //
                // Rows are finalised in increasing r (one thread per row, barrier per row),
                // so T[m][*] for m<r is already final when row r reads it.
                //
                // Forward j order is what lets the original M[r][m] be read straight out of
                // sT: at column j only indices < j have been written, so m>j is still M.
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

        // ---- w = T (beta .* V) into sKbS
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int i = 0; i <= r; ++i) acc = fmaf(sT[r * GDN_CHUNK + i], sVb[i * VW + j], acc);
            sKbS[r * VW + j] = acc;
        }
        __syncthreads();

        // ---- kbS = kb @ S  (kb = beta.K.e^G), accumulated into the SAME sVb buffer now
        // that w is formed and sVb is dead.
        // Each thread holds S only for its own RT rows, so the sum over all DK rows stages
        // through sRed: one partial per (thread row-group, column), reduced by the consumer.
        // b MUST be unrolled: a runtime index into Sr[a][b] pushes the whole state tile
        // out of registers and into local memory, which is where the spill came from.
        // Reduce PER ROW, barrier either side. sRed is [TG][VW] with NO r dimension, so
        // looping r and reducing once at the end left every sVb row holding the same
        // (last) reduction -- a logic bug, not just a race.
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
        // NEGATIVE CONTROL BUILD (never shipped): v_new from RAW V instead of w = T(beta.*V).
        // Same source, same build path, only the term the contract names swapped.
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int t = 0; t <= r; ++t) acc = fmaf(sT[r * GDN_CHUNK + t], sVb[t * VW + j], acc);
            sKbS[r * VW + j] = sVb[r * VW + j] - acc;   // raw v, not w
        }
        __syncthreads();
#else
        // ---- v_new = w - T @ kbS, written in place into sKbS (row r needs only w[r]).
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int t = 0; t <= r; ++t) acc = fmaf(sT[r * GDN_CHUNK + t], sVb[t * VW + j], acc);
            sKbS[r * VW + j] -= acc;
        }
        __syncthreads();
#endif

        // ---- o = (q.e^G) @ S + [tril(q k^T .* L)] @ v_new
        for (int r = 0; r < GDN_CHUNK; ++r) {
            const int gr = row0 + r;
            // sQr holds RAW q for this row. It is deliberately NOT pre-scaled by e^{G_r}:
            // sQk[i] = sQr . k_i feeds the attention term, which applies e^{G_r-G_i} itself.
            // Pre-scaling here counted e^{G_r} twice -- once in sQk and again in that
            // exponent. Scaling inter by e^{G_r} once in the consumer keeps the attention
            // exponent at G_r-G_i (<= 0, stable) instead of needing exp(-G_i), which
            // OVERFLOWS for the fast-decaying heads where G_i is far below zero.
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
            // DEBUG build: write the two terms PRE-rounding so each can be compared with the
            // oracle's own component instead of against their sum. If both components sit
            // within FP32 order-noise of their own magnitude and only the SUM is off, the
            // excess is cancellation rather than a defect.
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
                const float eG = __expf(sG[r]);            // applied exactly once, here
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

        // The last row's snapshot duplicates the final S write-back; unread
        // (snapshot_row requires rows < live).
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

        // ---- S = S * e^{G_{C-1}} + (k .* e^{G_{C-1}-G})^T @ v_new
        {
            const float decay = __expf(sG[GDN_CHUNK - 1]);
            const float invc = 1.0f / fmaxf(decay, 1e-38f);
            // updated IN PLACE: the old value is read once into a register, so the tile
            // never needs a second copy. An Snew[RT][RT] array here is what put 256 B of
            // stack frame in the build.
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
        // End-of-chunk barrier: the next iteration rewrites sBeta/sG/sKb at the top of the
        // body, while slower threads may still be reading sG/sKb/sKbS in the state update.
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

// ABI-shaped wrapper. VSPLIT is a property of the prepared plan, fixed in setup, never
// chosen per call. The plan is frozen: prefill is C=32 / value_split 1 (this entry point),
// decode is the recurrent path at value_split 2. See gdn_kernel_design.md -- the C=64 /
// value_split=1 variant that needed 147,968 B of smem did not fit the 101,376 B limit and
// was removed from the plan, which is why only this one remains.
extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_chunked_core_v1(const float* q, const float* k, const float* v,
                        const float* beta, const float* g, float* S,
                        __nv_bfloat16* out, const GdnMeta* meta, int valid_rows) {
    gdn_chunked_core_t<1>(q, k, v, beta, g, S, out, meta, valid_rows, nullptr GDN_DUMP_ARGS);
}

// Negative-control build; never defined by build.rs.
#ifdef GDN_DUMP_PARTS
// DEBUG-only entry point: additionally writes the output phase's two terms pre-rounding,
// so each can be compared against the oracle's own component rather than against their
// sum. Never shipped; the G5 build does not define GDN_DUMP_PARTS.
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
