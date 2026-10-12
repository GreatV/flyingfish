
// P3 retains shipped FP32 reduction/FMA order inside each G4 group. Only the
// input state, precomputed T, chunk range and final state owner differ.
#define GDN_CHUNK 32
#define GDN_DK 128
#define GDN_DV 128
#define GDN_HEADS 48
#define GDN_THREADS 512
extern "C" __global__ __launch_bounds__(512, 1)
void gdn_b_p3(const float* q, const float* k, const float* v,
              const float* beta, const float* g, float* S,
              __nv_bfloat16* out, const float* all_t, const float* entries,
              const Meta* meta, int bound, int leaves) {
    const int valid_rows=b_rows(meta,bound,leaves);
    const int head=blockIdx.y, group=blockIdx.x, tid=threadIdx.x;
    constexpr int VW=128, j0=0;
    if (blockDim.x!=512 || blockDim.y!=1 || blockDim.z!=1 ||
        gridDim.x!=unsigned(leaves) || gridDim.y!=H || gridDim.z!=1 ||
        !b_aligned(q) || !b_aligned(k) || !b_aligned(v) || !b_aligned(beta) ||
        !b_aligned(g) || !b_aligned(S) || !b_aligned(out) || !b_aligned(all_t) ||
        !b_aligned(entries)) { __trap(); return; }
    if (group*G*C>=valid_rows) return;
    extern __shared__ float smem[];
    float* const sKb   = smem;
    float* const sVb   = sKb   + GDN_CHUNK * GDN_DK;
    float* const sKbS  = sVb   + GDN_CHUNK * VW;
    float* const sT    = sKbS  + GDN_CHUNK * VW;
    float* const sQr   = sT    + GDN_CHUNK * GDN_CHUNK;
    float* const sBeta = sQr   + GDN_DK;
    float* const sG    = sBeta + GDN_CHUNK;
    float* const sRed  = sG    + GDN_CHUNK;
    float* const sQk   = sRed  + 16 * VW;

    constexpr int RT = 8, TC = GDN_DV / 32;
    const int ti = tid / 32, tj = tid % 32;
    const int i0 = ti * RT, jb = tj * TC;
    float Sr[RT][TC];
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            Sr[a][b] = entries[entry_at(head, leaves, group) + (i0 + a) * GDN_DV + jb + b];

    const int nchunk = (valid_rows + GDN_CHUNK - 1) / GDN_CHUNK;

    for (int c = group * G; c < min(nchunk, (group + 1) * G); ++c) {
        const int row0 = c * GDN_CHUNK;

        for (int r = tid; r < GDN_CHUNK; r += GDN_THREADS) {
            const int gr = row0 + r;
            const bool live = gr < valid_rows;
            sBeta[r] = live ? beta[(size_t)gr * GDN_HEADS + head] : 0.0f;
            sG[r]   = live ? g[(size_t)gr * GDN_HEADS + head] : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            for (int r = 1; r < GDN_CHUNK; ++r) sG[r] += sG[r - 1];
        }
        __syncthreads();

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

        for (int idx = tid; idx < C*C; idx += GDN_THREADS)
            sT[idx] = all_t[(static_cast<size_t>(head)*leaves*G+c)*C*C+idx];
        __syncthreads();

        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int i = 0; i <= r; ++i) acc = fmaf(sT[r * GDN_CHUNK + i], sVb[i * VW + j], acc);
            sKbS[r * VW + j] = acc;
        }
        __syncthreads();

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
                sVb[r * VW + j] = acc;
            }
            __syncthreads();
        }
        __syncthreads();

#ifdef GDN_RAW_V
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int t = 0; t <= r; ++t) acc = fmaf(sT[r * GDN_CHUNK + t], sVb[t * VW + j], acc);
            sKbS[r * VW + j] = sVb[r * VW + j] - acc;
        }
        __syncthreads();
#else
        for (int idx = tid; idx < GDN_CHUNK * VW; idx += GDN_THREADS) {
            const int r = idx / VW, j = idx - r * VW;
            float acc = 0.0f;
            for (int t = 0; t <= r; ++t) acc = fmaf(sT[r * GDN_CHUNK + t], sVb[t * VW + j], acc);
            sKbS[r * VW + j] -= acc;
        }
        __syncthreads();
#endif

        for (int r = 0; r < GDN_CHUNK; ++r) {
            const int gr = row0 + r;
            if (gr < valid_rows) {
                for (int t = tid; t < GDN_DK; t += GDN_THREADS)
                    sQr[t] = q[((size_t)gr * GDN_HEADS + head) * GDN_DK + t];
            } else {
                for (int t = tid; t < GDN_DK; t += GDN_THREADS) sQr[t] = 0.0f;
            }
            __syncthreads();
            #pragma unroll
            for (int b = 0; b < TC; ++b) {
                float p = 0.0f;
                #pragma unroll
                for (int a = 0; a < RT; ++a) p = fmaf(sQr[i0 + a], Sr[a][b], p);
                sRed[ti * VW + jb + b] = p;
            }
            for (int i = tid; i <= r; i += GDN_THREADS) {
                float qk = 0.0f;
                for (int t = 0; t < GDN_DK; ++t) qk = fmaf(sQr[t], sKb[i * GDN_DK + t], qk);
                sQk[i] = qk;
            }
            __syncthreads();
#ifdef GDN_DUMP_PARTS
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

        {
            const float decay = __expf(sG[GDN_CHUNK - 1]);
            #pragma unroll
            for (int a = 0; a < RT; ++a)
                #pragma unroll
                for (int b = 0; b < TC; ++b) {
                    float acc = Sr[a][b] * decay;
                    const int i = i0 + a;
                    #pragma unroll 4
                    for (int r = 0; r < GDN_CHUNK; ++r) {
                        const float f = __expf(sG[GDN_CHUNK - 1] - sG[r]);
                        acc = fmaf(sKb[r * GDN_DK + i] * f, sKbS[r * VW + jb + b], acc);
                    }
                    Sr[a][b] = acc;
                }
        }
        __syncthreads();
    }

    if (group == (valid_rows - 1) / (G*C)) {
    #pragma unroll
    for (int a = 0; a < RT; ++a)
        #pragma unroll
        for (int b = 0; b < TC; ++b)
            S[(size_t)head * GDN_DK * GDN_DV + (i0 + a) * GDN_DV + j0 + jb + b] = Sr[a][b];
    }
}
