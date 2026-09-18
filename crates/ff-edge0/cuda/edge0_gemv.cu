// Fused groupwise-affine int4/int8 dequant GEMV for decode (batch 1).
//
// Layout (byte-level pinned in docs/edge0-design.md): payload U32 words
// pack 8x unsigned int4 (low nibble first) or 4x unsigned int8; scales and
// biases are f32 [out, in/64] uploads. Exact reconstruction w = s*q + b:
// y[o] = sum_g s[o,g]*dot_g + b[o,g]*groupsum_g.
//
// RPB rows per block; x preloaded once into registers (x is
// identical across a block's rows — the one-block-per-row shape staged 8x
// more x traffic than weights and ran at ~16% of DRAM bandwidth). Threads
// cover ceil(words/256) words each (int4 in 4096 = 512 words, int8 in
// 4096 = 1024); per word the segment shuffle stays 8/16-aligned because
// blockDim is a multiple of the segment width. Group leaders write
// per-group partials; thread 0 sums them in fixed group order per row —
// deterministic run-to-run. in_dim must be a multiple of 64, <= 4096.

// RPB is a macro parameter (16 default, 4 for small out_dims)

#define GEMV(NAME, BITS, PER_WORD, SEG, WPT_CAP, RPB)                         \
extern "C" __global__ void edge0_gemv##NAME(                                  \
    const unsigned int* __restrict__ packed,                                  \
    const float* __restrict__ scales,                                         \
    const float* __restrict__ biases,                                         \
    const float* __restrict__ x,                                              \
    float* __restrict__ y,                                                    \
    int out_dim,                                                              \
    int in_dim)                                                               \
{                                                                             \
    const int tid = threadIdx.x;                                              \
    const int row0 = blockIdx.x * RPB;                                        \
    const int nrows = min(RPB, out_dim - row0);                               \
    const int groups = in_dim >> 6;                                           \
    /* in_dim chunked at 4096 columns: keeps wpt <= WPT_CAP and the 64-group \
       partials table at any width. Chunk totals fold into rt in ascending   \
       chunk order; single-chunk shapes are bit-identical to the unchunked   \
       kernel (0 + fold == fold). */                                         \
    __shared__ float rt[RPB];                                                 \
    if (tid < nrows) rt[tid] = 0.0f;                                          \
    __syncthreads();                                                          \
    for (int c0 = 0; c0 < in_dim; c0 += 4096) {                               \
        const int cin = min(4096, in_dim - c0);                               \
        const int cwords = cin / PER_WORD;                                    \
        const int cwpt = (cwords + blockDim.x - 1) / blockDim.x;              \
        float xr[WPT_CAP][PER_WORD];                                          \
        _Pragma("unroll")                                                     \
        for (int k = 0; k < WPT_CAP; k++) {                                   \
            if (k >= cwpt) break;                                             \
            const int w = tid + k * blockDim.x;                               \
            if (w < cwords) {                                                 \
                _Pragma("unroll")                                             \
                for (int j = 0; j < PER_WORD; j++)                            \
                    xr[k][j] = x[c0 + w * PER_WORD + j];                      \
            }                                                                 \
        }                                                                     \
        unsigned int words[RPB][WPT_CAP];                                     \
        _Pragma("unroll")                                                     \
        for (int r = 0; r < RPB; r++) {                                       \
            if (r >= nrows) break;                                            \
            _Pragma("unroll")                                                 \
            for (int k = 0; k < WPT_CAP; k++) {                               \
                const int w = tid + k * blockDim.x;                           \
                words[r][k] = (k < cwpt && w < cwords)                        \
                    ? packed[(long long)(row0 + r) * (in_dim / PER_WORD)      \
                             + c0 / PER_WORD + w]                             \
                    : 0u;                                                     \
            }                                                                 \
        }                                                                     \
        __shared__ float partials[RPB][64];                                   \
        _Pragma("unroll")                                                     \
        for (int r = 0; r < RPB; r++) {                                       \
            if (r >= nrows) break;                                            \
            const int row = row0 + r;                                         \
            _Pragma("unroll")                                                 \
            for (int k = 0; k < WPT_CAP; k++) {                               \
                if (k >= cwpt) break;                                         \
                const int w = tid + k * blockDim.x;                           \
                float dot = 0.0f;                                             \
                float sumx = 0.0f;                                            \
                if (w < cwords) {                                             \
                    const unsigned int word = words[r][k];                    \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < PER_WORD; j++) {                      \
                        const float q = (float)((word >> (BITS * j)) &        \
                            ((1u << BITS) - 1u));                             \
                        dot += q * xr[k][j];                                  \
                        sumx += xr[k][j];                                     \
                    }                                                         \
                }                                                             \
                for (int offset = SEG / 2; offset > 0; offset >>= 1) {        \
                    dot += __shfl_down_sync(0xffffffffu, dot, offset, SEG);   \
                    sumx += __shfl_down_sync(0xffffffffu, sumx, offset, SEG); \
                }                                                             \
                if (w < cwords && (tid & (SEG - 1)) == 0) {                   \
                    const int group = c0 / 64 + w / (64 / PER_WORD);          \
                    const int gi = row * groups + group;                      \
                    partials[r][group - c0 / 64] =                            \
                        scales[gi] * dot + biases[gi] * sumx;                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        if (tid < nrows) {                                                    \
            float acc = rt[tid];                                              \
            const int cg = cin >> 6;                                          \
            for (int g = 0; g < cg; g++) acc += partials[tid][g];             \
            rt[tid] = acc;                                                    \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (tid < nrows) y[row0 + tid] = rt[tid];                                 \
}

GEMV(4, 4, 8, 8, 2, 16)
GEMV(8, 8, 4, 16, 4, 16)
GEMV(4r4, 4, 8, 8, 2, 4)
GEMV(8r4, 8, 4, 16, 4, 4)
