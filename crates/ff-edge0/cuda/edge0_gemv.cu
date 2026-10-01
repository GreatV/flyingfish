#include <cuda_bf16.h>
// Groupwise affine GEMV: packed U32 int4/int8, F32 scales/biases per 64 inputs; w = s*q + b. Block rows reuse x; group partials are summed in fixed order. in_dim is a multiple of 64 and at most 4096.

// RPB is a macro parameter (16 default, 4 for small out_dims)

#define GEMV(NAME, BITS, PER_WORD, SEG, WPT_CAP, RPB)                         \
extern "C" __global__ void edge0_gemv##NAME(                                  \
    const unsigned int* __restrict__ packed,                                  \
    const __nv_bfloat16* __restrict__ scales,                                         \
    const __nv_bfloat16* __restrict__ biases,                                         \
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
    __shared__ float s_sc[RPB][64];                                           \
    __shared__ float s_bi[RPB][64];                                           \
    for (int c0 = 0; c0 < in_dim; c0 += 4096) {                               \
        const int cin = min(4096, in_dim - c0);                               \
        for (int i = tid; i < nrows * 64; i += blockDim.x) {                  \
            const int sr = i / 64;                                            \
            const int sg = i % 64;                                            \
            const int gi = (row0 + sr) * groups + c0 / 64 + sg;               \
            s_sc[sr][sg] = __bfloat162float(scales[gi]);                      \
            s_bi[sr][sg] = __bfloat162float(biases[gi]);                      \
        }                                                                     \
        __syncthreads();                                                      \
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
                    partials[r][group - c0 / 64] =                            \
                        s_sc[r][w / (64 / PER_WORD)] * dot + s_bi[r][w / (64 / PER_WORD)] * sumx;                 \
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
