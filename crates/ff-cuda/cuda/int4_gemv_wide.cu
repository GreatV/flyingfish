#include <cuda_bf16.h>
// Packed int4 GEMVs over f32 activations with bf16 scales and biases; the split-K partials are combined in fixed order.

#define RPB 16
#define RPB_V4 16
#define RPB_V4D8 8

__device__ __forceinline__ void wide_sum(
    const float* part, float* out, int rows, int split)
{
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float total = 0.0f;
    for (int s = 0; s < split; s++)
        total += part[(long long)s * rows + row];
    out[row] = total;
}

// y_part[slice * rows + row] = sum over the slice's groups; in_dim is a multiple of 64 * split.
extern "C" __global__ void int4_gemv_rows_sum(
    const float* partial, float* out, int rows, int split)
{
    wide_sum(partial + (long long)blockIdx.x * split * rows,
        out + (long long)blockIdx.x * rows, rows, split);
}

// Up to 4 row segments over the same x in one launch; segments are 16-row aligned, no split-K, in_dim <= 6144.
extern "C" __global__ void int4_group4(
    const unsigned int* __restrict__ packed0,
    const __nv_bfloat16* __restrict__ scales0,
    const __nv_bfloat16* __restrict__ biases0,
    float* __restrict__ y0,
    float* __restrict__ yb0,
    int rows0,
    const unsigned int* __restrict__ packed1,
    const __nv_bfloat16* __restrict__ scales1,
    const __nv_bfloat16* __restrict__ biases1,
    float* __restrict__ y1,
    float* __restrict__ yb1,
    int rows1,
    const unsigned int* __restrict__ packed2,
    const __nv_bfloat16* __restrict__ scales2,
    const __nv_bfloat16* __restrict__ biases2,
    float* __restrict__ y2,
    float* __restrict__ yb2,
    int rows2,
    const unsigned int* __restrict__ packed3,
    const __nv_bfloat16* __restrict__ scales3,
    const __nv_bfloat16* __restrict__ biases3,
    float* __restrict__ y3,
    float* __restrict__ yb3,
    int rows3,
    const float* __restrict__ x,
    const float* __restrict__ xb,
    int in_dim)
{
    const int tid = threadIdx.x;
    const float* xcol = blockIdx.x == 0 ? x : xb;
    const int pad0 = (rows0 + 15) & ~15;
    const int pad1 = (rows1 + 15) & ~15;
    const int pad2 = (rows2 + 15) & ~15;

    const unsigned int* packed;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    float* y;
    int rows;
    int row_base;
    {
        int g = blockIdx.y * RPB;
        int seg = 0;
        if (g >= pad0) { g -= pad0; seg = 1; }
        if (seg == 1 && g >= pad1) { g -= pad1; seg = 2; }
        if (seg == 2 && g >= pad2) { g -= pad2; seg = 3; }
        row_base = g;
        switch (seg) {
        case 0: packed = packed0; scales = scales0; biases = biases0; y = blockIdx.x == 0 ? y0 : yb0; rows = rows0; break;
        case 1: packed = packed1; scales = scales1; biases = biases1; y = blockIdx.x == 0 ? y1 : yb1; rows = rows1; break;
        case 2: packed = packed2; scales = scales2; biases = biases2; y = blockIdx.x == 0 ? y2 : yb2; rows = rows2; break;
        default: packed = packed3; scales = scales3; biases = biases3; y = blockIdx.x == 0 ? y3 : yb3; rows = rows3; break;
        }
        if (row_base >= rows) return;
    }

    const int words_per_row = in_dim / 8;
    const int wpt = (words_per_row + blockDim.x - 1) / blockDim.x;
    float xr[3][8];
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (k >= wpt) break;
        const int w = tid + k * blockDim.x;
        if (w < words_per_row) {
            #pragma unroll
            for (int j = 0; j < 8; j++)
                xr[k][j] = xcol[w * 8 + j];
        }
    }

    __shared__ float partials[RPB][96];
    const int groups = in_dim >> 6;
    #pragma unroll
    for (int r = 0; r < RPB; r++) {
        if (r + row_base >= rows) break;
        const unsigned int* prow = packed + (long long)(row_base + r) * words_per_row;
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            if (k >= wpt) break;
            const int w = tid + k * blockDim.x;
            float dot = 0.0f;
            float sumx = 0.0f;
            if (w < words_per_row) {
                const unsigned int word = prow[w];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    dot += (float)((word >> (4 * j)) & 0xFu) * xr[k][j];
                    sumx += xr[k][j];
                }
            }
            for (int off = 4; off > 0; off >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
                sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
            }
            if (w < words_per_row && (tid & 7) == 0) {
                const int group = w >> 3;
                const int gi = (row_base + r) * groups + group;
                partials[r][group] = __bfloat162float(scales[gi]) * dot + __bfloat162float(biases[gi]) * sumx;
            }
        }
    }
    __syncthreads();
    if (tid < RPB && tid + row_base < rows) {
        float total = 0.0f;
        for (int g2 = 0; g2 < groups; g2++) total += partials[tid][g2];
        y[row_base + tid] = total;
    }
}

// Split-K slice with one 16-byte load per thread; a thread pair covers one 64-column group.
// Requires slice_words % 4 == 0.
__device__ __forceinline__ void wide_dot_v4(
    const unsigned int* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
    const float* __restrict__ xcol, float* __restrict__ ycol,
    int rows, int in_dim, int split)
{
    const int words_per_row = in_dim / 8;
    const int slice_words = words_per_row / split;
    const int tid = threadIdx.x;
    const int blocks_per_row = (rows + RPB - 1) / RPB;
    const int slice = blockIdx.y / blocks_per_row;
    const int rblock = blockIdx.y % blocks_per_row;
    const int row_base = rblock * RPB;
    const int nrows = min(RPB, rows - row_base);
    if (row_base >= rows) return;

    const int w_lo = slice * slice_words;
    const int words4 = slice_words >> 2;
    const uint4* prow4 = reinterpret_cast<const uint4*>(packed + (long long)row_base * words_per_row + w_lo);

    const bool active = tid < words4;
    float xr[32];
    if (active) {
        const float* xf = xcol + w_lo * 8 + tid * 32;
        #pragma unroll
        for (int j = 0; j < 32; j++) xr[j] = xf[j];
    }

    __shared__ float partials[RPB][192];
    const int groups_total = in_dim >> 6;
    const int g_lo = (w_lo * 8) >> 6;
    #pragma unroll
    for (int r = 0; r < RPB; r++) {
        if (r >= nrows) break;
        float dot = 0.0f;
        float sumx = 0.0f;
        if (active) {
            const uint4 w4 = prow4[(long long)r * (words_per_row >> 2) + tid];
            const unsigned int ws[4] = { w4.x, w4.y, w4.z, w4.w };
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int word = ws[i];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    dot += (float)((word >> (4 * j)) & 0xFu) * xr[i * 8 + j];
                    sumx += xr[i * 8 + j];
                }
            }
        }
        dot += __shfl_down_sync(0xffffffffu, dot, 1, 2);
        sumx += __shfl_down_sync(0xffffffffu, sumx, 1, 2);
        if (active && (tid & 1) == 0) {
            const int group = g_lo + (tid >> 1);
            const int gi = (row_base + r) * groups_total + group;
            partials[r][group - g_lo] = __fmaf_rn(__bfloat162float(scales[gi]), dot, __fmul_rn(__bfloat162float(biases[gi]), sumx));
        }
    }
    __syncthreads();
    if (tid < nrows) {
        const int groups_here = slice_words >> 3;
        float total = 0.0f;
        for (int g = 0; g < groups_here; g++) total += partials[tid][g];
        ycol[(long long)slice * rows + row_base + tid] = total;
    }
}

extern "C" __global__ void int4_gemv_rows(
    const unsigned int* packed, const __nv_bfloat16* scales, const __nv_bfloat16* biases,
    const float* x, float* partial, int rows, int cols, int split)
{
    wide_dot_v4(packed, scales, biases, x + (long long)blockIdx.x * cols,
        partial + (long long)blockIdx.x * split * rows, rows, cols, split);
}

// Grouped affine-int4 GEMV over 64-column groups and RPB_V4 row tiles.
template <int X, bool Lora>
__device__ __forceinline__ void group4_body(
    const unsigned int* __restrict__ packed0,
    const __nv_bfloat16* __restrict__ scales0,
    const __nv_bfloat16* __restrict__ biases0,
    float* __restrict__ y0,
    float* __restrict__ yb0,
    int rows0,
    const unsigned int* __restrict__ packed1,
    const __nv_bfloat16* __restrict__ scales1,
    const __nv_bfloat16* __restrict__ biases1,
    float* __restrict__ y1,
    float* __restrict__ yb1,
    int rows1,
    const unsigned int* __restrict__ packed2,
    const __nv_bfloat16* __restrict__ scales2,
    const __nv_bfloat16* __restrict__ biases2,
    float* __restrict__ y2,
    float* __restrict__ yb2,
    int rows2,
    const unsigned int* __restrict__ packed3,
    const __nv_bfloat16* __restrict__ scales3,
    const __nv_bfloat16* __restrict__ biases3,
    float* __restrict__ y3,
    float* __restrict__ yb3,
    int rows3,
    const float* __restrict__ lb0,
    const float* __restrict__ ax0,
    const float* __restrict__ lb1,
    const float* __restrict__ ax1,
    const float* __restrict__ lb2,
    const float* __restrict__ ax2,
    const float* __restrict__ lb3,
    const float* __restrict__ ax3,
    const float* __restrict__ x,
    const float* __restrict__ xb,
    int in_dim,
    int rank)
{
    const int tid = threadIdx.x;
    const float* xcol = blockIdx.x == 0 ? x : xb;
    const int pad0 = (rows0 + 15) & ~15;
    const int pad1 = (rows1 + 15) & ~15;
    const int pad2 = (rows2 + 15) & ~15;

    const unsigned int* packed;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    float* y;
    const float* lb;
    const float* ax;
    int rows;
    int row_base;
    int seg_has_lora;
    {
        int g = blockIdx.y * RPB_V4;
        int seg = 0;
        if (g >= pad0) { g -= pad0; seg = 1; }
        if (seg == 1 && g >= pad1) { g -= pad1; seg = 2; }
        if (seg == 2 && g >= pad2) { g -= pad2; seg = 3; }
        row_base = g;
        int has_lora = 0;
        switch (seg) {
        case 0: packed = packed0; scales = scales0; biases = biases0; y = blockIdx.x == 0 ? y0 : yb0; rows = rows0; lb = lb0; ax = ax0; has_lora = (lb0 != nullptr); break;
        case 1: packed = packed1; scales = scales1; biases = biases1; y = blockIdx.x == 0 ? y1 : yb1; rows = rows1; lb = lb1; ax = ax1; has_lora = (lb1 != nullptr); break;
        case 2: packed = packed2; scales = scales2; biases = biases2; y = blockIdx.x == 0 ? y2 : yb2; rows = rows2; lb = lb2; ax = ax2; has_lora = (lb2 != nullptr); break;
        default: packed = packed3; scales = scales3; biases = biases3; y = blockIdx.x == 0 ? y3 : yb3; rows = rows3; lb = lb3; ax = ax3; has_lora = (lb3 != nullptr); break;
        }
        seg_has_lora = has_lora;
        if (row_base >= rows) return;
    }

    __shared__ float sax[64];
    if constexpr (Lora) {
        if (rank > 0 && seg_has_lora) {
            for (int k = tid; k < rank; k += blockDim.x) sax[k] = ax[k];
            __syncthreads();
        }
    }

    const int words_per_row = in_dim / 8;
    __shared__ float partials[RPB_V4][128];
    const int groups = in_dim >> 6;
#define GROUP4_ROWS(Vector, Words, Lanes, Active, Pointer, Stride, Offset, Group, ...) \
    _Pragma("unroll") \
    for (int r = 0; r < RPB_V4; r++) { \
        if (r + row_base >= rows) break; \
        float dot = 0.0f; \
        float sumx = 0.0f; \
        if (Active) { \
            const Vector w = Pointer[(long long)r * Stride + Offset + tid]; \
            const unsigned int ws[Words] = { __VA_ARGS__ }; \
            _Pragma("unroll") \
            for (int i = 0; i < Words; i++) { \
                const unsigned int word = ws[i]; \
                _Pragma("unroll") \
                for (int j = 0; j < 8; j++) { \
                    dot += (float)((word >> (4 * j)) & 0xFu) * xr[i * 8 + j]; \
                    sumx += xr[i * 8 + j]; \
                } \
            } \
        } \
        dot += __shfl_down_sync(0xffffffffu, dot, 1, Lanes); \
        sumx += __shfl_down_sync(0xffffffffu, sumx, 1, Lanes); \
        if constexpr (Lanes == 4) { \
            dot += __shfl_down_sync(0xffffffffu, dot, 2, Lanes); \
            sumx += __shfl_down_sync(0xffffffffu, sumx, 2, Lanes); \
        } \
        if (Active && (tid & (Lanes - 1)) == 0) { \
            const int group = Group; \
            const int gi = (row_base + r) * groups + group; \
            partials[r][group] = __bfloat162float(scales[gi]) * dot + __bfloat162float(biases[gi]) * sumx; \
        } \
    }
    if constexpr (X == 32) {
        const int stride4 = words_per_row >> 2;
        const bool active = tid < stride4;
        float xr[32];
        if (active) {
            const float* xf = xcol + tid * 32;
            #pragma unroll
            for (int j = 0; j < 32; j++) xr[j] = xf[j];
        }
        const uint4* prow4 = reinterpret_cast<const uint4*>(packed + (long long)row_base * words_per_row);
        GROUP4_ROWS(uint4, 4, 2, active, prow4, stride4, 0, (tid >> 1), w.x, w.y, w.z, w.w)
    } else {
        float xr[16];
        const uint2* prow2 = reinterpret_cast<const uint2*>(packed + (long long)row_base * words_per_row);
        const int wpr2 = words_per_row >> 1;
        for (int cw = 0; cw < words_per_row; cw += (int)blockDim.x * 2) {
            const int nthr = min((int)blockDim.x, (words_per_row - cw) >> 1);
            if (tid < nthr) {
                const float* xf = xcol + cw * 8 + tid * 16;
                #pragma unroll
                for (int j = 0; j < 16; j++) xr[j] = xf[j];
            }
            GROUP4_ROWS(uint2, 2, 4, (tid < nthr), prow2, wpr2, (cw >> 1), ((cw >> 3) + (tid >> 2)), w.x, w.y)
        }
    }
#undef GROUP4_ROWS
    __syncthreads();
    if (tid < RPB_V4 && tid + row_base < rows) {
        float total = 0.0f;
        for (int g2 = 0; g2 < groups; g2++) total += partials[tid][g2];
        if constexpr (Lora) {
            if (rank > 0 && seg_has_lora) {
                const float* br = lb + (long long)(row_base + tid) * rank;
                for (int k = 0; k < rank; k++) total += br[k] * sax[k];
            }
        }
        y[row_base + tid] = total;
    }
}

#define GROUP4(Name, X) \
extern "C" __global__ void Name( \
    const unsigned int* __restrict__ packed0, \
    const __nv_bfloat16* __restrict__ scales0, \
    const __nv_bfloat16* __restrict__ biases0, \
    float* __restrict__ y0, \
    float* __restrict__ yb0, \
    int rows0, \
    const unsigned int* __restrict__ packed1, \
    const __nv_bfloat16* __restrict__ scales1, \
    const __nv_bfloat16* __restrict__ biases1, \
    float* __restrict__ y1, \
    float* __restrict__ yb1, \
    int rows1, \
    const unsigned int* __restrict__ packed2, \
    const __nv_bfloat16* __restrict__ scales2, \
    const __nv_bfloat16* __restrict__ biases2, \
    float* __restrict__ y2, \
    float* __restrict__ yb2, \
    int rows2, \
    const unsigned int* __restrict__ packed3, \
    const __nv_bfloat16* __restrict__ scales3, \
    const __nv_bfloat16* __restrict__ biases3, \
    float* __restrict__ y3, \
    float* __restrict__ yb3, \
    int rows3, \
    const float* __restrict__ x, \
    const float* __restrict__ xb, \
    int in_dim) \
{ \
    group4_body<X, false>(packed0, scales0, biases0, y0, yb0, rows0, \
        packed1, scales1, biases1, y1, yb1, rows1, \
        packed2, scales2, biases2, y2, yb2, rows2, \
        packed3, scales3, biases3, y3, yb3, rows3, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        x, xb, in_dim, 0); \
}
GROUP4(int4_group4_stock, 32)
GROUP4(int4_group4_xr16, 16)
#undef GROUP4

#define GROUP4_L(Name, X) \
extern "C" __global__ void Name( \
    const unsigned int* __restrict__ packed0, \
    const __nv_bfloat16* __restrict__ scales0, \
    const __nv_bfloat16* __restrict__ biases0, \
    float* __restrict__ y0, \
    int rows0, \
    const float* __restrict__ lb0, \
    const float* __restrict__ ax0, \
    const unsigned int* __restrict__ packed1, \
    const __nv_bfloat16* __restrict__ scales1, \
    const __nv_bfloat16* __restrict__ biases1, \
    float* __restrict__ y1, \
    int rows1, \
    const float* __restrict__ lb1, \
    const float* __restrict__ ax1, \
    const unsigned int* __restrict__ packed2, \
    const __nv_bfloat16* __restrict__ scales2, \
    const __nv_bfloat16* __restrict__ biases2, \
    float* __restrict__ y2, \
    int rows2, \
    const float* __restrict__ lb2, \
    const float* __restrict__ ax2, \
    const unsigned int* __restrict__ packed3, \
    const __nv_bfloat16* __restrict__ scales3, \
    const __nv_bfloat16* __restrict__ biases3, \
    float* __restrict__ y3, \
    int rows3, \
    const float* __restrict__ lb3, \
    const float* __restrict__ ax3, \
    const float* __restrict__ x, \
    int in_dim, \
    int rank) \
{ \
    group4_body<X, true>(packed0, scales0, biases0, y0, y0, rows0, \
        packed1, scales1, biases1, y1, y1, rows1, \
        packed2, scales2, biases2, y2, y2, rows2, \
        packed3, scales3, biases3, y3, y3, rows3, \
        lb0, ax0, lb1, ax1, lb2, ax2, lb3, ax3, \
        x, x, in_dim, rank); \
} \
 \
// int4_group4_v4d: the group kernel for in_dim > 8192 (down projection). \
// Chunk c owns absolute groups [64c, 64c+64); partials[r][tid>>2] \
// accumulates across chunks in slot-major order, so the y-sum reassociates \
// the group sum. 4-row tiles; requires in_dim % 64 == 0 and rows % 4 == 0.
GROUP4_L(int4_group4_stock_l, 32)
GROUP4_L(int4_group4_xr16_l, 16)
#undef GROUP4_L

extern "C" __global__ void int4_group4_v4d8(
    const unsigned int* __restrict__ packed0,
    const __nv_bfloat16* __restrict__ scales0,
    const __nv_bfloat16* __restrict__ biases0,
    float* __restrict__ y0,
    float* __restrict__ yb0,
    int rows0,
    const unsigned int* __restrict__ packed1,
    const __nv_bfloat16* __restrict__ scales1,
    const __nv_bfloat16* __restrict__ biases1,
    float* __restrict__ y1,
    float* __restrict__ yb1,
    int rows1,
    const unsigned int* __restrict__ packed2,
    const __nv_bfloat16* __restrict__ scales2,
    const __nv_bfloat16* __restrict__ biases2,
    float* __restrict__ y2,
    float* __restrict__ yb2,
    int rows2,
    const unsigned int* __restrict__ packed3,
    const __nv_bfloat16* __restrict__ scales3,
    const __nv_bfloat16* __restrict__ biases3,
    float* __restrict__ y3,
    float* __restrict__ yb3,
    int rows3,
    const float* __restrict__ x,
    const float* __restrict__ xb,
    int in_dim)
{
    const int tid = threadIdx.x;
    const float* xcol = blockIdx.x == 0 ? x : xb;
    const int pad0 = (rows0 + RPB_V4D8 - 1) & ~(RPB_V4D8 - 1);
    const int pad1 = (rows1 + RPB_V4D8 - 1) & ~(RPB_V4D8 - 1);
    const int pad2 = (rows2 + RPB_V4D8 - 1) & ~(RPB_V4D8 - 1);

    const unsigned int* packed;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    float* y;
    int rows;
    int row_base;
    {
        int g = blockIdx.y * RPB_V4D8;
        int seg = 0;
        if (g >= pad0) { g -= pad0; seg = 1; }
        if (seg == 1 && g >= pad1) { g -= pad1; seg = 2; }
        if (seg == 2 && g >= pad2) { g -= pad2; seg = 3; }
        row_base = g;
        switch (seg) {
        case 0: packed = packed0; scales = scales0; biases = biases0; y = blockIdx.x == 0 ? y0 : yb0; rows = rows0; break;
        case 1: packed = packed1; scales = scales1; biases = biases1; y = blockIdx.x == 0 ? y1 : yb1; rows = rows1; break;
        case 2: packed = packed2; scales = scales2; biases = biases2; y = blockIdx.x == 0 ? y2 : yb2; rows = rows2; break;
        default: packed = packed3; scales = scales3; biases = biases3; y = blockIdx.x == 0 ? y3 : yb3; rows = rows3; break;
        }
        if (row_base >= rows) return;
    }

    const int words_per_row = in_dim / 8;
    const int chunk_words = (int)blockDim.x * 2;
    float xr[16];
    __shared__ float partials[RPB_V4D8][64];
    const int groups = in_dim >> 6;
    const uint2* prow2 = reinterpret_cast<const uint2*>(packed + (long long)row_base * words_per_row);
    const int wpr2 = words_per_row >> 1;
    for (int cw = 0; cw < words_per_row; cw += chunk_words) {
        const int nthr = min((int)blockDim.x, (words_per_row - cw) >> 1);
        if (tid < nthr) {
            const float* xf = xcol + cw * 8 + tid * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) xr[j] = xf[j];
        }
        #pragma unroll
        for (int r = 0; r < RPB_V4D8; r++) {
            if (r + row_base >= rows) break;
            float dot = 0.0f;
            float sumx = 0.0f;
            if (tid < nthr) {
                const uint2 w2 = prow2[(long long)r * wpr2 + (cw >> 1) + tid];
                const unsigned int ws[2] = { w2.x, w2.y };
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    const unsigned int word = ws[i];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        dot += (float)((word >> (4 * j)) & 0xFu) * xr[i * 8 + j];
                        sumx += xr[i * 8 + j];
                    }
                }
            }
            dot += __shfl_down_sync(0xffffffffu, dot, 1, 4);
            sumx += __shfl_down_sync(0xffffffffu, sumx, 1, 4);
            dot += __shfl_down_sync(0xffffffffu, dot, 2, 4);
            sumx += __shfl_down_sync(0xffffffffu, sumx, 2, 4);
            if (tid < nthr && (tid & 3) == 0) {
                const int slot = tid >> 2;
                const int gi = (row_base + r) * groups + (cw >> 3) + slot;
                const float p = __bfloat162float(scales[gi]) * dot + __bfloat162float(biases[gi]) * sumx;
                if (cw == 0) partials[r][slot] = p;
                else partials[r][slot] += p;
            }
        }
    }
    __syncthreads();
    if (tid < RPB_V4D8 && tid + row_base < rows) {
        float total = 0.0f;
        for (int s = 0; s < 64; s++) total += partials[tid][s];
        y[row_base + tid] = total;
    }
}
