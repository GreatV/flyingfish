// Wide-shape int4 GEMVs for ff-qwen35 (hidden 5120, down in 17408).
// The current path chunks every projection to in-4096 and runs the edge0
// primitive; cold numbers say block count drives bandwidth (wide_bw):
// 320 blocks -> 161 GB/s (down chunk), 15520 -> 456 (lm_head). Split-K
// restores occupancy on short-row-count shapes: each block computes a
// K-slice of RPB rows into a global scratch, a combine kernel sums the
// slices in fixed order (deterministic, §3c(3)-safe — no atomics).

#define RPB 16

// Slice partial: y_part[slice * rows + row] = sum over this slice's groups.
// in_dim must be a multiple of 64*SPLIT_K_MAX (5120/17408 are).
extern "C" __global__ void edge0_wide_gemv4_splitk(
    const unsigned int* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    const float* __restrict__ x,
    const float* __restrict__ xb,         // column B input; aliases x when
    float* __restrict__ y_part,           // [split, rows], column A
    float* __restrict__ yb_part,          // [split, rows], column B; aliases
    int rows,                             // y_part when launched with 1 column
    int in_dim,
    int split)
{
    // gridDim.y selects the column; with gridDim.y==1 no block takes the
    // B path, so xb/yb_part are never dereferenced (they alias x/y_part).
    // Column-major dispatch: gridDim.x = columns so both columns' blocks
    // for one tile dispatch adjacently (x-fastest) and L2 merges B's reads.
    const float* xcol = blockIdx.x == 0 ? x : xb;
    float* ycol = blockIdx.x == 0 ? y_part : yb_part;
    const int words_per_row = in_dim / 8;
    const int slice_words = words_per_row / split;
    const int tid = threadIdx.x;
    // gridDim.x = (rows/RPB) * split, decomposed without div-by-zero.
    const int blocks_per_row = (rows + RPB - 1) / RPB;
    const int slice = blockIdx.y / blocks_per_row;
    const int rblock = blockIdx.y % blocks_per_row;
    const int row_base = rblock * RPB;
    const int nrows = min(RPB, rows - row_base);
    if (row_base >= rows) return;

    const int w_lo = slice * slice_words;
    // xr: this slice's words, one per thread. wpt <= 3 -> slice_words <= 768:
// down 17408 words/row needs split >= 3 (use 4; 2176/3 is not whole).
    float xr[3][8];
    const int wpt = (slice_words + blockDim.x - 1) / blockDim.x;
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (k >= wpt) break;
        const int w = tid + k * blockDim.x;
        if (w < slice_words) {
            #pragma unroll
            for (int j = 0; j < 8; j++)
                xr[k][j] = xcol[(w_lo + w) * 8 + j];
        }
    }

    __shared__ float partials[RPB][192];  // <=136 slice groups (down split=2)
    const int groups_total = in_dim >> 6;
    const int g_lo = (w_lo * 8) >> 6;   // first group index in slice
    // Rows are always a full RPB multiple (host asserts) — no break, so
    // the unrolled r-loop's shuffles stay convergence-safe (the mega
    // lesson: break-guarded loops with shuffles miscompile at some
    // unroll factors; RPB=16 hid it, RPB=8 did not).
    #pragma unroll
    for (int r = 0; r < RPB; r++) {
        const unsigned int* prow = packed + (long long)(row_base + r) * words_per_row;
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            if (k >= wpt) break;
            const int w = tid + k * blockDim.x;   // slice-local word
            const int gw = w_lo + w;              // global word
            float dot = 0.0f;
            float sumx = 0.0f;
            if (w < slice_words) {
                const unsigned int word = prow[gw];
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
            if (w < slice_words && (tid & 7) == 0) {
                const int group = gw >> 3;
                const int gi = (row_base + r) * groups_total + group;
                partials[r][group - g_lo] = scales[gi] * dot + biases[gi] * sumx;
            }
        }
    }
    __syncthreads();
    // Parallel finalize: thread r sums its row's slice groups, left fold.
    if (tid < nrows) {
        const int groups_here = slice_words >> 3;
        float total = 0.0f;
        for (int g = 0; g < groups_here; g++) total += partials[tid][g];
        ycol[(long long)slice * rows + row_base + tid] = total;
    }
}

// Combine: y[row] = sum over slices in ascending order (deterministic).
extern "C" __global__ void edge0_wide_combine(
    const float* __restrict__ y_part,   // [split, rows], column A
    const float* __restrict__ yb_part,  // [split, rows], column B (aliases A)
    float* __restrict__ y,              // [rows], column A
    float* __restrict__ yb,             // [rows], column B (aliases y)
    int rows,
    int split)
{
    // gridDim.y = columns; the B pointers are only touched at y==1.
    const float* part = blockIdx.x == 0 ? y_part : yb_part;
    float* out = blockIdx.x == 0 ? y : yb;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float total = 0.0f;
    for (int s = 0; s < split; s++)
        total += part[(long long)s * rows + row];
    out[row] = total;
}

// Wide GROUPED int4 GEMV: up to 4 segments over the same x in ONE launch,
// full width (in_dim <= 6144 -> wpt 3, NO chunking — the current path
// chunks at 4096 and pays two launches plus a partial-sum pass per group).
// Segments are 16-row aligned; blocks stride (segment-block, k) with no
// K-split (words <= 768). No LoRA path: qwen35 ships no adapters (assert
// on the host side). Deterministic per-row finalize, left fold.
extern "C" __global__ void edge0_wide_group4(
    const unsigned int* __restrict__ packed0,
    const float* __restrict__ scales0,
    const float* __restrict__ biases0,
    float* __restrict__ y0,
    float* __restrict__ yb0,            // column-B outputs; alias y when
    int rows0,
    const unsigned int* __restrict__ packed1,
    const float* __restrict__ scales1,
    const float* __restrict__ biases1,
    float* __restrict__ y1,
    float* __restrict__ yb1,
    int rows1,
    const unsigned int* __restrict__ packed2,
    const float* __restrict__ scales2,
    const float* __restrict__ biases2,
    float* __restrict__ y2,
    float* __restrict__ yb2,
    int rows2,
    const unsigned int* __restrict__ packed3,
    const float* __restrict__ scales3,
    const float* __restrict__ biases3,
    float* __restrict__ y3,
    float* __restrict__ yb3,
    int rows3,
    const float* __restrict__ x,
    const float* __restrict__ xb,        // column B input; aliases x
    int in_dim)
{
    const int tid = threadIdx.x;
    const float* xcol = blockIdx.x == 0 ? x : xb;
    const int pad0 = (rows0 + 15) & ~15;
    const int pad1 = (rows1 + 15) & ~15;
    const int pad2 = (rows2 + 15) & ~15;

    const unsigned int* packed;
    const float* scales;
    const float* biases;
    float* y;
    int rows;
    int row_base;
    {
        int g = blockIdx.y * 16;
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

    const int words_per_row = in_dim / 8;   // <= 768
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

    __shared__ float partials[16][96];  // <= 96 groups (in 6144)
    const int groups = in_dim >> 6;
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        if (r + row_base >= rows) break;   // segments are 16-padded; rows guard only
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
                partials[r][group] = scales[gi] * dot + biases[gi] * sumx;
            }
        }
    }
    __syncthreads();
    if (tid < 16 && tid + row_base < rows) {
        float total = 0.0f;
        for (int g2 = 0; g2 < groups; g2++) total += partials[tid][g2];
        y[row_base + tid] = total;
    }
}

// uint4-widened splitk: each thread loads FOUR consecutive words (32
// int4 columns) per pass — one LDG.128 instead of three LDG.32 — for the
// instruction-issue-bound regime the back-to-back probe identified
// (43.9 us warm = ~4x the L2 floor; geometry tweaks are exhausted).
// A thread pair now covers one 64-col group: the segment reduce is a
// width-2 shuffle, even threads lead. Deterministic per row, left fold.
// Requires slice_words % 4 == 0 and 4-byte-aligned row starts (rows are
// word-aligned by layout; slice boundaries must land on 4-word edges —
// the launcher asserts).
extern "C" __global__ void edge0_wide_gemv4_splitk_v4(
    const unsigned int* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    const float* __restrict__ x,
    const float* __restrict__ xb,
    float* __restrict__ y_part,
    float* __restrict__ yb_part,
    int rows,
    int in_dim,
    int split)
{
    const float* xcol = blockIdx.x == 0 ? x : xb;
    float* ycol = blockIdx.x == 0 ? y_part : yb_part;
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
    const int words4 = slice_words >> 2;          // uint4 tiles per row-slice
    const uint4* prow4 = reinterpret_cast<const uint4*>(packed + (long long)row_base * words_per_row + w_lo);

    // x tile for this thread: 4 words = 32 columns, in registers.
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
        // Thread pair covers one 64-col group: width-2 reduce, even leads.
        dot += __shfl_down_sync(0xffffffffu, dot, 1, 2);
        sumx += __shfl_down_sync(0xffffffffu, sumx, 1, 2);
        if (active && (tid & 1) == 0) {
            const int group = g_lo + (tid >> 1);
            const int gi = (row_base + r) * groups_total + group;
            partials[r][group - g_lo] = scales[gi] * dot + biases[gi] * sumx;
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


// uint4-widened group4: same segment layout and column-major dispatch as
// edge0_wide_group4, but each thread loads four consecutive words per row
// (one LDG.128). Thread pairs cover one 64-col group (width-2 reduce).
// Requires words_per_row % 4 == 0 and words_per_row <= 1024.
extern "C" __global__ void edge0_wide_group4_v4(
    const unsigned int* __restrict__ packed0,
    const float* __restrict__ scales0,
    const float* __restrict__ biases0,
    float* __restrict__ y0,
    float* __restrict__ yb0,
    int rows0,
    const unsigned int* __restrict__ packed1,
    const float* __restrict__ scales1,
    const float* __restrict__ biases1,
    float* __restrict__ y1,
    float* __restrict__ yb1,
    int rows1,
    const unsigned int* __restrict__ packed2,
    const float* __restrict__ scales2,
    const float* __restrict__ biases2,
    float* __restrict__ y2,
    float* __restrict__ yb2,
    int rows2,
    const unsigned int* __restrict__ packed3,
    const float* __restrict__ scales3,
    const float* __restrict__ biases3,
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
    const float* scales;
    const float* biases;
    float* y;
    int rows;
    int row_base;
    {
        int g = blockIdx.y * 16;
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

    const int words_per_row = in_dim / 8;      // <= 1024
    const int stride4 = words_per_row >> 2;    // uint4 row stride
    const bool active = tid < stride4;
    float xr[32];
    if (active) {
        const float* xf = xcol + tid * 32;
        #pragma unroll
        for (int j = 0; j < 32; j++) xr[j] = xf[j];
    }

    __shared__ float partials[16][128];  // <= 128 groups (in 8192)
    const int groups = in_dim >> 6;
    const uint4* prow4 = reinterpret_cast<const uint4*>(packed + (long long)row_base * words_per_row);
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        if (r + row_base >= rows) break;
        float dot = 0.0f;
        float sumx = 0.0f;
        if (active) {
            const uint4 w4 = prow4[(long long)r * stride4 + tid];
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
            const int group = tid >> 1;
            const int gi = (row_base + r) * groups + group;
            partials[r][group] = scales[gi] * dot + biases[gi] * sumx;
        }
    }
    __syncthreads();
    if (tid < 16 && tid + row_base < rows) {
        float total = 0.0f;
        for (int g2 = 0; g2 < groups; g2++) total += partials[tid][g2];
        y[row_base + tid] = total;
    }
}
