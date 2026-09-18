// Batched expert GEMV: one launch covers the 4 routed experts' SAME
// projection (gate, up or down) over the fused stacked tensor. The
// checkpoint stores switch_mlp.{proj} as ONE [256 experts, rows, in/8]
// tensor; gridDim.y = expert slot, gridDim.x covers rows in
// ROWS_PER_BLOCK chunks, x preloaded to registers once per block (the
// one-row-per-block shape ran at ~16% of DRAM bandwidth).
//
// Group leaders write per-group partials; thread 0 sums them in fixed
// group order per row — deterministic run-to-run. Threads past
// words_per_row carry zeros through the segment shuffles.

#define ROWS_PER_BLOCK 4

extern "C" __global__ void edge0_batched_gemv4(
    const unsigned int* __restrict__ packed,   // [256, rows, words]
    const float* __restrict__ scales,          // [256, rows, groups]
    const float* __restrict__ biases,
    const int* __restrict__ expert_ids,        // [slots]
    const float* __restrict__ x,               // shared input
    float* __restrict__ y,                     // [slots, rows] output
    int rows,
    int in_dim,
    int slots)
{
    const int slot = blockIdx.y;
    if (slot >= slots) return;
    const int expert = expert_ids[slot];
    const int tid = threadIdx.x;
    const int words_per_row = in_dim / 8;

    float xr[8];
    if (tid < words_per_row) {
        #pragma unroll
        for (int j = 0; j < 8; j++) xr[j] = x[tid * 8 + j];
    }

    const int nrows = min(ROWS_PER_BLOCK, rows - (int)blockIdx.x * ROWS_PER_BLOCK);
    unsigned int words[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        const long long base =
            ((long long)expert * rows + blockIdx.x * ROWS_PER_BLOCK + r) * words_per_row;
        words[r] = (r < nrows && tid < words_per_row) ? packed[base + tid] : 0u;
    }
    __shared__ float partials[ROWS_PER_BLOCK][32];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        if (r >= nrows) break;
        const int row = blockIdx.x * ROWS_PER_BLOCK + r;
        const int sb_base = ((long long)expert * rows + row) * (in_dim >> 6);
        float dot = 0.0f;
        float sumx = 0.0f;
        if (tid < words_per_row) {
            const unsigned int word = words[r];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dot += (float)((word >> (4 * j)) & 0xFu) * xr[j];
                sumx += xr[j];
            }
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
            sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
        }
        if (tid < words_per_row && (tid & 7) == 0) {
            const int group = tid >> 3;
            partials[r][group] =
                scales[sb_base + group] * dot +
                biases[sb_base + group] * sumx;
        }
    }
    __syncthreads();
    if (tid < nrows) {
        const int row = blockIdx.x * ROWS_PER_BLOCK + tid;
        const int groups = in_dim >> 6;
        float total = 0.0f;
        for (int g = 0; g < groups; g++) total += partials[tid][g];
        y[(long long)slot * rows + row] = total;
    }
}

// Down variant: the input x is per-slot ([slots, in_dim]) — each routed
// expert's silu output differs. Same multi-row shape.
extern "C" __global__ void edge0_batched_gemv4_slotx(
    const unsigned int* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    const int* __restrict__ expert_ids,
    const float* __restrict__ x,           // [slots, in_dim]
    float* __restrict__ y,                 // [slots, rows]
    int rows,
    int in_dim,
    int slots)
{
    const int slot = blockIdx.y;
    if (slot >= slots) return;
    const int expert = expert_ids[slot];
    const int tid = threadIdx.x;
    const int words_per_row = in_dim / 8;

    float xr[8];
    if (tid < words_per_row) {
        #pragma unroll
        for (int j = 0; j < 8; j++) xr[j] = x[slot * in_dim + tid * 8 + j];
    }

    const int nrows = min(ROWS_PER_BLOCK, rows - (int)blockIdx.x * ROWS_PER_BLOCK);
    unsigned int words[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        const long long base =
            ((long long)expert * rows + blockIdx.x * ROWS_PER_BLOCK + r) * words_per_row;
        words[r] = (r < nrows && tid < words_per_row) ? packed[base + tid] : 0u;
    }
    __shared__ float partials[ROWS_PER_BLOCK][32];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        if (r >= nrows) break;
        const int row = blockIdx.x * ROWS_PER_BLOCK + r;
        const int sb_base = ((long long)expert * rows + row) * (in_dim >> 6);
        float dot = 0.0f;
        float sumx = 0.0f;
        if (tid < words_per_row) {
            const unsigned int word = words[r];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dot += (float)((word >> (4 * j)) & 0xFu) * xr[j];
                sumx += xr[j];
            }
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
            sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
        }
        if (tid < words_per_row && (tid & 7) == 0) {
            const int group = tid >> 3;
            partials[r][group] =
                scales[sb_base + group] * dot +
                biases[sb_base + group] * sumx;
        }
    }
    __syncthreads();
    if (tid < nrows) {
        const int row = blockIdx.x * ROWS_PER_BLOCK + tid;
        const int groups = in_dim >> 6;
        float total = 0.0f;
        for (int g = 0; g < groups; g++) total += partials[tid][g];
        y[(long long)slot * rows + row] = total;
    }
}

// slotx + silu + router-weight fold: x is computed inline as
// w[slot] * silu(g) * u — the §3c(3) fold (weight applied inside the
// inner, not to the down outputs). Deletes the routed silu_mul launch.
extern "C" __global__ void edge0_batched_gemv4_slotx_silu(
    const unsigned int* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    const int* __restrict__ expert_ids,
    const float* __restrict__ g,           // [slots, in_dim]
    const float* __restrict__ u,           // [slots, in_dim]
    const float* __restrict__ w,           // [slots]
    float* __restrict__ y,                 // [slots, rows]
    int rows,
    int in_dim,
    int slots)
{
    const int slot = blockIdx.y;
    if (slot >= slots) return;
    const int expert = expert_ids[slot];
    const int tid = threadIdx.x;
    const int words_per_row = in_dim / 8;
    const float ws = w[slot];

    float xr[8];
    if (tid < words_per_row) {
        const int c = slot * in_dim + tid * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float gv = g[c + j];
            xr[j] = ws * (gv / (1.0f + expf(-gv))) * u[c + j];
        }
    }

    const int nrows = min(ROWS_PER_BLOCK, rows - (int)blockIdx.x * ROWS_PER_BLOCK);
    unsigned int words[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        const long long base =
            ((long long)expert * rows + blockIdx.x * ROWS_PER_BLOCK + r) * words_per_row;
        words[r] = (r < nrows && tid < words_per_row) ? packed[base + tid] : 0u;
    }
    __shared__ float partials[ROWS_PER_BLOCK][32];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        if (r >= nrows) break;
        const int row = blockIdx.x * ROWS_PER_BLOCK + r;
        const int sb_base = ((long long)expert * rows + row) * (in_dim >> 6);
        float dot = 0.0f;
        float sumx = 0.0f;
        if (tid < words_per_row) {
            const unsigned int word = words[r];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dot += (float)((word >> (4 * j)) & 0xFu) * xr[j];
                sumx += xr[j];
            }
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
            sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
        }
        if (tid < words_per_row && (tid & 7) == 0) {
            const int group = tid >> 3;
            partials[r][group] =
                scales[sb_base + group] * dot +
                biases[sb_base + group] * sumx;
        }
    }
    __syncthreads();
    if (tid < nrows) {
        const int row = blockIdx.x * ROWS_PER_BLOCK + tid;
        const int groups = in_dim >> 6;
        float total = 0.0f;
        for (int g2 = 0; g2 < groups; g2++) total += partials[tid][g2];
        y[(long long)slot * rows + row] = total;
    }
}
