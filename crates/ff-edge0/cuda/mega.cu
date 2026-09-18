// Mega-kernel scaffolding: sense-reversing arrival-counter grid barrier
// (§6) with a spin cap that traps — a wrong grid-vs-residency answer must
// fail loud, not hang. No cooperative launch: the barrier works in normal
// launches and inside graphs, provided the grid fits co-residency (the
// host asserts this via the occupancy API).

struct GridBarrier {
    volatile int count;
    volatile int sense;
};

// One thread per CTA participates; the rest wait at __syncthreads.
__device__ __forceinline__ void grid_barrier(
    GridBarrier* b, int nblocks, int cap)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        const int s = b->sense;
        __threadfence();
        const int arrived = atomicAdd((int*)&b->count, 1);
        if (arrived == nblocks - 1) {
            b->count = 0;
            __threadfence();
            b->sense = 1 - s;
        } else {
            int spins = 0;
            while (b->sense == s) {
                if (cap > 0 && ++spins > cap) __trap();
                __nanosleep(64);
            }
            __threadfence();  // acquire: order later reads after the flip
        }
    }
    __syncthreads();
}

// Microbench: BARRIER_ITERS barriers per CTA over a small touched buffer.
extern "C" __global__ void edge0_barrier_bench(
    GridBarrier* bar,
    float* scratch,
    int iters,
    int cap)
{
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = 0; i < iters; i++) {
        acc += scratch[blockIdx.x * blockDim.x + tid];
        scratch[blockIdx.x * blockDim.x + tid] = acc;
        grid_barrier(bar, gridDim.x, cap);
    }
    if (acc == 12345.678f) scratch[0] = acc;  // keep alive
}

// Stub with the per-phase footprint UNION of the three mega-kernels:
// GEMV register staging (xr-style), GDN-heads shared arrays, MoE partials
// + ax. One phase streams `stream_len` floats per CTA (GEMV-like traffic)
// so phase bandwidth inside the resident grid is measurable.
extern "C" __global__ void edge0_mega_stub(
    GridBarrier* bar,
    const float* stream_in,
    float* stream_out,
    int phases,
    long long stream_len,
    int cap)
{
    const int tid = threadIdx.x;

    // GEMV-phase register staging shape.
    volatile float xr[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) xr[j] = (float)j * 0.5f;

    // GDN-heads shared shape (q_n/k_n/out_sh + partials + ax union).
    __shared__ float sh_q[256];
    __shared__ float sh_k[256];
    __shared__ float sh_out[256];
    __shared__ float partials[64];
    sh_q[tid] = xr[tid & 15];
    sh_k[tid] = xr[(tid + 1) & 15];
    __syncthreads();

    for (int p = 0; p < phases; p++) {
        // GEMV-like streaming phase: coalesced pass over stream_len.
        float acc = 0.0f;
        for (long long i = blockIdx.x * blockDim.x + tid;
             i < stream_len;
             i += (long long)gridDim.x * blockDim.x)
        {
            acc += stream_in[i];
        }
        partials[tid & 63] = acc;
        __syncthreads();
        sh_out[tid & 255] = partials[tid & 63] + sh_q[tid] + sh_k[tid];
        __syncthreads();
        if (tid == 0) stream_out[blockIdx.x] = sh_out[0];

        grid_barrier(bar, gridDim.x, cap);
    }
    if (sh_out[tid & 255] == 12345.678f) stream_out[0] = 1.0f;  // keep alive
}

// Row-loop helpers ported verbatim from the standalone kernels so the
// mega-kernel's numerics are bit-identical. Both return the row total via
// a shared broadcast slot (partials[63]; all moe_mega shapes have <= 32
// groups so the slot never collides with data). int4 shapes here have
// in_dim <= 2048 (words <= 256, one word per thread).

__device__ __forceinline__ void gemv8_row_total(
    const unsigned int* packed, const float* scales, const float* biases,
    int row, const float* x, int in_dim, int rank,
    const float* la, const float* lb, const float* ax,
    int tid, float* partials, float* out_total)
{
    const int words_per_row = in_dim / 4;
    const int wpt = (words_per_row + blockDim.x - 1) / blockDim.x;
    for (int k = 0; k < wpt; k++) {
        const int w = tid + k * blockDim.x;
        float dot = 0.0f;
        float sumx = 0.0f;
        if (w < words_per_row) {
            const unsigned int word = packed[(long long)row * words_per_row + w];
            const int col = w * 4;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float q = (float)((word >> (8 * j)) & 0xFFu);
                const float xv = x[col + j];
                dot += q * xv;
                sumx += xv;
            }
        }
        for (int off = 8; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, off, 16);
            sumx += __shfl_down_sync(0xffffffffu, sumx, off, 16);
        }
        if (w < words_per_row && (tid & 15) == 0) {
            const int group = w / 16;
            const int gi = row * (in_dim >> 6) + group;
            partials[group] = scales[gi] * dot + biases[gi] * sumx;
        }
    }
    __syncthreads();
    if (tid == 0) {
        const int groups = in_dim >> 6;
        float total = 0.0f;
        for (int g = 0; g < groups; g++) total += partials[g];
        if (rank > 0) {
            const float* br = lb + (long long)row * rank;
            for (int k = 0; k < rank; k++) total += br[k] * ax[k];
        }
        partials[63] = total;
    }
    __syncthreads();
    *out_total = partials[63];
    __syncthreads();
}

__device__ __forceinline__ void gemv4_row_total(
    const unsigned int* packed, const float* scales, const float* biases,
    int row, const float* xr, int in_dim, int rank,
    const float* lb, const float* ax,
    int tid, float* partials, float* out_total)
{
    const int words_per_row = in_dim / 8;
    const int w = tid;
    float dot = 0.0f;
    float sumx = 0.0f;
    if (w < words_per_row) {
        const unsigned int word = packed[(long long)row * words_per_row + w];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            dot += (float)((word >> (4 * j)) & 0xFu) * xr[j];
            sumx += xr[j];
        }
    }
    for (int off = 4; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
        sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
    }
    if (w < words_per_row && (tid & 7) == 0) {
        const int group = w / 8;
        const int gi = row * (in_dim >> 6) + group;
        partials[group] = scales[gi] * dot + biases[gi] * sumx;
    }
    __syncthreads();
    if (tid == 0) {
        const int groups = in_dim >> 6;
        float total = 0.0f;
        for (int g = 0; g < groups; g++) total += partials[g];
        if (rank > 0) {
            const float* br = lb + (long long)row * rank;
            for (int k = 0; k < rank; k++) total += br[k] * ax[k];
        }
        partials[63] = total;
    }
    __syncthreads();
    *out_total = partials[63];
    __syncthreads();
}

// ax = A * xv for a rank <= 16 pair, computed cooperatively by the CTA
// (warp w covers k = w and w + warps), same order as the group kernel.
__device__ __forceinline__ void lora_ax(
    const float* la, const float* xv, int in_dim, int rank,
    int tid, float* ax)
{
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    for (int k = warp; k < rank; k += warps) {
        const float* ar = la + (long long)k * in_dim;
        float acc = 0.0f;
        for (int c = lane; c < in_dim; c += 32) acc += ar[c] * xv[c];
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) ax[k] = acc;
    }
    __syncthreads();
}

// moe_mega: the whole MoE block in one launch, four barriers:
//   P1 router(int8+lora) | ss(int8+lora) | sg,su(int4+lora)   [dx only]
//   P2 top-k over router y (one CTA, writes ids + weights)
//   P3 routed gate+up batched (int4, reads ids)
//   P4 routed slotx_silu down + shared down over silu(sg,su)
//   P5 combine into hidden
// Work items are strided by gridDim.x; every phase's row math is the
// standalone kernels' loops verbatim (bit-identical numerics).
extern "C" __global__ void edge0_moe_mega(
    GridBarrier* bar, int cap,
    const unsigned int* __restrict__ r_packed,
    const float* __restrict__ r_scales, const float* __restrict__ r_biases,
    const float* __restrict__ r_la, const float* __restrict__ r_lb,
    int r_rows,
    const unsigned int* __restrict__ ss_packed,
    const float* __restrict__ ss_scales, const float* __restrict__ ss_biases,
    const float* __restrict__ ss_la, const float* __restrict__ ss_lb,
    float* __restrict__ ss_y,
    const unsigned int* __restrict__ sg_packed,
    const float* __restrict__ sg_scales, const float* __restrict__ sg_biases,
    const float* __restrict__ sg_la, const float* __restrict__ sg_lb,
    int sg_rows, float* __restrict__ sg_y,
    const unsigned int* __restrict__ su_packed,
    const float* __restrict__ su_scales, const float* __restrict__ su_biases,
    const float* __restrict__ su_la, const float* __restrict__ su_lb,
    float* __restrict__ su_y,
    const float* __restrict__ x, int in_dim, int rank,
    int* __restrict__ ids, float* __restrict__ w,
    const unsigned int* __restrict__ exg_packed,
    const float* __restrict__ exg_scales, const float* __restrict__ exg_biases,
    const unsigned int* __restrict__ exu_packed,
    const float* __restrict__ exu_scales, const float* __restrict__ exu_biases,
    float* __restrict__ gate_y, float* __restrict__ up_y,
    int ex_rows,
    const unsigned int* __restrict__ exd_packed,
    const float* __restrict__ exd_scales, const float* __restrict__ exd_biases,
    float* __restrict__ down_y, int down_rows,
    const unsigned int* __restrict__ sd_packed,
    const float* __restrict__ sd_scales, const float* __restrict__ sd_biases,
    const float* __restrict__ sd_la, const float* __restrict__ sd_lb,
    float* __restrict__ sd_y,
    float* __restrict__ hidden,
    float* __restrict__ router_y)
{
    const int tid = threadIdx.x;
    const int nblocks = gridDim.x;
    __shared__ float partials[64];
    __shared__ float ax[16];
    float total;

    const int words4 = in_dim / 8;   // int4 words per row (<= 256)

    // ---------- P1: router | ss | sg | su ----------
    {
        const int router_items = (r_rows + 15) >> 4;
        const int ss_items = 1;
        const int sg_items = (sg_rows + 15) >> 4;
        const int n1 = router_items + ss_items + 2 * sg_items;
        for (int item = blockIdx.x; item < n1; item += nblocks) {
            float xr[8];
            if (tid < words4) {
                #pragma unroll
                for (int j = 0; j < 8; j++) xr[j] = x[tid * 8 + j];
            }
            if (item < router_items) {
                // mlp.gate ships no adapter pair in this checkpoint: r_la/
                // r_lb are dummy pointers and must never be dereferenced —
                // rank is forced to 0 for this segment (the old code read
                // 128KB through an 8KB dummy and trapped).
                for (int r = 0; r < 16; r++) {
                    const int row = item * 16 + r;
                    if (row >= r_rows) break;
                    gemv8_row_total(r_packed, r_scales, r_biases, row, x,
                                    in_dim, 0, r_la, r_lb, ax, tid, partials, &total);
                    if (tid == 0) router_y[row] = total;
                }
            } else if (item == router_items) {
                // shared_expert_gate likewise ships no adapter pair.
                gemv8_row_total(ss_packed, ss_scales, ss_biases, 0, x,
                                in_dim, 0, ss_la, ss_lb, ax, tid, partials, &total);
                if (tid == 0) ss_y[0] = total;
            } else if (item < router_items + ss_items + sg_items) {
                const int blk = item - router_items - ss_items;
                if (rank > 0) lora_ax(sg_la, x, in_dim, rank, tid, ax);
                for (int r = 0; r < 16; r++) {
                    const int row = blk * 16 + r;
                    if (row >= sg_rows) break;
                    gemv4_row_total(sg_packed, sg_scales, sg_biases, row, xr,
                                    in_dim, rank, sg_lb, ax, tid, partials, &total);
                    if (tid == 0) sg_y[row] = total;
                }
            } else {
                const int blk = item - router_items - ss_items - sg_items;
                if (rank > 0) lora_ax(su_la, x, in_dim, rank, tid, ax);
                for (int r = 0; r < 16; r++) {
                    const int row = blk * 16 + r;
                    if (row >= sg_rows) break;
                    gemv4_row_total(su_packed, su_scales, su_biases, row, xr,
                                    in_dim, rank, su_lb, ax, tid, partials, &total);
                    if (tid == 0) su_y[row] = total;
                }
            }
        }
    }
    grid_barrier(bar, nblocks, cap);

    // ---------- P2: top-k ----------
    if (blockIdx.x == 0 && tid == 0) {
        bool used[256];
        for (int i = 0; i < r_rows; i++) used[i] = false;
        float maxv[4];
        for (int s = 0; s < 4; s++) {
            int best = -1;
            for (int i = 0; i < r_rows; i++) {
                if (!used[i] && (best < 0 || router_y[i] > router_y[best])) best = i;
            }
            ids[s] = best;
            maxv[s] = router_y[best];
            used[best] = true;
        }
        const float m = maxv[0];
        float sum = 0.0f;
        for (int s = 0; s < 4; s++) {
            w[s] = __expf(maxv[s] - m);
            sum += w[s];
        }
        for (int s = 0; s < 4; s++) w[s] /= sum;
    }
    grid_barrier(bar, nblocks, cap);

    // ---------- P3: routed gate + up ----------
    {
        const int per_proj = 4 * ((ex_rows + 15) >> 4);
        const int n3 = 2 * per_proj;
        for (int item = blockIdx.x; item < n3; item += nblocks) {
            const int blocks_per_slot = (ex_rows + 15) >> 4;
            const int proj = item / per_proj;             // 0 = gate, 1 = up
            const int rem = item % per_proj;
            const int slot = rem / blocks_per_slot;
            const int blk = rem % blocks_per_slot;
            const int expert = ids[slot];
            float xr[8];
            if (tid < words4) {
                #pragma unroll
                for (int j = 0; j < 8; j++) xr[j] = x[tid * 8 + j];
            }
            const unsigned int* pk = proj == 0 ? exg_packed : exu_packed;
            const float* sc = proj == 0 ? exg_scales : exu_scales;
            const float* bi = proj == 0 ? exg_biases : exu_biases;
            float* yy = proj == 0 ? gate_y : up_y;
            for (int r = 0; r < 16; r++) {
                const int row = blk * 16 + r;
                if (row >= ex_rows) break;
                const long long base = (long long)expert * ex_rows + row;
                float dot = 0.0f;
                float sumx = 0.0f;
                if (tid < words4) {
                    const unsigned int word = pk[base * words4 + tid];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        dot += (float)((word >> (4 * j)) & 0xFu) * xr[j];
                        sumx += xr[j];
                    }
                }
                for (int off = 4; off > 0; off >>= 1) {
                    dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
                    sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
                }
                if (tid < words4 && (tid & 7) == 0) {
                    const int group = tid >> 3;
                    partials[group] =
                        sc[base * (in_dim >> 6) + group] * dot +
                        bi[base * (in_dim >> 6) + group] * sumx;
                }
                __syncthreads();
                if (tid == 0) {
                    float t2 = 0.0f;
                    for (int g = 0; g < (in_dim >> 6); g++) t2 += partials[g];
                    yy[(long long)slot * ex_rows + row] = t2;
                }
                __syncthreads();
            }
        }
    }
    grid_barrier(bar, nblocks, cap);

    // ---------- P4: slotx_silu down + shared down ----------
    {
        const int down_blocks = (down_rows + 15) >> 4;
        const int n4a = 4 * down_blocks;
        const int n4b = down_blocks;
        for (int item = blockIdx.x; item < n4a + n4b; item += nblocks) {
            if (item < n4a) {
                const int slot = item / down_blocks;
                const int blk = item % down_blocks;
                const int expert = ids[slot];
                const float ws = w[slot];
                const int words_d = 64;  // down input width 512
                float xrd[8];
                if (tid < words_d) {
                    const int c = slot * 512 + tid * 8;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float gv = gate_y[c + j];
                        xrd[j] = ws * (gv / (1.0f + expf(-gv))) * up_y[c + j];
                    }
                }
                for (int r = 0; r < 16; r++) {
                    const int row = blk * 16 + r;
                    if (row >= down_rows) break;
                    const unsigned int word =
                        tid < words_d
                            ? exd_packed[((long long)expert * down_rows + row) * words_d + tid]
                            : 0u;
                    float dot = 0.0f, sumx = 0.0f;
                    if (tid < words_d) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            dot += (float)((word >> (4 * j)) & 0xFu) * xrd[j];
                            sumx += xrd[j];
                        }
                    }
                    for (int off = 4; off > 0; off >>= 1) {
                        dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
                        sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
                    }
                    if (tid < words_d && (tid & 7) == 0) {
                        const int group = tid >> 3;
                        const long long sb = (long long)expert * down_rows + row;
                        partials[group] =
                            exd_scales[sb * 8 + group] * dot +
                            exd_biases[sb * 8 + group] * sumx;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        float t2 = 0.0f;
                        for (int g = 0; g < 8; g++) t2 += partials[g];
                        down_y[(long long)slot * down_rows + row] = t2;
                    }
                    __syncthreads();
                }
            } else {
                const int blk = item - n4a;
                float xr[8];
                const int words_sd = 64;  // shared down in 512
                if (tid < words_sd) {
                    const int c = tid * 8;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float gv = sg_y[c + j];
                        xr[j] = (gv / (1.0f + expf(-gv))) * su_y[c + j];
                    }
                }
                if (rank > 0) {
                    // LoRA acts on silu(g)*u — recompute in lora_ax's caller
                    // is not possible; inline the silu'd x into shared first.
                    __shared__ float xs[512];
                    if (tid < words_sd) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) xs[tid * 8 + j] = xr[j];
                    }
                    __syncthreads();
                    lora_ax(sd_la, xs, 512, rank, tid, ax);
                }
                for (int r = 0; r < 16; r++) {
                    const int row = blk * 16 + r;
                    if (row >= down_rows) break;
                    gemv4_row_total(sd_packed, sd_scales, sd_biases, row, xr,
                                    512, rank, sd_lb, ax, tid, partials, &total);
                    if (tid == 0) sd_y[row] = total;
                }
            }
        }
    }
    grid_barrier(bar, nblocks, cap);

    // ---------- P5: combine ----------
    {
        const float g = ss_y[0];
        const float sig = 1.0f / (1.0f + __expf(-g));
        for (int i = blockIdx.x * blockDim.x + tid; i < down_rows; i += nblocks * blockDim.x) {
            float acc = 0.0f;
            for (int s = 0; s < 4; s++) acc += down_y[(long long)s * down_rows + i];
            acc += sig * sd_y[i];
            hidden[i] += acc;
        }
    }
}

// TLB probe: read n_ptrs buffers of len_each floats through a device
// pointer table — identical access pattern for scattered allocations and
// slab offsets, isolating allocation locality as the variable.
extern "C" __global__ void edge0_read_scatter(
    const unsigned long long* __restrict__ ptr_table,
    int n_ptrs,
    long long len_each,
    float* __restrict__ out)
{
    const long long total = (long long)n_ptrs * len_each;
    float acc = 0.0f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total;
         i += (long long)gridDim.x * blockDim.x)
    {
        const int p = (int)(i / len_each);
        const long long o = i - (long long)p * len_each;
        acc += ((const float*)ptr_table[p])[o];
    }
    if (acc == 12345.0f) out[0] = acc;
}
