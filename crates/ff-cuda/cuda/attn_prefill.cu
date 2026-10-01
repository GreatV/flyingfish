#include <cuda_bf16.h>

#define BM 48   // M rows: BT=8 tokens x G=6 q-heads
#define BT 8
#define G 6
#define D 256   // head_dim
#define BN 32   // keys per tile
#define TPB 192 // 6 warps: warp = (m-frag w%3, n-half w/3)

__device__ __forceinline__ int ap_lane_g() { return (threadIdx.x & 31) >> 2; }
__device__ __forceinline__ int ap_lane_tig() { return threadIdx.x & 3; }

__device__ __forceinline__ unsigned ap_smem(const void* p)
{
    return (unsigned)__cvta_generic_to_shared(p);
}

// x4 row-major: lane l addresses row l%16, col 8*(l/16) of a b16 tile.
__device__ __forceinline__ void ap_ldsm4(
    const void* smem, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(ap_smem(smem)));
}

__device__ __forceinline__ void ap_mma_bf16(
    float* acc, const unsigned* a, const unsigned* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned ap_bf16_bits(float v)
{
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// B-fragment address for a [rows_n][stride_k] bf16 tile (the mma B
// operand is col-major [k][n], which row-major [n][k] storage provides).
__device__ __forceinline__ const void* ap_b_addr(
    const void* tile, int tile_n, int k_off, int stride_k)
{
    const int lane = threadIdx.x & 31;
    const int row = tile_n + (lane & 7) + ((lane >> 4) << 3);
    const int col = k_off + (((lane >> 3) & 1) << 3);
    return static_cast<const char*>(tile) + (row * stride_k + col) * 2;
}

// A-fragment address for a [rows_m][stride_k] bf16 tile (row-major M).
__device__ __forceinline__ const void* ap_a_addr(
    const void* tile, int tile_m, int k_off, int stride_k)
{
    const int lane = threadIdx.x & 31;
    const int row = tile_m + (lane & 15);
    const int col = k_off + ((lane >> 4) << 3);
    return static_cast<const char*>(tile) + (row * stride_k + col) * 2;
}


// Partials: [qtiles, head_panels, splits, BM, D]; pml has two values per row.
extern "C" __global__ void attn_prefill(
    const float* __restrict__ q,
    const float* __restrict__ gate,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ partials,
    float* __restrict__ pml,
    int T, int base, int kv_groups, int kv_stride, int heads,
    float scale, int chunk)
{
    const int qt = blockIdx.x, panel = blockIdx.y, split = blockIdx.z;
    const int ratio = heads / kv_groups, panels = (ratio + G - 1) / G;
    const int kg = panel / panels, head0 = (panel % panels) * G;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int mf = warp % 3, half = warp / 3;
    const int g = ap_lane_g(), pair = ap_lane_tig() * 2;
    const int tok0 = qt * BT;
    extern __shared__ unsigned short smem[];
    unsigned short* sq = smem;
    unsigned short* sk = sq + BM * D;
    unsigned short* sv = sk + BN * D;
    unsigned short* sp = sv + D * BN;
    float* reduce = reinterpret_cast<float*>(sp + BM * BN);
    for (int i = tid; i < BM * D; i += TPB) {
        const int row = i / D, dim = i % D, token = tok0 + row / G;
        const int head = head0 + row % G;
        sq[i] = token < T && head < ratio ? ap_bf16_bits(q[((long long)token * heads + kg * ratio + head) * D + dim]) : 0;
    }
    __syncthreads();
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[16][4] = {};
    const int k0 = split * chunk;
    const int k1 = min(k0 + chunk, min(base + T, base + tok0 + BT));
    for (int kb = k0; kb < k1; kb += BN) {
        const int count = min(BN, k1 - kb);
        for (int i = tid; i < BN * D; i += TPB) {
            const int key = i / D, dim = i % D;
            const long long offset = (long long)(kb + key) * kv_stride + kg * D + dim;
            sk[i] = key < count ? ap_bf16_bits(keys[offset]) : 0;
            sv[dim * BN + key] = key < count ? ap_bf16_bits(values[offset]) : 0;
        }
        __syncthreads();
        float scores[2][4] = {};
        for (int d = 0; d < D; d += 16) {
            unsigned a[4], b[4];
            ap_ldsm4(ap_a_addr(sq, mf * 16, d, D), a[0], a[1], a[2], a[3]);
            ap_ldsm4(ap_b_addr(sk, half * 16, d, D), b[0], b[1], b[2], b[3]);
            ap_mma_bf16(scores[0], a, b);
            ap_mma_bf16(scores[1], a, b + 2);
        }
        for (int n = 0; n < 2; ++n) {
            for (int r = 0; r < 4; ++r) {
                const int row = mf * 16 + g + (r / 2) * 8;
                const int col = half * 16 + n * 8 + pair + (r & 1);
                const int token = tok0 + row / G;
                scores[n][r] = token < T && head0 + row % G < ratio && col < count && kb + col <= base + token
                    ? scores[n][r] * scale : -INFINITY;
            }
        }
        float corr[2];
        for (int row = 0; row < 2; ++row) {
            const int r = row * 2;
            float mx = fmaxf(fmaxf(scores[0][r], scores[0][r + 1]), fmaxf(scores[1][r], scores[1][r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            if (ap_lane_tig() == 0) reduce[(mf * 16 + g + row * 8) * 2 + half] = mx;
        }
        __syncthreads();
        for (int row = 0; row < 2; ++row) {
            const int ri = (mf * 16 + g + row * 8) * 2;
            const float next = fmaxf(m[row], fmaxf(reduce[ri], reduce[ri + 1]));
            corr[row] = m[row] == -INFINITY ? 0.0f : __expf(m[row] - next);
            m[row] = next;
            l[row] *= corr[row];
        }
        float sum[2] = {};
        for (int n = 0; n < 2; ++n) {
            for (int r = 0; r < 4; ++r) {
                const int row = mf * 16 + g + (r / 2) * 8;
                const int col = half * 16 + n * 8 + pair + (r & 1);
                const float p = scores[n][r] == -INFINITY ? 0.0f : __expf(scores[n][r] - m[r / 2]);
                sum[r / 2] += p;
                sp[row * BN + col] = ap_bf16_bits(p);
            }
        }
        __syncthreads();
        for (int row = 0; row < 2; ++row) {
            sum[row] += __shfl_xor_sync(0xffffffffu, sum[row], 1);
            sum[row] += __shfl_xor_sync(0xffffffffu, sum[row], 2);
            if (ap_lane_tig() == 0) reduce[(mf * 16 + g + row * 8) * 2 + half] = sum[row];
        }
        __syncthreads();
        for (int row = 0; row < 2; ++row) {
            const int ri = (mf * 16 + g + row * 8) * 2;
            l[row] += reduce[ri] + reduce[ri + 1];
        }
        for (int step = 0; step < D / 32; ++step) {
            unsigned a[4], b[4];
            for (int n = 0; n < 2; ++n)
                for (int r = 0; r < 4; ++r) o[step * 2 + n][r] *= corr[r / 2];
            for (int key = 0; key < BN; key += 16) {
                ap_ldsm4(ap_a_addr(sp, mf * 16, key, BN), a[0], a[1], a[2], a[3]);
                ap_ldsm4(ap_b_addr(sv, half * (D / 2) + step * 16, key, BN), b[0], b[1], b[2], b[3]);
                ap_mma_bf16(o[step * 2], a, b);
                ap_mma_bf16(o[step * 2 + 1], a, b + 2);
            }
        }
        __syncthreads();
    }
    const long long rows = (((long long)qt * gridDim.y + panel) * gridDim.z + split) * BM;
    for (int row = 0; row < 2; ++row) {
        const long long ri = rows + mf * 16 + g + row * 8;
        if (half == 0 && ap_lane_tig() == 0) {
            pml[ri * 2] = m[row];
            pml[ri * 2 + 1] = l[row];
        }
        for (int step = 0; step < D / 32; ++step)
            for (int n = 0; n < 2; ++n)
                for (int v = 0; v < 2; ++v)
                    partials[ri * D + half * (D / 2) + step * 16 + n * 8 + pair + v] = o[step * 2 + n][row * 2 + v];
    }
}

extern "C" __global__ void attn_prefill_combine(
    const float* __restrict__ partials, const float* __restrict__ pml,
    const float* __restrict__ gate, float* __restrict__ out,
    int T, int base, int kv_groups, int heads, int splits)
{
    const int qt = blockIdx.x, panel = blockIdx.y;
    const int ratio = heads / kv_groups, panels = (ratio + G - 1) / G;
    const int kg = panel / panels, head0 = (panel % panels) * G;
    const long long tile = ((long long)qt * gridDim.y + panel) * splits * BM;
    for (int i = threadIdx.x; i < BM * D; i += blockDim.x) {
        const int row = i / D, dim = i % D, token = qt * BT + row / G;
        if (token >= T || head0 + row % G >= ratio) continue;
        float max = -INFINITY;
        for (int s = 0; s < splits; ++s) max = fmaxf(max, pml[(tile + s * BM + row) * 2]);
        float value = 0.0f, denom = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const long long ri = tile + s * BM + row;
            const float length = pml[ri * 2 + 1];
            if (length == 0.0f) continue;
            const float weight = __expf(pml[ri * 2] - max);
            value += weight * partials[ri * D + dim];
            denom += weight * length;
        }
        const long long oi = ((long long)token * heads + kg * ratio + head0 + row % G) * D + dim;
        const float g = gate[oi];
        out[oi] = value / denom / (1.0f + __expf(-g));
    }
}
