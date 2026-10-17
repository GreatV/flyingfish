// Token sampling kernels over BF16 logits.
//
// argmax: launch <<<1, T>>>, T <= 1024. Comparator: strictly greater value wins;
// ties resolve to the lower index, including all-equal and -inf logits. 16B loads
// (8 bf16 per uint4) when vocab % 8 == 0 and the pointer is 16B-aligned; scalar
// loads otherwise, same semantics.
//
// advance: thread 0 increments the device position scalar.
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void argmax(const __nv_bfloat16* logits, uint32_t* token, int vocab) {
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    float best = -INFINITY;
    uint32_t id = 0xFFFFFFFFu;
    const bool vec_ok = (vocab % 8 == 0) &&
                        ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);
    if (vec_ok) {
        const int nv = vocab >> 3;
        const uint4* p = reinterpret_cast<const uint4*>(logits);
        for (int i = tid; i < nv; i += nthr) {
            const uint4 r = p[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(h[j]);
                const uint32_t base = (uint32_t)i * 8 + (uint32_t)j * 2;
                if (f.x > best || (f.x == best && base < id))     { best = f.x; id = base; }
                if (f.y > best || (f.y == best && base + 1 < id)) { best = f.y; id = base + 1; }
            }
        }
    } else {
        for (int i = tid; i < vocab; i += nthr) {
            const float v = __bfloat162float(logits[i]);
            if (v > best || (v == best && (uint32_t)i < id)) { best = v; id = i; }
        }
    }
    // warp reduce with the same comparator
    for (int off = 16; off > 0; off >>= 1) {
        const float ob    = __shfl_down_sync(0xffffffffu, best, off);
        const uint32_t oi = __shfl_down_sync(0xffffffffu, id, off);
        if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
    }
    __shared__ float wv[32];
    __shared__ uint32_t wi[32];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { wv[warp] = best; wi[warp] = id; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (nthr + 31) >> 5;
        best = -INFINITY; id = 0xFFFFFFFFu;
        for (int w = lane; w < nw; w += 32) {
            const float v = wv[w];
            const uint32_t j = wi[w];
            if (v > best || (v == best && j < id)) { best = v; id = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ob    = __shfl_down_sync(0xffffffffu, best, off);
            const uint32_t oi = __shfl_down_sync(0xffffffffu, id, off);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        if (lane == 0) *token = id;
    }
}

// Per-row argmax, launch <<<budget, 1024>>>: block b reduces row b of a
// [budget][vocab] bf16 logits slab into tokens[b]. Same comparator as argmax.
// Rows b >= *valid_rows_dev return before any write.
extern "C" __global__ void __launch_bounds__(1024) argmax_rows_bf16(
    const __nv_bfloat16* __restrict__ logits, uint32_t* __restrict__ tokens,
    int vocab, const int* __restrict__ valid_rows_dev) {
    const int row = blockIdx.x;
    if (row >= *valid_rows_dev) return;

    const __nv_bfloat16* __restrict__ p = logits + (size_t)row * vocab;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    float best = -INFINITY;
    uint32_t id = 0xFFFFFFFFu;
    const bool vec_ok = (vocab % 8 == 0) &&
                        ((reinterpret_cast<uintptr_t>(p) & 15) == 0);
    if (vec_ok) {
        const int nv = vocab >> 3;
        const uint4* pv = reinterpret_cast<const uint4*>(p);
        for (int i = tid; i < nv; i += nthr) {
            const uint4 r = pv[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(h[j]);
                const uint32_t base = (uint32_t)i * 8 + (uint32_t)j * 2;
                if (f.x > best || (f.x == best && base < id))     { best = f.x; id = base; }
                if (f.y > best || (f.y == best && base + 1 < id)) { best = f.y; id = base + 1; }
            }
        }
    } else {
        for (int i = tid; i < vocab; i += nthr) {
            const float v = __bfloat162float(p[i]);
            if (v > best || (v == best && (uint32_t)i < id)) { best = v; id = i; }
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ob    = __shfl_down_sync(0xffffffffu, best, off);
        const uint32_t oi = __shfl_down_sync(0xffffffffu, id, off);
        if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
    }
    __shared__ float wv[32];
    __shared__ uint32_t wi[32];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { wv[warp] = best; wi[warp] = id; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (nthr + 31) >> 5;
        best = -INFINITY; id = 0xFFFFFFFFu;
        for (int w = lane; w < nw; w += 32) {
            const float v = wv[w];
            const uint32_t j = wi[w];
            if (v > best || (v == best && j < id)) { best = v; id = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ob    = __shfl_down_sync(0xffffffffu, best, off);
            const uint32_t oi = __shfl_down_sync(0xffffffffu, id, off);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        if (lane == 0) tokens[row] = id;
    }
}

extern "C" __global__ void advance(int* position) {
    if (threadIdx.x == 0) ++*position;
}

// A call-scoped sticky error flag. The caller clears it before the first scan.
extern "C" __global__ void finite_bf16(
    const __nv_bfloat16* values, uint32_t* nonfinite, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(__bfloat162float(values[i]))) {
        atomicOr(nonfinite, 1u);
    }
}
