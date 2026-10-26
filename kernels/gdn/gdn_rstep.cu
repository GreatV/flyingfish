// Ordered recurrent rows: h32 1..16, h48 1..5. Each (head,value split) CTA holds FP32
// state in registers. The live row count is read from the device meta; bound is the
// prepared allocation. No metadata mutation and no cross-CTA state dependency.
#include "gdn_recurrent.cuh"

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_rstep_core_v1(const float* q, const float* k, const float* v,
                       const float* beta, const float* g, float* S,
                       __nv_bfloat16* out, const Q38SequenceFields* meta, int bound) {
    gdn_recurrent_t<2, false>(q, k, v, beta, g, S, out, meta, bound, nullptr);
}

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_rstep_core_snapshot(const float* q, const float* k, const float* v,
                            const float* beta, const float* g, float* S,
                            __nv_bfloat16* out, const Q38SequenceFields* meta, int bound,
                            float* snapshots) {
    gdn_recurrent_t<2, false, true>(q, k, v, beta, g, S, out, meta, bound, snapshots);
}

// Full-head state uses sixteen key groups and four adjacent value columns per lane.
extern "C" __global__ __launch_bounds__(512, 1)
void gdn_step_fused(const __nv_bfloat16* conv, const __nv_bfloat16* a,
                    const __nv_bfloat16* b, const __nv_bfloat16* a_log,
                    const __nv_bfloat16* dt, float* state, __nv_bfloat16* out,
                    const __nv_bfloat16* norm, const __nv_bfloat16* z,
                    const Q38SequenceMeta* meta, int bound, uint64_t as0,
                    uint64_t bs0, float l2_eps, float rms_eps) {
    q38_meta_guard(meta);
    if (GDN_HEADS != 32 || gridDim.x != 1 || gridDim.y != GDN_HEADS || gridDim.z != 1 ||
        blockDim.x != 512 || blockDim.y != 1 || blockDim.z != 1 ||
        bound < 1 || bound > 16 || meta->valid_rows > bound ||
        !aligned16(conv) || !aligned16(state) || !aligned16(out) ||
        !a || !b || !a_log || !dt || (bool(norm) != bool(z))) { __trap(); return; }
    constexpr int cd = 2 * 16 * 128 + GDN_HEADS * 128;
    const int tid = threadIdx.x, lane = tid & 31, group = tid >> 5;
    const int head = blockIdx.y, kh = head / (GDN_HEADS / 16);
    __shared__ float red[16][128];
    __shared__ float qn[128], kn[128], sums[2][4];
    float s[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            s[i][j] = state[(size_t)head * 128 * 128 + (group * 8 + i) * 128 + lane * 4 + j];
    for (int row = 0; row < meta->valid_rows; ++row) {
        float qs = 0.0f, ks = 0.0f, qv = 0.0f, kv = 0.0f;
        if (tid < 128) {
            qv = __bfloat162float(conv[(size_t)row * cd + kh * 128 + tid]);
            kv = __bfloat162float(conv[(size_t)row * cd + 16 * 128 + kh * 128 + tid]);
            qs = qv * qv; ks = kv * kv;
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                qs += __shfl_down_sync(0xffffffff, qs, delta);
                ks += __shfl_down_sync(0xffffffff, ks, delta);
            }
            if (lane == 0) { sums[0][group] = qs; sums[1][group] = ks; }
        }
        __syncthreads();
        if (group == 0) {
            qs = lane < 4 ? sums[0][lane] : 0.0f;
            ks = lane < 4 ? sums[1][lane] : 0.0f;
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                qs += __shfl_down_sync(0xffffffff, qs, delta);
                ks += __shfl_down_sync(0xffffffff, ks, delta);
            }
            if (lane == 0) {
                sums[0][0] = rsqrtf(qs + l2_eps);
                sums[1][0] = rsqrtf(ks + l2_eps);
            }
        }
        __syncthreads();
        if (tid < 128) {
            qn[tid] = (qv * sums[0][0]) * (1.0f / sqrtf(128.0f));
            kn[tid] = kv * sums[1][0];
        }
        const float t = __fadd_rn(__bfloat162float(a[(size_t)row * as0 + head]), __bfloat162float(dt[head]));
        const float soft = t > 20.0f ? t : log1pf(expf(t));
        const float beta = bround(sigmoid(__bfloat162float(b[(size_t)row * bs0 + head])));
        const float g = -expf(__bfloat162float(a_log[head])) * soft;
        const float alpha = __expf(g);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] *= alpha;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) p = fmaf(kn[group * 8 + i], s[i][j], p);
            red[group][lane * 4 + j] = p;
        }
        __syncthreads();
        float delta[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float total = 0.0f;
            for (int u = 0; u < 16; ++u) total += red[u][lane * 4 + j];
            const float v = __bfloat162float(conv[(size_t)row * cd + 2 * 16 * 128 + head * 128 + lane * 4 + j]);
            delta[j] = (v - total) * beta;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = fmaf(kn[group * 8 + i], delta[j], s[i][j]);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) p = fmaf(qn[group * 8 + i], s[i][j], p);
            red[group][lane * 4 + j] = p;
        }
        __syncthreads();
        if (group == 0) {
            float values[4], squares[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float total = 0.0f;
                for (int u = 0; u < 16; ++u) total += red[u][lane * 4 + j];
                values[j] = bround(total);
                squares[j] = __fmul_rn(values[j], values[j]);
            }
            if (norm) {
                #pragma unroll
                for (int shift = 4; shift >= 1; shift >>= 1)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        squares[j] = __fadd_rn(squares[j], __shfl_down_sync(0xffffffff, squares[j], shift));
                float sum = __fadd_rn(__fadd_rn(squares[0], squares[2]), __fadd_rn(squares[1], squares[3]));
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, 16));
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, 8));
                float inv = lane == 0 ? rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / 128.0f), rms_eps)) : 0.0f;
                inv = __shfl_sync(0xffffffff, inv, 0);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = lane * 4 + j;
                    float scaled = bround(__fmul_rn(bround(__fmul_rn(values[j], inv)), __bfloat162float(norm[d])));
                    values[j] = __fmul_rn(scaled, silu(__bfloat162float(z[((size_t)row * GDN_HEADS + head) * 128 + d])));
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                out[((size_t)row * GDN_HEADS + head) * 128 + lane * 4 + j] = __float2bfloat16_rn(values[j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            state[(size_t)head * 128 * 128 + (group * 8 + i) * 128 + lane * 4 + j] = s[i][j];
}
