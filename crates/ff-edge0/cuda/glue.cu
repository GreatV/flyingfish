// Decode glue: embed row-gather, rmsnorm variants, residual add, and the
// full-attention decode path (q/gate deinterleave + per-head rmsnorm +
// partial rope + KV append, then scores/softmax/weighted-V + output gate).
// Reductions use fixed shapes (deterministic run-to-run); soft-max sums
// and per-dimension accumulations stay in ascending step order to track
// the host reference.

// In-place per-head rmsnorm over hd elements; blockDim must cover hd.
__device__ __forceinline__ void rmsnorm_head(
    float* __restrict__ x, const float* __restrict__ w, int tid, int hd, int zc)
{
    __shared__ float partials[256];
    float acc = 0.0f;
    if (tid < hd) {
        const float v = x[tid];
        acc = v * v;
    }
    partials[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)hd + 1e-6f);
    // zc: zero-centered variant (qwen3_5 dense) applies (1 + w).
    if (tid < hd) x[tid] = zc ? x[tid] * inv * (1.0f + w[tid]) : x[tid] * inv * w[tid];
}

// Partial rotary in place over the first rotary_dim dims; the host computes
// freq/angle in f64 and casts the sin/cos, so the same is done here.
__device__ __forceinline__ void rope_head(
    float* __restrict__ x, int position, int half, int rotary_dim,
    double theta, int tid)
{
    if (tid >= half) return;
    const double freq = pow(theta, -(2.0 * tid) / rotary_dim);
    const double angle = position * freq;
    const float sin_f = (float)sin(angle);
    const float cos_f = (float)cos(angle);
    const float x1 = x[tid];
    const float x2 = x[tid + half];
    x[tid] = x1 * cos_f - x2 * sin_f;
    x[tid + half] = x2 * cos_f + x1 * sin_f;
}


extern "C" __global__ void edge0_embed_row(
    const unsigned int* __restrict__ packed,  // [vocab, in_dim/8]
    const float* __restrict__ scales,         // [vocab, in_dim/64]
    const float* __restrict__ biases,
    const int* __restrict__ token,            // device: argmax output
    float* __restrict__ out,                  // [in_dim]
    int in_dim)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int words = in_dim >> 3;
    if (w >= words) return;
    const long long t = *token;
    const unsigned int word = packed[t * words + w];
    const int group = w >> 3;
    const float s = scales[t * (in_dim >> 6) + group];
    const float b = biases[t * (in_dim >> 6) + group];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        out[w * 8 + j] = s * (float)((word >> (4 * j)) & 0xFu) + b;
}

// n <= 8192. Deterministic fixed-order block reduction.
extern "C" __global__ void edge0_rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    float eps)
{
    __shared__ float partials[256];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        acc += v * v;
    }
    partials[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * inv * w[i];
}

// Final norm: plain RMSNorm (this checkpoint's norm weights are unshifted;
// zero-centered is the qwen3_5 dense variant below).
extern "C" __global__ void edge0_final_norm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    float eps)
{
    __shared__ float partials[256];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        acc += v * v;
    }
    partials[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * inv * w[i];
}

// Zero-centered variants (qwen3_5 dense): out = rms(x) * (1 + w).
extern "C" __global__ void edge0_rmsnorm_zc(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    float eps)
{
    __shared__ float partials[256];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        acc += v * v;
    }
    partials[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * inv * (1.0f + w[i]);
}

extern "C" __global__ void edge0_add_rmsnorm_zc(
    float* __restrict__ acc,
    const float* __restrict__ delta,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    float eps)
{
    __shared__ float partials[256];
    const int tid = threadIdx.x;
    float sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float s = acc[i] + delta[i];
        acc[i] = s;
        sq += s * s;
    }
    partials[tid] = sq;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = acc[i] * inv * (1.0f + w[i]);
}

extern "C" __global__ void edge0_add_inplace(
    float* __restrict__ acc,      // n
    const float* __restrict__ delta,  // n
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += delta[i];
}

// One block per q head (heads blocks) handles q: deinterleave, rmsnorm,
// partial rope; one block per kv head (next kv_heads blocks) handles k:
// rmsnorm, rope, append to the KV slot; the last kv_heads blocks copy v.
__device__ void attn_qk_impl(
    int zc,
    const float* __restrict__ q_raw,       // [heads * 2 * head_dim]
    const float* __restrict__ q_norm_w,    // [head_dim]
    const float* __restrict__ k_raw,       // [kv_heads * head_dim]
    const float* __restrict__ k_norm_w,    // [head_dim]
    const float* __restrict__ v_raw,       // [kv_heads * head_dim]
    float* __restrict__ q_out,             // [heads * head_dim]
    float* __restrict__ gate_out,          // [heads * head_dim]
    float* __restrict__ kv_keys,           // [max_ctx, kv_stride]
    float* __restrict__ kv_values,
    const int* __restrict__ position,         // device counter
    int kv_stride,
    int heads, int kv_heads, int head_dim, int rotary_dim,
    double theta)
{
    const int pos = *position;
    const int hd = head_dim;
    const int half = rotary_dim / 2;
    const int tid = threadIdx.x;

    if (blockIdx.x < heads) {
        const int h = blockIdx.x;
        const float* src = q_raw + (long long)h * 2 * hd;
        const float* gate_src = src + hd;
        float* dst = q_out + (long long)h * hd;
        float* gdst = gate_out + (long long)h * hd;
        for (int i = tid; i < hd; i += blockDim.x) {
            dst[i] = src[i];
            gdst[i] = gate_src[i];
        }
        __syncthreads();
        rmsnorm_head(dst, q_norm_w, tid, hd, zc);
        __syncthreads();
        rope_head(dst, pos, half, rotary_dim, theta, tid);
    } else if (blockIdx.x < heads + kv_heads) {
        const int kh = blockIdx.x - heads;
        float* dst = kv_keys + (long long)pos * kv_stride + (long long)kh * hd;
        const float* src = k_raw + (long long)kh * hd;
        for (int i = tid; i < hd; i += blockDim.x) dst[i] = src[i];
        __syncthreads();
        rmsnorm_head(dst, k_norm_w, tid, hd, zc);
        __syncthreads();
        rope_head(dst, pos, half, rotary_dim, theta, tid);
    } else {
        const int kh = blockIdx.x - heads - kv_heads;
        float* dst = kv_values + (long long)pos * kv_stride + (long long)kh * hd;
        const float* src = v_raw + (long long)kh * hd;
        for (int i = tid; i < hd; i += blockDim.x) dst[i] = src[i];
    }
}
extern "C" __global__ void edge0_attn_qk(
    const float* __restrict__ q_raw, const float* __restrict__ q_norm_w,
    const float* __restrict__ k_raw, const float* __restrict__ k_norm_w,
    const float* __restrict__ v_raw, float* __restrict__ q_out,
    float* __restrict__ gate_out, float* __restrict__ kv_keys,
    float* __restrict__ kv_values, const int* __restrict__ position,
    int kv_stride, int heads, int kv_heads, int head_dim, int rotary_dim,
    double theta)
{
    attn_qk_impl(0, q_raw, q_norm_w, k_raw, k_norm_w, v_raw, q_out, gate_out,
                 kv_keys, kv_values, position, kv_stride, heads, kv_heads,
                 head_dim, rotary_dim, theta);
}

// qwen3_5 dense: q/k norms are the zero-centered variant.
extern "C" __global__ void edge0_attn_qk_zc(
    const float* __restrict__ q_raw, const float* __restrict__ q_norm_w,
    const float* __restrict__ k_raw, const float* __restrict__ k_norm_w,
    const float* __restrict__ v_raw, float* __restrict__ q_out,
    float* __restrict__ gate_out, float* __restrict__ kv_keys,
    float* __restrict__ kv_values, const int* __restrict__ position,
    int kv_stride, int heads, int kv_heads, int head_dim, int rotary_dim,
    double theta)
{
    attn_qk_impl(1, q_raw, q_norm_w, k_raw, k_norm_w, v_raw, q_out, gate_out,
                 kv_keys, kv_values, position, kv_stride, heads, kv_heads,
                 head_dim, rotary_dim, theta);
}



// mrope variant: rope positions come from a 3-int device buffer (t,h,w)
// with the interleaved section axis map (i%3, section caps). KV slot and
// length still ride `position`. Bit-identical to attn_qk_impl when
// rope_pos == [pos, pos, pos] (same f64 math order).
__device__ __forceinline__ void rope_mrope_head(
    float* __restrict__ x, const int* __restrict__ rope_pos, int half,
    int rotary_dim, double theta, int tid, int sec_h, int sec_w)
{
    if (tid >= half) return;
    const int axis = (tid % 3 == 1 && tid < 3 * sec_h) ? 1
                   : (tid % 3 == 2 && tid < 3 * sec_w) ? 2 : 0;
    const double freq = pow(theta, -(2.0 * tid) / rotary_dim);
    const double angle = rope_pos[axis] * freq;
    const float sin_f = (float)sin(angle);
    const float cos_f = (float)cos(angle);
    const float x1 = x[tid];
    const float x2 = x[tid + half];
    x[tid] = x1 * cos_f - x2 * sin_f;
    x[tid + half] = x2 * cos_f + x1 * sin_f;
}

__device__ void attn_qk_mrope_impl(
    const float* __restrict__ q_raw, const float* __restrict__ q_norm_w,
    const float* __restrict__ k_raw, const float* __restrict__ k_norm_w,
    const float* __restrict__ v_raw, float* __restrict__ q_out,
    float* __restrict__ gate_out, float* __restrict__ kv_keys,
    float* __restrict__ kv_values, const int* __restrict__ position,
    const int* __restrict__ rope_pos, int kv_stride, int heads, int kv_heads,
    int head_dim, int rotary_dim, double theta, int sec_h, int sec_w)
{
    const int pos = *position;
    const int hd = head_dim;
    const int half = rotary_dim / 2;
    const int tid = threadIdx.x;

    if (blockIdx.x < heads) {
        const int h = blockIdx.x;
        const float* src = q_raw + (long long)h * 2 * hd;
        const float* gate_src = src + hd;
        float* dst = q_out + (long long)h * hd;
        float* gdst = gate_out + (long long)h * hd;
        for (int i = tid; i < hd; i += blockDim.x) {
            dst[i] = src[i];
            gdst[i] = gate_src[i];
        }
        __syncthreads();
        rmsnorm_head(dst, q_norm_w, tid, hd, 1);
        __syncthreads();
        rope_mrope_head(dst, rope_pos, half, rotary_dim, theta, tid, sec_h, sec_w);
    } else if (blockIdx.x < heads + kv_heads) {
        const int kh = blockIdx.x - heads;
        float* dst = kv_keys + (long long)pos * kv_stride + (long long)kh * hd;
        const float* src = k_raw + (long long)kh * hd;
        for (int i = tid; i < hd; i += blockDim.x) dst[i] = src[i];
        __syncthreads();
        rmsnorm_head(dst, k_norm_w, tid, hd, 1);
        __syncthreads();
        rope_mrope_head(dst, rope_pos, half, rotary_dim, theta, tid, sec_h, sec_w);
    } else {
        const int kh = blockIdx.x - heads - kv_heads;
        float* dst = kv_values + (long long)pos * kv_stride + (long long)kh * hd;
        const float* src = v_raw + (long long)kh * hd;
        for (int i = tid; i < hd; i += blockDim.x) dst[i] = src[i];
    }
}

// qwen3_5 dense multimodal: zero-centered q/k norms + 3-axis mrope.
extern "C" __global__ void edge0_attn_qk_zc_mrope(
    const float* __restrict__ q_raw, const float* __restrict__ q_norm_w,
    const float* __restrict__ k_raw, const float* __restrict__ k_norm_w,
    const float* __restrict__ v_raw, float* __restrict__ q_out,
    float* __restrict__ gate_out, float* __restrict__ kv_keys,
    float* __restrict__ kv_values, const int* __restrict__ position,
    const int* __restrict__ rope_pos, int kv_stride, int heads, int kv_heads,
    int head_dim, int rotary_dim, double theta, int sec_h, int sec_w)
{
    attn_qk_mrope_impl(q_raw, q_norm_w, k_raw, k_norm_w, v_raw, q_out,
                       gate_out, kv_keys, kv_values, position, rope_pos,
                       kv_stride, heads, kv_heads, head_dim, rotary_dim,
                       theta, sec_h, sec_w);
}

// Increment a 3-int rope position counter (decode continuation after an
// mrope prefill — text positions advance all three axes together).
extern "C" __global__ void edge0_inc3(int* __restrict__ c)
{
    if (threadIdx.x == 0) { c[0]++; c[1]++; c[2]++; }
}

// One block per q head; thread d owns output dimension d. len <= 8192
// (the shared scores cap; the host asserts max_ctx).
extern "C" __global__ void edge0_attn_scores(
    const float* __restrict__ q,            // [heads * head_dim]
    const float* __restrict__ gate,         // [heads * head_dim]
    const float* __restrict__ kv_keys,      // [max_ctx, kv_stride]
    const float* __restrict__ kv_values,
    float* __restrict__ out,                // [heads * head_dim]
    const int* __restrict__ position,       // device counter; len = pos + 1
    int kv_stride,
    int heads, int kv_heads, int head_dim,
    float scale)
{
    const int len = *position + 1;
    __shared__ float q_sh[256];
    __shared__ float scores[8192];
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int hd = head_dim;
    const int kv_head = h / (heads / kv_heads);
    if (d < hd) q_sh[d] = q[(long long)h * hd + d];
    __syncthreads();

    for (int s = d; s < len; s += blockDim.x) {
        const float* k = kv_keys + (long long)s * kv_stride + (long long)kv_head * hd;
        float dot = 0.0f;
        for (int j = 0; j < hd; j++) dot += q_sh[j] * k[j];
        scores[s] = dot * scale;
    }
    __syncthreads();

    if (d == 0) {
        float max_v = -INFINITY;
        for (int s = 0; s < len; s++) max_v = fmaxf(max_v, scores[s]);
        float sum = 0.0f;
        for (int s = 0; s < len; s++) {
            const float e = __expf(scores[s] - max_v);
            scores[s] = e;
            sum += e;
        }
        for (int s = 0; s < len; s++) scores[s] /= sum;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int s = 0; s < len; s++)
        acc += scores[s] * kv_values[(long long)s * kv_stride + (long long)kv_head * hd + d];
    const float gv = gate[(long long)h * hd + d];
    out[(long long)h * hd + d] = acc * (1.0f / (1.0f + __expf(-gv)));
}

// Router top-k over n logits (n <= 1024): k rounds of parallel argmax over
// a shared copy, the round winner knocked to -INF. Each round's reduction
// is a fixed-order (value, index) tree that keeps the SMALLER index on
// ties — identical to the serial first-occurrence scan — and the softmax
// weights follow the host's max-subtraction form. The old one-CTA serial
// scan cost 43.7us (latency-bound); this runs ~4us.
extern "C" __global__ void edge0_router_topk(
    const float* __restrict__ logits,   // [n]
    int* __restrict__ ids,              // [k]
    float* __restrict__ w,              // [k]
    int n, int k)
{
    __shared__ float sh[1024];
    __shared__ float rv[32];
    __shared__ int ri[32];
    const int tid = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) sh[i] = logits[i];
    __syncthreads();

    float maxv[4];
    for (int s = 0; s < k && s < 4; s++) {
        float bv = -INFINITY;
        int bi = -1;
        for (int i = tid; i < n; i += blockDim.x) {
            const float v = sh[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const int warps = blockDim.x >> 5;
        if (lane == 0) {
            rv[warp] = bv;
            ri[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            bv = lane < warps ? rv[lane] : -INFINITY;
            bi = lane < warps ? ri[lane] : -1;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                ids[s] = bi;
                maxv[s] = bv;
                sh[bi] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float m = maxv[0];
        float sum = 0.0f;
        for (int s = 0; s < k; s++) {
            w[s] = __expf(maxv[s] - m);
            sum += w[s];
        }
        for (int s = 0; s < k; s++) w[s] /= sum;
    }
}

// hidden[i] += sum_slot down_y[slot*rows + i] + sigmoid(s)*shared[i] — the
// router weights are already folded into down_y (slotx_silu). Slot order
// ascending then shared, matching the host combine order.
extern "C" __global__ void edge0_moe_combine(
    float* __restrict__ hidden,
    const float* __restrict__ down_y,   // [slots, rows]
    const float* __restrict__ shared_y, // [rows]
    const float* __restrict__ gate_logit,  // [1]
    int rows, int slots)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float acc = 0.0f;
    for (int s = 0; s < slots; s++) acc += down_y[(long long)s * rows + i];
    const float g = *gate_logit;
    acc += (1.0f / (1.0f + __expf(-g))) * shared_y[i];
    hidden[i] += acc;
}

// Argmax over n logits. One block; thread-local (value, index) scans over
// ascending strided ranges, then a fixed-shape warp+shared tree reduce.
// Ties keep the GREATER index — identical to the host max_by's last-wins —
// and the tree is fixed, so the result is deterministic run-to-run.
extern "C" __global__ void edge0_argmax(
    const float* __restrict__ logits,   // [n]
    int* __restrict__ out,
    int n)
{
    __shared__ float s_v[32];
    __shared__ int s_i[32];
    const int tid = threadIdx.x;
    float bv = -INFINITY;
    int bi = -1;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v >= bv) {
            bv = v;
            bi = i;
        }
    }
    // Warp reduce: keep greater value, or greater index on ties.
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi > bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    if (lane == 0) {
        s_v[warp] = bv;
        s_i[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        bv = (lane < warps) ? s_v[lane] : -INFINITY;
        bi = (lane < warps) ? s_i[lane] : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi > bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) *out = bi;
    }
}

extern "C" __global__ void edge0_inc(int* counter)
{
    if (threadIdx.x == 0) (*counter)++;
}

// Two-pass argmax for large n (one 1024-thread block scans 248K logits
// latency-bound at ~58 us). Comparator is (value, then greater index) =
// host max_by's last-max-wins; it is associative, so slice order is free.
extern "C" __global__ void edge0_argmax_part(
    const float* __restrict__ logits,
    float* __restrict__ part_v,
    int* __restrict__ part_i,
    int n)
{
    const int tid = threadIdx.x;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per;
    const int hi = min(n, lo + per);
    float bv = -INFINITY;
    int bi = -1;
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        const float v = logits[i];
        if (v >= bv) { bv = v; bi = i; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
    }
    __shared__ float s_v[32];
    __shared__ int s_i[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) { s_v[warp] = bv; s_i[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        bv = (lane < 8) ? s_v[lane] : -INFINITY;
        bi = (lane < 8) ? s_i[lane] : -1;
        for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { part_v[blockIdx.x] = bv; part_i[blockIdx.x] = bi; }
    }
}

extern "C" __global__ void edge0_argmax_final(
    const float* __restrict__ part_v,
    const int* __restrict__ part_i,
    int* __restrict__ out,
    int nparts)
{
    const int tid = threadIdx.x;
    float bv = (tid < nparts) ? part_v[tid] : -INFINITY;
    int bi = (tid < nparts) ? part_i[tid] : -1;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
    }
    __shared__ float s_v[4];
    __shared__ int s_i[4];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) { s_v[warp] = bv; s_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float wv = -INFINITY;
        int wi = -1;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (s_v[k] > wv || (s_v[k] == wv && s_i[k] > wi)) { wv = s_v[k]; wi = s_i[k]; }
        }
        *out = wi;
    }
}

// Grouped int4 GEMV: up to four projections over the SAME x and in_dim in
// one launch, LoRA folded into the epilogue (replaces one GEMV + one
// lora_add launch per member). Segments are 16-row aligned; padding rows
// return early. Deterministic: fixed-order group partials per row, thread
// 0 sums in group order and applies the rank-k correction in ascending k,
// matching the host's lora_delta.
extern "C" __global__ void edge0_gemv_group4_lora(
    const unsigned int* __restrict__ packed0,
    const float* __restrict__ scales0,
    const float* __restrict__ biases0,
    const float* __restrict__ la0,     // [rank, in] or null
    const float* __restrict__ lb0,     // [rows0, rank] or null
    float* __restrict__ y0,
    int rows0,
    const unsigned int* __restrict__ packed1,
    const float* __restrict__ scales1,
    const float* __restrict__ biases1,
    const float* __restrict__ la1,
    const float* __restrict__ lb1,
    float* __restrict__ y1,
    int rows1,
    const unsigned int* __restrict__ packed2,
    const float* __restrict__ scales2,
    const float* __restrict__ biases2,
    const float* __restrict__ la2,
    const float* __restrict__ lb2,
    float* __restrict__ y2,
    int rows2,
    const unsigned int* __restrict__ packed3,
    const float* __restrict__ scales3,
    const float* __restrict__ biases3,
    const float* __restrict__ la3,
    const float* __restrict__ lb3,
    float* __restrict__ y3,
    int rows3,
    const float* __restrict__ x,
    int in_dim,
    int rank,
    int l0, int l1, int l2, int l3)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;

    const int pad0 = (rows0 + 15) & ~15;
    const int pad1 = (rows1 + 15) & ~15;
    const int pad2 = (rows2 + 15) & ~15;

    const unsigned int* packed;
    const float* scales;
    const float* biases;
    const float* la;
    const float* lb;
    float* y;
    int rows;
    int row_base;
    int seg_has_lora;
    {
        int g = blockIdx.x * 16;
        int seg = 0;
        if (g >= pad0) { g -= pad0; seg = 1; }
        if (seg == 1 && g >= pad1) { g -= pad1; seg = 2; }
        if (seg == 2 && g >= pad2) { g -= pad2; seg = 3; }
        row_base = g;
        int has_lora = l0;
        switch (seg) {
        case 0: packed = packed0; scales = scales0; biases = biases0; la = la0; lb = lb0; y = y0; rows = rows0; has_lora = l0; break;
        case 1: packed = packed1; scales = scales1; biases = biases1; la = la1; lb = lb1; y = y1; rows = rows1; has_lora = l1; break;
        case 2: packed = packed2; scales = scales2; biases = biases2; la = la2; lb = lb2; y = y2; rows = rows2; has_lora = l2; break;
        default: packed = packed3; scales = scales3; biases = biases3; la = la3; lb = lb3; y = y3; rows = rows3; has_lora = l3; break;
        }
        seg_has_lora = has_lora;
        if (row_base >= rows) return;
    }

    // Single-chunk shapes (in_dim <= 4096) keep the pre-chunking fast path
    // (no rt round trip); wider shapes chunk at 4096 columns below.
    if (in_dim <= 4096) {
        const int words_per_row = in_dim / 8;
        const int wpt = (words_per_row + blockDim.x - 1) / blockDim.x;
        float xr[2][8];
        // Compile-time trip counts everywhere an array is indexed: a
        // runtime-bounded loop over xr/words forces local-memory spills.
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            if (k >= wpt) break;
            const int w = tid + k * blockDim.x;
            if (w < words_per_row) {
                #pragma unroll
                for (int j = 0; j < 8; j++) xr[k][j] = x[w * 8 + j];
            }
        }

        __shared__ float ax[16];
        if (rank > 0 && seg_has_lora != 0) {
            for (int k = warp; k < rank; k += warps) {
                const float* ar = la + (long long)k * in_dim;
                float acc = 0.0f;
                for (int c = lane; c < in_dim; c += 32) acc += ar[c] * x[c];
                for (int off = 16; off > 0; off >>= 1)
                    acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) ax[k] = acc;
            }
            __syncthreads();
        }

        const int groups = in_dim >> 6;
        const int nrows = min(16, rows - row_base);
        if (wpt <= 2) {
            // Preload all rows' words before any reduction: independent loads
            // overlap DRAM latency instead of serializing behind per-row
            // barriers. Reduction/finalize order per row is unchanged.
            unsigned int words[16][2];
            #pragma unroll
            for (int r = 0; r < 16; r++) {
                if (r >= nrows) break;
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    const int w = tid + k * blockDim.x;
                    words[r][k] = (k < wpt && w < words_per_row)
                        ? packed[(long long)(row_base + r) * words_per_row + w]
                        : 0u;
                }
            }
            __shared__ float partials[16][64];
            #pragma unroll
            for (int r = 0; r < 16; r++) {
                if (r >= nrows) break;
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    if (k >= wpt) break;
                    const int w = tid + k * blockDim.x;
                    float dot = 0.0f;
                    float sumx = 0.0f;
                    if (w < words_per_row) {
                        const unsigned int word = words[r][k];
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
                        const int group = w / 8;
                        const int gi = (row_base + r) * groups + group;
                        partials[r][group] = scales[gi] * dot + biases[gi] * sumx;
                    }
                }
            }
            __syncthreads();
            if (tid < nrows) {
                const int rrow = row_base + tid;
                float total = 0.0f;
                for (int g = 0; g < groups; g++) total += partials[tid][g];
                if (rank > 0 && seg_has_lora != 0) {
                    const float* br = lb + (long long)rrow * rank;
                    for (int k = 0; k < rank; k++) total += br[k] * ax[k];
                }
                y[rrow] = total;
            }
            return;
        }

        __shared__ float partials[64];
        for (int r = 0; r < nrows; r++) {
            const int rrow = row_base + r;
            for (int k = 0; k < wpt; k++) {
                const int w = tid + k * blockDim.x;
                float dot = 0.0f;
                float sumx = 0.0f;
                if (w < words_per_row) {
                    const unsigned int word =
                        packed[(long long)rrow * words_per_row + w];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        dot += (float)((word >> (4 * j)) & 0xFu) * xr[k][j];
                        sumx += xr[k][j];
                    }
                }
                // Per-word segment reduce: group = w/8 stays 8-aligned per k.
                for (int off = 4; off > 0; off >>= 1) {
                    dot += __shfl_down_sync(0xffffffffu, dot, off, 8);
                    sumx += __shfl_down_sync(0xffffffffu, sumx, off, 8);
                }
                if (w < words_per_row && (tid & 7) == 0) {
                    const int group = w / 8;
                    const int gi = rrow * (in_dim >> 6) + group;
                    partials[group] = scales[gi] * dot + biases[gi] * sumx;
                }
            }
            __syncthreads();
            if (tid == 0) {
                const int groups = in_dim >> 6;
                float total = 0.0f;
                for (int g = 0; g < groups; g++) total += partials[g];
                if (rank > 0 && seg_has_lora != 0) {
                    const float* br = lb + (long long)rrow * rank;
                    for (int k = 0; k < rank; k++) total += br[k] * ax[k];
                }
                y[rrow] = total;
            }
            __syncthreads();
        }
        return;
    }

    const int words_per_row = in_dim / 8;
    const int nrows = min(16, rows - row_base);
    const int groups = in_dim >> 6;
    // in_dim chunked at 4096 columns (wpt <= 2 at 256 threads, 64-group
    // partials at any width). Per-lane column order is chunk-ascending,
    // identical to a flat sweep — single-chunk shapes stay bit-identical.
    float axr[2];
    int axk[2];
    int axn = 0;
    if (rank > 0 && seg_has_lora != 0) {
        for (int k = warp; k < rank; k += warps) {
            axk[axn] = k;
            axr[axn] = 0.0f;
            axn++;
        }
    }
    __shared__ float rt[16];
    if (tid < nrows) rt[tid] = 0.0f;
    __syncthreads();

    for (int c0 = 0; c0 < in_dim; c0 += 4096) {
        const int cin = min(4096, in_dim - c0);
        const int cwords = cin / 8;
        const int cwpt = (cwords + blockDim.x - 1) / blockDim.x;
        float xr[2][8];
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            if (k >= cwpt) break;
            const int w = tid + k * blockDim.x;
            if (w < cwords) {
                #pragma unroll
                for (int j = 0; j < 8; j++) xr[k][j] = x[c0 + w * 8 + j];
            }
        }
        for (int i = 0; i < axn; i++) {
            const float* ar = la + (long long)axk[i] * in_dim + c0;
            float acc = 0.0f;
            for (int c = lane; c < cin; c += 32) acc += ar[c] * x[c0 + c];
            axr[i] += acc;
        }
        unsigned int words[16][2];
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            if (r >= nrows) break;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                const int w = tid + k * blockDim.x;
                words[r][k] = (k < cwpt && w < cwords)
                    ? packed[(long long)(row_base + r) * words_per_row
                             + c0 / 8 + w]
                    : 0u;
            }
        }
        __shared__ float partials[16][64];
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            if (r >= nrows) break;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                if (k >= cwpt) break;
                const int w = tid + k * blockDim.x;
                float dot = 0.0f;
                float sumx = 0.0f;
                if (w < cwords) {
                    const unsigned int word = words[r][k];
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
                if (w < cwords && (tid & 7) == 0) {
                    const int group = w / 8;
                    const int gi = (row_base + r) * groups + c0 / 64 + group;
                    partials[r][group] = scales[gi] * dot + biases[gi] * sumx;
                }
            }
        }
        __syncthreads();
        if (tid < nrows) {
            float acc = rt[tid];
            const int cg = cin >> 6;
            for (int g = 0; g < cg; g++) acc += partials[tid][g];
            rt[tid] = acc;
        }
        __syncthreads();
    }

    __shared__ float ax[16];
    if (axn > 0) {
        for (int i = 0; i < axn; i++) {
            float acc = axr[i];
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) ax[axk[i]] = acc;
        }
        __syncthreads();
    }
    if (tid < nrows) {
        const int rrow = row_base + tid;
        float total = rt[tid];
        if (rank > 0 && seg_has_lora != 0) {
            const float* br = lb + (long long)rrow * rank;
            for (int k = 0; k < rank; k++) total += br[k] * ax[k];
        }
        y[rrow] = total;
    }
}

// Fused residual add + rmsnorm: hidden += delta, x1 = rmsnorm(hidden)·w —
// one launch replacing the add/rmsnorm pair; same fixed-order reduction.
extern "C" __global__ void edge0_add_rmsnorm(
    float* __restrict__ acc,
    const float* __restrict__ delta,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    float eps)
{
    __shared__ float partials[256];
    const int tid = threadIdx.x;
    float sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float s = acc[i] + delta[i];
        acc[i] = s;
        sq += s * s;
    }
    partials[tid] = sq;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = acc[i] * inv * w[i];
}

// Single int4 projection over a silu'd input with the LoRA folded in
// (group-of-one with silu prologue) — the shared-expert down path.
extern "C" __global__ void edge0_gemv1_silu_lora(
    const unsigned int* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    const float* __restrict__ la,
    const float* __restrict__ lb,
    float* __restrict__ y,
    int rows,
    const float* __restrict__ g,           // [in_dim]
    const float* __restrict__ u,           // [in_dim]
    int in_dim,
    int rank)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int words_per_row = in_dim / 8;

    float xr[8];
    if (tid < words_per_row) {
        const int c = tid * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float gv = g[c + j];
            xr[j] = (gv / (1.0f + expf(-gv))) * u[c + j];
        }
    }

    __shared__ float ax[16];
    if (rank > 0) {
        for (int k = warp; k < rank; k += warps) {
            const float* ar = la + (long long)k * in_dim;
            float acc = 0.0f;
            // The LoRA acts on the down projection's true input: silu(g)*u.
            for (int c = lane; c < in_dim; c += 32) {
                const float gv = g[c];
                acc += ar[c] * ((gv / (1.0f + expf(-gv))) * u[c]);
            }
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) ax[k] = acc;
        }
        __syncthreads();
    }

    const int nrows = min(8, rows - (int)blockIdx.x * 8);
    unsigned int words[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        const long long base = ((long long)blockIdx.x * 8 + r) * words_per_row;
        words[r] = (r < nrows && tid < words_per_row) ? packed[base + tid] : 0u;
    }
    __shared__ float partials[8][64];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r >= nrows) break;
        const int row = blockIdx.x * 8 + r;
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
            const int gi = row * (in_dim >> 6) + group;
            partials[r][group] = scales[gi] * dot + biases[gi] * sumx;
        }
    }
    __syncthreads();
    if (tid < nrows) {
        const int row = blockIdx.x * 8 + tid;
        const int groups = in_dim >> 6;
        float total = 0.0f;
        for (int g2 = 0; g2 < groups; g2++) total += partials[tid][g2];
        if (rank > 0) {
            const float* br = lb + (long long)row * rank;
            for (int k = 0; k < rank; k++) total += br[k] * ax[k];
        }
        y[row] = total;
    }
}
