// Batch-2 decode kernels for MTP speculative verify: two token positions
// share ONE weight read. x0/x1 are the two positions' inputs; y0/y1 the
// outputs. Reduction shapes are the single-column kernel's, per column —
// per-column results bit-match the batch-1 path.
//
// GDN two-step: token B's recurrence starts from post-A state; the
// post-A state goes to state_out_a (scratch swap target on reject), the
// post-B state to state_live (accepted state on hit).

#define RPB 16

extern "C" __global__ void qwen_gemv4_batch2(
    const unsigned int* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    const float* __restrict__ x0,
    const float* __restrict__ x1,
    float* __restrict__ y0,
    float* __restrict__ y1,
    int out_dim,
    int in_dim)
{
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * RPB;
    const int nrows = min(RPB, out_dim - row0);
    const int groups = in_dim >> 6;
    __shared__ float rt0[RPB];
    __shared__ float rt1[RPB];
    if (tid < nrows) { rt0[tid] = 0.0f; rt1[tid] = 0.0f; }
    __syncthreads();
    for (int c0 = 0; c0 < in_dim; c0 += 4096) {
        const int cin = min(4096, in_dim - c0);
        const int cwords = cin / 8;
        const int cwpt = (cwords + blockDim.x - 1) / blockDim.x;
        float xr0[2][8];
        float xr1[2][8];
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            if (k >= cwpt) break;
            const int w = tid + k * blockDim.x;
            if (w < cwords) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    xr0[k][j] = x0[c0 + w * 8 + j];
                    xr1[k][j] = x1[c0 + w * 8 + j];
                }
            }
        }
        unsigned int words[RPB][2];
        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            if (r >= nrows) break;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                const int w = tid + k * blockDim.x;
                words[r][k] = (k < cwpt && w < cwords)
                    ? packed[(long long)(row0 + r) * (in_dim / 8) + c0 / 8 + w]
                    : 0u;
            }
        }
        __shared__ float partials0[RPB][64];
        __shared__ float partials1[RPB][64];
        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            if (r >= nrows) break;
            const int row = row0 + r;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                if (k >= cwpt) break;
                const int w = tid + k * blockDim.x;
                float dot0 = 0.0f, sum0 = 0.0f, dot1 = 0.0f, sum1 = 0.0f;
                if (w < cwords) {
                    const unsigned int word = words[r][k];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float q = (float)((word >> (4 * j)) & 0xFu);
                        dot0 += q * xr0[k][j];
                        sum0 += xr0[k][j];
                        dot1 += q * xr1[k][j];
                        sum1 += xr1[k][j];
                    }
                }
                for (int off = 4; off > 0; off >>= 1) {
                    dot0 += __shfl_down_sync(0xffffffffu, dot0, off, 8);
                    sum0 += __shfl_down_sync(0xffffffffu, sum0, off, 8);
                    dot1 += __shfl_down_sync(0xffffffffu, dot1, off, 8);
                    sum1 += __shfl_down_sync(0xffffffffu, sum1, off, 8);
                }
                if (w < cwords && (tid & 7) == 0) {
                    const int gi = row * groups + c0 / 64 + w / 8;
                    partials0[r][w / 8] = scales[gi] * dot0 + biases[gi] * sum0;
                    partials1[r][w / 8] = scales[gi] * dot1 + biases[gi] * sum1;
                }
            }
        }
        __syncthreads();
        if (tid < nrows) {
            float a0 = rt0[tid], a1 = rt1[tid];
            const int cg = cin >> 6;
            for (int g = 0; g < cg; g++) {
                a0 += partials0[tid][g];
                a1 += partials1[tid][g];
            }
            rt0[tid] = a0;
            rt1[tid] = a1;
        }
        __syncthreads();
    }
    if (tid < nrows) {
        y0[row0 + tid] = rt0[tid];
        y1[row0 + tid] = rt1[tid];
    }
}

// Batch-2 grouped GEMV (no LoRA in this checkpoint): four segments, two
// input columns, one weight pass. Same chunking as edge0's group4.
extern "C" __global__ void qwen_group4_batch2(
    const unsigned int* __restrict__ packed0, const float* __restrict__ scales0, const float* __restrict__ biases0, float* __restrict__ y00, float* __restrict__ y01, int rows0,
    const unsigned int* __restrict__ packed1, const float* __restrict__ scales1, const float* __restrict__ biases1, float* __restrict__ y10, float* __restrict__ y11, int rows1,
    const unsigned int* __restrict__ packed2, const float* __restrict__ scales2, const float* __restrict__ biases2, float* __restrict__ y20, float* __restrict__ y21, int rows2,
    const unsigned int* __restrict__ packed3, const float* __restrict__ scales3, const float* __restrict__ biases3, float* __restrict__ y30, float* __restrict__ y31, int rows3,
    const float* __restrict__ xa,
    const float* __restrict__ xb,
    int in_dim)
{
    const int tid = threadIdx.x;
    const int pad0 = (rows0 + 15) & ~15;
    const int pad1 = (rows1 + 15) & ~15;
    const int pad2 = (rows2 + 15) & ~15;

    const unsigned int* packed;
    const float* scales;
    const float* biases;
    float* y0;
    float* y1;
    int rows;
    int row_base;
    {
        int g = blockIdx.x * 16;
        int seg = 0;
        if (g >= pad0) { g -= pad0; seg = 1; }
        if (seg == 1 && g >= pad1) { g -= pad1; seg = 2; }
        if (seg == 2 && g >= pad2) { g -= pad2; seg = 3; }
        row_base = g;
        switch (seg) {
        case 0: packed = packed0; scales = scales0; biases = biases0; y0 = y00; y1 = y01; rows = rows0; break;
        case 1: packed = packed1; scales = scales1; biases = biases1; y0 = y10; y1 = y11; rows = rows1; break;
        case 2: packed = packed2; scales = scales2; biases = biases2; y0 = y20; y1 = y21; rows = rows2; break;
        default: packed = packed3; scales = scales3; biases = biases3; y0 = y30; y1 = y31; rows = rows3; break;
        }
        if (row_base >= rows) return;
    }

    const int words_per_row = in_dim / 8;
    const int nrows = min(16, rows - row_base);
    const int groups = in_dim >> 6;
    __shared__ float rt0[16];
    __shared__ float rt1[16];
    if (tid < nrows) { rt0[tid] = 0.0f; rt1[tid] = 0.0f; }
    __syncthreads();
    for (int c0 = 0; c0 < in_dim; c0 += 4096) {
        const int cin = min(4096, in_dim - c0);
        const int cwords = cin / 8;
        const int cwpt = (cwords + blockDim.x - 1) / blockDim.x;
        float xr0[2][8];
        float xr1[2][8];
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            if (k >= cwpt) break;
            const int w = tid + k * blockDim.x;
            if (w < cwords) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    xr0[k][j] = xa[c0 + w * 8 + j];
                    xr1[k][j] = xb[c0 + w * 8 + j];
                }
            }
        }
        unsigned int words[16][2];
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            if (r >= nrows) break;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                const int w = tid + k * blockDim.x;
                words[r][k] = (k < cwpt && w < cwords)
                    ? packed[(long long)(row_base + r) * words_per_row + c0 / 8 + w]
                    : 0u;
            }
        }
        __shared__ float p0[16][64];
        __shared__ float p1[16][64];
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            if (r >= nrows) break;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                if (k >= cwpt) break;
                const int w = tid + k * blockDim.x;
                float d0 = 0.0f, s0 = 0.0f, d1 = 0.0f, s1 = 0.0f;
                if (w < cwords) {
                    const unsigned int word = words[r][k];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float q = (float)((word >> (4 * j)) & 0xFu);
                        d0 += q * xr0[k][j];
                        s0 += xr0[k][j];
                        d1 += q * xr1[k][j];
                        s1 += xr1[k][j];
                    }
                }
                for (int off = 4; off > 0; off >>= 1) {
                    d0 += __shfl_down_sync(0xffffffffu, d0, off, 8);
                    s0 += __shfl_down_sync(0xffffffffu, s0, off, 8);
                    d1 += __shfl_down_sync(0xffffffffu, d1, off, 8);
                    s1 += __shfl_down_sync(0xffffffffu, s1, off, 8);
                }
                if (w < cwords && (tid & 7) == 0) {
                    const int gi = (row_base + r) * groups + c0 / 64 + w / 8;
                    p0[r][w / 8] = scales[gi] * d0 + biases[gi] * s0;
                    p1[r][w / 8] = scales[gi] * d1 + biases[gi] * s1;
                }
            }
        }
        __syncthreads();
        if (tid < nrows) {
            float a0 = rt0[tid], a1 = rt1[tid];
            const int cg = cin >> 6;
            for (int g = 0; g < cg; g++) {
                a0 += p0[tid][g];
                a1 += p1[tid][g];
            }
            rt0[tid] = a0;
            rt1[tid] = a1;
        }
        __syncthreads();
    }
    if (tid < nrows) {
        y0[row_base + tid] = rt0[tid];
        y1[row_base + tid] = rt1[tid];
    }
}

// Two-step causal conv k<=8. Step A: window [state rows, qkvA]. Step B:
// window [post-A rows, qkvB]. state <- post-B (live), state_a <- post-A
// (scratch for reject swap).
extern "C" __global__ void qwen_gdn_conv2(
    const float* __restrict__ w,
    float* __restrict__ state,
    float* __restrict__ state_a,
    const float* __restrict__ qkvA,
    const float* __restrict__ qkvB,
    float* __restrict__ outA,
    float* __restrict__ outB,
    int conv_dim,
    int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int k1 = kernel - 1;
    float win[8];
    for (int j = 0; j < k1; j++) win[j] = state[j * conv_dim + c];
    // Step A.
    float acc = 0.0f;
    for (int j = 0; j < k1; j++) acc += w[c * kernel + j] * win[j];
    acc += w[c * kernel + k1] * qkvA[c];
    outA[c] = acc / (1.0f + expf(-acc));
    // Post-A: shift left, append qkvA.
    for (int j = 0; j < k1 - 1; j++) win[j] = win[j + 1];
    win[k1 - 1] = qkvA[c];
    for (int j = 0; j < k1; j++) state_a[j * conv_dim + c] = win[j];
    // Step B over the post-A window.
    acc = 0.0f;
    for (int j = 0; j < k1; j++) acc += w[c * kernel + j] * win[j];
    acc += w[c * kernel + k1] * qkvB[c];
    outB[c] = acc / (1.0f + expf(-acc));
    // Post-B (live state).
    for (int j = 0; j < k1 - 1; j++) win[j] = win[j + 1];
    win[k1 - 1] = qkvB[c];
    for (int j = 0; j < k1; j++) state[j * conv_dim + c] = win[j];
}

// Two sequential GDN recurrence steps (speculative verify pair) per value
// head. State flow: reads state_in, writes post-A to state_a (scratch)
// and post-B to state_out (new live). Mirrors edge0_gdn_heads' float4
// sweeps and per-head norm/gate per step.
__device__ __forceinline__ float gdn_sig(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

extern "C" __global__ void qwen_gdn_heads2(
    const float* __restrict__ convA,  // [2*key_dim + value_dim]
    const float* __restrict__ convB,
    const float* __restrict__ zA,
    const float* __restrict__ zB,
    const float* __restrict__ bA,
    const float* __restrict__ bB,
    const float* __restrict__ aA,
    const float* __restrict__ aB,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w,
    const float* __restrict__ state_in,
    float* __restrict__ state_a,
    float* __restrict__ state_out,
    float* __restrict__ outA,
    float* __restrict__ outB,
    int num_v, int num_k, int dk, int dv,
    float scale, float eps)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int c = tid / 8;
    const int g = tid % 8;
    const int r0 = g * (dk / 8);
    const int k_head = h / (num_v / num_k);
    const int key_dim = num_k * dk;

    __shared__ float q_n[256], k_n[256];
    __shared__ float out_sh[256];
    __shared__ float kv_sh[256];
    __shared__ float kv_p[128][8];
    __shared__ float o_p[128][8];
    __shared__ float sh_decay, sh_beta, sh_inv;

    const float* S_in = state_in + (long long)h * dv * dk + (long long)c * dk;
    float* S_a = state_a + (long long)h * dv * dk + (long long)c * dk;
    float* S_out = state_out + (long long)h * dv * dk + (long long)c * dk;

    for (int step = 0; step < 2; step++) {
        const float* conv = step == 0 ? convA : convB;
        const float* z = step == 0 ? zA : zB;
        const float* bv = step == 0 ? bA : bB;
        const float* av = step == 0 ? aA : aB;
        float* out = step == 0 ? outA : outB;
        const float* Scur = step == 0 ? S_in : S_a;
        float* Snew = step == 0 ? S_a : S_out;

        if (threadIdx.x == 0) {
            float qs = 0.0f, ks = 0.0f;
            for (int r = 0; r < dk; r++) {
                const float qv = conv[k_head * dk + r];
                const float kv2 = conv[key_dim + k_head * dk + r];
                qs += qv * qv;
                ks += kv2 * kv2;
                q_n[r] = qv;
                k_n[r] = kv2;
            }
            const float qn = sqrtf(qs + 1e-6f);
            const float kn = sqrtf(ks + 1e-6f);
            for (int r = 0; r < dk; r++) {
                q_n[r] /= qn;
                k_n[r] /= kn;
            }
            const float ap = av[h] + dt_bias[h];
            const float sp = (ap > 20.0f) ? ap : logf(1.0f + expf(ap));
            sh_decay = expf(-expf(a_log[h]) * sp);
            sh_beta = 1.0f / (1.0f + expf(-bv[h]));
        }
        __syncthreads();
        const float decay = sh_decay;
        const float beta = sh_beta;

        float kv4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const int r = r0 + v * 4;
            float4 s = *reinterpret_cast<const float4*>(Scur + r);
            s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
            *reinterpret_cast<float4*>(Snew + r) = s;
            kv4[v] = k_n[r] * s.x;
            kv4[v] += k_n[r + 1] * s.y;
            kv4[v] += k_n[r + 2] * s.z;
            kv4[v] += k_n[r + 3] * s.w;
        }
        kv_p[c][g] = ((kv4[0] + kv4[1]) + kv4[2]) + kv4[3];
        __syncthreads();

        if (g == 0) {
            float kv_mem = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) kv_mem += kv_p[c][i];
            kv_sh[c] = (conv[2 * key_dim + h * dv + c] - kv_mem) * beta;
        }
        __syncthreads();

        const float kvs = kv_sh[c];
        float o4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const int r = r0 + v * 4;
            float4 s = *reinterpret_cast<float4*>(Snew + r);
            float4 kq = *reinterpret_cast<const float4*>(k_n + r);
            float4 qq = *reinterpret_cast<const float4*>(q_n + r);
            s.x += kq.x * kvs; s.y += kq.y * kvs; s.z += kq.z * kvs; s.w += kq.w * kvs;
            *reinterpret_cast<float4*>(Snew + r) = s;
            o4[v] = qq.x * s.x;
            o4[v] += qq.y * s.y;
            o4[v] += qq.z * s.z;
            o4[v] += qq.w * s.w;
        }
        o_p[c][g] = ((o4[0] + o4[1]) + o4[2]) + o4[3];
        __syncthreads();

        if (g == 0) {
            float o = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) o += o_p[c][i];
            out_sh[c] = o * scale;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float mean_sq = 0.0f;
            for (int i = 0; i < dv; i++) mean_sq += out_sh[i] * out_sh[i];
            sh_inv = 1.0f / sqrtf(mean_sq / (float)dv + eps);
        }
        __syncthreads();
        if (c < dv) {
            const float zv = z[h * dv + c];
            out[h * dv + c] = out_sh[c] * sh_inv * norm_w[c] *
                              (zv / (1.0f + expf(-zv)));
        }
        __syncthreads();
    }
}

// out = [a; b] — the MTP fusion buffer assembly.
extern "C" __global__ void qwen_concat(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i];
        out[n + i] = b[i];
    }
}
