// GDN conv1d + gated delta-net recurrence, device-resident. Reductions are
// fixed-order and deterministic run-to-run; the q/k l2norms and the output
// rmsnorm are warp-tree reductions (lane-contiguous partials, shuffle fold)
// — ULP-different from the host's serial order, acceptance-gate arbitrated.
// Per-column sweeps stay ascending; the decay sweep remains
// separate because the reference reads the DECAYED state before kv_mem.

extern "C" __global__ void edge0_gdn_conv(
    const float* __restrict__ qkv,   // [conv_dim]
    const float* __restrict__ w,     // [conv_dim * kernel]
    float* __restrict__ state,       // [(kernel-1) * conv_dim]
    float* __restrict__ out,         // [conv_dim]
    int conv_dim,
    int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float acc = 0.0f;
    for (int j = 0; j < kernel - 1; j++)
        acc += w[c * kernel + j] * state[j * conv_dim + c];
    acc += w[c * kernel + kernel - 1] * qkv[c];
    out[c] = acc / (1.0f + expf(-acc));
    for (int j = 0; j < kernel - 2; j++)
        state[j * conv_dim + c] = state[(j + 1) * conv_dim + c];
    state[(kernel - 2) * conv_dim + c] = qkv[c];
}

// One block (1024 threads) per value head: 8 threads per state column,
// each owning a 16-row slice (dk=128). Row-slice partials combine by
// strict left fold in shared memory — deterministic; the fold differs
// from the serial per-column order at rounding level (gate-arbitrated).
// The old one-thread-per-column form ran the sweeps latency-bound at
// ~37 GB/s (nsys: 53.7us/layer).
extern "C" __global__ void edge0_gdn_heads(
    const float* __restrict__ conv_out,  // [2*key_dim + value_dim]
    const float* __restrict__ z,         // [value_dim]
    const float* __restrict__ b,         // [num_v]
    const float* __restrict__ a,         // [num_v]
    const float* __restrict__ a_log,     // [num_v]
    const float* __restrict__ dt_bias,   // [num_v]
    const float* __restrict__ norm_w,    // [dv] — shared across heads
    float* __restrict__ state,           // [num_v * dv * dk]
    float* __restrict__ out,             // [value_dim]
    int num_v, int num_k, int dk, int dv,
    float scale, float eps)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int c = tid / 8;          // column 0..dv-1
    const int g = tid % 8;          // row-slice 0..7
    const int r0 = g * (dk / 8);
    const int r1 = r0 + dk / 8;
    const int k_head = h / (num_v / num_k);
    const int key_dim = num_k * dk;

    __shared__ float q_n[256], k_n[256];
    __shared__ float out_sh[256];
    __shared__ float kv_sh[256];
    __shared__ float kv_p[128][8];
    __shared__ float o_p[128][8];
    __shared__ float sh_decay, sh_beta, sh_inv;

    // Warp-parallel q/k norms: lane-contiguous 4-element sums + shuffle
    // tree (fixed order, deterministic; differs from the serial order at
    // rounding level — acceptance-gate arbitrated). The serial thread-0 form
    // idled 1023 threads per block.
    __shared__ float sh_qn, sh_kn;
    for (int r = threadIdx.x; r < dk; r += blockDim.x) {
        q_n[r] = conv_out[k_head * dk + r];
        k_n[r] = conv_out[key_dim + k_head * dk + r];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int per = dk / 32;
        float qs = 0.0f, ks = 0.0f;
        #pragma unroll
        for (int j = 0; j < per; j++) {
            const float qv = q_n[threadIdx.x * per + j];
            const float kv2 = k_n[threadIdx.x * per + j];
            qs += qv * qv;
            ks += kv2 * kv2;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            qs += __shfl_down_sync(0xffffffffu, qs, off);
            ks += __shfl_down_sync(0xffffffffu, ks, off);
        }
        if (threadIdx.x == 0) {
            sh_qn = sqrtf(qs + 1e-6f);
            sh_kn = sqrtf(ks + 1e-6f);
            const float ap = a[h] + dt_bias[h];
            const float sp = (ap > 20.0f) ? ap : logf(1.0f + expf(ap));
            sh_decay = expf(-expf(a_log[h]) * sp);
            sh_beta = 1.0f / (1.0f + expf(-b[h]));
        }
    }
    __syncthreads();
    {
        const float qn = sh_qn;
        const float kn = sh_kn;
        for (int r = threadIdx.x; r < dk; r += blockDim.x) {
            q_n[r] /= qn;
            k_n[r] /= kn;
        }
    }
    __syncthreads();
    const float decay = sh_decay;
    const float beta = sh_beta;

    float* S = state + (long long)h * dv * dk + (long long)c * dk;
    // float4 sweeps: 16 scalar loads per thread per pass sat on the LSU
    // queue; 4 vector loads issue the same bytes with 4x fewer
    // instructions. Lane accumulators fold in fixed order.
    float kv4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        const int r = r0 + v * 4;
        float4 s = *reinterpret_cast<float4*>(S + r);
        s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
        *reinterpret_cast<float4*>(S + r) = s;
        kv4[v] = k_n[r] * s.x;
        kv4[v] += k_n[r + 1] * s.y;
        kv4[v] += k_n[r + 2] * s.z;
        kv4[v] += k_n[r + 3] * s.w;
    }
    float kv_acc = ((kv4[0] + kv4[1]) + kv4[2]) + kv4[3];
    kv_p[c][g] = kv_acc;
    __syncthreads();

    if (g == 0) {
        float kv_mem = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) kv_mem += kv_p[c][i];
        kv_sh[c] = (conv_out[2 * key_dim + h * dv + c] - kv_mem) * beta;
    }
    __syncthreads();

    const float kvs = kv_sh[c];
    float o4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        const int r = r0 + v * 4;
        float4 s = *reinterpret_cast<const float4*>(S + r);
        float4 kq = *reinterpret_cast<const float4*>(k_n + r);
        float4 qq = *reinterpret_cast<const float4*>(q_n + r);
        s.x += kq.x * kvs; s.y += kq.y * kvs; s.z += kq.z * kvs; s.w += kq.w * kvs;
        *reinterpret_cast<float4*>(S + r) = s;
        o4[v] = qq.x * s.x;
        o4[v] += qq.y * s.y;
        o4[v] += qq.z * s.z;
        o4[v] += qq.w * s.w;
    }
    float o_acc = ((o4[0] + o4[1]) + o4[2]) + o4[3];
    o_p[c][g] = o_acc;
    __syncthreads();

    if (g == 0) {
        float o = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) o += o_p[c][i];
        out_sh[c] = o * scale;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        const int per = dv / 32;
        float ss = 0.0f;
        #pragma unroll
        for (int j = 0; j < per; j++) {
            const float v = out_sh[threadIdx.x * per + j];
            ss += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        if (threadIdx.x == 0)
            sh_inv = 1.0f / sqrtf(ss / (float)dv + eps);
    }
    __syncthreads();
    if (c < dv) {
        const float zv = z[h * dv + c];
        out[h * dv + c] = out_sh[c] * sh_inv * norm_w[c] *
                          (zv / (1.0f + expf(-zv)));
    }
}
