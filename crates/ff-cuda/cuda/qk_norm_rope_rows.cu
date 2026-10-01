#include <math.h>

extern "C" __global__ void qk_norm_rope_rows(
    const float* __restrict__ q_raw, const float* __restrict__ q_norm_w,
    const float* __restrict__ k_raw, const float* __restrict__ k_norm_w,
    const float* __restrict__ v_raw, float* __restrict__ q_out,
    float* __restrict__ gate_out, float* __restrict__ kv_keys,
    float* __restrict__ kv_values, const int* __restrict__ position,
    const int* __restrict__ rope_pos, int kv_stride, int heads, int kv_heads,
    int head_dim, int rotary_dim, double theta, int sec_h, int sec_w)
{
    const long long t = blockIdx.y;
    const int pos = position[t];
    const int hd = head_dim;
    const int half = rotary_dim / 2;
    const int tid = threadIdx.x;
    const int h = blockIdx.x;
    float* dst;
    const float* weight;
    if (h < heads) {
        const float* src = q_raw + (t * heads + h) * 2 * hd;
        const float* gate_src = src + hd;
        dst = q_out + (t * heads + h) * hd;
        float* gdst = gate_out + (t * heads + h) * hd;
        for (int i = tid; i < hd; i += blockDim.x) {
            dst[i] = src[i];
            gdst[i] = gate_src[i];
        }
        weight = q_norm_w;
    } else if (h < heads + kv_heads) {
        const int kh = h - heads;
        dst = kv_keys + (long long)pos * kv_stride + (long long)kh * hd;
        const float* src = k_raw + (t * kv_heads + kh) * hd;
        for (int i = tid; i < hd; i += blockDim.x) dst[i] = src[i];
        weight = k_norm_w;
    } else {
        const int kh = h - heads - kv_heads;
        dst = kv_values + (long long)pos * kv_stride + (long long)kh * hd;
        const float* src = v_raw + (t * kv_heads + kh) * hd;
        for (int i = tid; i < hd; i += blockDim.x) dst[i] = src[i];
        return;
    }
    __syncthreads();
    __shared__ float partials[256];
    float acc = 0.0f;
    if (tid < hd) {
        const float v = dst[tid];
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
    if (tid < hd) dst[tid] = dst[tid] * inv * (1.0f + weight[tid]);
    __syncthreads();
    if (tid >= half) return;
    const int axis = (tid % 3 == 1 && tid < 3 * sec_h) ? 1
                   : (tid % 3 == 2 && tid < 3 * sec_w) ? 2 : 0;
    const double freq = pow(theta, -(2.0 * tid) / rotary_dim);
    const double angle = rope_pos[3 * t + axis] * freq;
    const float sin_f = (float)sin(angle);
    const float cos_f = (float)cos(angle);
    const float x1 = dst[tid];
    const float x2 = dst[tid + half];
    dst[tid] = x1 * cos_f - x2 * sin_f;
    dst[tid + half] = x2 * cos_f + x1 * sin_f;
}
