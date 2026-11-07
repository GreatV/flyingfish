#include "gpu_compat.cuh"
#include <stdint.h>
#include <math.h>

// Gates are [input,forget,candidate,output], each hidden_size wide.
extern "C" __global__ void lstm_step_f32(
    const float* input_gates, const float* recurrent_gates,
    const float* bias_ih, const float* bias_hh, float* cell, float* hidden,
    uint32_t hidden_size) {
    if (blockDim.x != 256 || !input_gates || !recurrent_gates || !bias_ih || !bias_hh ||
        !cell || !hidden || !hidden_size) { __trap(); return; }
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < hidden_size;
         i += gridDim.x * 256) {
        float pre[4];
        #pragma unroll
        for (uint32_t gate = 0; gate < 4; ++gate) {
            const uint32_t index = gate * hidden_size + i;
            float value = __fadd_rn(input_gates[index], recurrent_gates[index]);
            value = __fadd_rn(value, bias_ih[index]);
            pre[gate] = __fadd_rn(value, bias_hh[index]);
        }
        const float input = 1.0f / (1.0f + expf(-pre[0]));
        const float forget = 1.0f / (1.0f + expf(-pre[1]));
        const float candidate = tanhf(pre[2]);
        const float output = 1.0f / (1.0f + expf(-pre[3]));
        const float next_cell = __fadd_rn(__fmul_rn(forget, cell[i]), __fmul_rn(input, candidate));
        hidden[i] = __fmul_rn(output, tanhf(next_cell));
        cell[i] = next_cell;
    }
}

// Native weights[out,in/groups,3,3], NHWC input/output, stride 2 and padding 1.
extern "C" __global__ void speech_conv2d_f32(
    const float* input, const float* weight, const float* bias, float* output,
    uint32_t time, uint32_t frequency, uint32_t in_channels, uint32_t out_channels,
    uint32_t groups, uint32_t valid_time) {
    if (blockDim.x != 256 || !input || !weight || !bias || !output || !time || !frequency ||
        !in_channels || !out_channels || !groups || in_channels % groups || out_channels % groups) {
        __trap(); return;
    }
    const uint32_t out_time = (time - 1) / 2 + 1, out_frequency = (frequency - 1) / 2 + 1;
    if (valid_time > out_time) { __trap(); return; }
    const uint32_t group_width = in_channels / groups, group_count = out_channels / groups;
    const uint64_t count = uint64_t(out_time) * out_frequency * out_channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t ot = uint32_t(i / (uint64_t(out_frequency) * out_channels));
        const uint32_t of = uint32_t((i / out_channels) % out_frequency), channel = uint32_t(i % out_channels);
        if (ot >= valid_time) { output[i] = 0.0f; continue; }
        const uint32_t first = channel / group_count * group_width;
        float sum = 0.0f;
        for (uint32_t ic = 0; ic < group_width; ++ic) {
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    const uint32_t it = ot * 2 + ky, iff = of * 2 + kx;
                    if (it >= 1 && it <= time && iff >= 1 && iff <= frequency) {
                        const uint64_t source = (uint64_t(it - 1) * frequency + iff - 1) * in_channels + first + ic;
                        const uint64_t tap = ((uint64_t(channel) * group_width + ic) * 3 + ky) * 3 + kx;
                        sum = __fmaf_rn(input[source], weight[tap], sum);
                    }
                }
            }
        }
        output[i] = __fadd_rn(sum, bias[channel]);
    }
}

extern "C" __global__ void speech_mask_time_f32(
    float* values, uint32_t spatial, uint32_t time, uint32_t frequency,
    uint32_t channels, uint32_t valid_time) {
    if (blockDim.x != 256 || !values || !spatial || !time || !frequency || !channels ||
        spatial != time * frequency || valid_time > time) { __trap(); return; }
    const uint64_t count = uint64_t(spatial) * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t row = uint32_t(i / (uint64_t(frequency) * channels));
        if (row >= valid_time) values[i] = 0.0f;
    }
}

// Input [time,frequency,channel] to output [time,channel,frequency].
extern "C" __global__ void speech_pack_subsample_f32(
    const float* input, float* output, uint32_t time, uint32_t frequency, uint32_t channels) {
    if (blockDim.x != 256 || !input || !output || !time || !frequency || !channels) {
        __trap(); return;
    }
    const uint64_t count = uint64_t(time) * frequency * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t t = uint32_t(i / (uint64_t(channels) * frequency));
        const uint32_t channel = uint32_t((i / frequency) % channels), f = uint32_t(i % frequency);
        output[i] = input[(uint64_t(t) * frequency + f) * channels + channel];
    }
}

// Native weights[channel,1,kernel], symmetric time padding, [time,channel] input/output.
extern "C" __global__ void speech_depthwise_conv_f32(
    const float* input, const float* weight, const uint8_t* mask, float* output,
    uint32_t rows, uint32_t channels, uint32_t kernel) {
    if (blockDim.x != 256 || !input || !weight || !mask || !output || !rows || !channels ||
        !kernel || !(kernel & 1)) { __trap(); return; }
    const uint64_t count = uint64_t(rows) * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t row = uint32_t(i / channels), channel = uint32_t(i % channels);
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel; ++k) {
            const int t = int(row) + int(k) - int(kernel / 2);
            if (t >= 0 && t < int(rows) && mask[t])
                sum = __fmaf_rn(input[uint64_t(t) * channels + channel], weight[channel * kernel + k], sum);
        }
        output[i] = sum;
    }
}

extern "C" __global__ void speech_batch_norm_f32(
    const float* input, const float* weight, const float* bias,
    const float* mean, const float* variance, float* output,
    uint32_t rows, uint32_t channels, float eps) {
    if (blockDim.x != 256 || !input || !weight || !bias || !mean || !variance || !output ||
        !rows || !channels || !(eps > 0.0f)) { __trap(); return; }
    const uint64_t count = uint64_t(rows) * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t channel = uint32_t(i % channels);
        const float normalized = __fdiv_rn(__fsub_rn(input[i], mean[channel]), sqrtf(variance[channel] + eps));
        output[i] = __fmaf_rn(normalized, weight[channel], bias[channel]);
    }
}

extern "C" __global__ void speech_residual_f32(
    const float* residual, const float* delta, float* output, uint64_t count, float scale) {
    if (blockDim.x != 256 || !residual || !delta || !output || !count || !isfinite(scale)) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256)
        output[i] = __fadd_rn(residual[i], __fmul_rn(scale, delta[i]));
}

// Full bidirectional Transformer-XL attention with interleaved relative positions and head_dim=128.
extern "C" __global__ void speech_relative_attention_f32(
    const float* q, const float* k, const float* v, const float* relative,
    const float* bias_u, const float* bias_v, const uint8_t* mask, float* output,
    uint32_t rows, uint32_t heads, float scaling) {
    extern __shared__ float shared[];
    float* reduction = shared;
    float* probabilities = shared + 8;
    if (blockDim.x != 128 || gridDim.x != rows || gridDim.y != heads || gridDim.z != 1 ||
        !q || !k || !v || !relative || !bias_u || !bias_v || !mask || !output ||
        !rows || !heads || !(scaling > 0.0f)) { __trap(); return; }
    const uint32_t query = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const uint32_t hidden = heads * 128;
    const uint8_t query_valid = mask[query];
    for (uint32_t key = 0; key < rows; ++key) {
        float content = 0.0f, position = 0.0f;
        if (query_valid && mask[key]) {
            const uint32_t relative_row = rows - 1 - query + key;
            const float qu = __fadd_rn(q[uint64_t(query) * hidden + head * 128 + d], bias_u[head * 128 + d]);
            const float qv = __fadd_rn(q[uint64_t(query) * hidden + head * 128 + d], bias_v[head * 128 + d]);
            content = __fmul_rn(qu, k[uint64_t(key) * hidden + head * 128 + d]);
            position = __fmul_rn(qv, relative[uint64_t(relative_row) * hidden + head * 128 + d]);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            content += __shfl_down_sync(0xffffffffu, content, offset);
            position += __shfl_down_sync(0xffffffffu, position, offset);
        }
        if ((d & 31) == 0) {
            reduction[d / 32] = content;
            reduction[4 + d / 32] = position;
        }
        __syncthreads();
        if (d == 0) {
            const float ac = (reduction[0] + reduction[1]) + (reduction[2] + reduction[3]);
            const float bd = (reduction[4] + reduction[5]) + (reduction[6] + reduction[7]);
            probabilities[key] = !query_valid ? 0.0f : mask[key] ? __fmul_rn(ac + bd, scaling) : -INFINITY;
        }
        __syncthreads();
    }
    if (d == 0) {
        float maximum = -INFINITY, sum = 0.0f;
        for (uint32_t key = 0; key < rows; ++key) maximum = fmaxf(maximum, probabilities[key]);
        for (uint32_t key = 0; key < rows; ++key) {
            probabilities[key] = expf(probabilities[key] - maximum);
            sum += probabilities[key];
        }
        for (uint32_t key = 0; key < rows; ++key) probabilities[key] /= sum;
    }
    __syncthreads();
    float result = 0.0f;
    for (uint32_t key = 0; key < rows; ++key)
        result = __fmaf_rn(probabilities[key], v[uint64_t(key) * hidden + head * 128 + d], result);
    output[uint64_t(query) * hidden + head * 128 + d] = result;
}
