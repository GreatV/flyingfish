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

// Native weights[out,in/groups,3,3], NHWC input/output, stride 2.
// Output length is (n + left_pad + right_pad - 3) / 2 + 1. Source index is output * 2 - left_pad + tap.
// lanes batches speaker lanes over the same [time, frequency] extent: input/output
// hold `lanes` planes; weights and bias are shared.
extern "C" __global__ void speech_conv2d_f32(
    const float* input, const float* weight, const float* bias, float* output,
    uint32_t time, uint32_t frequency, uint32_t in_channels, uint32_t out_channels,
    uint32_t groups, uint32_t valid_time, uint32_t left_pad, uint32_t right_pad, uint32_t lanes) {
    if (blockDim.x != 256 || !input || !weight || !bias || !output || !time || !frequency ||
        !in_channels || !out_channels || !groups || in_channels % groups || out_channels % groups ||
        !lanes || lanes > 4) {
        __trap(); return;
    }
    const int time_span = int(time) + int(left_pad) + int(right_pad) - 3;
    const int frequency_span = int(frequency) + int(left_pad) + int(right_pad) - 3;
    if (time_span < 0 || frequency_span < 0) { __trap(); return; }
    const uint32_t out_time = uint32_t(time_span) / 2 + 1;
    const uint32_t out_frequency = uint32_t(frequency_span) / 2 + 1;
    if (valid_time > out_time) { __trap(); return; }
    const uint32_t group_width = in_channels / groups, group_count = out_channels / groups;
    const uint64_t plane = uint64_t(out_time) * out_frequency * out_channels;
    const uint64_t count = plane * lanes;
    const uint64_t in_plane = uint64_t(time) * frequency * in_channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t lane = i / plane;
        const uint64_t local = i - lane * plane;
        const uint32_t ot = uint32_t(local / (uint64_t(out_frequency) * out_channels));
        const uint32_t of = uint32_t((local / out_channels) % out_frequency), channel = uint32_t(local % out_channels);
        if (ot >= valid_time) { output[i] = 0.0f; continue; }
        const float* lane_input = input + lane * in_plane;
        const uint32_t first = channel / group_count * group_width;
        float sum = 0.0f;
        for (uint32_t ic = 0; ic < group_width; ++ic) {
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    const int it = int(ot) * 2 - int(left_pad) + int(ky);
                    const int iff = int(of) * 2 - int(left_pad) + int(kx);
                    if (it >= 0 && it < int(time) && iff >= 0 && iff < int(frequency)) {
                        const uint64_t source = (uint64_t(it) * frequency + uint32_t(iff)) * in_channels + first + ic;
                        const uint64_t tap = ((uint64_t(channel) * group_width + ic) * 3 + ky) * 3 + kx;
                        sum = __fmaf_rn(lane_input[source], weight[tap], sum);
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

// Native weights[channel,1,kernel], [time,channel] input/output. Tap index is row + k - left.
// lanes batches speaker lanes: input/output hold `lanes` planes of [rows, channels],
// weights are shared, the time-validity mask is shared.
extern "C" __global__ void speech_depthwise_conv_f32(
    const float* input, const float* weight, const uint8_t* mask, float* output,
    uint32_t rows, uint32_t channels, uint32_t kernel, uint32_t left, uint32_t lanes) {
    if (blockDim.x != 256 || !input || !weight || !mask || !output || !rows || !channels ||
        !kernel || !(kernel & 1) || !lanes || lanes > 4) { __trap(); return; }
    const uint64_t count = uint64_t(lanes) * rows * channels;
    const uint64_t plane = uint64_t(rows) * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t row = uint32_t((i % plane) / channels), channel = uint32_t(i % channels);
        const float* lane_input = input + (i / plane) * plane;
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel; ++k) {
            const int t = int(row) + int(k) - int(left);
            if (t >= 0 && t < int(rows) && mask[t])
                sum = __fmaf_rn(lane_input[uint64_t(t) * channels + channel], weight[channel * kernel + k], sum);
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

// blockDim.x is 128; head_dim must divide 128 and lanes >= head_dim stay idle.
// Relative row is rows-1-query+key into a table of length 2*key_rows-1.
// chunk_size 0 attends every valid key. A positive chunk_size keeps an absolute key when
// its chunk index is in [q_abs/chunk - left_chunks, q_abs/chunk + right_chunks].
// Keys at or beyond valid_keys are excluded from attention; their own query rows
// still compute over the valid keys.
// lanes batches speaker lanes: q/k/v/relative/output hold `lanes` planes of
// [rows|key_rows|2*key_rows-1, hidden] and the time-validity mask is shared;
// gridDim.z is the lane. Per-block arithmetic is independent of the lane count.
extern "C" __global__ void speech_relative_attention_f32(
    const float* q, const float* k, const float* v, const float* relative,
    const float* bias_u, const float* bias_v, const uint8_t* mask, float* output,
    uint32_t rows, uint32_t heads, uint32_t head_dim, float scaling, uint32_t chunk_size, uint32_t left_chunks,
    uint32_t right_chunks, uint32_t key_rows, int64_t query_offset, int64_t kv_offset, uint32_t valid_keys,
    uint32_t lanes, uint32_t left_slide) {
    extern __shared__ float shared[];
    float* reduction = shared;
    float* probabilities = shared + 8;
    if (blockDim.x != 128 || gridDim.x != rows || gridDim.y != heads || gridDim.z != lanes ||
        !q || !k || !v || !relative || !bias_u || !bias_v || !mask || !output ||
        !rows || !key_rows || !valid_keys || !heads || !head_dim || head_dim > 128 || (128 % head_dim) != 0 || !(scaling > 0.0f) || right_chunks > 1 ||
        !lanes || lanes > 4 ||
        key_rows < rows || valid_keys > key_rows || (chunk_size == 0 && right_chunks != 0) ||
        (chunk_size == 0 && (query_offset != 0 || kv_offset != 0 || key_rows != rows))) {
        __trap(); return;
    }
    const uint32_t query = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const uint64_t q_plane = uint64_t(rows) * heads * head_dim;
    const uint64_t k_plane = uint64_t(key_rows) * heads * head_dim;
    const uint64_t r_plane = (uint64_t(2 * key_rows) - 1) * heads * head_dim;
    q += uint64_t(blockIdx.z) * q_plane;
    k += uint64_t(blockIdx.z) * k_plane;
    v += uint64_t(blockIdx.z) * k_plane;
    relative += uint64_t(blockIdx.z) * r_plane;
    output += uint64_t(blockIdx.z) * q_plane;
    const uint32_t hidden = heads * head_dim;
    const bool active = d < head_dim;
    const uint32_t query_mask = key_rows - rows + query;
    const uint8_t query_valid = mask[query_mask];
    uint32_t key_begin = 0, key_end = valid_keys;
    if (chunk_size != 0) {
        // Signed window arithmetic: a cold-start layout carries zero-filled
        // slots whose absolute chunks are negative and still attendable.
        const int64_t q_abs = int64_t(query_offset) + int64_t(query);
        const int64_t query_chunk = q_abs / int64_t(chunk_size);
        // left_slide > 0 (multitalker): the window anchors at the query's
        // 14-row block within the step — [block - left_slide, block + chunk)
        // — verified against the reference's captured masks on five steps.
        // left_slide == 0: chunk-aligned left_chunks semantics. Both saturate
        // at the stream start: keys at negative absolute rows never attend.
        const int64_t block_start = query_offset + (int64_t(query) / int64_t(chunk_size)) * int64_t(chunk_size);
        int64_t begin_abs = left_slide > 0
            ? block_start - int64_t(left_slide)
            : (query_chunk - int64_t(left_chunks)) * int64_t(chunk_size);
        if (begin_abs < 0) begin_abs = 0;
        const int64_t end_abs = left_slide > 0
            ? block_start + int64_t(chunk_size) * (1 + int64_t(right_chunks))
            : (query_chunk + 1 + int64_t(right_chunks)) * int64_t(chunk_size);
        const int64_t local_begin = begin_abs - int64_t(kv_offset);
        const int64_t local_end = end_abs - int64_t(kv_offset);
        key_begin = local_begin > 0 ? uint32_t(local_begin > key_rows ? key_rows : local_begin) : 0;
        if (local_end > 0) {
            const uint32_t window_end = local_end > key_rows ? key_rows : uint32_t(local_end);
            key_end = window_end < key_end ? window_end : key_end;
        } else {
            key_end = 0;
        }
    }
    for (uint32_t key = 0; key < key_rows; ++key) {
        float content = 0.0f, position = 0.0f;
        const bool in_window = key >= key_begin && key < key_end;
        if (query_valid && mask[key] && in_window) {
            const uint32_t relative_row = rows - 1 - query + key;
            if (active) {
                const float qu = __fadd_rn(q[uint64_t(query) * hidden + head * head_dim + d], bias_u[head * head_dim + d]);
                const float qv = __fadd_rn(q[uint64_t(query) * hidden + head * head_dim + d], bias_v[head * head_dim + d]);
                content = __fmul_rn(qu, k[uint64_t(key) * hidden + head * head_dim + d]);
                position = __fmul_rn(qv, relative[uint64_t(relative_row) * hidden + head * head_dim + d]);
            }
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
            probabilities[key] = !query_valid ? 0.0f
                : (mask[key] && in_window) ? __fmul_rn(ac + bd, scaling) : -INFINITY;
        }
        __syncthreads();
    }
    if (d == 0) {
        float maximum = -INFINITY, sum = 0.0f;
        for (uint32_t key = 0; key < key_rows; ++key) maximum = fmaxf(maximum, probabilities[key]);
        for (uint32_t key = 0; key < key_rows; ++key) {
            probabilities[key] = expf(probabilities[key] - maximum);
            sum += probabilities[key];
        }
        for (uint32_t key = 0; key < key_rows; ++key) probabilities[key] /= sum;
    }
    __syncthreads();
    float result = 0.0f;
    if (active) {
        for (uint32_t key = 0; key < key_rows; ++key)
            result = __fmaf_rn(probabilities[key], v[uint64_t(key) * hidden + head * head_dim + d], result);
        output[uint64_t(query) * hidden + head * head_dim + d] = result;
    }
}

// Masked-asr feature mask: input/output [lanes, frequency, time] mel features.
// speaker_masks [lanes, mask_frames] diar frames binarized at 0.5 and repeated
// 8x over time; a mask wider than the window is trimmed from the front, a
// narrower mask left-pads zeros. Frames where the original feature is exactly
// zero take the log floor.
extern "C" __global__ void speech_feature_speaker_mask_f32(
    const float* input, const float* speaker_masks, float* output,
    uint32_t frequency, uint32_t time, uint32_t mask_frames, uint32_t lanes, float mask_value) {
    if (blockDim.x != 256 || !input || !speaker_masks || !output || !frequency || !time ||
        !mask_frames || !lanes || lanes > 4 || !isfinite(mask_value)) {
        __trap(); return;
    }
    const uint64_t count = uint64_t(lanes) * frequency * time;
    const uint32_t mask_mel = 8u * mask_frames;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t lane = i / (uint64_t(frequency) * time);
        const uint64_t local = i - lane * uint64_t(frequency) * time;
        const uint32_t t = uint32_t(local % time);
        const float mask = mask_mel >= time
            ? (speaker_masks[lane * mask_frames + t / 8] > 0.5f ? 1.0f : 0.0f)
            : (t < time - mask_mel || (t - (time - mask_mel)) / 8 >= mask_frames
                   ? 0.0f
                   : (speaker_masks[lane * mask_frames + (t - (time - mask_mel)) / 8] > 0.5f ? 1.0f : 0.0f));
        const float value = input[i] * mask;
        output[i] = input[i] == 0.0f ? mask_value : value;
    }
}

// Rowwise scale for the layer-0 speaker kernels: input/output [lanes*rows, width],
// scale [lanes*rows] one factor per row. Per-row arithmetic is lane-count invariant.
extern "C" __global__ void speech_row_scale_f32(
    const float* input, const float* scale, float* output, uint64_t count, uint32_t width) {
    if (blockDim.x != 256 || !input || !scale || !output || !count || !width) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t row = uint32_t(i / width);
        output[i] = __fmul_rn(input[i], scale[row]);
    }
}
