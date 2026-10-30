#include "q38_contract.cuh"

template<int Block, bool PowerOfTwo>
__device__ __forceinline__ void pack_rows(
    const __nv_bfloat16* input, uint8_t* output, uint8_t* scales,
    const Q38SequenceMeta* sequence, int rows, int width, int group_rows) {
    q38_meta_guard(sequence);
    const int valid = sequence->key_count / group_rows - sequence->prefix / group_rows;
    if (valid > rows) __trap();
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(valid) * width) return;
    uint8_t code, scale;
    fp4_act_quant<Block, PowerOfTwo>(__bfloat162float(input[index]), &code, &scale);
    const int adjacent = __shfl_down_sync(FF_WARP_MASK, int(code), 1);
    if ((index & 1) == 0) output[index / 2] = code | (adjacent << 4);
    if (index % Block == 0) scales[index / Block] = scale;
}

extern "C" __global__ void fp4_pack_e4m3_16(
    const __nv_bfloat16* input, uint8_t* output, uint8_t* scales,
    const Q38SequenceMeta* sequence, int rows, int width, int group_rows) {
    pack_rows<16, false>(input, output, scales, sequence, rows, width, group_rows);
}

extern "C" __global__ void fp4_pack_e8m0_32(
    const __nv_bfloat16* input, uint8_t* output, uint8_t* scales,
    const Q38SequenceMeta* sequence, int rows, int width, int group_rows) {
    pack_rows<32, true>(input, output, scales, sequence, rows, width, group_rows);
}

// Records contain width E4M3 bytes followed by width/32 E8M0 bytes.
__device__ __forceinline__ float lookup_value(const uint8_t* records, int64_t index, int width) {
    const int64_t base = (index / width) * (width + width / 32);
    const int column = index % width;
    const float value = bround(__fmul_rn(e4m3_decode(records[base + column]),
        e8m0_decode(records[base + width + column / 32])));
    if (!isfinite(value)) __trap();
    return value;
}

extern "C" __global__ void fp8_lookup_prepare(
    const uint8_t* records, uint8_t* codes, uint8_t* scales,
    const Q38SequenceMeta* sequence, int rows, int lookups, int width) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows) __trap();
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(sequence->valid_rows) * lookups * width) return;
    const float value = lookup_value(records, index, width);
    uint8_t code, scale;
    act_quant32(value, &code, &scale);
    codes[index] = code;
    if ((index & 31) == 0) scales[index / 32] = scale;
}

// Acceptance gate exposes the same register decode before activation preparation.
extern "C" __global__ void fp8_lookup_decode(
    const uint8_t* records, __nv_bfloat16* output, const Q38SequenceMeta* sequence,
    int rows, int lookups, int width) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows) __trap();
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < int64_t(sequence->valid_rows) * lookups * width)
        output[index] = __float2bfloat16_rn(lookup_value(records, index, width));
}
