#pragma once
#include <stdint.h>

template <bool CHANNEL_G, int HEADS, int DV>
__device__ __forceinline__ float gdn_gate_global(
        const float* __restrict__ g, uint32_t row, uint32_t head, uint32_t channel) {
    if constexpr (CHANNEL_G) {
        float value = g[((size_t)row * HEADS + head) * DV + channel];
        if ((row & 63) != 0) {
            value -= g[((size_t)(row - 1) * HEADS + head) * DV + channel];
        }
        return value;
    } else {
        return g[(size_t)row * HEADS + head];
    }
}

// Per-channel gates are cumulative over 64 rows. A 32-row core subtracts the first half's
// cumulative value in its second block; per-head gates arrive raw and use shared cumsum.
template <bool CHANNEL_G, int HEADS, int DV, int CHUNK = 64>
__device__ __forceinline__ float gdn_gate_at(
        const float* __restrict__ g, const float* __restrict__ sG,
        uint32_t row0, uint32_t r, uint32_t valid, uint32_t head, uint32_t channel) {
    if constexpr (CHANNEL_G) {
        if (row0 + r >= valid) return 0.0f;
        float value = g[((size_t)(row0 + r) * HEADS + head) * DV + channel];
        if constexpr (CHUNK == 32) {
            if ((row0 & 63) != 0) {
                value -= g[((size_t)(row0 - 1) * HEADS + head) * DV + channel];
            }
        }
        return value;
    } else {
        return sG[r];
    }
}
