#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
struct alignas(16) Q38SequenceMeta {
    int32_t prefix, valid_rows, key_count, capacity, causal, reserved[3];
};
static_assert(sizeof(Q38SequenceMeta)==32,"Q38 ABI sequence size");
__device__ __forceinline__ void q38_meta_guard(const Q38SequenceMeta* s) {
    if (!s || (reinterpret_cast<uintptr_t>(s)&15)) { __trap(); return; }
    if (s->prefix<0 || s->valid_rows<=0 || s->capacity<=0 || s->causal!=1 ||
        int64_t(s->prefix)+s->valid_rows!=s->key_count || s->key_count>s->capacity ||
        s->reserved[0] || s->reserved[1] || s->reserved[2]) __trap();
}
