#include "gpu_compat.cuh"
#include "q38_contract.cuh"

template<typename T>
__device__ __forceinline__ void rotary_pair(T a, T b, T c, T s, T* lo, T* hi) {
    if constexpr (sizeof(T) == sizeof(float)) {
        *lo = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        *hi = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
    } else {
#if defined(__HIP__)
        *lo = __hsub_rn(__hmul_rn(a, c), __hmul_rn(b, s));
        *hi = __hadd_rn(__hmul_rn(b, c), __hmul_rn(a, s));
#else
        *lo = __hsub(__hmul(a, c), __hmul(b, s));
        *hi = __hadd(__hmul(b, c), __hmul(a, s));
#endif
    }
}

template<typename T>
__device__ __forceinline__ void rotary_element(T* q,
    const T* key, const T* value, T* k, T* v, const T* table,
    int start, int rows, int q_heads, int kv_heads, int dim, int rotary,
    int capacity, int i) {
    const int heads = q_heads + kv_heads;
    if (i >= rows * heads * dim) return;
    const int d = i % dim;
    const int head = (i / dim) % heads;
    const int row = i / (heads * dim);
    const int position = start + row;
    if (position < 0 || position >= capacity) { __trap(); return; }
    const bool query = head < q_heads;
    const int h = query ? head : head - q_heads;
    const size_t src = ((size_t)row * (query ? q_heads : kv_heads) + h) * dim;
    const size_t dst = ((size_t)h * capacity + position) * dim;
    if (!query) v[dst + d] = value[src + d];
    if (d >= rotary) {
        if (!query) k[dst + d] = key[src + d];
        return;
    }
    const int half = rotary / 2;
    if (d >= half) return;
    const T* input = query ? q : key;
    const T a = input[src + d];
    const T b = input[src + d + half];
    const T c = table[(size_t)position * rotary + d];
    const T s = table[(size_t)position * rotary + d + half];
    T lo, hi;
    rotary_pair(a, b, c, s, &lo, &hi);
    T* output = query ? q + src : k + dst;
    output[d] = lo;
    output[d + half] = hi;
}

template<typename T>
__device__ __forceinline__ void rotary_prefix(T* q,
    const T* key, const T* value,
    T* k, T* v, const T* table,
    int start, int rows, int q_heads, int kv_heads, int dim, int rotary,
    int capacity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    rotary_element(q, key, value, k, v, table, start, rows, q_heads, kv_heads, dim, rotary, capacity, i);
}
