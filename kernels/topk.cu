#include "dsa_topk.cuh"
#include <cmath>

template<int Capacity>
__device__ void dsa_select_scores(const float* scores, const int* ids, int candidates,
                                  int keep, int* selected, float* values, float* margins) {
    if (threadIdx.x != 0) return;
    if (keep <= 0 || keep > Capacity || candidates <= 0) __trap();
    __shared__ DsaTopK<Capacity> heap;
    heap.size = 0;
    float rejected = -INFINITY;
    size_t row = blockIdx.x;
    for (int candidate = 0; candidate < candidates; ++candidate) {
        float score = scores[row * candidates + candidate];
        int id = ids[row * candidates + candidate];
        if (isnan(score) || score == INFINITY || id < -1) __trap();
        if (id >= 0 && score != -INFINITY) dsa_topk_offer(&heap, keep, score, id, &rejected);
    }
    int count = heap.size;
    for (int slot = count - 1; slot >= 0; --slot) {
        selected[row * keep + slot] = heap.id[0];
        values[row * keep + slot] = heap.score[0];
        dsa_topk_pop(&heap);
    }
    for (int slot = count; slot < keep; ++slot) {
        selected[row * keep + slot] = -1;
        values[row * keep + slot] = -INFINITY;
    }
    margins[row] = count > 0 && isfinite(rejected) ? values[row * keep + count - 1] - rejected : INFINITY;
}

extern "C" __global__ __launch_bounds__(32, 1)
void dsa_bounded_topk512(const float* scores, const int* ids, int candidates,
                         int keep, int* selected, float* values, float* margins) {
    dsa_select_scores<512>(scores, ids, candidates, keep, selected, values, margins);
}

extern "C" __global__ __launch_bounds__(32, 1)
void dsa_bounded_topk2048(const float* scores, const int* ids, int candidates,
                          int keep, int* selected, float* values, float* margins) {
    dsa_select_scores<2048>(scores, ids, candidates, keep, selected, values, margins);
}
