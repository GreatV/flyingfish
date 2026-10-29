#pragma once
#include <cuda_runtime.h>

template<int Capacity>
struct DsaTopK {
    float score[Capacity];
    int id[Capacity];
    int size;
};

__device__ __forceinline__ bool dsa_topk_better(float sa, int ia, float sb, int ib) {
    return sa > sb || (sa == sb && ia < ib);
}

__device__ __forceinline__ bool dsa_topk_worse(float sa, int ia, float sb, int ib) {
    return sa < sb || (sa == sb && ia > ib);
}

template<int Capacity>
__device__ __forceinline__ void dsa_topk_sift(DsaTopK<Capacity>* heap, int root) {
    float score = heap->score[root];
    int id = heap->id[root];
    int child = root * 2 + 1;
    while (child < heap->size) {
        if (child + 1 < heap->size &&
            dsa_topk_worse(heap->score[child + 1], heap->id[child + 1],
                           heap->score[child], heap->id[child])) {
            ++child;
        }
        if (!dsa_topk_worse(heap->score[child], heap->id[child], score, id)) break;
        heap->score[root] = heap->score[child];
        heap->id[root] = heap->id[child];
        root = child;
        child = root * 2 + 1;
    }
    heap->score[root] = score;
    heap->id[root] = id;
}

template<int Capacity>
__device__ __forceinline__ bool dsa_topk_offer(DsaTopK<Capacity>* heap, int k, float score,
                                               int id, float* rejected) {
    if (heap->size < k) {
        int at = heap->size++;
        while (at > 0) {
            int parent = (at - 1) / 2;
            if (!dsa_topk_worse(score, id, heap->score[parent], heap->id[parent])) break;
            heap->score[at] = heap->score[parent];
            heap->id[at] = heap->id[parent];
            at = parent;
        }
        heap->score[at] = score;
        heap->id[at] = id;
        return true;
    }
    if (!dsa_topk_better(score, id, heap->score[0], heap->id[0])) {
        *rejected = fmaxf(*rejected, score);
        return false;
    }
    *rejected = fmaxf(*rejected, heap->score[0]);
    heap->score[0] = score;
    heap->id[0] = id;
    dsa_topk_sift(heap, 0);
    return true;
}

template<int Capacity>
__device__ __forceinline__ void dsa_topk_pop(DsaTopK<Capacity>* heap) {
    --heap->size;
    if (heap->size == 0) return;
    heap->score[0] = heap->score[heap->size];
    heap->id[0] = heap->id[heap->size];
    dsa_topk_sift(heap, 0);
}
