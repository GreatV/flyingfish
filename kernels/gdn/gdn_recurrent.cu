// GDN recurrent decode entry: M = 1, value_split = 2.
#include "gdn_recurrent.cuh"

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_recurrent_core_v2(const float* q, const float* k, const float* v,
                           const float* beta, const float* g, float* S,
                           __nv_bfloat16* out, const Q38SequenceFields* meta) {
    gdn_recurrent_t<2, true>(q, k, v, beta, g, S, out, meta, 1, nullptr);
}
