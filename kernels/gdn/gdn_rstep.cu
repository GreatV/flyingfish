// Ordered recurrent rows: h32 1..16, h48 1..5. Each (head,value split) CTA holds FP32
// state in registers. The live row count is read from the device meta; bound is the
// prepared allocation. No metadata mutation and no cross-CTA state dependency.
#include "gdn_recurrent.cuh"

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_rstep_core_v1(const float* q, const float* k, const float* v,
                       const float* beta, const float* g, float* S,
                       __nv_bfloat16* out, const Q38SequenceFields* meta, int bound) {
    gdn_recurrent_t<2, false>(q, k, v, beta, g, S, out, meta, bound, nullptr);
}

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_rstep_core_snapshot(const float* q, const float* k, const float* v,
                            const float* beta, const float* g, float* S,
                            __nv_bfloat16* out, const Q38SequenceFields* meta, int bound,
                            float* snapshots) {
    gdn_recurrent_t<2, false, true>(q, k, v, beta, g, S, out, meta, bound, snapshots);
}
