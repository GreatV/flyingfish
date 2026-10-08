#include <cmath>
#include <string>
#include <nvtx3/nvToolsExt.h>

#include "src/flash_fwd_launch_template.h"

using Traits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
static thread_local std::string error;

extern "C" void ff_profile_push(const char* name) { nvtxRangePushA(name); }
extern "C" void ff_profile_pop() { nvtxRangePop(); }

extern "C" int ff_fa2_tile() {
    return Traits::kBlockM;
}

extern "C" const char* ff_fa2_error() {
    return error.c_str();
}

extern "C" int ff_fa2_run(void* q, void* k, void* v, void* out, void* lse,
                           int rows, int start, int q_heads, int kv_heads,
                           int capacity, int causal, cudaStream_t stream) {
    try {
        if (rows <= 0 || start < 0 || start + rows > capacity ||
            kv_heads <= 0 || q_heads <= 0 || q_heads % kv_heads != 0) {
            throw std::runtime_error("invalid FA2 shape or cache range");
        }
        flash::Flash_fwd_params p{};
        p.q_ptr = q;
        p.k_ptr = k;
        p.v_ptr = v;
        p.o_ptr = out;
        p.softmax_lse_ptr = lse;
        p.q_row_stride = (q_heads + 2 * kv_heads) * 128;
        p.k_row_stride = p.v_row_stride = 128;
        p.q_head_stride = p.o_head_stride = 128;
        p.k_head_stride = p.v_head_stride = int64_t(capacity) * 128;
        p.q_batch_stride = int64_t(rows) * p.q_row_stride;
        p.k_batch_stride = p.v_batch_stride = int64_t(kv_heads) * capacity * 128;
        p.o_row_stride = q_heads * 128;
        p.o_batch_stride = int64_t(rows) * p.o_row_stride;
        p.h = q_heads;
        p.h_k = kv_heads;
        p.h_h_k_ratio = q_heads / kv_heads;
        p.b = 1;
        p.seqlen_q = rows;
        p.seqlen_k = start + rows;
        p.seqlen_q_rounded = (rows + 127) / 128 * 128;
        p.seqlen_k_rounded = (p.seqlen_k + 127) / 128 * 128;
        p.d = p.d_rounded = 128;
        p.total_q = rows;
        p.scale_softmax = 1.0f / std::sqrt(128.0f);
        p.scale_softmax_log2 = p.scale_softmax * std::log2(std::exp(1.0f));
        p.p_dropout = p.rp_dropout = 1.0f;
        p.p_dropout_in_uint8_t = 255;
        p.scale_softmax_rp_dropout = p.scale_softmax;
        p.window_size_left = -1;
        p.window_size_right = 0;
        p.is_bf16 = p.is_seqlens_k_cumulative = true;
        p.is_causal = causal != 0;
        p.num_splits = 1;
        if (causal) flash::run_flash_fwd<Traits, false, true>(p, stream);
        else flash::run_flash_fwd<Traits, false, false>(p, stream);
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
        return -1;
    }
}
