// FF-FA adapter: HD128 prefill for MiniCPM (one fixed trait set) and HD256
// prefill for Qwen3.8, which replicates FA2's own run_mha_fwd_hdim256 branch.
//
// d128 keeps Flash_fwd_kernel_traits<128, 64, 64, 4> and the previous parameter
// fill; with q_row_stride = (q_heads + 2 * kv_heads) * 128 the launch is the one
// the tree has always performed.
//
// d256 queries the shared-memory attributes and takes the 128x64 / 8-warp tile
// only under FA2's own condition (>= 2*256*(128 + 2*64) B per block and
// < 4*256*(64 + 2*64) B per SM), else the 64x64 / 4-warp fallback. Both current
// targets (sm_86, sm_89) sit below the per-block threshold and take the
// fallback; ff_fa2_tile records which tile a head dim resolves to.
//
// Q is read as [rows, q_heads, head_dim] with head stride head_dim and a row
// pitch the caller states: the HD256 prefill reads the split [M, 24, 256] Q
// view (pitch 6144), not a fused qkv. K and V are the head-major caches
// [kv_heads][capacity][head_dim] that tcmqa_d256_rope_k_scatter / _v_scatter
// fill: position stride head_dim, head stride capacity * head_dim.
#include <cmath>
#include <cstdint>
#include <string>
#include <nvtx3/nvToolsExt.h>

#include "src/flash_fwd_launch_template.h"

using Traits128 = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
static thread_local std::string error;

extern "C" void ff_profile_push(const char* name) { nvtxRangePushA(name); }
extern "C" void ff_profile_pop() { nvtxRangePop(); }

extern "C" const char* ff_fa2_error() {
    return error.c_str();
}

namespace {
// FA2's own hdim256 choice (run_mha_fwd_hdim256): the 128x64 / 8-warp tile when
// a block may opt into 2*256*(128 + 2*64) B of dynamic shared memory and an SM
// holds fewer than four of the 64x64 fallback's worth, else the fallback.
bool hdim256_big_tile() {
    int device = 0;
    cudaGetDevice(&device);
    int smem_block = 0, smem_sm = 0;
    cudaDeviceGetAttribute(&smem_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    return smem_block >= 2 * 256 * (128 + 2 * 64) && smem_sm < 4 * 256 * (64 + 2 * 64);
}

// One launch-parameter fill for every (head dim, traits) instantiation.
//
// seqlen_k = start + rows is what right-aligns the causal mask at the query
// rows' absolute positions; seqlen_q stays rows, so row r sees keys <= start+r.
template <int D, typename Traits>
int run_impl(void* q, void* k, void* v, void* out, void* lse, int rows, int start, int q_heads,
             int kv_heads, int capacity, int64_t q_row_stride, int causal,
             cudaStream_t stream, int* seqused_k = nullptr) {
    if (rows <= 0 || start < 0 || start + rows > capacity || kv_heads <= 0 || q_heads <= 0 ||
        q_heads % kv_heads != 0 || q_row_stride < int64_t(q_heads) * D) {
        error = "invalid FA2 shape or cache range";
        return -1;
    }
    flash::Flash_fwd_params p{};
    p.q_ptr = q;
    p.k_ptr = k;
    p.v_ptr = v;
    p.o_ptr = out;
    p.softmax_lse_ptr = lse;
    p.q_row_stride = q_row_stride;
    p.k_row_stride = p.v_row_stride = D;
    p.q_head_stride = p.o_head_stride = D;
    p.k_head_stride = p.v_head_stride = int64_t(capacity) * D;
    p.q_batch_stride = int64_t(rows) * q_row_stride;
    p.k_batch_stride = p.v_batch_stride = int64_t(kv_heads) * capacity * D;
    p.o_row_stride = int64_t(q_heads) * D;
    p.o_batch_stride = int64_t(rows) * p.o_row_stride;
    p.h = q_heads;
    p.h_k = kv_heads;
    p.h_h_k_ratio = q_heads / kv_heads;
    p.b = 1;
    p.seqlen_q = rows;
    p.seqlen_k = seqused_k ? capacity : start + rows;
    p.seqused_k = seqused_k;
    p.seqlen_q_rounded = (rows + 127) / 128 * 128;
    p.seqlen_k_rounded = (p.seqlen_k + 127) / 128 * 128;
    p.d = p.d_rounded = D;
    p.total_q = rows;
    p.scale_softmax = 1.0f / std::sqrt(float(D));
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
}

int dispatch(void* q, void* k, void* v, void* out, void* lse, int rows, int start,
                          int q_heads, int kv_heads, int capacity, int head_dim,
                          int64_t q_row_stride, int causal, cudaStream_t stream, int* seqused_k) {
    try {
        if (head_dim == 128) {
            return run_impl<128, Traits128>(q, k, v, out, lse, rows, start, q_heads, kv_heads,
                                            capacity, q_row_stride, causal, stream, seqused_k);
        }
        if (head_dim == 256) {
            if (hdim256_big_tile()) {
                using Traits =
                    Flash_fwd_kernel_traits<256, 128, 64, 8, false, false, cutlass::bfloat16_t>;
                return run_impl<256, Traits>(q, k, v, out, lse, rows, start, q_heads, kv_heads,
                                             capacity, q_row_stride, causal, stream, seqused_k);
            }
            using Traits =
                Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;
            return run_impl<256, Traits>(q, k, v, out, lse, rows, start, q_heads, kv_heads,
                                         capacity, q_row_stride, causal, stream, seqused_k);
        }
        error = "unsupported FA2 head dim";
        return -1;
    } catch (const std::exception& e) {
        error = e.what();
        return -1;
    }
}

}  // namespace

// kBlockM FA2 uses for this head dim on the current device: 64 for d128, 128 or
// 64 for d256 per the query above. 0 = unsupported head dim.
extern "C" int ff_fa2_tile(int head_dim) {
    if (head_dim == 128) {
        return Traits128::kBlockM;
    }
    if (head_dim == 256) {
        return hdim256_big_tile() ? 128 : 64;
    }
    return 0;
}

extern "C" int ff_fa2_run(void* q, void* k, void* v, void* out, void* lse, int rows, int start,
                          int q_heads, int kv_heads, int capacity, int head_dim,
                          int64_t q_row_stride, int causal, cudaStream_t stream) {
    return dispatch(q, k, v, out, lse, rows, start, q_heads, kv_heads, capacity,
                    head_dim, q_row_stride, causal, stream, nullptr);
}
