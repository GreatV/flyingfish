// Latent window attention: batch-1 decode over the per-layer ring of FP8
// latent K rows, with one attention sink per head.
// Semantics per the DeepSeek-V4.1-Flash model definition (publisher
// inference/model.py Attention.forward, _window_kv and get_window_topk_idxs):
//   - the decode query is one token; the visible window is the whole ring,
//     oldest first, minus the slots the ring has not filled yet;
//   - the ring holds post-RoPE K, quantized to E4M3 over the whole head vector
//     with one E8M0 scale per 32 codes, RoPE tail included;
//   - the score scale is head_dim^-0.5;
//   - the per-head sink is one extra softmax logit without a key and without a
//     value: it enters the denominator only;
//   - the value is the K content itself (latent attention, one KV head);
//   - the last rope_tail dims of the output get the inverse RoPE.
// Entry: latent_window_attention(q, k_codes, k_scales, slot_ids, sink,
// inv_rotary, out, heads, head_dim, window, slots, rope_tail, scale).
//   q [heads, head_dim] bf16 post-RoPE, k_codes [window, head_dim] E4M3,
//   k_scales [window, head_dim/32] E8M0, slot_ids [slots] ring slot ids
//   oldest first, sink [heads] f32, inv_rotary [rope_tail] f32 interleaved
//   (cos, sin) of the inverse angle, out [heads, head_dim] bf16.
// One block per head, 256 threads; the ring rows are the first `window` rows
// of k_codes and k_scales.
//
// The same module stages the ring and the attention output: latent_ring_write
// (quantize the post-RoPE K row per 32 and place it at the position's slot),
// latent_act_quant (the same quantization into linear codes/scales, used for
// the attention output before wo_a), latent_window_slots (visible slots,
// oldest first), and latent_group_project (block-diagonal
// wo_a over groups with one E8M0 scale per 32x32 weight block). The
// adjacent-pair rotary lives in kernels/rope.cu (rope_pair_rows /
// rope_pair_tail).
#include "q38_contract.cuh"
#include <math.h>
#include <stdint.h>

#define LWA_THREADS 256
// E4M3 codes per E8M0 scale along the head vector.
#define LWA_SCALE_BLOCK 32
// Key rows per softmax block of the definition's window attention kernel.
#define LWA_PUB_BLOCK 64

// Dequantized ring value of one code.
__device__ __forceinline__ float lwa_value(uint8_t code, uint8_t scale) {
  return e4m3_decode(code) * e8m0_decode(scale);
}

extern "C" __global__ __launch_bounds__(LWA_THREADS) void latent_window_attention(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_codes,
    const uint8_t* __restrict__ k_scales, const int* __restrict__ slot_ids,
    const float* __restrict__ sink, const float* __restrict__ inv_rotary,
    __nv_bfloat16* __restrict__ out, int heads, int head_dim, int window,
    int slots, int rope_tail, float scale, int p_mode) {
  if ((int)blockIdx.x >= heads || heads < 1 || head_dim < 1 || window < 1 ||
      slots < 1 || slots > window || (head_dim % LWA_SCALE_BLOCK) != 0 ||
      rope_tail < 0 || rope_tail > head_dim || (rope_tail % 2) != 0 ||
      !(scale > 0.0f) || (unsigned)p_mode > 1u || blockDim.x != LWA_THREADS)
    __trap();

  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int head = (int)blockIdx.x;
  const int nblocks = head_dim / LWA_SCALE_BLOCK;

  for (int s = tid; s < slots; s += nthr) {
    const int row = slot_ids[s];
    if (row < 0 || row >= window) __trap();
  }

  extern __shared__ __align__(16) float lwa_smem[];
  float* s_q = lwa_smem;             // [head_dim]
  float* s_scores = s_q + head_dim;  // [window]
  float* s_acc = s_scores + window;  // [head_dim]

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarp = nthr >> 5;
  const int per_lane = head_dim / 32;

  const __nv_bfloat16* q_row = q + (size_t)head * head_dim;
  for (int d = tid; d < head_dim; d += nthr) s_q[d] = __bfloat162float(q_row[d]);
  // Every thread writes its own s_q elements and then reads all of them in the score loop.
  __syncthreads();

  // One warp per key row: lane `l` sums dims l, l+32, ... (one E8M0 scale block per
  // step), then a 32-lane shuffle tree gives the score. No shared staging and no
  // barrier inside the loop; warps take keys r, r+nwarp, ... .
  for (int s = warp; s < slots; s += nwarp) {
    const int row = slot_ids[s];
    const uint8_t* k_row = k_codes + (size_t)row * head_dim;
    const uint8_t* sc_row = k_scales + (size_t)row * nblocks;
    float acc = 0.0f;
    for (int i = 0; i < per_lane; ++i) {
      const int d = i * 32 + lane;
      acc = fmaf(s_q[d], lwa_value(k_row[d], sc_row[i]), acc);
    }
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(FF_WARP_MASK, acc, off);
    if (lane == 0) s_scores[s] = acc * scale;
  }
  __syncthreads();

  const float sk = sink[head];
  // The definition's blocked softmax (inference/kernel.py sparse_attn_kernel): the visible
  // keys form blocks of LWA_PUB_BLOCK; each block rescales the FP32 accumulator and the
  // running sum by exp(m_prev - m_new), rounds P to BF16 against its running max and adds
  // its unrounded sum; the sink enters the sum after the last block, then one divide.
  // p_mode 1 keeps the structure and drops only the BF16 P rounding (the gate's control).
  for (int d = tid; d < head_dim; d += nthr) s_acc[d] = 0.0f;
  float m_run = -INFINITY;
  float sum_exp = 0.0f;
  for (int t = 0; t < slots; t += LWA_PUB_BLOCK) {
    const int hi = t + LWA_PUB_BLOCK < slots ? t + LWA_PUB_BLOCK : slots;
    float bmax = m_run;
    for (int s = t; s < hi; ++s) bmax = fmaxf(bmax, s_scores[s]);
    if (t > 0) {
      const float rescale = expf(m_run - bmax);
      for (int d = tid; d < head_dim; d += nthr) s_acc[d] *= rescale;
      sum_exp *= rescale;
    }
    float bsum = 0.0f;
    for (int s = t; s < hi; ++s) {
      const int row = slot_ids[s];
      const uint8_t* k_row = k_codes + (size_t)row * head_dim;
      const uint8_t* sc_row = k_scales + (size_t)row * nblocks;
      const float p = expf(s_scores[s] - bmax);
      bsum += p;
      const float w = p_mode == 1 ? p : __bfloat162float(__float2bfloat16_rn(p));
      for (int d = tid; d < head_dim; d += nthr)
        s_acc[d] = fmaf(w, lwa_value(k_row[d], sc_row[d / LWA_SCALE_BLOCK]), s_acc[d]);
    }
    sum_exp += bsum;
    m_run = bmax;
  }
  const float denom = sum_exp + expf(sk - m_run);
  for (int d = tid; d < head_dim; d += nthr)
    s_acc[d] = __bfloat162float(__float2bfloat16_rn(s_acc[d] / denom));
  __syncthreads();
  const int tail = head_dim - rope_tail;
  for (int i = tid; i < rope_tail / 2; i += nthr) {
    const int d = tail + 2 * i;
    const float c = inv_rotary[2 * i], sn = inv_rotary[2 * i + 1];
    const float a = s_acc[d], b = s_acc[d + 1];
    s_acc[d] = a * c - b * sn;
    s_acc[d + 1] = a * sn + b * c;
  }
  __syncthreads();
  __nv_bfloat16* out_row = out + (size_t)head * head_dim;
  for (int d = tid; d < head_dim; d += nthr) out_row[d] = __float2bfloat16_rn(s_acc[d]);
}

// ---------------------------------------------------------------------------
// Ring staging and the attention-output path of the same operator family.
// The rotary convention is the model definition's (publisher model.py
// apply_rotary_emb): adjacent element pairs rotated as complex numbers in
// FP32, one BF16 round at the write; the adjacent-pair rotary rows and tail
// rotation live in kernels/rope.cu (rope_pair_rows / rope_pair_tail).

// One warp per 32-element group of one row: quantize (act_quant32) and place
// row r at ring slot (position + r) % window.
extern "C" __global__ __launch_bounds__(LWA_THREADS) void latent_ring_write(
    const __nv_bfloat16* __restrict__ kv, unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales, int position, int rows, int head_dim,
    int window) {
  if (position < 0 || rows < 1 || head_dim < 1 || window < 1 ||
      (head_dim % LWA_SCALE_BLOCK) != 0 || blockDim.x != LWA_THREADS)
    __trap();
  const int groups = head_dim / LWA_SCALE_BLOCK;
  const int warp = int(blockIdx.x) * (int(blockDim.x) / 32) + int(threadIdx.x / 32);
  if (warp >= rows * groups) return;
  const int lane = int(threadIdx.x % 32);
  const int row = warp / groups;
  const int group = warp % groups;
  const int slot = (position + row) % window;
  const float value =
      __bfloat162float(kv[(size_t)row * head_dim + (size_t)group * LWA_SCALE_BLOCK + lane]);
  unsigned char code, scale_code;
  act_quant32(value, &code, &scale_code);
  codes[(size_t)slot * head_dim + (size_t)group * LWA_SCALE_BLOCK + lane] = code;
  if (lane == 0) scales[(size_t)slot * groups + group] = scale_code;
}

// One warp per 32-element group of one row: quantize (act_quant32) into the
// linear codes/scales layout.
extern "C" __global__ __launch_bounds__(LWA_THREADS) void latent_act_quant(
    const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales, long long rows, int width) {
  if (rows < 1 || width < 1 || (width % LWA_SCALE_BLOCK) != 0 ||
      blockDim.x != LWA_THREADS)
    __trap();
  const int groups = width / LWA_SCALE_BLOCK;
  const int warp = int(blockIdx.x) * (int(blockDim.x) / 32) + int(threadIdx.x / 32);
  if ((long long)warp >= rows * groups) return;
  const int lane = int(threadIdx.x % 32);
  const size_t base = (size_t)warp * LWA_SCALE_BLOCK;
  const float value = __bfloat162float(x[base + lane]);
  unsigned char code, scale_code;
  act_quant32(value, &code, &scale_code);
  codes[base + lane] = code;
  if (lane == 0) scales[warp] = scale_code;
}

// The visible ring slots of the query at `position`, oldest first (model
// definition, publisher model.py get_window_topk_idxs decode branch); entries
// beyond the visible count are left untouched.
extern "C" __global__ void latent_window_slots(int* slot_ids, int position,
                                               int window) {
  if (position < 0 || window < 1) __trap();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= window) return;
  const int count = position + 1 < window ? position + 1 : window;
  if (i >= count) return;
  slot_ids[i] = (position + 1 - count + i) % window;
}

// Block-diagonal group projection of the quantized attention output (model
// definition, publisher model.py Attention.forward: o quantized per 32 before
// the einsum, each group projected by its own wo_a rows with one E8M0 scale
// per 32x32 weight block). act [rows, groups, width] codes with scales
// [rows, groups, width/32]; weights [groups*rank, width] codes with scales
// [groups*rank/32, width/32]; out [rows, groups*rank] BF16. One warp per
// output, each 32-element block split over four lanes of eight elements.
extern "C" __global__ __launch_bounds__(LWA_THREADS) void latent_group_project(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ w_codes,
    const unsigned char* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ out, int rows, int groups, int rank,
    int width) {
  if (rows < 1 || groups < 1 || rank < 1 || width < 1 ||
      (width % (LWA_SCALE_BLOCK * 8)) != 0 || (rank % LWA_SCALE_BLOCK) != 0 ||
      blockDim.x != LWA_THREADS ||
      gridDim.x != (unsigned)((rows * groups * rank + 7) / 8))
    __trap();
  const int warp = int(blockIdx.x) * 8 + int(threadIdx.x / 32);
  if (warp >= rows * groups * rank) return;
  const int lane = int(threadIdx.x % 32);
  const int col = warp % (groups * rank);
  const int row = warp / (groups * rank);
  const int nblocks = width / LWA_SCALE_BLOCK;
  const unsigned char* act_row = act_codes + ((size_t)row * groups + col / rank) * width;
  const unsigned char* act_sc =
      act_scales + ((size_t)row * groups + col / rank) * nblocks;
  const unsigned char* w_row = w_codes + (size_t)col * width;
  const unsigned char* w_sc = w_scales + (size_t)(col / LWA_SCALE_BLOCK) * nblocks;
  const int sub = lane & 3;
  const int stride = lane >> 2;
  // Stride s owns the contiguous block range [s * per, (s + 1) * per) and adds
  // the two 16-element sums of each block in ascending order, which is the
  // definition's mma k-step: exact 16-element inner sums, one FP32 rounding per
  // 16 elements into a contiguous partial (model.py Attention.forward wo_a
  // einsum, 32 activation groups of 16).
  const int per = nblocks / 8;
  float acc = 0.0f;
  for (int j = 0; j < per; ++j) {
    const int block = stride * per + j;
    float sum = 0.0f;
    const unsigned char* a =
        act_row + (size_t)block * LWA_SCALE_BLOCK + sub * 8;
    const unsigned char* w = w_row + (size_t)block * LWA_SCALE_BLOCK + sub * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum = fmaf(e4m3_decode(w[i]), e4m3_decode(a[i]), sum);
    // Lanes with sub in {0, 1} hold the block's low 16-element sum, {2, 3} its
    // high one; sub 0 adds both to its partial in ascending order.
    sum += __shfl_xor_sync(FF_WARP_MASK, sum, 1, 4);
    const float upper = __shfl_xor_sync(FF_WARP_MASK, sum, 2, 4);
    if (sub == 0) {
      const float scale =
          e8m0_decode(act_sc[block]) * e8m0_decode(w_sc[block]);
      acc = fmaf(sum, scale, acc);
      acc = fmaf(upper, scale, acc);
    }
  }
  // Lanes with sub != 0 hold acc 0; the xor reduction sums the eight
  // stride lanes' partial dots.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(FF_WARP_MASK, acc, off);
  if (lane == 0) out[(size_t)row * groups * rank + col] = __float2bfloat16_rn(acc);
}

// Raw and compressed rows are separate entries, including overlapping token ranges.
__device__ __forceinline__ float indexed_value(
    const uint8_t* raw, const uint8_t* raw_scales,
    const uint8_t* packed, const uint8_t* packed_scales,
    int source, int row, int column, int width) {
  if (!source) return bround(e4m3_decode(raw[int64_t(row) * width + column])
      * e8m0_decode(raw_scales[int64_t(row) * (width / 32) + column / 32]));
  const uint8_t pair = packed[int64_t(row) * (width / 2) + column / 2];
  return bround(e2m1_decode((column & 1) ? pair >> 4 : pair & 15)
      * e4m3_decode(packed_scales[int64_t(row) * (width / 16) + column / 16]));
}

template<bool Trace>
__device__ __forceinline__ void indexed_attention_core(
    const __nv_bfloat16* q, const uint8_t* raw, const uint8_t* raw_scales,
    const uint8_t* packed, const uint8_t* packed_scales, const int32_t* selected,
    const uint32_t* committed, const float* sink, const float* inverse,
    const Q38SequenceMeta* sequence, __nv_bfloat16* output,
    int heads, int width, int window, int capacity, int ratio, int query,
    int rotary_dim, float scale, float* trace) {
  if constexpr (Trace) { if (heads != 64) __trap(); }
  q38_meta_guard(sequence);
  if (blockDim.x != 256 || width != 512 || window != 128 || heads < 1
      || query < 0 || query >= sequence->valid_rows || (ratio != 1 && ratio != 2)
      || *committed != uint32_t(sequence->key_count / ratio)
      || *committed > uint32_t(capacity) || rotary_dim < 0 || rotary_dim > width
      || rotary_dim % 2 || !(scale > 0)) __trap();
  const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  const int position = sequence->prefix + query;
  const int visible = min(position + 1, window), first = position + 1 - visible;
  const int compressed = (position + 1) / ratio;
  const int count = window + 512;
  const int32_t* ids = selected + int64_t(query) * 512;
  for (int i = tid; i < 512; i += 256) {
    const int id = ids[i];
    if (id < -1 || id >= compressed || id >= capacity
        || (i && id >= 0 && (ids[i-1] < 0 || ids[i-1] >= id))) __trap();
  }
  extern __shared__ __align__(16) float shared[];
  float* qv = shared;
  float* scores = qv + width;
  float* acc = scores + count;
  for (int d = tid; d < width; d += 256) {
    qv[d] = __bfloat162float(q[int64_t(head) * width + d]);
    acc[d] = 0.0f;
  }
  __syncthreads();
  for (int i = warp; i < count; i += 8) {
    const bool source = i >= window;
    const int id = source ? ids[i-window] : (i < visible ? (first+i) % window : -1);
    float dot = 0;
    if (id >= 0) {
      for (int d = lane; d < width; d += 32) {
        const float value = indexed_value(raw, raw_scales, packed, packed_scales, source, id, d, width);
        if constexpr (Trace) { if (!head) trace[163840+i*512+d] = value; }
        dot = fmaf(qv[d], value, dot);
      }
      for (int off = 16; off; off >>= 1) dot += __shfl_down_sync(FF_WARP_MASK, dot, off);
    }
    if constexpr (Trace) {
      if (id < 0 && !head) for (int d = lane; d < width; d += 32)
        trace[163840+i*512+d] = 0;
    }
    if (!lane) {
      scores[i] = id < 0 ? -INFINITY : dot * scale;
      if constexpr (Trace) {
        trace[(i/64*64+head)*64+i%64] = id < 0 ? -INFINITY : dot;
        trace[40960+(i/64*64+head)*64+i%64] = scores[i];
      }
    }
  }
  __syncthreads();
  float maximum = -INFINITY, denominator = 0;
  for (int start = 0; start < count; start += 64) {
    float next = maximum;
    for (int i = start; i < start + 64; ++i) next = fmaxf(next, scores[i]);
    if (next == -INFINITY) continue;
    const float rescale = maximum == -INFINITY ? 0.0f : expf(maximum - next);
    denominator *= rescale;
    for (int d = tid; d < width; d += 256) acc[d] *= rescale;
    float sum = 0;
    for (int i = start; i < start + 64; ++i) {
      const float p = expf(scores[i] - next);
      sum += p;
      const float stored = bround(p);
      if constexpr (Trace) {
        if (!tid) {
          trace[81920+(start/64*64+head)*64+i-start] = p;
          trace[122880+(start/64*64+head)*64+i-start] = stored;
        }
      }
      const bool source = i >= window;
      const int id = source ? ids[i-window] : (i < visible ? (first+i) % window : -1);
      if (id >= 0) for (int d = tid; d < width; d += 256)
        acc[d] = fmaf(stored, indexed_value(raw, raw_scales, packed, packed_scales, source, id, d, width), acc[d]);
    }
    denominator += sum;
    if constexpr (Trace) {
      if (!tid) {
        trace[819200+start/64*64+head] = denominator;
        trace[819840+start/64*64+head] = next;
        trace[820480+start/64*64+head] = rescale;
      }
      for (int d = tid; d < width; d += 256)
        trace[491520+(start/64*64+head)*512+d] = acc[d];
    }
    maximum = next;
  }
  denominator += expf(sink[head] - maximum);
  if constexpr (Trace) { if (!tid) trace[853888+head] = denominator; }
  for (int d = tid; d < width; d += 256) {
    const float value = acc[d] / denominator;
    if constexpr (Trace) {
      trace[821120+head*512+d] = acc[d];
      trace[853952+head*512+d] = value;
    }
    acc[d] = bround(value);
  }
  __syncthreads();
  for (int i = tid; i < rotary_dim / 2; i += 256) {
    const int d = width - rotary_dim + 2*i;
    const float a = acc[d], b = acc[d+1], c = inverse[2*i], sn = inverse[2*i+1];
    acc[d] = a*c - b*sn; acc[d+1] = a*sn + b*c;
  }
  __syncthreads();
  for (int d = tid; d < width; d += 256) {
    if (!isfinite(acc[d])) __trap();
    output[int64_t(head)*width+d] = __float2bfloat16_rn(acc[d]);
  }
}

extern "C" __global__ __launch_bounds__(256) void latent_indexed_attention(
    const __nv_bfloat16* q, const uint8_t* raw, const uint8_t* raw_scales,
    const uint8_t* packed, const uint8_t* packed_scales, const int32_t* selected,
    const uint32_t* committed, const float* sink, const float* inverse,
    const Q38SequenceMeta* sequence, __nv_bfloat16* output,
    int heads, int width, int window, int capacity, int ratio, int query,
    int rotary_dim, float scale) {
  indexed_attention_core<false>(q, raw, raw_scales, packed, packed_scales, selected, committed, sink, inverse, sequence, output, heads, width, window, capacity, ratio, query, rotary_dim, scale, nullptr);
}

extern "C" __global__ __launch_bounds__(256) void latent_indexed_attention_trace(
    const __nv_bfloat16* q, const uint8_t* raw, const uint8_t* raw_scales,
    const uint8_t* packed, const uint8_t* packed_scales, const int32_t* selected,
    const uint32_t* committed, const float* sink, const float* inverse,
    const Q38SequenceMeta* sequence, __nv_bfloat16* output,
    int heads, int width, int window, int capacity, int ratio, int query,
    int rotary_dim, float scale, float* trace) {
  indexed_attention_core<true>(q, raw, raw_scales, packed, packed_scales, selected, committed, sink, inverse, sequence, output, heads, width, window, capacity, ratio, query, rotary_dim, scale, trace);
}
