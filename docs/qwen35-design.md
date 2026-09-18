# Qwen3.8-27B support design — 2026-09-17

Status: draft. Checkpoint: `models/Qwen/Qwen3.8-27B` (51.7 GiB, 18 shards,
1199 tensors, all BF16, zero quantized payloads — verified via shard index).

## 1. What the model is

`Qwen3_5ForConditionalGeneration` (`qwen3_5`), dense sibling of the Edge0
family — same GDN/full-attention hybrid skeleton, no MoE:

- **64 layers**: 48 `linear_attention` + 16 `full_attention`
  (`full_attention_interval: 4`). GDN: 16 key heads × 128 + 48 value heads ×
  128 (conv_dim 10240), conv k=4, ssm f32. Full-attn: 24 heads / 4 KV,
  head_dim 256, attn_output_gate (swish), partial rotary 0.25, interleaved
  mrope [11,11,10] — all identical in kind to Edge0's layers.
- hidden 5120, dense MLP gate/up [17408, 5120] + down [5120, 17408]
  (~537 MB bf16 per layer), vocab 248320 (embed/lm_head untied),
  **1 MTP layer** (full-attn + dense MLP, `mtp_use_dedicated_embeddings:
  false`).
- Vision tower present in the checkpoint (`model.visual.*`, SigLIP-class).
- Tensor prefix differs from Edge0: `model.language_model.layers.N.*`
  (Edge0: `language_model.model.layers.N.*`).

## 2. The 1000 tok/s question (physics before code)

Batch-1 decode on the RTX 4090 (1008 GB/s peak, ~680 GB/s demonstrated on
vocab-scale int4 GEMV):

| precision | bytes/token | floor | ceiling tok/s |
|---|---|---|---|
| bf16 (as shipped) | ~48 GB | 48 ms | ~21 |
| int4 requant | ~13.5 GB | ~20 ms @ 680 | ~50-68 |
| int4 + MTP-1 (measured acceptance ~1.7-1.85x typical) | — | — | ~85-125 |

**1000 tok/s at batch 1 on this card is unreachable by 8-50x at any
precision.** The number as seen must be aggregate-throughput (large batch),
prefill, or bigger-hardware. Before chasing it we need the reference's
hardware/batch/precision context from the user. What this card CAN do:
~50-70 tok/s straight decode, ~100-125 with MTP — matching the per-bandwidth
efficiency of any honest single-GPU claim.

## 3. Plan

- **Q0 requant tool** (prerequisite — checkpoint ships no quant): offline
  bf16 → int4 groupwise-affine (`s·q+b`, group 64, the edge0 byte layout).
  Per-group scale/bias by least squares on the bf16 values (not just
  min/max — the biases tensor is authoritative, so use the full affine fit).
  Output: a derived checkpoint dir (packed U32 + BF16 scales/biases per
  tensor) + a manifest. Norms/conv/dt_bias/A_log stay bf16 (as in edge0).
- **Q1 dense text adapter** (`ff-qwen35` crate, reusing ff-edge0's kernel
  layer): config.rs (qwen3_5_text), weights.rs (requant layout), model.rs
  (GDN/attn/dense-MLP), GPU path = the edge0 closed decode loop minus MoE
  plus dense MLP (gate/up group + silu + down). mrope text-only first
  (vision positions reserved, per the edge0 P0 lesson).
- **Q2 kernel generalization** (the real kernel work): hidden 5120 → 640
  words/row → wpt=3 at 256 threads; dense down in=17408 → 2176 words →
  wpt=9. WPT_CAP=2 assumptions break. Two moves: (a) WPT_CAP=4
  instantiation for in<=8192; (b) split-K with deterministic fixed-order
  cross-block reduce for in>8192 (down_proj). Both are new reduction shapes
  — gate-arbitrated, not bit-chased (no stricter reference exists than
  bf16 for a requant).
- **Q3 MTP self-speculative decode** (the speed multiplier; the checkpoint
  ships the MTP layer). Greedy accept keeps outputs identical to
  non-speculative decode — the acceptance gate stays exact.
- **Q4 vision** (tower present this time): ViT + merger + mrope positions.

## 4. Acceptance hooks (decided before code)

- Requant quality: per-tensor max_rel and end-to-end KL on a fixed prompt
  set vs the bf16 reference (Python, HF-style decode over the raw shards —
  the edge0_reference_generate.py pattern). Budget: int4 group-64 affine
  typically lands last-layer logit KL < 1e-3; measure, don't assume.
- Decode acceptance: greedy token ids vs the bf16 Python reference on 2
  prompts × 32 tokens (the extended gate — the 8-token gate demonstrably
  misses adapter-magnitude errors).

## 5. Measured (2026-09-17)

- bf16 Rust CPU vs HF bf16: **8/8 token ids exact** with confident margins
  (token-0 margin 4.50 vs HF's 4.38). The pre-fix divergence traced to the
  hand-rolled chat template missing the checkpoint's system block — not
  numerics, not dtype chaos (single-token per-layer diffs sat at the bf16
  rounding floor ~0.2% throughout).
- int4 (LSQ affine) vs HF: prefix 2 exact, first flip at token 2 with
  margin 0.82 — a REAL quality gap, not a tie-flip. 4 LSQ rounds + scale
  search moved rms_ratio 0.0870 -> 0.08695 (floor of the method). Next
  level is activation-aware calibration (GPTQ/AWQ class).
- **2026-09-18 requant re-run (bf16-rounded candidate scoring, #10)**:
  rms_ratio 0.08695 -> **0.08517**, now printed by requant itself
  (reproducible, not ad-hoc). The in-repo int4 dir was regenerated with
  the fixed code; the id-level picture vs HF is unchanged (prefix 2
  exact) — the residual gap is activation-aware territory. All perf
  numbers below re-measured on the new weights: plain graph decode
  24-25 ms/token, spec **46.8 tok/s vs ~40 plain (~+17%)** on a 256-token
  run (156 rounds, acceptance 64.7% — 95% CI ±7.5pp iid-binomial, but
  rounds within one run correlate with position/context, so treat that as
  an optimistic lower bound; 48-token runs scatter 47.7-48.7 tok/s /
  66-69%, statistically indistinguishable from the long run), spec ids
  exact vs plain greedy, CPU-vs-GPU int4 ids 8/8 bit-exact.
- int4 vs bf16 within ff-qwen35: identical ids at 4 tokens — the pipeline
  is stable under its own quantization noise; the gap is vs the bf16
  reference.
- The HF bf16 reference must never run resident (52 GB weights + f32
  upcast transients killed a session at 62 GB RAM); MemoryMax-scoped runs
  only, and the Rust bf16 path streams off the mmap (no f32 residency).

## 6. GPU path + speculation (2026-09-18, measured)

- GPU decode (int4, full resident): 37 ms/token v1 -> **25 ms (graph)**;
  decode-window in-kernel 25.0 ms vs the 20.3 ms traffic floor (13.8 GB
  at 680 GB/s). Wide split-K + grouped kernels run 630-635 GB/s
  in-pipeline; the residual gap is norms (1.35), gdn_heads (1.08),
  splitk/group4 residuals (~1.6), gaps (0.56). Ids bit-exact vs the CPU
  int4 path at every step.
- MTP speculation machinery landed and output-exact (16/16 ids == plain
  greedy): batch-2 verify (cols=2 wide kernels, column-major dispatch —
  x-fastest block order was an L2-sharing killer), GDN post-A/post-B
  state split (reject = dtod restore, no recompute), GPU draft on its own
  0-based KV timeline. Acceptance 87% off-path, ~70% in-loop.
- **But net-neutral at 25 ms**: column 2's compute is not free while
  singles sit at 63% of bandwidth peak. Speculation pays ~1.6-1.8x once
  the kernel gap closes (at the floor: ~22.5 ms/round for 1.7 tokens =
  ~75 tok/s); the machinery stays env-gated until then.
- **2026-09-18, re-measured after the dup-slot fix**: the batch-2 verify
  was passing the SAME projection into multiple group slots (out_proj x4,
  o_proj x4, gate_proj 3x — the grid sums blocks over slots, so those
  matrices were re-read for identical output, ~5.7 GB/round of waste).
  With one real slot + empty_seg_like(): verify 42.5 -> **32.5 ms/round**,
  spec decode **48.7 tok/s vs 41.7 plain — net-POSITIVE (+19%)**, ids
  still exact vs plain greedy (48/48), acceptance 69% in-loop. The
  remaining verify gap to the 22.5 ms floor is column 2's real compute.
- Edge0's checkpoint has NO MTP weights (shard index verified) — MTP is
  Qwen-only until the official release ships them.
- Perf gate: cold-truth bw numbers per shape (L2-flushed bench from day
  one — the warm-L2 fiction cost us two commits).
