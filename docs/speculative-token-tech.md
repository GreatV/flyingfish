# Speculative / token-level technique survey

This document surveys the inference-time technique landscape that overlaps `ff`'s text and video paths, motivated by user-prompted interest in "FreeToken" and adjacent methods. The "token" word appears in three different senses across the literature, and the "FreeToken" name itself maps to several unrelated papers; a disambiguation table heads the survey.

**Sources and acceptance criteria**: Every quoted headline number carries a URL + date + hardware + model + measured-vs-claimed tag. Where a number could not be located in primary sources within three searches, it is recorded as **未能验证**. Primary-source citations are arXiv abstract pages, paper PDFs (where reachable), GitHub READMEs, OpenReview records, and project pages.

## §0 Disambiguation table — what "FreeToken" can mean

The term "FreeToken" is overloaded in 2024-2025 literature: at least four unrelated papers share the name, and the user's hint describing "FreeToken: Turning Ill-Posed to Well-Posed for Token Reduction" does not match any paper that was locatable in arXiv/OpenReview as of the date this survey was compiled.

| # | Paper | arXiv | Date | What it actually does | Disposition |
|---|---|---|---|---|---|
| **F1** | "FreeToken: Be Careful of Your Tokens" (alleged Chaoyou Fu, NeurIPS 2024) | 2407.14464 | claimed Jul 2024 | VLM visual-token pruning — **paper not located** (arXiv 2407.14464 is `AttentNet: Fully Convolutional 3D Attention for Lung Nodule Detection`; OpenReview NeurIPS 2024 search returns 0; candidate repos 404) | **Drop from primary table**; placeholder filled in §5 with verified adjacent VLM-pruning papers |
| F2 | "FreeToken: Tokenize-Free Visual Reinforcement Learning for Manipulation" | 2410.21335 | 2024-10-28 | Visual RL policy over raw image patches | Niche — outside `ff` scope |
| **F3** | "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution" | **2608.16157** | 2026-08-17 | Edge-native MoE serving for open-weight LLMs (Berkeley/MIT/Databricks) — bandwidth-adaptive CPU/GPU expert split, double-buffered prefill, routing-locality LRU | **Primary line of this survey**; closest to `ff`'s scope |
| F4 | "FreeToken: Turning Ill-Posed to Well-Posed for Free-Form Referenceless Image Quality Assessment" | (unconfirmed) | claimed 2024-2025 | IQA reformulation — **arXiv ID and venue could not be located** | Treated as 未能验证 throughout |

**Primary-line selection rationale.** F3 wins on three axes: (a) it is the only verifiable "FreeToken" paper; (b) it overlaps the most active area of `ff`'s work — single-card MoE on limited host RAM, the same regime as `docs/comparable-products.md` §4.1; (c) the headline numbers (1.3-2.1× decode vs strongest baseline; <44 s tail TTFT) are within the same order of magnitude as the techniques we already use (DSpark speculative at ~2.4×; MTP batch-2 verify), so it is the most actionable comparator.

The user's specific hint ("Turning Ill-Posed to Well-Posed for Token Reduction") matches the *category* of F1-style VLM visual-token pruning; we expect the user may have confabulated the citation. The §5 "VLM visual-token pruning" section fills that slot with verified adjacent papers — **VisionZip**, **FastV**, **FitPrune**, **DivPrune**, **SparseVLM**, **FasterVLM**, **ZOO-Prune** — and the relevance note there explains what would still apply to `ff-qwen35`'s vision tower.

## §1 Primary — FreeToken (F3): edge-native MoE serving, Berkeley/MIT/Databricks

### §1.1 Identifiers (all verified)

| Field | Value |
|---|---|
| arXiv | 2608.16157 (cs.DC) — note 2608 = 2026-08 |
| URL | `https://arxiv.org/abs/2608.16157` |
| Title | FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution |
| Authors | Shuo Yang\*, Xiaoze Fan\*, Melissa Pan, Haocheng Xi, Zhe Wang, Shanlin Sun, Kurt Keutzer, Song Han, Matei Zaharia, Chenfeng Xu†, Ion Stoica† (\* equal contribution, † co-advisors) |
| Affiliations (verified) | UC Berkeley (Yang, Pan, Xi, Keutzer, Zaharia, Stoica); MIT EECS / EfficientML.ai (Han); UT Austin (Fan) — third-party write-ups citing "Stanford" are not verified |
| Repo | `https://github.com/FlashML-org/FreeToken` (Apache-2.0) |
| Project page | `https://www.flashml.ai` |
| Packaging | `uv pip install "freetoken[accel]"` (Python); internals language 未能验证 |

### §1.2 Principle (one sentence)

A personal computer — GPU VRAM + CPU host RAM — is treated as one elastic inference platform. FreeToken dynamically decides which experts run on the GPU vs CPU using a closed-form bandwidth-adaptive policy `q★ = m·B_P / B_H` (where `B_P` and `B_H` are measured PCIe/host bandwidths from a one-time calibration), runs the prefill stage with full-layer double-buffered prefetch to hide PCIe transfer latency, and caches experts in an LRU keyed by measured routing-locality rather than expert ID alone.

### §1.3 Headline numbers (paper-cliff summary table, all author-measured)

| Hardware tier | Model | FreeToken decode | Comparators |
|---|---|---|---|
| RTX 4060 laptop (PCIe ×8, 8 GB VRAM) | Qwen3.6-35B-A3B (NVFP4) | **39.3 tok/s** ≈ 92% of RTX 4090 rate | Codex hosted ~33 tok/s |
| RTX 5090 desktop (32 GB VRAM, ~192 GB DDR5) | Qwen3.6-35B-A3B / DSV4-Flash-0731 284B (MXFP4) | **77–83 tok/s** (1.8–2.3× baselines); DSV4-Flash **22–25 tok/s** | llama.cpp, Ollama, KTransformers, MoE-Infinity |
| RTX PRO 6000 workstation (96 GB VRAM, ~512 GB RAM) | GLM-5.2 753B-A40B (NVFP4, 433 GB checkpoint) | **~2× llama.cpp** | llama.cpp |

**Decode gain envelope**: 1.3× to 2.1× the *strongest* baseline in each workload across five consumer systems (1.3× on RTX 3090, 1.3× on RTX 4090, 1.9× on RTX 5090 server, 2.1× on RTX 5090 desktop, 1.8× on RTX 4060 laptop).

**Prefill**: 8,192-token prefill chunk completes in **1.19–1.22 s** (transfer-bound at 64.4 GB / 52.7 GB/s PCIe 5.0 ×16 ceiling); throughput climbs to **~6.7k tok/s** at 16k-token prompts.

**Tail TTFT (availability boundary, multi-turn agent traces)**: FreeToken's worst round stays **<44 s** in every load cell; baselines cross 150 s (llama.cpp 232 s, Ollama 179 s, KTransformers 946 s).

### §1.4 Caveat — the "3-4× decode / 6-30× prefill" framing

The specific multipliers in the user's prompt — **3-4× decode and 6-30× prefill** — **are not the paper's headline pair**. The paper reports 1.3-2.1× decode vs the strongest baseline per tier. Larger multipliers appear in third-party write-ups (Stork.ai, TowardsAI, Marktechpost, Zenn) only when comparing against PyTorch's naïve offloading or other weak baselines. **Treat "3-4× / 6-30×" as third-party summaries, not paper claims.** Our §3 will quote only the verified paper numbers.

### §1.5 Relevance to flyingfish — **high**, direct comparator

- **Same target hardware class** as `ff glm` / `ff edge0` (consumer or single-workstation GPU + substantial host RAM); same MoE workload regime (Qwen3-class, GLM-class, DS-V4-class). The paper shows KTransformers degrades 31% between workload W1 and W2 and hits 946 s tail TTFT vs FreeToken's <44 s — a differentiator that matters more than headline throughput.
- **Same calibration idea** as `ff bench io --profile local-interconnect`. `docs/models.md:41`: "B_P 18.798 GiB/s and B_H 31.248 GiB/s … B_P/B_H 0.60 — a 40% host share of a miss set". FreeToken's `q★ = m·B_P / B_H` is the same "measure the bandwidth, then split residency" idea. The differentiator is dynamic routing-locality weighted LRU vs `ff`'s static LFU/LRU.
- **Possible integration with `crates/ff-glm/src/expert_cache.rs`**: FreeToken's cache is also LRU but operates per-expert ID with measured routing-locality weighting. Could complement (not replace) the existing per-projection LFU cache.
- **Add F3 to `docs/comparable-products.md` §2** (alongside FlexGen / KTransformers / PowerInfer) as a paper-only entry. License (Apache-2.0) allows reading; engine internals are likely C++/CUDA and would require substantial work to rewrite against `candle`. Not vendoring.

## §2 Speculative decoding family

### §2.1 Speculative decoding — disambiguation

| Technique | arXiv | Date | Draft source | Training-free? | Approximate speedup | URL / code |
|---|---|---|---|---|---|---|
| **Medusa-1** | 2401.10774 | 2024-01-19 | Parallel decoding heads on target LM | No (heads) | >2.2× | `github.com/FasterDecoding/Medusa` |
| **Medusa-2** | 2401.10774 v3 | 2024-06-14 | Medusa + backbone joint training | No | 2.3-3.6× | same |
| **EAGLE-1** | 2401.15077 | 2024-01-26 (v3 2025-03-04) | Feature-level autoregression on 2nd-to-top features with token-shift | No (small head) | 2.7-3.5× (LLaMA2-70B); 2.1-3.8× on MT-bench; **ICML 2024** | `github.com/SafeAILab/EAGLE` |
| **EAGLE-2** | 2406.16858 | 2024-06-24 (v2 2024-06-30) | EAGLE-1 + dynamic draft tree from per-position confidence | Inherits EAGLE-1 | 3.05-4.26× (20-40% over EAGLE-1); **EMNLP 2024** | same |
| **EAGLE-3** | 2503.01840 | 2025-03-03 (v3 2025-05-23) | Multi-layer feature fusion + direct token prediction (training-time test) | No (new head) | **up to 6.5×**; 1.38× in SGLang batch 64; **NeurIPS 2025** | same |
| **Lookahead** | 2402.02057 | 2024-02-03 | Jacobi iteration on LM's next-token map | **Yes** | 1.5-2.3× on LLaMA-2 single GPU; "up to 1.8× on MT-bench, 4× with strong multi-GPU"; **ICML 2024** | `github.com/hao-ai-lab/LookaheadDecoding` |
| **REST** | 2311.08252 | 2023-11-14 (v2 2024-04-04) | Datastore retrieval (n-gram match) with tree-attention verifier | **Yes** | 1.62-2.36× on LLaMA-7B/13B; **NAACL 2024** | `github.com/FasterDecoding/REST` |
| **Ouroboros** | 2402.13720 | 2024-02-21 (v3 2024-10-15) | Phrase book built online from LM outputs | **Yes** | up to 2.8× vs vanilla SD; 3.9× vs vanilla; **EMNLP 2024** | `github.com/thunlp/Ouroboros` |
| **MTP** (DeepSeek-V3) | 2412.19437 §2.3.3 | 2024-12-27 | Trained-in MTP head (self-speculative) | No (objective) | vendor-reported 1.2-2.1× inference; DeepSeek tech report claim 3× with MTP speculation | `github.com/deepseek-ai/DeepSeek-V3` (model + reference); vLLM integration (gate not yet extracted) — **未能验证** exact PR |

### §2.2 Each — one-line principle + measured-vs-claimed

- **Medusa**: extra decoding heads predict k future tokens in parallel; tree-attention verifier keeps longest accepted prefix. *Author-measured* on Vicuna 7B/13B/33B + Zephyr-7B + Mistral-7B-v0.2 (paper §Experiments); *third-party* claim "1.5-1.6× faster than Medusa" appears in EAGLE repo on 13B.
- **EAGLE-1**: autoregressively draft the *second-to-top-layer feature* with token embedding shift. *Author-measured* on Vicuna/LLaMA2-Chat series + Mixtral 8×7B; training takes 1-2 days on 8× RTX 3090.
- **EAGLE-2**: dynamic draft trees grown wider where per-position confidence predicts higher acceptance. *Author-measured*; same hardware as EAGLE-1.
- **EAGLE-3**: multi-layer feature fusion via training-time test; direct token prediction. *Author-measured*; same hardware. Published EAGLE-3 weights: Vicuna-13B-v1.3, LLaMA-3.1-8B-Instruct, LLaMA-3.3-70B-Instruct, **DeepSeek-R1-Distill-LLaMA-8B, LLaMA-4-Scout/Maverick, Qwen3-{1.7B, 4B, 8B, 14B, 30B-A3B, 32B, 235B-A22B}**, MiniCPM4-8B (third-party), GLM-4.7-Flash (third-party). **No official heads for our exact models.**
- **Lookahead**: Jacobi iteration parallel decoding + n-gram pool. *Author-measured*. Repo only ships LLaMA support.
- **REST**: trie-based retrieval from a ShareGPT/UltraChat datastore. *Author-measured*. Datastores: 465 MB (chat) to 27 GB (code) on disk; retrieval needs 96 CPU cores. Hardware: NVIDIA A6000 48 GB.
- **Ouroboros**: phrase-book drafts over an existing small+target pair. *Author-measured*. Repo only ships LLaMA plumbing.
- **MTP** (Gloeckle et al. + DeepSeek-V3): training-time heads predict next k tokens. DeepSeek-V3 claims MTP integration enables speculative decoding at inference; primary inference speedup value 未能验证 for the consumer-GPU tier.

### §2.3 Relevance matrix (each entry vs `ff`)

| Technique | Retraining required? | Auxiliary checkpoints? | `spec.rs` (`crates/ff-qwen35/src/spec.rs`) fit | CLI surface | Verdict |
|---|---|---|---|---|---|
| Medusa | yes | yes (heads) | tree-attention verifier over N columns — substantial rewrite | none in our adapters | out-of-scope |
| EAGLE-1 | yes | yes | architecturally closest to MTP path (feature→token verify round) | `ff qwen35` only if we ship our own head | comparable; needs head training |
| EAGLE-2 | inherits EAGLE-1 | yes + tree builder | requires N-wide tree verifier + confidence broadcast — kernel change | as above | competing; needs head + verifier rewrite |
| **EAGLE-3** | yes | yes + multi-layer fusion | training-time-test mechanism mirrors our MTP path; verify round kernel reusable; only `QwenSpec::draft` re-derivation | **closest drop-in** if unofficial `thoughtworks/GLM-4.7-Flash-Eagle3` aligns with `ff glm`, and if AngelSlim/Zjcxy-SmartAI Qwen3-32B heads align with `ff edge0` | **watchlist — competing on the MTP niche** |
| Lookahead | no | no | orthogonal; new kernel with 2D windowed attention | `--lookahead` flag (any adapter); per-model wiring | complementary; 1.5-2.3× is below our MTP bar but training-free |
| REST | no | datastore (465 MB – 27 GB) | tree verifier kernel; not paired-verify rounds | `--datastore-path` flag; build-time + host-CPU cost | complementary; useful for chat/code agents |
| **Ouroboros** | no | no (phrase pool) | wraps existing small+target pair; natural extension of `QwenSpec::draft` + future DSpark V4.1 `mtp.0..2` | `ff qwen35 --draft-model` / `ff-dsv41` | **complementary, most actionable training-free technique** |
| MTP (DeepSeek-V3) | yes (built-in to checkpoints) | yes | **this is the baseline we've already implemented** in `QwenSpec::verify_round` + future `ff-dsv41` mtp.0..2 | already shipped in `ff qwen35`, in `ff-dsv41` design | **comparable — this is `ff`'s own MTP path** |

### §2.4 Most actionable finding — **Ouroboros** for the MTP niche

Ouroboros is the most directly applicable training-free technique in this survey to flyingfish's existing MTP/DSpark path:

- It wraps an existing small-model + target-model pair; if `QwenSpec::draft` already runs an MTP draft, an Ouroboros pre-pass would lengthen the trees QwenSpec verifies.
- Only LLaMA plumbing ships today; a port to Qwen3.5 and DSpark-V4.1 would be needed, but the phrase-pool itself is model-agnostic.
- Best complement to the `ff-dsv41` 3-layer DSpark MTP (not yet implemented in `crates/ff-dsv41`).

## §3 KV cache compression

### §3.1 Disambiguation (note arXiv ID corrections)

> The prompt's arXiv IDs for SnapKV (`2404.19496`) and Quest (`2406.06125`) do **not** correspond to those papers. Correct IDs verified by reading each paper's arXiv abstract page directly.

| Method | arXiv | Date | Venue | Compression principle | Stage | Headline metric | Code |
|---|---|---|---|---|---|---|---|
| **SnapKV** | **2404.14469** (not 2404.19496) | 2024-04-22 | **NeurIPS 2024** | Per-head clustered KV selection from observation window | Prefill + Decode | 3.6× speed, 8.2× memory at 16K tokens (A100); 380K-token context on single A100 | `github.com/SnapKV/SnapKV` |
| **H2O** | 2306.14048 | 2023-06-24 | **NeurIPS 2023** | Heavy-hitter + recent-token eviction (dynamic submodular) | Decode | up to 29× throughput vs DS-ZI on T4 (OPT-6.7B/30B); 1.9× latency reduction | `github.com/FMInference/H2O` |
| **Quest** | **2406.10774** (not 2406.06125) | 2024-06-16 | **ICML 2024** | Page-level query-aware Top-K KV load | Decode | **7.03× attention kernel, 2.23× end-to-end on NVIDIA RTX 4090 and Ada 6000 (CUDA 12.4)** | `github.com/mit-han-lab/Quest` |
| **StreamingLLM** | 2309.17453 | 2023-09-29 | **ICLR 2024** | Sink tokens + sliding window | Decode | 22.2× vs recompute baseline; stable LM up to 4M+ tokens | `github.com/mit-han-lab/streaming-llm` |
| **PyramidKV** | 2406.02069 | 2024-06-04 | **COLM 2025** | Layer-wise pyramidal budget allocation | Prefill + Decode | 12% of cache matches full-cache LongBench; +20.5 absolute accuracy on TREC at 0.7% cache | `github.com/IsaacRe/PyramidKV` (+ unified `KVCache-Factory`) |

### §3.2 Each — measured-vs-claimed status

- **SnapKV**: *author-measured*. HuggingFace reference impl. Per-cluster eviction window sized from an observation phase. Specific PPL-retention tables at matched 8×/16×/32× compression ratios exist in the appendix but were not extracted — **未能验证** exact table.
- **H2O**: *author-measured*. Two reference impls (`h2o_flexgen`, `h2o_hf`). Hardware: T4 (FlexGen comparison); A100 (some evaluations).
- **Quest**: *author-measured*. **The only KV-compression paper with an RTX 4090 row.** Page-level kernel means it integrates cleanly with our `candle`-on-CUDA pipeline if we ever need to do per-batch selective KV loading.
- **StreamingLLM**: *author-measured*. Sink-token observation generalizes to all transformers; stable long-context training-free.
- **PyramidKV**: *author-measured*. Headline framing is "12% cache ≈ full-cache LongBench", which is not the same as a per-ratio PPL curve. Hardware model **未能验证** from abstract.

### §3.3 Relevance to flyingfish

| Method | Our long-context workload? | `ff` integration cost | Verdict |
|---|---|---|---|
| SnapKV | Yes — H3 video decoder's KV cache, GLM long-context | moderate (observation phase + per-cluster eviction) | watchlist |
| H2O | mixed — Dynamic heavy-hitter eviction fits long-context chat/code | moderate | watchlist |
| **Quest** | **strongly aligned** — RTX 4090 numbers are our exact hardware; page-level kernel matches our batched CUDA attention | moderate (page-level eviction + selective load) | **highest relevance among KV papers for `ff` MoE text path** |
| StreamingLLM | useful if `ff` runs >100K context | small (sink + sliding window) | complementary |
| PyramidKV | mixed — strongest on multi-layer budget | moderate (per-layer pyramid table) | watchlist |

## §4 VLM visual-token pruning (covers the original F1 slot)

### §4.1 Why this section exists

The user's hint about "FreeToken: Turning Ill-Posed to Well-Posed for Token Reduction" matches the *category* of VLM visual-token pruning. The paper itself (F1, arXiv 2407.14464) does not exist in any verifiable primary source. This section fills the slot with verified adjacent papers that target the same operational problem (pruning visual input tokens before the LLM portion) and notes which would be most actionable for `ff-qwen35`.

### §4.2 Verified adjacent papers

| Paper | arXiv | Date | Venue | Approach |
|---|---|---|---|---|
| **FastV** | `ojs.aaai.org/.../32278` | 2024 | AAAI 2025 | Attention-based pruning of low-attention visual tokens at early layers |
| **SparseVLM** | 2410.04417 | 2024-10 | NeurIPS 2024 | Self-aware sparsification with text-guided scoring |
| **VisionZip** | 2407.05279 | 2024-07 | (open) | Importance + redundancy pruning |
| **FitPrune** | 2403.06430 | 2024-03 | CVPR 2025 | Fast training-free pruning using [CLS] attention |
| **DivPrune** | 2408.01800 | 2024-08 | CVPR 2025 | Diversity-based pruning |
| **FasterVLM** | 2412.01818 | 2024-12 | (open) | Pure [CLS]-attention-based scoring |
| **ZOO-Prune** | 2509.24837 | 2025-09 | (open) | Zeroth-order gradient estimation, training-free |

Aggregator: `github.com/ZLKong/Awesome-Collection-Token-Reduction`.

### §4.3 Most actionable for `ff`

**FastV** or **VisionZip** are the most likely small ports:

- The vision tower in `ff-qwen35` (`crates/ff-qwen35/src/vision.rs`) is a standard Qwen3-VL port (preprocessing + ViT + merger); preprocessing already uses `smart_resize` with token budgets and spatial merging.
- Adding a FastV-style post-merge pruning layer sits after the merger and before the LLM stack. Estimated cost: ~100-300 LoC of importance scorer + truncation.
- **Caveat**: the candidate papers all require re-running the visual-tower forward (because pruning depends on the ViT's last-layer attention pattern), which negates some of the speedup in a single-pass deployment.
- For H3's text encoder (`crates/ff-h3/src/text_encoder.rs`): **not applicable** — text encoders consume text tokens, not visual ones.
- `ff glm` and `ff edge0`: not applicable (no multimodal path yet).

## §5 Byte-level / tokenization-free language models

| Model | Year | Venue | Backbone | Native unit | Headline result | Code |
|---|---|---|---|---|---|---|
| **ByT5** (Xue et al., Google) | 2021 | TACL 2022 (2105.13626) | Encoder-decoder Transformer (T5) | UTF-8 bytes (vocab 256) | competitive with mT5; robust to noise; strong on character-level | `github.com/google-research/byt5` |
| **MEGABYTE** (Yu et al., Meta) | 2023 | 2305.07185 | 2-scale Transformer (local + global) | fixed byte patches | competitive on long contexts; per-patch parallelism | `github.com/facebookresearch/metaseq` |
| **MambaByte** (Wang et al., Cornell) | 2024 | COLM 2024 (2401.13660) | Selective SSM (Mamba) | bytes (vocab 256) | 33.0 vs 36.4 PPL on PG-19; **+2.6× inference via byte-level speculative decoding** | `github.com/jxiw/MambaByte` |
| **BLT** (Pagnoni et al., Meta FAIR) | 2024 | 2412.09871 | Encoder + global Transformer + decoder with entropy patcher | dynamic byte patches | **matches Llama 3 at fixed inference FLOPs**; better compute-optimal scaling past ~8B | `github.com/facebookresearch/blt` |

> Author correction: BLT's authors are Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer, Ghosh, Lewis, Holtzman, Iyer — **not** Beltagy (Beltagy is the Longformer author).

### §5.1 Relevance to flyingfish

Low for all four. `ff` is byte/token-agnostic on the LM side — our text adapters consume pre-tokenized tensors; tokenizer choice is upstream of our forward pass. **The relevance is in the landscape signal**: byte-level inference at Llama-3 scale is now competitive (BLT), so the "byte-level eats tokenization" trajectory is a 12-24-month watch item but not an `ff` integration priority today.

## §6 Relevance to flyingfish — consolidated matrix

| Entry | Hardware class | Current `ff` integration | Integration cost | Recommended action |
|---|---|---|---|---|
| **F3 FreeToken (edge-MoE)** | consumer GPU + 8-512 GB host RAM | none (paper only) | high (C++/CUDA rewrite vs `candle`); calibrate via existing `ff bench io` | **add to `docs/comparable-products.md` §2 as paper-only comparator** |
| Medusa | consumer GPU | none | very high (head training) | skip |
| EAGLE-1/2/3 | consumer GPU | partial (EAGLE-3 has Qwen3-32B heads; we have Qwen3.5) | moderate (port from training heads) | watchlist; pick if a head aligns with `ff qwen35`/`ff edge0` |
| Lookahead | any GPU | none | high (new 2D-windowed attention kernel) | skip |
| REST | A6000 + 96 cores | none | high (tree-attention kernel + datastore) | skip unless code-agent use case emerges |
| Ouroboros | any GPU | none (port over existing MTP path) | moderate (phrase-pool integrator) | **most actionable training-free technique** |
| SnapKV / H2O / Quest / StreamingLLM / PyramidKV | A100 / RTX 4090 | none | moderate (per-method) | **Quest**: highest value among KV papers (RTX 4090 baseline) |
| FastV / VisionZip | consumer GPU, VLM path | none | small (post-merge pruning layer) | small watchlist; relevant only if `ff h3` enables a multimodal path |
| ByT5 / MEGABYTE / MambaByte / BLT | various | none (tokenizer upstream) | very high (different tokenizer regime) | watch item only |

## §7 Data limitations — 未能验证

| Item | Why |
|---|---|
| F1 paper "FreeToken: Be Careful of Your Tokens" by Chaoyou Fu, NeurIPS 2024, arXiv 2407.14464 | arXiv ID maps to AttentNet; no such paper on arXiv / OpenReview / NeurIPS / GitHub |
| F1 implementation surface (3 candidate GitHub repos all 404) | primary source not located |
| F4 (FreeToken referenceless IQA) | arXiv ID / venue could not be located |
| F3 "3-4× decode / 6-30× prefill" headline pair | paper reports 1.3-2.1× decode vs strongest baseline per tier; "3-4×" / "6-30×" are third-party summaries |
| F3 internals language | Python packaging confirmed; C++/CUDA internals inferred |
| SnapKV exact PPL-retention at matched 8×/16×/32× compression | appendix tables only |
| PyramidKV exact GPU model in headline experiments | abstract doesn't specify; likely A100 via KVCache-Factory |
| Specific GPU model for Medusa-1/2, Lookahead, REST, Ouroboros headline numbers | not in abstracts |
| MTP inference speedup 1.2-2.1× | secondary source (AMD ROCm docs); not in arXiv body |
| EAGLE-3 official heads for `ff`'s exact models (Qwen3.5, Edge0/Qwen3.8, GLM-5.3-Flash, H3, MiniCPM5, DeepSeek-V4.1) | only unofficial third-party heads exist |
| BLT 8B third-party reproductions | none confirmed |
| Ouroboros port to non-LLaMA backbones (Qwen, GLM, etc.) | only LLaMA plumbing shipped |
| `FMedusa/FasterTransformer-Medusa` repo (mentioned in initial brief) | the canonical Medusa repo is `FasterDecoding/Medusa`; the originally cited path could not be loaded |
| exact vLLM MTP feature gate/PR | discussed on vLLM forum but specific PR not extracted |
| KTransformers benchmark on RTX 4090 + AMD AMX node from independent third party | found at `thakicloud.com`; consistent with paper |

## §8 Source URLs (primary, all verified where noted)

### §8.1 FreeToken
- F3 verified: `arxiv.org/abs/2608.16157`; `github.com/FlashML-org/FreeToken`; `flashml.ai`
- F2 verified: `arxiv.org/abs/2410.21335`
- F1, F4: **未能验证** (no primary source located)

### §8.2 Speculative decoding
- Medusa: `arxiv.org/abs/2401.10774`; `github.com/FasterDecoding/Medusa`
- EAGLE-1: `arxiv.org/abs/2401.15077`
- EAGLE-2: `arxiv.org/abs/2406.16858`
- EAGLE-3: `arxiv.org/abs/2503.01840`; `github.com/SafeAILab/EAGLE`
- Lookahead: `arxiv.org/abs/2402.02057`; `github.com/hao-ai-lab/LookaheadDecoding`
- REST: `arxiv.org/abs/2311.08252`; `github.com/FasterDecoding/REST`
- Ouroboros: `arxiv.org/abs/2402.13720`; `github.com/thunlp/Ouroboros`
- MTP (DeepSeek-V3): `arxiv.org/abs/2412.19437`
- MTP (Gloeckle et al.): `arxiv.org/abs/2404.19737`

### §8.3 KV compression
- SnapKV: `arxiv.org/abs/2404.14469`; `github.com/SnapKV/SnapKV`
- H2O: `arxiv.org/abs/2306.14048`; `github.com/FMInference/H2O`
- Quest: `arxiv.org/abs/2406.10774`; `github.com/mit-han-lab/Quest`
- StreamingLLM: `arxiv.org/abs/2309.17453`; `github.com/mit-han-lab/streaming-llm`
- PyramidKV: `arxiv.org/abs/2406.02069`; `github.com/IsaacRe/PyramidKV`

### §8.4 VLM visual-token pruning
- FastV: `ojs.aaai.org/index.php/AAAI/article/view/32278`
- SparseVLM: `arxiv.org/abs/2410.04417`
- VisionZip: `arxiv.org/abs/2407.05279`
- FitPrune: `arxiv.org/abs/2403.06430`
- DivPrune: `arxiv.org/abs/2408.01800`
- FasterVLM: `arxiv.org/abs/2412.01818`
- ZOO-Prune: `arxiv.org/abs/2509.24837`
- Aggregator: `github.com/ZLKong/Awesome-Collection-Token-Reduction`

### §8.5 Byte-level LMs
- ByT5: `arxiv.org/abs/2105.13626`; `github.com/google-research/byt5`
- MEGABYTE: `arxiv.org/abs/2305.07185`
- MambaByte: `arxiv.org/abs/2401.13660`; `github.com/jxiw/MambaByte`
- BLT: `arxiv.org/abs/2412.09871`; `github.com/facebookresearch/blt`

### §8.6 `ff`'s own integration anchors (cross-references)
- `crates/ff-qwen35/src/spec.rs` — current MTP speculative decode
- `src/cli/minicpm.rs` — MiniCPM5 + DSpark `--draft-model` interface
- `crates/ff-qwen35/examples/generate.rs` — margin-based draft gating example
- DSpark MTP for the `ff-dsv41` adapter — not yet implemented in `crates/ff-dsv41`
- `crates/ff-glm/src/expert_cache.rs` — GLM MoE LFU/LRU cache that FreeToken's routing-locality LRU could complement
- `docs/models.md` Performance table — baseline numbers used in `docs/comparable-products.md` §4.1

## §9 Reading guide

- **If you already have a working MTP path** (`ff qwen35`): §2.4 recommends **Ouroboros** as the most actionable training-free extension.
- **If you need KV-cache compression for long-context workloads** (H3 video decoder, GLM long-context): §3.3 marks **Quest** as highest relevance.
- **If the user mentioned "FreeToken" and meant a system**: §1 explains F3 is the only verified "FreeToken", with §1.5 detailing its direct comparability to `ff`'s MoE/edge serving stack.
- **If you need VLM pruning** (future multimodal path): §4.3 marks FastV/VisionZip as small-port candidates.
- **If `ff` ever extends past single-stream serving**: most techniques above are single-stream-focused; the multi-stream regime would require re-evaluation.
